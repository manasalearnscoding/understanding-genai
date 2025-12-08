import torch
import torch.nn.functional as F
from collections import defaultdict

try:
    from util import nethook
    NETHOOK_AVAILABLE = True
except ImportError:
    print("Warning: nethook not found. Fine-grained projection will not be available.")
    print("Clone MEMIT repo and add to PYTHONPATH: export PYTHONPATH=$PYTHONPATH:/path/to/memit/")
    NETHOOK_AVAILABLE = False


class LlamaVocabProjector:
    """
    Vocabulary projection for LLaMA models.
    Directly adapted from understanding_mcqa's approach.
    
    Supports both coarse (full layer) and fine-grained (MLP/Attn) projection.
    """
    
    def __init__(self, mt):
        """
        Args:
            mt: ModelAndTokenizer instance from your scoring_and_patching_utils
        """
        self.mt = mt
        self.model = mt.model
        self.tokenizer = mt.tokenizer
        self.num_layers = mt.num_layers
        
        # LLaMA-specific components - always use module, not weight tensor
        # Using module ensures PyTorch handles device placement automatically
        self.lm_head_module = self.model.lm_head  # Use module for forward pass
        self.final_norm = self.model.model.norm   # RMSNorm

        # Get shape info from module attributes (works even if weight is on meta device)
        if hasattr(self.model.lm_head, 'out_features'):
            self.vocab_size = self.model.lm_head.out_features
            self.hidden_dim = self.model.lm_head.in_features
        else:
            # Fallback: use tokenizer vocab size
            self.vocab_size = self.tokenizer.vocab_size
            # Try to infer hidden_dim from model config or use default
            if hasattr(self.model.config, 'hidden_size'):
                self.hidden_dim = self.model.config.hidden_size
            else:
                self.hidden_dim = 4096  # Default for LLaMA-7B
        
        self.num_layers = mt.num_layers
        
        # Build layer name mappings for hooks
        self._build_layer_names()
    
    def _build_layer_names(self):
        """Build layer name mappings for fine-grained hooks."""
        self.layer_names = {
            "full": [f"model.layers.{i}" for i in range(self.num_layers)],
            "mlp": [f"model.layers.{i}.mlp" for i in range(self.num_layers)],
            "attn": [f"model.layers.{i}.self_attn" for i in range(self.num_layers)],
        }
    
    # ==================== COARSE PROJECTION ====================
    
    def layer_decode(self, hidden_states, apply_ln=True):
        """
        Project hidden states at last token position to vocabulary space.
        Mirrors understanding_mcqa's LLaMAWrapper.layer_decode().
        
        Args:
            hidden_states: tuple of tensors from output_hidden_states=True
                          Each tensor is [batch, seq_len, hidden_dim]
            apply_ln: Whether to apply layer normalization before projection
            
        Returns:
            list of [vocab_size, batch] tensors, one per layer
        """
        logits = []
        for i, h in enumerate(hidden_states):
            h = h[:, -1, :].detach()  # [batch, hidden_dim] - last token only
            
            # Apply normalization (skip for last layer if already applied)
            if apply_ln:
                h = self.final_norm(h)
            
            # Project using module forward pass - this handles device placement automatically
            # When weights are offloaded to CPU, PyTorch will load them as needed
            with torch.inference_mode():
                l = self.lm_head_module(h)  # [batch, vocab_size]
                l = l.T.detach()  # [vocab_size, batch]
            
            logits.append(l)
        
        return logits
    
    def get_layerwise_logits(self, input_ids):
        """
        Get vocabulary logits at each layer for the last token position.
        Uses output_hidden_states=True (no hooks needed).
        
        Args:
            input_ids: Tokenized input [batch, seq_len]
            
        Returns:
            torch.Tensor of shape [num_layers+1, vocab_size, batch]
            (includes embedding layer as layer 0)
        """
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        
        logits = self.layer_decode(outputs.hidden_states, apply_ln=True)
        return torch.stack(logits)  # [num_layers+1, vocab_size, batch]
    
    # ==================== FINE-GRAINED PROJECTION ====================
    
    def get_layerwise_logits_finegrained(self, input_ids, component="mlp"):
        """
        Get vocabulary logits from specific components (MLP or attention) at each layer.
        Requires hooks to capture intermediate outputs.
        
        This shows what each component "wants to say" before being added to residual.
        
        Args:
            input_ids: Tokenized input [batch, seq_len]
            component: "mlp" or "attn"
            
        Returns:
            torch.Tensor of shape [num_layers, vocab_size, batch]
        """
        if not NETHOOK_AVAILABLE:
            raise RuntimeError("nethook not available. Cannot do fine-grained projection.")
        
        if component not in ["mlp", "attn"]:
            raise ValueError(f"Unknown component: {component}. Use 'mlp' or 'attn'.")
        
        layer_names = self.layer_names[component]
        
        # Run forward pass with hooks to capture outputs
        with torch.inference_mode(), nethook.TraceDict(
            self.model,
            layers=layer_names,
            retain_input=False,
            retain_output=True,
        ) as traces:
            _ = self.model(input_ids=input_ids)
        
        # Project each component's output to vocabulary
        component_logits = []
        for i, layer_name in enumerate(layer_names):
            # Get component output
            h = traces[layer_name].output
            
            # Handle tuple outputs (attention returns (output, attn_weights, ...))
            if isinstance(h, tuple):
                h = h[0]
            
            # Take last token position
            h = h[:, -1, :].detach()  # [batch, hidden_dim]
            
            # Apply final norm and project
            # Note: This projects the component output as if it were the final output
            # This is an approximation but matches the paper's approach
            normed = self.final_norm(h)
            # Use module forward pass instead of direct matmul
            with torch.inference_mode():
                logits = self.lm_head_module(normed)  # [batch, vocab_size]
                logits = logits.T.detach()  # [vocab_size, batch]
            component_logits.append(logits)
        
        return torch.stack(component_logits)  # [num_layers, vocab_size, batch]
    
    def get_attention_head_logits(self, input_ids, layer_idx):
        """
        Get vocabulary logits from individual attention heads at a specific layer.
        
        Following Eq. (3) in the paper:
        MHSA_output = sum_h (W_O_h @ head_h_output)
        
        Each head's weighted contribution can be projected independently.
        
        Args:
            input_ids: Tokenized input [batch, seq_len]
            layer_idx: Which layer to analyze
            
        Returns:
            torch.Tensor of shape [num_heads, vocab_size, batch]
        """
        if not NETHOOK_AVAILABLE:
            raise RuntimeError("nethook not available. Cannot do attention head projection.")
        
        # We need to hook inside the attention to get per-head outputs
        # For LLaMA, the attention computation is:
        #   1. Q, K, V projections
        #   2. Attention computation per head
        #   3. Concatenate heads
        #   4. Output projection (o_proj)
        
        # Hook the output of attention (before o_proj would be ideal, but after works too)
        layer_name = f"model.layers.{layer_idx}.self_attn.o_proj"
        
        # Get attention layer for accessing weights
        attn_layer = self.model.model.layers[layer_idx].self_attn
        num_heads = attn_layer.num_heads
        head_dim = attn_layer.head_dim
        
        # Hook to capture input to o_proj (which is the concatenated head outputs)
        with torch.inference_mode(), nethook.TraceDict(
            self.model,
            layers=[layer_name],
            retain_input=True,  # We want input to o_proj, not output
            retain_output=False,
        ) as traces:
            _ = self.model(input_ids=input_ids)
        
        # Get concatenated head outputs (input to o_proj)
        # Shape: [batch, seq_len, num_heads * head_dim]
        concat_heads = traces[layer_name].input
        if isinstance(concat_heads, tuple):
            concat_heads = concat_heads[0]
        
        # Take last token position
        concat_heads = concat_heads[:, -1, :].detach()  # [batch, num_heads * head_dim]
        
        # Split into individual heads
        batch_size = concat_heads.shape[0]
        head_outputs = concat_heads.view(batch_size, num_heads, head_dim)  # [batch, num_heads, head_dim]
        
        try:
            o_proj_weight = attn_layer.o_proj.weight.detach()
            
            # Check if it's a meta tensor (empty placeholder)
            if o_proj_weight.device.type == 'meta':
                raise RuntimeError("Weight is on meta device")
            
            # If we get here, weights are accessible - do per-head analysis
            o_proj_by_head = o_proj_weight.view(o_proj_weight.shape[0], num_heads, head_dim)  # [hidden_dim, num_heads, head_dim]
            
        except RuntimeError as e:
            # Weights not accessible - fall back to less granular analysis
            print(f"Warning: Cannot do per-head analysis ({e})")
            print("Falling back to full attention output")
            
            # Use module forward (always works)
            full_output = attn_layer.o_proj(concat_heads)
            normed = self.final_norm(full_output)
            logits = self.lm_head_module(normed).T.detach()
            
            # Return same result for all heads (less informative but doesn't crash)
            return logits.unsqueeze(0).expand(num_heads, -1, -1)


        '''
        # Get o_proj weight and split by heads
        # o_proj.weight shape: [hidden_dim, num_heads * head_dim]
        o_proj_weight = attn_layer.o_proj.weight  # [hidden_dim, num_heads * head_dim]
        '''
        
        # Project each head's output through its portion of o_proj, then to vocab
        head_logits = []
        for h in range(num_heads):
            # Get this head's output: [batch, head_dim]
            head_out = head_outputs[:, h, :]
            
            # Get this head's o_proj weights: [hidden_dim, head_dim]
            head_o_proj = o_proj_by_head[:, h, :]
            
            # Project to hidden dim: [batch, head_dim] @ [head_dim, hidden_dim] -> [batch, hidden_dim]
            head_hidden = torch.matmul(head_out, head_o_proj.T)
            
            # Apply norm and project to vocab
            normed = self.final_norm(head_hidden)
            # Use module forward pass instead of direct matmul
            with torch.inference_mode():
                logits = self.lm_head_module(normed)  # [batch, vocab_size]
                logits = logits.T.detach()  # [vocab_size, batch]
            head_logits.append(logits)
        
        return torch.stack(head_logits)  # [num_heads, vocab_size, batch]
    
    # ==================== ANALYSIS UTILITIES ====================
    
    def prob_of_token_per_layer(self, logits, token_id):
        """
        Get probability of a specific token at each layer.
        Mirrors understanding_mcqa's prob_of_answer_per_layer().
        
        Args:
            logits: [num_layers, vocab_size, batch] or [num_layers, vocab_size]
            token_id: int token ID to track
            
        Returns:
            list of probabilities, one per layer (averaged over batch if batched)
        """
        probs = F.softmax(logits, dim=1)  # Softmax over vocab dimension
        
        if probs.dim() == 3:
            # [num_layers, vocab_size, batch] -> average over batch
            return probs[:, token_id, :].mean(dim=-1).cpu().tolist()
        else:
            return probs[:, token_id].cpu().tolist()
    
    def logit_of_token_per_layer(self, logits, token_id):
        """
        Get logit of a specific token at each layer.
        Mirrors understanding_mcqa's log_of_answer_per_layer().
        """
        if logits.dim() == 3:
            return logits[:, token_id, :].mean(dim=-1).cpu().tolist()
        else:
            return logits[:, token_id].cpu().tolist()
    
    def logit_diff_per_layer(self, logits, token_id_a, token_id_b):
        """
        Get logit difference (a - b) at each layer.
        Useful for measuring bias: positive = favors token_a.
        """
        if logits.dim() == 3:
            diff = logits[:, token_id_a, :] - logits[:, token_id_b, :]
            return diff.mean(dim=-1).cpu().tolist()
        else:
            return (logits[:, token_id_a] - logits[:, token_id_b]).cpu().tolist()
    
    def topk_per_layer(self, logits, k=10, use_probs=True):
        """
        Get top-k tokens at each layer.
        Mirrors understanding_mcqa's topk_per_layer().
        
        Args:
            logits: [num_layers, vocab_size, batch]
            k: Number of top tokens to return
            use_probs: Whether to convert to probabilities first
            
        Returns:
            list of dicts: layerwise_topk[layer_idx][batch_idx] = [(token_str, value), ...]
        """
        if use_probs:
            values = F.softmax(logits, dim=1)
        else:
            values = logits
        
        layerwise_topk = []
        for layer_idx in range(values.shape[0]):
            layer_topk = {}
            
            if values.dim() == 3:
                batch_size = values.shape[2]
            else:
                batch_size = 1
                values = values.unsqueeze(-1)
            
            for batch_idx in range(batch_size):
                top_vals, top_ids = torch.topk(values[layer_idx, :, batch_idx], k)
                tokens = self.tokenizer.convert_ids_to_tokens(top_ids.cpu().tolist())
                layer_topk[batch_idx] = [
                    (tok, val.item()) for tok, val in zip(tokens, top_vals)
                ]
            layerwise_topk.append(layer_topk)
        
        return layerwise_topk
    
    def rr_per_layer(self, logits, token_id):
        """
        Get reciprocal rank of token at each layer.
        Mirrors understanding_mcqa's rr_per_layer().
        
        Args:
            logits: [num_layers, vocab_size, batch]
            token_id: Token ID to track
            
        Returns:
            list of reciprocal ranks per layer
        """
        probs = F.softmax(logits, dim=1)
        rrs = []
        
        for layer_idx in range(probs.shape[0]):
            if probs.dim() == 3:
                # Average RR over batch
                batch_rrs = []
                for batch_idx in range(probs.shape[2]):
                    sorted_ids = probs[layer_idx, :, batch_idx].argsort(descending=True)
                    rank = (sorted_ids == token_id).nonzero(as_tuple=True)[0].item()
                    batch_rrs.append(1.0 / (rank + 1))
                rrs.append(sum(batch_rrs) / len(batch_rrs))
            else:
                sorted_ids = probs[layer_idx, :].argsort(descending=True)
                rank = (sorted_ids == token_id).nonzero(as_tuple=True)[0].item()
                rrs.append(1.0 / (rank + 1))
        
        return rrs


# ==================== WINOBIAS-SPECIFIC ANALYSIS ====================

def analyze_winobias_bias_emergence(
    projector,
    formatted_example,
    tokenizer,
    projection_type="coarse",
    component="mlp",
    k=10
):
    """
    Analyze bias emergence for a WinoBias example using vocab projection.
    
    This is the WinoBias-specific analysis layer on top of the generic projector.
    
    Args:
        projector: LlamaVocabProjector instance
        formatted_example: output of format_winobias_as_mcqa()
        tokenizer: tokenizer for encoding
        projection_type: "coarse" (full layers) or "finegrained" (MLP/Attn)
        component: "mlp" or "attn" (only used if projection_type="finegrained")
        
    Returns:
        dict with layer-wise probability curves for correct vs incorrect entities
    """
    from scoring_and_patching_utils import make_inputs
    
    # Encode both pro and anti prompts
    pro_prompt = formatted_example['pro_prompt']
    anti_prompt = formatted_example['anti_prompt']
    inp = make_inputs(tokenizer, [pro_prompt, anti_prompt])
    
    # Get entity token IDs (first token only, matching single-token scoring)
    correct_entity = formatted_example['correct_entity']
    other_entity = formatted_example['other_entity']
    
    # Encode with space prefix (standard for LLaMA tokenization)
    correct_tokens = tokenizer.encode(" " + correct_entity, add_special_tokens=False)
    print(f"Tokens for ' {correct_entity}': {correct_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in correct_tokens]}")

    # Find the token that actually contains the entity
    for i, tok in enumerate(correct_tokens):
        decoded = tokenizer.decode([tok])
        if correct_entity.lower() in decoded.lower():
            correct_token_id = tok
            break
    else:
        correct_token_id = correct_tokens[-1]  # fallback to last

    other_tokens = tokenizer.encode(" " + other_entity, add_special_tokens=False)
    print(f"Tokens for ' {other_entity}': {other_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in other_tokens]}")

    for i, tok in enumerate(other_tokens):
        decoded = tokenizer.decode([tok])
        if other_entity.lower() in decoded.lower():
            other_token_id = tok
            break
    else:
        other_token_id = other_tokens[-1]

    print(f"correct_token: {correct_token_id} -> '{tokenizer.decode([correct_token_id])}'")
    print(f"other_token: {other_token_id} -> '{tokenizer.decode([other_token_id])}'")
    
    # Get layerwise logits based on projection type
    if projection_type == "coarse":
        logits = projector.get_layerwise_logits(inp["input_ids"])
    elif projection_type == "finegrained":
        logits = projector.get_layerwise_logits_finegrained(inp["input_ids"], component=component)
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")
    
    # Compute probabilities
    probs = F.softmax(logits, dim=1)  # [layers, vocab, batch]
    
    # Track entity probabilities across layers
    # Batch 0 = pro-stereotyped, Batch 1 = anti-stereotyped
    results = {
        # Probabilities
        "pro_correct_probs": [],
        "pro_incorrect_probs": [],
        "anti_correct_probs": [],
        "anti_incorrect_probs": [],
        # Logits
        "pro_correct_logits": [],
        "pro_incorrect_logits": [],
        "anti_correct_logits": [],
        "anti_incorrect_logits": [],
        # Differences (positive = correct entity favored)
        "pro_logit_diff": [],
        "anti_logit_diff": [],
        "pro_prob_diff": [],
        "anti_prob_diff": [],
    }
    
    for layer_idx in range(logits.shape[0]):
        # Pro-stereotyped (batch 0)
        pro_corr_prob = probs[layer_idx, correct_token_id, 0].item()
        pro_incorr_prob = probs[layer_idx, other_token_id, 0].item()
        pro_corr_logit = logits[layer_idx, correct_token_id, 0].item()
        pro_incorr_logit = logits[layer_idx, other_token_id, 0].item()
        
        # Anti-stereotyped (batch 1)
        anti_corr_prob = probs[layer_idx, correct_token_id, 1].item()
        anti_incorr_prob = probs[layer_idx, other_token_id, 1].item()
        anti_corr_logit = logits[layer_idx, correct_token_id, 1].item()
        anti_incorr_logit = logits[layer_idx, other_token_id, 1].item()
        
        # Store probabilities
        results["pro_correct_probs"].append(pro_corr_prob)
        results["pro_incorrect_probs"].append(pro_incorr_prob)
        results["anti_correct_probs"].append(anti_corr_prob)
        results["anti_incorrect_probs"].append(anti_incorr_prob)
        
        # Store logits
        results["pro_correct_logits"].append(pro_corr_logit)
        results["pro_incorrect_logits"].append(pro_incorr_logit)
        results["anti_correct_logits"].append(anti_corr_logit)
        results["anti_incorrect_logits"].append(anti_incorr_logit)
        
        # Store differences
        results["pro_logit_diff"].append(pro_corr_logit - pro_incorr_logit)
        results["anti_logit_diff"].append(anti_corr_logit - anti_incorr_logit)
        results["pro_prob_diff"].append(pro_corr_prob - pro_incorr_prob)
        results["anti_prob_diff"].append(anti_corr_prob - anti_incorr_prob)
    
    # Add top-k for debugging/inspection
    results["topk_per_layer"] = projector.topk_per_layer(logits, k=10)
    
    # Add metadata
    results["correct_entity"] = correct_entity
    results["other_entity"] = other_entity
    results["correct_token_id"] = correct_token_id
    results["other_token_id"] = other_token_id
    results["correct_token_str"] = tokenizer.decode([correct_token_id])
    results["other_token_str"] = tokenizer.decode([other_token_id])
    results["num_layers"] = logits.shape[0]
    results["projection_type"] = projection_type
    results["component"] = component if projection_type == "finegrained" else None
    results["pro_prompt"] = pro_prompt
    results["anti_prompt"] = anti_prompt
    
    return results


def analyze_winobias_component_comparison(
    projector,
    formatted_example,
    tokenizer,
):
    """
    Compare MLP vs Attention contributions to bias using vocab projection.
    
    This mirrors Figure 7 from the Understanding MCQA paper.
    
    Args:
        projector: LlamaVocabProjector instance
        formatted_example: output of format_winobias_as_mcqa()
        tokenizer: tokenizer for encoding
        
    Returns:
        dict with MLP vs Attention comparison
    """
    # Run analysis for each component
    mlp_results = analyze_winobias_bias_emergence(
        projector, formatted_example, tokenizer,
        projection_type="finegrained", component="mlp"
    )
    
    attn_results = analyze_winobias_bias_emergence(
        projector, formatted_example, tokenizer,
        projection_type="finegrained", component="attn"
    )
    
    # Compare components
    comparison = {
        "mlp_results": mlp_results,
        "attn_results": attn_results,
        "comparison": {
            # Which component shows stronger bias (larger logit diff)?
            "mlp_max_pro_logit_diff": max(mlp_results["pro_logit_diff"]),
            "attn_max_pro_logit_diff": max(attn_results["pro_logit_diff"]),
            "mlp_max_anti_logit_diff": max(mlp_results["anti_logit_diff"]),
            "attn_max_anti_logit_diff": max(attn_results["anti_logit_diff"]),
            # Layer where each component has max effect
            "mlp_max_effect_layer": mlp_results["pro_logit_diff"].index(
                max(mlp_results["pro_logit_diff"])
            ),
            "attn_max_effect_layer": attn_results["pro_logit_diff"].index(
                max(attn_results["pro_logit_diff"])
            ),
        }
    }
    
    # Determine which component drives bias more
    mlp_bias = abs(comparison["comparison"]["mlp_max_pro_logit_diff"])
    attn_bias = abs(comparison["comparison"]["attn_max_pro_logit_diff"])
    comparison["comparison"]["dominant_component"] = "mlp" if mlp_bias > attn_bias else "attn"
    comparison["comparison"]["bias_ratio_mlp_to_attn"] = mlp_bias / (attn_bias + 1e-8)
    
    return comparison


def analyze_attention_heads_for_bias(
    projector,
    formatted_example,
    tokenizer,
    layer_idx,
):
    """
    Analyze which attention heads contribute most to bias at a specific layer.
    
    This mirrors Figure 8 from the Understanding MCQA paper.
    
    Args:
        projector: LlamaVocabProjector instance
        formatted_example: output of format_winobias_as_mcqa()
        tokenizer: tokenizer for encoding
        layer_idx: Which layer to analyze
        
    Returns:
        dict with per-head bias analysis
    """
    from scoring_and_patching_utils import make_inputs
    
    # Encode prompts
    pro_prompt = formatted_example['pro_prompt']
    anti_prompt = formatted_example['anti_prompt']
    inp = make_inputs(tokenizer, [pro_prompt, anti_prompt])
    
    # Get entity token IDs
    correct_entity = formatted_example['correct_entity']
    other_entity = formatted_example['other_entity']
    correct_token_id = tokenizer.encode(" " + correct_entity, add_special_tokens=False)[0]
    other_token_id = tokenizer.encode(" " + other_entity, add_special_tokens=False)[0]
    
    # Get per-head logits
    head_logits = projector.get_attention_head_logits(inp["input_ids"], layer_idx)
    # Shape: [num_heads, vocab_size, batch]
    
    num_heads = head_logits.shape[0]
    probs = F.softmax(head_logits, dim=1)
    
    results = {
        "layer_idx": layer_idx,
        "num_heads": num_heads,
        "per_head_analysis": [],
    }
    
    for head_idx in range(num_heads):
        head_analysis = {
            "head_idx": head_idx,
            # Pro-stereotyped (batch 0)
            "pro_correct_logit": head_logits[head_idx, correct_token_id, 0].item(),
            "pro_incorrect_logit": head_logits[head_idx, other_token_id, 0].item(),
            "pro_correct_prob": probs[head_idx, correct_token_id, 0].item(),
            "pro_incorrect_prob": probs[head_idx, other_token_id, 0].item(),
            # Anti-stereotyped (batch 1)
            "anti_correct_logit": head_logits[head_idx, correct_token_id, 1].item(),
            "anti_incorrect_logit": head_logits[head_idx, other_token_id, 1].item(),
            "anti_correct_prob": probs[head_idx, correct_token_id, 1].item(),
            "anti_incorrect_prob": probs[head_idx, other_token_id, 1].item(),
        }
        
        # Compute bias metrics
        head_analysis["pro_logit_diff"] = (
            head_analysis["pro_correct_logit"] - head_analysis["pro_incorrect_logit"]
        )
        head_analysis["anti_logit_diff"] = (
            head_analysis["anti_correct_logit"] - head_analysis["anti_incorrect_logit"]
        )
        
        results["per_head_analysis"].append(head_analysis)
    
    # Find most biased heads
    pro_diffs = [h["pro_logit_diff"] for h in results["per_head_analysis"]]
    results["max_bias_head"] = pro_diffs.index(max(pro_diffs, key=abs))
    results["max_bias_value"] = max(pro_diffs, key=abs)
    
    # Summary: heads sorted by absolute bias
    sorted_heads = sorted(
        range(num_heads),
        key=lambda h: abs(results["per_head_analysis"][h]["pro_logit_diff"]),
        reverse=True
    )
    results["heads_by_bias_magnitude"] = sorted_heads[:10]  # Top 10
    
    return results