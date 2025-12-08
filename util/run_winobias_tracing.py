import argparse
from read_and_process import read_winocoref_file
from scoring_and_patching_utils import (
    ModelAndTokenizer,
    trace_winobias_mcqa_style,
    compute_winobias_accuracy,
    format_winobias_as_mcqa,
    compare_mlp_vs_attn_patching,
    trace_winobias_attention_heads,
)
from vocab_projection_utils import (
    LlamaVocabProjector,
    analyze_winobias_bias_emergence,
    analyze_winobias_component_comparison,
    # analyze_attention_heads_for_bias,
)
import jsonlines
import time

#ADD HUGGINGFACE AUTHENTICATION HERE
import os
from huggingface_hub import login
import os
os.environ['HF_HOME'] = '/fs/clip-scratch/mvinodku/'
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pro_file", type=str, default="/fs/clip-ml/mvinodku/understanding_genai/data/pro_stereotyped_type1.txt.dev")
    parser.add_argument("--anti_file", type=str, default="/fs/clip-ml/mvinodku/understanding_genai/data/anti_stereotyped_type1.txt.dev")
    parser.add_argument("--output_file", type=str, default="results/winobias_causal_tracing_results.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--llama_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=10)
    
    parser.add_argument("--model_variant", type=str, default="baseline", 
                       choices=["baseline", "safety", "jailbreak"],
                       help="Model variant: baseline (M_B), safety (M_S), or jailbreak (M_J)")
    parser.add_argument("--safety_prompt_key", type=str, default="fair",
                       choices=["helpful", "responsible", "ethical", "fair"],
                       help="Which safety prompt to use when model_variant=safety")
    parser.add_argument("--jailbreak_prompt_key", type=str, default="roleplay",
                       choices=["roleplay", "character", "hypothetical", "fictional", 
                               "permission", "override", "academic", "contrast"],
                       help="Which jailbreak prompt to use when model_variant=jailbreak")
    
    parser.add_argument("--analysis_type", type=str, default="causal_trace",
                       choices=[
                           "causal_trace",
                           "vocab_projection",           # Coarse (full layer)
                           "vocab_projection_mlp",       # Fine-grained MLP only
                           "vocab_projection_attn",      # Fine-grained Attention only
                           "vocab_projection_mlp_vs_attn",  # Compare MLP vs Attention
                           "patching_mlp",
                           "patching_attn", 
                           "patching_mlp_vs_attn",
                           "patching_heads",
                       ],
                       help="Type of analysis to perform")
    parser.add_argument("--layers_to_trace", type=str, default=None,
                   help="Comma-separated layer indices for head analysis (e.g., '22,23,24,25')")
    parser.add_argument("--component_type", type=str, default="mlp",
                       choices=["mlp", "attn", "attn_heads"],
                       help="Component type for component-specific tracing")
    parser.add_argument("--vocab_k", type=int, default=10,
                       help="Number of top-k tokens to analyze in vocabulary projection")
    # parser.add_argument("--accuracy_only", action="store_true", help="If set, only computes accuracy, no causal tracing.")
    args = parser.parse_args()

    # 1. Load WinoBias/Coref examples
    examples = read_winocoref_file(args.pro_file, args.anti_file)
    # if args.max_examples is not None:
    examples = examples[:args.max_examples]

    # 2. Initialize model and tokenizer
    mt = ModelAndTokenizer(
        model_name=args.model_name,
        llama_path=args.llama_path
    )
    print(f"Loaded model: {mt}")

    projector = None
    if args.analysis_type.startswith("vocab_projection"):
        print("Initializing vocabulary projector...")
        projector = LlamaVocabProjector(mt)
        print(f"Projector initialized. Vocab size: {projector.vocab_size}, Hidden dim: {projector.hidden_dim}")

    # 3. Run analysis based on type
    with jsonlines.open(args.output_file, mode="w") as writer:
        for i, example in enumerate(examples):
            print(f"\n--- Processing Example {i+1}/{len(examples)} (Model: {args.model_variant}, Analysis: {args.analysis_type}) ---")
            if args.model_variant == "safety":
                print(f"Using safety prompt: '{args.safety_prompt_key}'")
            elif args.model_variant == "jailbreak":
                print(f"Using jailbreak prompt: '{args.jailbreak_prompt_key}'")
            print(example)

            start_time = time.time()
            
            # Choose analysis type
            if args.analysis_type == "causal_trace":
                result = trace_winobias_mcqa_style(
                    mt=mt,
                    example=example,
                    kind=None,
                    include_negatives=False,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )

            elif args.analysis_type == "patching_mlp":
                result = trace_winobias_mcqa_style(
                    mt=mt,
                    example=example,
                    kind="mlp",  # This is the key difference
                    include_negatives=False,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result["analysis_type"] = "patching_mlp"
                
            elif args.analysis_type == "patching_attn":
                result = trace_winobias_mcqa_style(
                    mt=mt,
                    example=example,
                    kind="attn",  # This is the key difference
                    include_negatives=False,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result["analysis_type"] = "patching_attn"
                
            elif args.analysis_type == "patching_mlp_vs_attn":
                result = compare_mlp_vs_attn_patching(
                    mt=mt,
                    example=example,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result["analysis_type"] = "patching_mlp_vs_attn"
                
            elif args.analysis_type == "patching_heads":
                # Parse layers to trace
                if args.layers_to_trace:
                    layers = [int(x) for x in args.layers_to_trace.split(",")]
                else:
                    layers = None  # Defaults to last 10 layers
                
                result = trace_winobias_attention_heads(
                    mt=mt,
                    example=example,
                    layers_to_trace=layers,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result["analysis_type"] = "patching_heads"
                
            elif args.analysis_type == "vocab_projection":
                # Coarse projection (full layer outputs)
                formatted_example = format_winobias_as_mcqa(
                    example,
                    prompt_type=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result = analyze_winobias_bias_emergence(
                    projector=projector,
                    formatted_example=formatted_example,
                    tokenizer=mt.tokenizer,
                    projection_type="coarse",
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection"
                result["model_variant"] = args.model_variant
                result["example_index"] = i
                
            elif args.analysis_type == "vocab_projection_mlp":
                # Fine-grained MLP projection
                formatted_example = format_winobias_as_mcqa(
                    example,
                    prompt_type=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result = analyze_winobias_bias_emergence(
                    projector=projector,
                    formatted_example=formatted_example,
                    tokenizer=mt.tokenizer,
                    projection_type="finegrained",
                    component="mlp",
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection_mlp"
                result["model_variant"] = args.model_variant
                result["example_index"] = i
                
            elif args.analysis_type == "vocab_projection_attn":
                # Fine-grained Attention projection
                formatted_example = format_winobias_as_mcqa(
                    example,
                    prompt_type=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result = analyze_winobias_bias_emergence(
                    projector=projector,
                    formatted_example=formatted_example,
                    tokenizer=mt.tokenizer,
                    projection_type="finegrained",
                    component="attn",
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection_attn"
                result["model_variant"] = args.model_variant
                result["example_index"] = i
                
            elif args.analysis_type == "vocab_projection_mlp_vs_attn":
                # Compare MLP vs Attention contributions
                formatted_example = format_winobias_as_mcqa(
                    example,
                    prompt_type=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result = analyze_winobias_component_comparison(
                    projector=projector,
                    formatted_example=formatted_example,
                    tokenizer=mt.tokenizer,
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection_mlp_vs_attn"
                result["model_variant"] = args.model_variant
                result["example_index"] = i
                
            else:
                raise ValueError(f"Unknown analysis type: {args.analysis_type}")
            
            end_time = time.time()
            print(f"Analysis time: {end_time - start_time:.2f} seconds")
            
            # Write result
            writer.write(result)    

            ########################################################################################################
            ##################################### PRINTING SUMMARY RESULTS #########################################
            ########################################################################################################

            if args.analysis_type == "causal_trace":
                if result.get("correct_prediction", False):
                    print("Prediction: CORRECT")
                elif result.get("skip_reason"):
                    print("Skipped:", result["skip_reason"])
                else:
                    print("Prediction: INCORRECT")

            elif args.analysis_type in ["patching_mlp", "patching_attn"]:
                component = "MLP" if "mlp" in args.analysis_type else "Attention"
                print(f"Component: {component}")
                if result.get("correct_prediction", False):
                    print("Prediction: CORRECT")
                else:
                    print("Prediction: INCORRECT/MIXED")
                probits_correct = result.get("probits_correct", [])
                probits_incorrect = result.get("probits_incorrect", [])
                if probits_correct and probits_incorrect:
                    diffs = [c - i for c, i in zip(probits_correct, probits_incorrect)]
                    max_diff = max(diffs)
                    max_layer = diffs.index(max_diff)
                    print(f"Max prob diff from {component}: {max_diff:.3f} at layer {max_layer}")
                    
            elif args.analysis_type == "patching_mlp_vs_attn":
                comparison = result.get("comparison", {})
                print(f"Dominant component: {comparison.get('dominant_component', '?')}")
                print(f"MLP max effect: {comparison.get('mlp_max_effect', 0):.3f} at layer {comparison.get('mlp_max_effect_layer', '?')}")
                print(f"Attn max effect: {comparison.get('attn_max_effect', 0):.3f} at layer {comparison.get('attn_max_effect_layer', '?')}")
                print(f"MLP/Attn ratio: {comparison.get('mlp_to_attn_ratio', 0):.2f}")
                
            elif args.analysis_type == "patching_heads":
                print(f"Layers traced: {result.get('layers_traced', [])}")
                print(f"Baseline logit diff: {result.get('baseline', {}).get('logit_diff', 0):.3f}")
                top_heads = result.get("top_heads_per_layer", {})
                for layer_idx, heads in list(top_heads.items())[:3]:  # Show top 3 layers
                    top_3 = heads[:3]
                    print(f"Layer {layer_idx} top heads: {[(h, f'{e:.3f}') for h, e in top_3]}")
                    
            elif args.analysis_type == "vocab_projection":
                print(f"Entities: {result.get('correct_entity', '?')} vs {result.get('other_entity', '?')}")
                pro_logit_diff = result.get("pro_logit_diff", [])
                anti_logit_diff = result.get("anti_logit_diff", [])
                if pro_logit_diff:
                    max_pro_diff = max(pro_logit_diff)
                    max_pro_layer = pro_logit_diff.index(max_pro_diff)
                    print(f"Pro-stereotyped: Max logit diff = {max_pro_diff:.3f} at layer {max_pro_layer}")
                if anti_logit_diff:
                    max_anti_diff = max(anti_logit_diff)
                    max_anti_layer = anti_logit_diff.index(max_anti_diff)
                    print(f"Anti-stereotyped: Max logit diff = {max_anti_diff:.3f} at layer {max_anti_layer}")
                    
            elif args.analysis_type in ["vocab_projection_mlp", "vocab_projection_attn"]:
                component = "MLP" if "mlp" in args.analysis_type else "Attention"
                print(f"Component: {component}")
                print(f"Entities: {result.get('correct_entity', '?')} vs {result.get('other_entity', '?')}")
                pro_logit_diff = result.get("pro_logit_diff", [])
                if pro_logit_diff:
                    max_diff = max(pro_logit_diff)
                    max_layer = pro_logit_diff.index(max_diff)
                    print(f"Max logit diff from {component} = {max_diff:.3f} at layer {max_layer}")
                    
            elif args.analysis_type == "vocab_projection_mlp_vs_attn":
                comparison = result.get("comparison", {})
                mlp_results = result.get("mlp_results", {})
                print(f"Entities: {mlp_results.get('correct_entity', '?')} vs {mlp_results.get('other_entity', '?')}")
                print(f"Dominant component: {comparison.get('dominant_component', '?')}")
                print(f"MLP max bias: {comparison.get('mlp_max_pro_logit_diff', 0):.3f} at layer {comparison.get('mlp_max_effect_layer', '?')}")
                print(f"Attn max bias: {comparison.get('attn_max_pro_logit_diff', 0):.3f} at layer {comparison.get('attn_max_effect_layer', '?')}")
                print(f"MLP/Attn ratio: {comparison.get('bias_ratio_mlp_to_attn', 0):.2f}")

            
            print(f"\nDone! Results saved to {args.output_file}")