import os
import re
import jsonlines
import torch
import warnings
from collections import defaultdict
import os
os.environ['HF_HOME'] = '/fs/clip-scratch/mvinodku/'
# import accelerate
# import bitsandbytes as bnb
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )

try:
    from util import nethook
    print(f"Loaded nethook from {nethook.__file__}")
except ImportError:
    print("Warning: nethook not found. Make sure MEMIT repository is in your PYTHONPATH.")
    nethook = None

warnings.filterwarnings("ignore")


#########################################################################################################
######################################## DATA PREP/UTILITIES ############################################
#########################################################################################################

def clean_entity_name(entity):
    stopwords = ['the', 'a', 'an', 'The', 'A', 'An']
    words = entity.split()
    cleaned_words = [word for word in words if word not in stopwords]
    print(' '.join(cleaned_words))
    return ' '.join(cleaned_words)

def format_winobias_as_mcqa(example):
    clean_correct = clean_entity_name(example['correct_referent'])
    clean_other = clean_entity_name(example['other_entity'])
    def create_prompt(sentence, pronoun):
        return f"{sentence}. In this sentence, '{pronoun}' refers to"
    pro_prompt = create_prompt(example['pro_sentence'], example['pro_pronoun'])
    anti_prompt = create_prompt(example['anti_sentence'], example['anti_pronoun'])
    # print("pro_prompt": pro_prompt,
    #     "anti_prompt": anti_prompt,
    #     "correct_entity": clean_correct,
    #     "other_entity": clean_other,
    #     "original_correct": example['correct_referent'],
    #     "original_other": example['other_entity'])
    return {
        "pro_prompt": pro_prompt,
        "anti_prompt": anti_prompt,
        "correct_entity": clean_correct,
        "other_entity": clean_other,
        "original_correct": example['correct_referent'],
        "original_other": example['other_entity'],
    }

def make_inputs(
    tokenizer,
    prompts,
    device="cuda" if torch.cuda.is_available() else "cpu", #QUESTION : is this okay or will it crash the cpu device?
    add_special_tokens=True,
    truncate=False, # better way to pad?
):
    token_lists = [tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in prompts]
    # print(prompts)
    # print(token_lists)
    input_ids = token_lists
    try:
        r1 = torch.tensor(input_ids)
    except: #QUESTION : haven't had a chance to run this yet but is this a crude way of making them same length, how can i pad?
        if truncate:
            min_len = min([len(t) for t in input_ids])
            truncated_input_ids = [t[:min_len] for t in input_ids]
            try:
                r1 = torch.tensor(truncated_input_ids)
            except:
                breakpoint()
        else:
            raise Exception("input_ids are not the same length; must specify padding")
    return dict(input_ids=r1.to(device))

#########################################################################################################
######################################## ENCODING/SCORING ###############################################
#########################################################################################################

def encode_winobias_mcqa(tokenizer, formatted_example, accuracy_only=False, return_token_seqs=False):
    """
    Encode WinoBias/coref example as generative prompt, and get all entity token sequences.
    """
    pro_prompt = formatted_example['pro_prompt']
    anti_prompt = formatted_example['anti_prompt']
    correct_entity = formatted_example['correct_entity']
    other_entity = formatted_example['other_entity']

    #encode the prompts: base prompt length = 26 tokens
    if accuracy_only:
        full_input_strings = [pro_prompt]
    else:
        full_input_strings = [pro_prompt, anti_prompt]
    inp = make_inputs(tokenizer, full_input_strings)

    #encode the answers with the correct entities
    answer_strings = [
        pro_prompt + " " + correct_entity,
        pro_prompt + " " + other_entity,
    ]
    if not accuracy_only:
        answer_strings += [
            anti_prompt + " " + correct_entity,
            anti_prompt + " " + other_entity,
        ]
    full_input_encodings_with_answers = make_inputs(tokenizer, answer_strings)

    #Extracts answer tokens by slicing off the prompt portion from complete strings
    prompt_len = inp["input_ids"].shape[1]
    answer_token_seqs = []
    #for answers with multiple tokens like 'software engineer'
    for idx in range(len(answer_strings)):
        tokens = full_input_encodings_with_answers["input_ids"][idx][prompt_len:] 
        answer_token_seqs.append(tokens.tolist())
    # print(f'Answer token seqs = {answer_token_seqs}')
    # w/o multi token tokenizing: return first token of each sequence for legacy compatibility
    answer_encodings = [seq[0] if len(seq) > 0 else -1 for seq in answer_token_seqs]
    # print(f'Answer encodings = {answer_encodings}')
    
    if accuracy_only:
        return (
            inp,
            (answer_encodings[0], answer_encodings[1]),  # (correct, incorrect) tokens
            pro_prompt,
            correct_entity,
            other_entity,
            answer_token_seqs[:2] if return_token_seqs else None
        )
    else:
        return (
            inp,
            answer_encodings,  # [pro_correct, pro_incorrect, anti_correct, anti_incorrect]
            pro_prompt,
            anti_prompt,
            correct_entity,
            other_entity,
            answer_token_seqs if return_token_seqs else None
        )

def score_winobias_target(
    mt,  # Model and tokenizer object
    inp,  # Input dictionary with tokenized prompts
    answers_t,  # Answer token IDs [correct_token, incorrect_token]
    answer_token_seqs=None,  # Full token sequences for multi-token answers
    counterfactual=False  # Whether to compare two different prompts
):
    """
    Score entity referents for generative pronoun resolution.
    If answer_token_seqs is given, score all tokens in the sequence.
    """
    with torch.inference_mode():  # Disable gradient computation for efficiency
        outputs = mt.model(  # Run the language model forward pass
            input_ids=inp["input_ids"],  # Tokenized input prompts
            return_dict=True,
        )
    logits = outputs.logits  # Raw model predictions [batch, seq_len, vocab_size]
    probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities

    # --- Per-token log-probability collection ---
    logprob_info = []  # Store detailed scoring info for each token
    num_batches = logits.shape[0]  # Number of input prompts in batch
    #QUESTION : for each token, we are scoring the tokens individually right now, is there a better way to score multiple tokens
    if answer_token_seqs is not None:  # Only if multi-token scoring requested
        for b_idx in range(num_batches):  # For each prompt in the batch
            cur_logprob = []  # Log probabilities for this prompt's answer tokens
            context_len = inp["input_ids"][b_idx].shape[0] #token length of input prompt - 26
            # For each answer in this batch, score full multi tok answer sequence
            tokens = answer_token_seqs[b_idx] if b_idx < len(answer_token_seqs) else []  # Get answer tokens for this prompt
            # Each answer token is scored at successive positions (as if generating one token at a time)
            for t_idx, tok in enumerate(tokens):  # For each token in the answer sequence
                pos = context_len + t_idx  # Position where this token should appear
                if pos >= logits.shape[1]:  # Skip if position exceeds sequence length
                    break
                prob_row = probs[b_idx, pos, :]  # Model's probability distribution at this position
                # Log-prob for gold token
                logprob = torch.log(prob_row[tok] + 1e-12)  # Log probability of the correct token (add epsilon to avoid log(0))
                cur_logprob.append(logprob.item())  # Convert to Python float and store
            logprob_info.append(cur_logprob)  # Add this prompt's token scores to overall list

    # --- Binary comparison: correct vs incorrect entity probabilities ---
    correct_entity_token = answers_t[0]  # Token ID for correct answer (e.g., 13897 = "developer")
    incorrect_entity_token = answers_t[1]  # Token ID for incorrect answer (e.g., 23383 = "designer")
    last_token_logits = logits[:, -1, :]  # Model predictions at the final position (after "refers to")
    last_token_probs = probs[:, -1, :]  # Probabilities at the final position
    base_corr_prob = last_token_probs[0, correct_entity_token].item()  # Probability of correct entity for first prompt
    base_incorr_prob = last_token_probs[0, incorrect_entity_token].item()  # Probability of incorrect entity for first prompt
    base_corr_logit = last_token_logits[0, correct_entity_token].item()  # Raw logit for correct entity
    base_incorr_logit = last_token_logits[0, incorrect_entity_token].item()  # Raw logit for incorrect entity
    base_probs = (base_corr_prob - base_incorr_prob, base_corr_prob, base_incorr_prob)  # (difference, correct, incorrect)
    base_logs = (base_corr_logit - base_incorr_logit, base_corr_logit, base_incorr_logit)  # Same for logits

    if counterfactual and last_token_logits.shape[0] > 1:  # If comparing two prompts (e.g., "he" vs "she")
        if len(answers_t) >= 4:  # If we have separate tokens for the second prompt
            contrast_corr_token = answers_t[2]  # Correct token for second prompt
            contrast_incorr_token = answers_t[3]  # Incorrect token for second prompt
        else:  # Otherwise use same tokens as first prompt
            contrast_corr_token = correct_entity_token
            contrast_incorr_token = incorrect_entity_token
        contrast_corr_prob = last_token_probs[1, contrast_corr_token].item()  # Prob of correct entity for second prompt
        contrast_incorr_prob = last_token_probs[1, contrast_incorr_token].item()  # Prob of incorrect entity for second prompt
        contrast_corr_logit = last_token_logits[1, contrast_corr_token].item()  # Raw logit for correct entity (second prompt)
        contrast_incorr_logit = last_token_logits[1, contrast_incorr_token].item()  # Raw logit for incorrect entity (second prompt)
        counterfactual_probs = (  # Same format as base_probs but for second prompt
            contrast_corr_prob - contrast_incorr_prob,  # Difference between correct and incorrect
            contrast_corr_prob,  # Probability of correct entity
            contrast_incorr_prob,  # Probability of incorrect entity
        )
        counterfactual_logs = (  # Same format as base_logs but for second prompt
            contrast_corr_logit - contrast_incorr_logit,  # Difference in logits
            contrast_corr_logit,  # Logit for correct entity
            contrast_incorr_logit,  # Logit for incorrect entity
        )
        return (base_probs, base_logs, logprob_info), (counterfactual_probs, counterfactual_logs, logprob_info)  # Return both base and counterfactual results
    return base_probs, base_logs, logprob_info  # Return only base results if no counterfactual comparison

def compute_winobias_accuracy(mt, example):  # Model+tokenizer, WinoBias example dict
    """
    Compute model accuracy and per-token log-probs for a generative WinoBias/coref example.
    """
    formatted_example = format_winobias_as_mcqa(example)  # Convert to MCQA format with prompts
    (  # Unpack the encoding results
        inp,  # Tokenized input dictionary
        answers_t,  # Answer token IDs (correct, incorrect)
        pro_prompt,  # The stereotyped prompt string
        correct_entity,  # Correct entity name ("developer")
        other_entity,  # Incorrect entity name ("designer")
        answer_token_seqs  # Full token sequences for answers
    ) = encode_winobias_mcqa(  # Encode the example for scoring
        mt.tokenizer,  # Tokenizer to use
        formatted_example,  # Formatted example dict
        accuracy_only=True,  # Only process stereotyped prompt, not anti-stereotyped
        return_token_seqs=True  # Return full token sequences for multi-token support
    )
    base_probs, base_logs, logprob_info = score_winobias_target(  # Get model's predictions
        mt,  # Model and tokenizer
        inp,  # Tokenized inputs
        answers_t,  # Answer token IDs to score
        answer_token_seqs=answer_token_seqs,  # Full sequences for detailed scoring
        counterfactual=False  # Don't compare with anti-stereotyped prompt
    )
    correct_prediction = base_probs[0] > 0  # True if correct entity has higher probability than incorrect
    return {  # Return structured results
        'correct_prediction': correct_prediction,  # Boolean: did model get it right?
        'pro_prompt': pro_prompt,  # The input prompt used
        'correct_entity': correct_entity,  # What the correct answer should be
        'other_entity': other_entity,  # What the incorrect answer is
        'prob_diff': base_probs[0],  # Difference in probabilities (correct - incorrect)
        'correct_prob': base_probs[1],  # Raw probability of correct entity
        'incorrect_prob': base_probs[2],  # Raw probability of incorrect entity
        'logprob_per_token': logprob_info  # Detailed per-token scoring info
    }

def trace_winobias_mcqa_style(
    mt,
    example,
    kind=None,
    include_negatives=False,
):
    """
    Run causal tracing on generative WinoBias/coref using prompt-entity format.
    Uses trace_with_patch from the original architecture.
    """
    try:
        formatted_example = format_winobias_as_mcqa(example)
        (
            inp,
            answers_t,
            pro_prompt,
            anti_prompt,
            correct_entity,
            other_entity,
            answer_token_seqs
        ) = encode_winobias_mcqa(
            mt.tokenizer,
            formatted_example,
            accuracy_only=False,
            return_token_seqs=True
        )
        # Get initial predictions
        base_inst, counterfact_instance = score_winobias_target(
            mt,
            inp,
            answers_t,
            answer_token_seqs=answer_token_seqs,
            counterfactual=True
        )
        base_prob_diff = base_inst[0][0]
        counterfact_prob_diff = counterfact_instance[0][0]
        pro_correct = base_prob_diff > 0
        anti_correct = counterfact_prob_diff > 0
        both_correct = pro_correct and anti_correct
        if not both_correct and not include_negatives:
            return dict(correct_prediction=False)
        prob_corr, prob_incorr, other_token_probs = trace_with_patch(
            mt=mt,
            inp=inp,
            answers_t=answers_t,
            indices_to_replace=None,
            kind=kind,
        )
        return dict(
            correct_prediction=both_correct,
            probits_correct=prob_corr,
            probits_incorrect=prob_incorr,
            other_token_probs=other_token_probs,
            base_probs_logs=base_inst,
            contrast_probs_logs=counterfact_instance,
            input_ids_base_inst=inp["input_ids"][0].tolist(),
            input_ids_counterfact_inst=inp["input_ids"][1].tolist(),
            input_tokens_base_inst=mt.tokenizer.decode(inp["input_ids"][0]),
            input_tokens_counterfact_inst=mt.tokenizer.decode(inp["input_ids"][1]),
            base_answer_tokens=(mt.tokenizer.decode([t for t in answer_token_seqs[0]]),
                                mt.tokenizer.decode([t for t in answer_token_seqs[1]])),
            prediction_type="both_correct" if both_correct else "mixed",
            kind=kind,
            formatted_prompts={
                'pro_prompt': pro_prompt,
                'anti_prompt': anti_prompt,
                'correct_entity': correct_entity,
                'other_entity': other_entity
            },
            example_data={
                'index': example['index'],
                'pro_sentence': example['pro_sentence'],
                'anti_sentence': example['anti_sentence'],
                'correct_referent': example['correct_referent'],
                'other_entity': example['other_entity']
            }
        )
    except Exception as e:
        return {
            'correct_prediction': False,
            'skip_reason': 'processing_error',
            'error': str(e),
            'example_index': example['index']
        }


#########################################################################################################
################################## PATCHING/TRACING ARCHITECTURE (UNCHANGED) ############################
#########################################################################################################

def find_pronoun_positions(tokenizer, input_ids):
    """
    Find positions of pronouns (he/she) in tokenized input.
    Returns list of positions where pronouns occur.
    """
    # Common pronoun tokens for Llama tokenizer
    he_tokens = tokenizer.encode(" he", add_special_tokens=False)
    she_tokens = tokenizer.encode(" she", add_special_tokens=False)
    
    pronoun_positions = []
    
    # Search for pronoun tokens in the input
    for batch_idx in range(input_ids.shape[0]):
        tokens = input_ids[batch_idx].tolist()
        
        # Look for "he" tokens
        for he_token in he_tokens:
            for i, token in enumerate(tokens):
                if token == he_token:
                    pronoun_positions.append(i)
        
        # Look for "she" tokens  
        for she_token in she_tokens:
            for i, token in enumerate(tokens):
                if token == she_token:
                    pronoun_positions.append(i)
    
    # Remove duplicates and sort
    pronoun_positions = sorted(list(set(pronoun_positions)))
    return pronoun_positions if pronoun_positions else [input_ids.shape[1] - 1]  # fallback to final token

def trace_with_patch(
    mt,
    inp,
    answers_t,
    indices_to_replace,
    kind,
):
    """
    Causal tracing: patch hidden states at selected layers/tokens,
    compare probability assigned to correct/incorrect entity referents.
    """
    if indices_to_replace is None:
        # Find pronoun positions instead of using final token
        indices_to_replace = find_pronoun_positions(mt.tokenizer, inp["input_ids"])
        print(f"Auto-detected pronoun positions: {indices_to_replace}")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep_seqScoring(x, layer):
        if layer not in patch_spec:
            return x
        h = untuple(x)
        for t in patch_spec[layer]:
            h[0, t] = h[1, t]
        return x

    probs_table_corr, probs_table_incorr = [], []
    other_token_probs = defaultdict(list)
    for layer in range(mt.num_layers):
        lname = layername(mt, layer, kind)
        states_to_patch = [(token_pos, lname) for token_pos in indices_to_replace]
        patch_spec = defaultdict(list)
        for t, l in states_to_patch:
            patch_spec[l].append(t)
        # Use the updated scoring function!
        with torch.inference_mode(), nethook.TraceDict(
            mt.model,
            list(patch_spec.keys()),
            edit_output=patch_rep_seqScoring,
        ):
            p, l, logprob_info = score_winobias_target(
                mt,
                inp,
                answers_t,
                answer_token_seqs=None,
                counterfactual=False,
            )
        probs_table_corr.append(p[1])
        probs_table_incorr.append(p[2])
    return probs_table_corr, probs_table_incorr, other_token_probs

def layername(mt, num, kind):
    model = mt.model
    if hasattr(model, "transformer"):
        if kind == "embed":
            layername = "transformer.wte"
        else:
            layername = f'transformer.h.{num}{"" if kind is None else "." + kind}'
    elif hasattr(model, "model"):
        if kind == "embed":
            layername = "model.embed_tokens"
        else:
            if kind == "attn":
                kind = "self_attn"
            layername = f'model.layers.{num}{"" if kind is None else "." + kind}'
    else:
        raise Exception("unknown transformer structure")
    if layername not in [n for n, _ in mt.model.named_modules()]:
        raise Exception("invalid layername: ", layername)
    return layername

class ModelAndTokenizer:
    """
    Holds a language model and tokenizer, counts the number of layers.
    """
    def __init__(
        self,
        model_name,
        model=None,
        tokenizer=None,
        no_model_load=False,
        llama_path=None,
        device_map="auto",        
        low_cpu_mem_usage=True      
    ):
        if tokenizer is None:
            assert model_name is not None
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token="***")
        if no_model_load:
            model = None
        elif model is None:
            assert model_name is not None
            model = LlamaForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                use_auth_token="***")
            model.eval()
        if not no_model_load:
            self.model = model
            self.layer_names = [
                n for n, m in model.named_modules()
                if re.match(r"^(transformer|gpt_neox|model)\.(h|layers|transformer\.blocks)\.\d+$", n)
            ]
            self.num_layers = len(self.layer_names)
        else:
            if "13b" in model_name:
                self.num_layers = 40
            elif "7b" in model_name:
                self.num_layers = 32
            else:
                raise Exception("unknown model size")
        self.tokenizer = tokenizer
        if not no_model_load:
            if ("13b" in model_name and self.num_layers != 40) or (
                "7b" in model_name and self.num_layers != 32
            ):
                breakpoint()
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )