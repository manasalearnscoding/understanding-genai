import os
import re
import jsonlines
import torch
import warnings
from collections import defaultdict
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

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
    device="cuda" if torch.cuda.is_available() else "cpu",
    add_special_tokens=True,
    truncate=False,
):
    token_lists = [tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in prompts]
    input_ids = token_lists
    try:
        r1 = torch.tensor(input_ids)
    except:
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

    if accuracy_only:
        full_input_strings = [pro_prompt]
    else:
        full_input_strings = [pro_prompt, anti_prompt]
    inp = make_inputs(tokenizer, full_input_strings)

    # -- MULTI-TOKEN: get entity token sequences --
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
    # For each answer, get token id sequence *after* the prompt
    prompt_len = inp["input_ids"].shape[1]
    answer_token_seqs = []
    for idx in range(len(answer_strings)):
        tokens = full_input_encodings_with_answers["input_ids"][idx][prompt_len:]
        answer_token_seqs.append(tokens.tolist())
    print(f'Answer token seqs = {answer_token_seqs}')
    # By default, for MCQA-style use: just return first token of each sequence for legacy compatibility
    answer_encodings = [seq[0] if len(seq) > 0 else -1 for seq in answer_token_seqs]
    
    # Structure: [pro_correct, pro_incorrect, anti_correct, anti_incorrect]
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
    mt,
    inp,
    answers_t,
    answer_token_seqs=None, 
    counterfactual=False
):
    """
    Score entity referents for generative pronoun resolution.
    If answer_token_seqs is given, score all tokens in the sequence.
    """
    with torch.inference_mode():
        outputs = mt.model(
            input_ids=inp["input_ids"],
            return_dict=True,
        )
    logits = outputs.logits  # [batch, seq_len, vocab]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # --- Per-token log-probability collection ---
    logprob_info = []
    num_batches = logits.shape[0]
    if answer_token_seqs is not None:
        for b_idx in range(num_batches):
            cur_logprob = []
            context_len = inp["input_ids"][b_idx].shape[0]
            # For each answer in this batch, score full sequence
            tokens = answer_token_seqs[b_idx] if b_idx < len(answer_token_seqs) else []
            # Each answer token is scored at successive positions (as if generating one token at a time)
            for t_idx, tok in enumerate(tokens):
                pos = context_len + t_idx
                if pos >= logits.shape[1]:
                    break
                prob_row = probs[b_idx, pos, :]
                # Log-prob for gold token
                logprob = torch.log(prob_row[tok] + 1e-12)
                cur_logprob.append(logprob.item())
            logprob_info.append(cur_logprob)

    # --- Binary comparison: correct vs incorrect entity probabilities ---
    correct_entity_token = answers_t[0]
    incorrect_entity_token = answers_t[1]
    last_token_logits = logits[:, -1, :]
    last_token_probs = probs[:, -1, :]
    base_corr_prob = last_token_probs[0, correct_entity_token].item()
    base_incorr_prob = last_token_probs[0, incorrect_entity_token].item()
    base_corr_logit = last_token_logits[0, correct_entity_token].item()
    base_incorr_logit = last_token_logits[0, incorrect_entity_token].item()
    base_probs = (base_corr_prob - base_incorr_prob, base_corr_prob, base_incorr_prob)
    base_logs = (base_corr_logit - base_incorr_logit, base_corr_logit, base_incorr_logit)

    if counterfactual and last_token_logits.shape[0] > 1:
        if len(answers_t) >= 4:
            contrast_corr_token = answers_t[2]
            contrast_incorr_token = answers_t[3]
        else:
            contrast_corr_token = correct_entity_token
            contrast_incorr_token = incorrect_entity_token
        contrast_corr_prob = last_token_probs[1, contrast_corr_token].item()
        contrast_incorr_prob = last_token_probs[1, contrast_incorr_token].item()
        contrast_corr_logit = last_token_logits[1, contrast_corr_token].item()
        contrast_incorr_logit = last_token_logits[1, contrast_incorr_token].item()
        counterfactual_probs = (
            contrast_corr_prob - contrast_incorr_prob,
            contrast_corr_prob,
            contrast_incorr_prob,
        )
        counterfactual_logs = (
            contrast_corr_logit - contrast_incorr_logit,
            contrast_corr_logit,
            contrast_incorr_logit,
        )
        return (base_probs, base_logs, logprob_info), (counterfactual_probs, counterfactual_logs, logprob_info)
    return base_probs, base_logs, logprob_info

def compute_winobias_accuracy(mt, example):
    """
    Compute model accuracy and per-token log-probs for a generative WinoBias/coref example.
    """
    formatted_example = format_winobias_as_mcqa(example)
    (
        inp,
        answers_t,
        pro_prompt,
        correct_entity,
        other_entity,
        answer_token_seqs
    ) = encode_winobias_mcqa(
        mt.tokenizer,
        formatted_example,
        accuracy_only=True,
        return_token_seqs=True
    )
    base_probs, base_logs, logprob_info = score_winobias_target(
        mt,
        inp,
        answers_t,
        answer_token_seqs=answer_token_seqs,
        counterfactual=False
    )
    correct_prediction = base_probs[0] > 0
    return {
        'correct_prediction': correct_prediction,
        'pro_prompt': pro_prompt,
        'correct_entity': correct_entity,
        'other_entity': other_entity,
        'prob_diff': base_probs[0],
        'correct_prob': base_probs[1],
        'incorrect_prob': base_probs[2],
        'logprob_per_token': logprob_info
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
        indices_to_replace = [inp["input_ids"].shape[1] - 1]

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
    ):
        if tokenizer is None:
            assert model_name is not None
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if no_model_load:
            model = None
        elif model is None:
            assert model_name is not None
            model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
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