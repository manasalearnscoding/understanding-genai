#modified pipeline
import os
import re
import torch
import jsonlines
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from scoring_and_patching_utils import (make_inputs, trace_with_patch, ModelAndTokenizer)

# For custom model implementations
# from model.modeling_modified_olmo import ModifiedOLMoForCausalLM
# from util import nethook

#########################################################################################################
####################################### PROCESS/STORE DATA ##############################################
#########################################################################################################

def read_winocoref_file(pro_file="../data/pro_stereotyped_type1.txt.dev", 
                       anti_file="../data/anti_stereotyped_type1.txt.dev"):
    """
    Read both pro-stereotype and anti-stereotype WinoCoRef dataset files and create paired examples.
    
    Args:
        pro_file: Path to the pro-stereotype dataset file
        anti_file: Path to the anti-stereotype dataset file
        
    Returns:
        List of dictionaries containing paired example data
    """
    examples = []
    
    # READ
    with open(pro_file, 'r') as f1, open(anti_file, 'r') as f2:
        pro_lines = f1.readlines()
        anti_lines = f2.readlines()
    
    # assert len(pro_lines) == len(anti_lines), "Pro and anti files must have same number of lines"
    
    for i, (pro_line, anti_line) in enumerate(zip(pro_lines, anti_lines)):
        pro_line = pro_line.strip()
        anti_line = anti_line.strip()
        
        # SKIP BLANK LINES
        if not pro_line or not anti_line:
            continue
        if not pro_line[0].isdigit() or not anti_line[0].isdigit():
            continue
            
        pro_parts = pro_line.split(' ', 1)
        anti_parts = anti_line.split(' ', 1)
        
        pro_index = int(pro_parts[0])
        anti_index = int(anti_parts[0])
        
        # Ensure indices match
        # assert pro_index == anti_index, f"Line indices don't match: {pro_index} vs {anti_index}"
        
        pro_sentence_raw = pro_parts[1]
        anti_sentence_raw = anti_parts[1]
        
        # EXTRACT INFO
        pro_info = extract_brackets_info(pro_sentence_raw)
        anti_info = extract_brackets_info(anti_sentence_raw)
        
        # Ensure sentence structures match (same entities, different pronouns)
        # assert pro_info['entity'] == anti_info['entity'], f"Entities don't match at line {pro_index}"
        
        # CLEAN SENTENCE
        pro_clean = pro_sentence_raw.replace('[', '').replace(']', '')
        anti_clean = anti_sentence_raw.replace('[', '').replace(']', '')
        
        # FIND OTHER ENTITY
        other_entity = find_other_entity(pro_lines, anti_lines, i, pro_info['entity'])
        
        examples.append({
            'index': pro_index,
            'pro_sentence': pro_clean,
            'anti_sentence': anti_clean,
            'pro_pronoun': pro_info['pronoun'],
            'anti_pronoun': anti_info['pronoun'],
            'correct_referent': pro_info['entity'],  
            'other_entity': other_entity,
            'pro_raw': pro_sentence_raw,
            'anti_raw': anti_sentence_raw
        })
    
    print(f"Loaded {len(examples)} paired examples")
    print(examples)
    return examples


def extract_brackets_info(sentence):
    """
    Extract entity and pronoun information from a bracketed sentence.
    
    Args:
        sentence: Raw sentence with brackets
        
    Returns:
        Dictionary with 'entity' and 'pronoun' keys
    """
    brackets = []
    start_idx = -1
    
    # FIND BRACKETS
    for i, char in enumerate(sentence):
        if char == '[':
            start_idx = i
        elif char == ']' and start_idx != -1:
            brackets.append((start_idx, i))
            start_idx = -1
    
    # EXTRACT BRACKETED TEXT
    if len(brackets) >= 2:
        entity_start, entity_end = brackets[0]
        pronoun_start, pronoun_end = brackets[1]
        
        entity = sentence[entity_start+1:entity_end]
        pronoun = sentence[pronoun_start+1:pronoun_end]
        
        return {
            'entity': entity,
            'pronoun': pronoun
        }
    else:
        raise ValueError(f"Expected at least 2 bracketed sections, found {len(brackets)}")


def find_other_entity(pro_lines, anti_lines, current_index, current_entity):
    """
    Find the other entity by looking at the paired line where the other entity is bracketed.
    
    Args:
        pro_lines: All lines from the pro-stereotype file
        anti_lines: All lines from the anti-stereotype file  
        current_index: Current line index (0-based)
        current_entity: The entity that's bracketed in the current line
        
    Returns:
        The other entity mentioned in the sentence
    """
    # FIND CURR LINE INDEX
    current_line = pro_lines[current_index].strip()
    current_line_number = int(current_line.split(' ', 1)[0])
    
    if current_line_number % 2 == 1:  
        paired_line_number = current_line_number + 1
        paired_index = current_index + 1
    else:    
        paired_line_number = current_line_number - 1
        paired_index = current_index - 1
    
    # if paired_index < 0 or paired_index >= len(pro_lines):
    #     return "other_entity"  
    
    # EXTRACT BRACKETED ENTITY FROM PAIR
    # try:
    paired_line = pro_lines[paired_index].strip()
    paired_parts = paired_line.split(' ', 1)
    paired_sentence_raw = paired_parts[1]
    
    paired_info = extract_brackets_info(paired_sentence_raw)
    other_entity = paired_info['entity']
    
    return other_entity
        
    # except (IndexError, ValueError, KeyError):
    #     # If anything goes wrong, return fallback
    #     return "other_entity"

#########################################################################################################
################################## ENCODING, TRACING COREF ##############################################
#########################################################################################################

def clean_entity_name(entity):
    """
    Clean entity name by removing articles and extra words.
    
    Args:
        entity: Raw entity string like "the doctor" or "The nurse"
        
    Returns:
        Clean entity name like "doctor" or "nurse"
    """
    # Remove common articles and determiners
    stopwords = ['the', 'a', 'an', 'The', 'A', 'An']
    words = entity.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return ' '.join(cleaned_words)


def format_winobias_as_mcqa(example):
    """
    Format WinoBias example as MCQA-style prompt ending with "refers to ___"
    
    Args:
        example: Dictionary from read_winocoref_file
        
    Returns:
        Formatted prompts and entity choices
    """
    # Clean entity names for cleaner predictions
    clean_correct = clean_entity_name(example['correct_referent'])
    clean_other = clean_entity_name(example['other_entity'])
    
    # Create base prompt template
    def create_prompt(sentence, pronoun):
        return f"{sentence}. In this sentence, '{pronoun}' refers to"
    
    # Create prompts for both pro and anti sentences
    pro_prompt = create_prompt(example['pro_sentence'], example['pro_pronoun'])
    anti_prompt = create_prompt(example['anti_sentence'], example['anti_pronoun'])
    
    return {
        'pro_prompt': pro_prompt,
        'anti_prompt': anti_prompt,
        'correct_entity': clean_correct,
        'other_entity': clean_other,
        'original_correct': example['correct_referent'],
        'original_other': example['other_entity']
    }


def encode_winobias_mcqa(
    tokenizer,
    formatted_example,
    accuracy_only=False
):
    """
    Encode WinoBias example in MCQA format for causal tracing.
    Adapted from the original encode() function.
    
    Args:
        tokenizer: Model tokenizer
        formatted_example: Output from format_winobias_as_mcqa
        accuracy_only: If True, only return accuracy data (no counterfactual)
        
    Returns:
        Similar structure to original encode() function
    """
    pro_prompt = formatted_example['pro_prompt']
    anti_prompt = formatted_example['anti_prompt']
    correct_entity = formatted_example['correct_entity']
    other_entity = formatted_example['other_entity']
    
    if accuracy_only:
        full_input_strings = [pro_prompt]
    else:
        # Include both pro and anti sentences as base/counterfactual pair
        full_input_strings = [pro_prompt, anti_prompt]
    
    # Create inputs using existing MCQA infrastructure
    inp = make_inputs(tokenizer, full_input_strings)
    
    # Get exact token indices by appending entities and isolating them
    full_input_strings_with_answers = [
        pro_prompt + " " + correct_entity,
        pro_prompt + " " + other_entity,
    ]
    
    if not accuracy_only:
        full_input_strings_with_answers += [
            anti_prompt + " " + correct_entity,
            anti_prompt + " " + other_entity,
        ]
    
    full_input_encodings_with_answers = make_inputs(
        tokenizer,
        full_input_strings_with_answers
    )
    
    # Verify we have single-token answers (or get first token if multi-token)
    if full_input_encodings_with_answers["input_ids"].shape[1] != inp["input_ids"].shape[1] + 1:
        print(f"Warning: Entity tokens may be multi-token. Taking first token only.")
    
    # Get entity token IDs (first token if multi-token)
    answer_encodings = [
        el[0] for el in full_input_encodings_with_answers["input_ids"][:, inp["input_ids"].shape[1]:].tolist()
    ]
    
    if accuracy_only:
        return (
            inp,
            (answer_encodings[0], answer_encodings[1]),  # (correct, incorrect) tokens
            pro_prompt,
            correct_entity,
            other_entity,
        )
    else:
        return (
            inp,
            answer_encodings,  # [pro_correct, pro_incorrect, anti_correct, anti_incorrect]
            pro_prompt,
            anti_prompt,
            correct_entity,
            other_entity,
        )


def score_winobias_target(mt, inp, answers_t, counterfactual=False):
    """
    Score WinoBias predictions using MCQA-style scoring.
    Adapted from the original score_target() function.
    """
    with torch.inference_mode():
        outputs = mt.model(
            input_ids=inp["input_ids"],
            return_dict=True,
        )

    correct_entity_token = answers_t[0]
    incorrect_entity_token = answers_t[1]
    
    # Get predictions at the last token position (like MCQA)
    last_token_logits = outputs.logits[:, -1, :]
    last_token_probs = torch.nn.Softmax(dim=1)(last_token_logits)
    
    # Score the base instance (pro sentence)
    base_corr_prob = last_token_probs[0, correct_entity_token].item()
    base_incorr_prob = last_token_probs[0, incorrect_entity_token].item()
    base_corr_logit = last_token_logits[0, correct_entity_token].item()
    base_incorr_logit = last_token_logits[0, incorrect_entity_token].item()
    
    base_probs = (base_corr_prob - base_incorr_prob, base_corr_prob, base_incorr_prob)
    base_logs = (base_corr_logit - base_incorr_logit, base_corr_logit, base_incorr_logit)
    
    if counterfactual and last_token_logits.shape[0] > 1:
        # Score the counterfactual instance (anti sentence)
        if len(answers_t) >= 4:
            contrast_corr_token = answers_t[2]
            contrast_incorr_token = answers_t[3]
        else:
            # Use same tokens for contrast
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
        
        return (base_probs, base_logs, [], []), (counterfactual_probs, counterfactual_logs)
    
    return base_probs, base_logs, [], []


def trace_winobias_mcqa_style(
    mt,
    example,
    kind=None,
    include_negatives=False,
):
    """
    Run causal tracing on WinoBias using MCQA-style formatting.
    Now fully compatible with trace_with_patch!
    """
    try:
        # Format as MCQA-style prompt
        formatted_example = format_winobias_as_mcqa(example)
        
        # Encode using MCQA infrastructure
        (
            inp,
            answers_t,
            pro_prompt,
            anti_prompt,
            correct_entity,
            other_entity,
        ) = encode_winobias_mcqa(
            mt.tokenizer,
            formatted_example,
            accuracy_only=False
        )
        
        # Get initial predictions using MCQA scoring
        base_inst, counterfact_instance = score_winobias_target(
            mt,
            inp,
            answers_t,
            counterfactual=True,
        )
        
        base_prob_diff = base_inst[0]
        counterfact_prob_diff = counterfact_instance[0]
        
        # Check prediction correctness
        pro_correct = base_prob_diff > 0
        anti_correct = counterfact_prob_diff > 0
        both_correct = pro_correct and anti_correct
        
        if not both_correct and not include_negatives:
            return dict(correct_prediction=False)
        
        # NOW WE CAN USE THE ORIGINAL trace_with_patch!
        prob_corr, prob_incorr, other_token_probs, top_k_tokens = trace_with_patch(
            mt=mt,
            inp=inp,
            answers_t=answers_t,
            indices_to_replace=None,  # Patch at last token position
            kind=kind,
        )
        
        return dict(
            correct_prediction=both_correct,
            probits_correct=prob_corr,
            probits_incorrect=prob_incorr,
            other_token_probs=other_token_probs,
            top_k_tokens=top_k_tokens,
            base_probs_logs=base_inst,
            contrast_probs_logs=counterfact_instance,
            input_ids_base_inst=inp["input_ids"][0].tolist(),
            input_ids_counterfact_inst=inp["input_ids"][1].tolist(),
            input_tokens_base_inst=mt.tokenizer.decode(inp["input_ids"][0]),
            input_tokens_counterfact_inst=mt.tokenizer.decode(inp["input_ids"][1]),
            base_answer_tokens=(mt.tokenizer.decode(answers_t[0]), mt.tokenizer.decode(answers_t[1])),
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


def compute_winobias_accuracy(mt, example):
    """
    Compute accuracy for WinoBias example using MCQA-style formatting.
    Similar to compute_accuracy() in original code.
    """
    formatted_example = format_winobias_as_mcqa(example)
    
    (
        inp,
        answers_t,
        pro_prompt,
        correct_entity,
        other_entity,
    ) = encode_winobias_mcqa(
        mt.tokenizer,
        formatted_example,
        accuracy_only=True
    )
    
    base_probs, base_logs, _, _ = score_winobias_target(
        mt,
        inp,
        answers_t,
        counterfactual=False
    )
    
    # Determine if prediction is correct
    correct_prediction = base_probs[0] > 0  # correct_prob - incorrect_prob > 0
    
    return {
        'correct_prediction': correct_prediction,
        'pro_prompt': pro_prompt,
        'correct_entity': correct_entity,
        'other_entity': other_entity,
        'prob_diff': base_probs[0],
        'correct_prob': base_probs[1],
        'incorrect_prob': base_probs[2]
    }


# Example usage
if __name__ == "__main__":
    # Test the formatting
    example = {
        'index': 1,
        'pro_sentence': "The doctor asked the nurse to help her with the patient.",
        'anti_sentence': "The doctor asked the nurse to help him with the patient.", 
        'pro_pronoun': "her",
        'anti_pronoun': "him",
        'correct_referent': "the nurse",
        'other_entity': "the doctor"
    }
    
    formatted = format_winobias_as_mcqa(example)
    print("Pro prompt:", formatted['pro_prompt'])
    print("Anti prompt:", formatted['anti_prompt'])
    print("Entities:", formatted['correct_entity'], "vs", formatted['other_entity'])
    
    # Output:
    # Pro prompt: The doctor asked the nurse to help her with the patient. In this sentence, 'her' refers to
    # Anti prompt: The doctor asked the nurse to help him with the patient. In this sentence, 'him' refers to  
    # Entities: nurse vs doctor

#########################################################################################################
############################################## MAIN #####################################################
#########################################################################################################

# if __name__ == "__main__":
#     # mt = ModelAndTokenizer(args.model, no_model_load=True, llama_path=args.llama_path)

#     read_winocoref_file()

#     # results = process_winocoref_dataset(mt)
    
#     # print(results)