#modified pipeline
import os
import re
import torch
import jsonlines
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from scoring_and_patching_utils import (make_inputs, trace_with_patch, ModelAndTokenizer)

# For custom model implementations
# from model.modeling_modified_olmo import ModifiedOLMoForCausalLM
import sys
import os

# memit_path = r'C:\Users\Vinod\memit'
# if memit_path not in sys.path:
#     sys.path.insert(0, memit_path)
from util import nethook
print(f"Loaded nethook from {nethook.__file__}")

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
    # print(examples)
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