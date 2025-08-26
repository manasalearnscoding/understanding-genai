import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring_and_patching_utils import clean_entity_name, format_winobias_as_mcqa, make_inputs, encode_winobias_mcqa

# Mock tokenizer class for testing without downloading models
class MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        # Simple word-based tokenization for testing
        words = text.split()
        # Simulate token IDs (just use word lengths as fake IDs)
        token_ids = [len(word) % 1000 + 1 for word in words]
        if add_special_tokens:
            token_ids = [1] + token_ids + [2]  # Add fake BOS and EOS tokens
        return token_ids

def test_functions():
    print("="*60)
    print("TESTING SCORING AND PATCHING UTILS FUNCTIONS")
    print("="*60)
    
    # Create a test example matching the expected format
    test_example = {
        'index': 1,
        'pro_sentence': 'The developer argued with the designer because he did not like the design.',
        'anti_sentence': 'The developer argued with the designer because she did not like the design.',
        'pro_pronoun': 'he',
        'anti_pronoun': 'she', 
        'correct_referent': 'the developer',
        'other_entity': 'the designer'
    }
    
    print("\n1. TESTING clean_entity_name():")
    print("-" * 40)
    print(f"Input: '{test_example['correct_referent']}'")
    clean_correct = clean_entity_name(test_example['correct_referent'])
    print(f"Output: '{clean_correct}'")
    
    print(f"\nInput: '{test_example['other_entity']}'")
    clean_other = clean_entity_name(test_example['other_entity'])
    print(f"Output: '{clean_other}'")
    
    print("\n2. TESTING format_winobias_as_mcqa():")
    print("-" * 40)
    formatted = format_winobias_as_mcqa(test_example)
    for key, value in formatted.items():
        print(f"{key}: '{value}'")
    
    print("\n3. TESTING make_inputs():")
    print("-" * 40)
    
    # Create mock tokenizer
    tokenizer = MockTokenizer()
    
    # Test with the formatted prompts
    test_prompts = [
        formatted['pro_prompt'],
        formatted['anti_prompt']
    ]
    
    print("Input prompts:")
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i}]: '{prompt}'")
    
    # Test make_inputs
    try:
        inputs = make_inputs(tokenizer, test_prompts, device="cpu")
        print(f"\nOutput tensor shape: {inputs['input_ids'].shape}")
        print(f"Output tensor: {inputs['input_ids']}")
        
        # Show token-by-token breakdown
        print("\nToken breakdown:")
        for i, prompt in enumerate(test_prompts):
            tokens = tokenizer.encode(prompt)
            print(f"  Prompt {i}: {len(tokens)} tokens -> {tokens}")
            
    except Exception as e:
        print(f"Error in make_inputs: {e}")
        # Try with truncate=True
        print("Retrying with truncate=True...")
        try:
            inputs = make_inputs(tokenizer, test_prompts, device="cpu", truncate=True)
            print(f"Success! Output shape: {inputs['input_ids'].shape}")
            print(f"Output tensor: {inputs['input_ids']}")
        except Exception as e2:
            print(f"Still failed: {e2}")

    print("\n4. TESTING encode_winobias_mcqa():")
    print("-" * 40)
    
    # Test with accuracy_only=True
    print("Testing with accuracy_only=True, return_token_seqs=True:")
    try:
        result_accuracy = encode_winobias_mcqa(
            tokenizer, 
            formatted, 
            accuracy_only=True, 
            return_token_seqs=True
        )
        
        inp, answers_t, pro_prompt, correct_entity, other_entity, answer_token_seqs = result_accuracy
        
        print(f"  inp tensor shape: {inp['input_ids'].shape}")
        print(f"  inp tensor: {inp['input_ids']}")
        print(f"  answers_t (correct, incorrect): {answers_t}")
        print(f"  pro_prompt: '{pro_prompt}'")
        print(f"  correct_entity: '{correct_entity}'")
        print(f"  other_entity: '{other_entity}'")
        print(f"  answer_token_seqs: {answer_token_seqs}")
        
        # Show what the answer strings would be
        answer_strings = [
            pro_prompt + " " + correct_entity,
            pro_prompt + " " + other_entity,
        ]
        print(f"  Full answer strings:")
        for i, ans_str in enumerate(answer_strings):
            print(f"    [{i}]: '{ans_str}'")
            print(f"         tokens: {tokenizer.encode(ans_str)}")
            
    except Exception as e:
        print(f"Error in encode_winobias_mcqa (accuracy_only=True): {e}")
        import traceback
        traceback.print_exc()
    
    # Test with accuracy_only=False
    print("\nTesting with accuracy_only=False, return_token_seqs=True:")
    try:
        result_full = encode_winobias_mcqa(
            tokenizer, 
            formatted, 
            accuracy_only=False, 
            return_token_seqs=True
        )
        
        inp, answer_encodings, pro_prompt, anti_prompt, correct_entity, other_entity, answer_token_seqs = result_full
        
        print(f"  inp tensor shape: {inp['input_ids'].shape}")
        print(f"  inp tensor: {inp['input_ids']}")
        print(f"  answer_encodings [pro_correct, pro_incorrect, anti_correct, anti_incorrect]: {answer_encodings}")
        print(f"  pro_prompt: '{pro_prompt}'")
        print(f"  anti_prompt: '{anti_prompt}'")
        print(f"  correct_entity: '{correct_entity}'")
        print(f"  other_entity: '{other_entity}'")
        print(f"  answer_token_seqs: {answer_token_seqs}")
        
        # Show what the answer strings would be
        answer_strings = [
            pro_prompt + " " + correct_entity,
            pro_prompt + " " + other_entity,
            anti_prompt + " " + correct_entity,
            anti_prompt + " " + other_entity,
        ]
        print(f"  Full answer strings:")
        for i, ans_str in enumerate(answer_strings):
            print(f"    [{i}]: '{ans_str}'")
            print(f"         tokens: {tokenizer.encode(ans_str)}")
            
    except Exception as e:
        print(f"Error in encode_winobias_mcqa (accuracy_only=False): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_functions()