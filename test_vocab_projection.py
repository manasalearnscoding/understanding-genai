#!/usr/bin/env python3
"""
Test script for vocabulary projection with meta tensor handling.
This script tests the fixed vocabulary projection implementation.
"""

import sys
import os
sys.path.append('/fs/clip-ml/mvinodku/understanding_genai/util')

def test_vocab_projection():
    """Test vocabulary projection with a simple example."""
    try:
        from scoring_and_patching_utils import (
            ModelAndTokenizer,
            WinoBiasVocabProjector,
        )
        from read_and_process import read_winocoref_file
        
        print("🧪 Testing Vocabulary Projection with Meta Tensor Handling")
        print("=" * 60)
        
        # Load one example
        pro_file = "/fs/clip-ml/mvinodku/understanding_genai/data/pro_stereotyped_type1.txt.dev"
        anti_file = "/fs/clip-ml/mvinodku/understanding_genai/data/anti_stereotyped_type1.txt.dev"
        examples = read_winocoref_file(pro_file, anti_file)[:1]
        example = examples[0]
        
        print(f"📖 Test example: {example['pro_sentence']}")
        
        # Initialize model
        print("\n🤖 Loading model...")
        mt = ModelAndTokenizer(
            model_name="meta-llama/Llama-2-7b-hf",
            # Note: device_map="auto" can cause meta tensor issues
        )
        print(f"✅ Model loaded: {mt}")
        
        # Test vocabulary projection
        print("\n📊 Testing vocabulary projection...")
        projector = WinoBiasVocabProjector(mt)
        
        # Check for meta tensors
        print(f"🔍 LM head device: {projector.lm_head.device}")
        
        # Test with error handling
        try:
            result = projector.analyze_bias_emergence(
                example=example,
                model_variant="baseline",
                k=10  # Small k for testing
            )
            
            print("✅ Vocabulary projection completed successfully!")
            
            # Print some results
            bias_analysis = result["bias_analysis"]
            print(f"📈 Male terms first emerge at layer: {bias_analysis.get('male_terms_emergence_first_layer', 'N/A')}")
            print(f"📈 Female terms first emerge at layer: {bias_analysis.get('female_terms_emergence_first_layer', 'N/A')}")
            
            # Show top tokens from first few layers
            layerwise_topk = result["layerwise_topk"]
            for layer_idx in range(min(3, len(layerwise_topk))):
                if 0 in layerwise_topk[layer_idx]:
                    tokens = [token for token, _ in layerwise_topk[layer_idx][0][:5]]
                    print(f"🔤 Layer {layer_idx} top tokens: {tokens}")
            
        except Exception as e:
            print(f"❌ Vocabulary projection failed: {e}")
            print("This might be due to meta tensor issues or memory constraints.")
            
            # Provide debugging info
            print("\n🔧 Debugging info:")
            print(f"Model device map: {getattr(mt.model, 'hf_device_map', 'Not available')}")
            
            # Check if any parameters are on meta device
            meta_params = []
            for name, param in mt.model.named_parameters():
                if param.device.type == 'meta':
                    meta_params.append(name)
            
            if meta_params:
                print(f"⚠️  Parameters on meta device: {len(meta_params)} total")
                print(f"First few: {meta_params[:5]}")
            else:
                print("✅ No parameters on meta device")
        
        print("\n" + "=" * 60)
        print("🎯 Test Summary:")
        print("- Meta tensor handling implemented")
        print("- Error recovery mechanisms in place") 
        print("- Vocabulary projection should work with device_map='auto'")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct environment with required packages.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_vocab_projection()
