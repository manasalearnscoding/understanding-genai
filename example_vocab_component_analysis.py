#!/usr/bin/env python3
"""
Example script demonstrating vocabulary projection and component-specific tracing
for WinoBias bias analysis.

This script shows how to use the new features to answer your research questions:
- RQ1: How does bias emerge? (vocabulary projection)
- RQ2: Does bias emerge in MLP or attention layers? (component tracing)
"""

import sys
import os
sys.path.append('/fs/clip-ml/mvinodku/understanding_genai/util')

from scoring_and_patching_utils import (
    ModelAndTokenizer,
    WinoBiasVocabProjector,
    trace_winobias_component_specific,
    compare_component_contributions,
    analyze_mlp_vs_attention_bias,
)
from read_and_process import read_winocoref_file

def main():
    print("🔬 WinoBias Vocabulary Projection & Component Analysis Demo")
    print("=" * 60)
    
    # Load a small example
    pro_file = "/fs/clip-ml/mvinodku/understanding_genai/data/pro_stereotyped_type1.txt.dev"
    anti_file = "/fs/clip-ml/mvinodku/understanding_genai/data/anti_stereotyped_type1.txt.dev"
    examples = read_winocoref_file(pro_file, anti_file)[:1]  # Just one example for demo
    
    print(f"📖 Loaded {len(examples)} example(s)")
    example = examples[0]
    print(f"Example: {example['pro_sentence']}")
    
    # Initialize model (you can set no_model_load=True for testing without GPU)
    print("\n🤖 Loading model...")
    mt = ModelAndTokenizer(
        model_name="meta-llama/Llama-2-7b-hf",
        # no_model_load=True  # Set to False when you want to run with actual model
    )
    print(f"Model loaded: {mt}")
    
    if mt.model is None:
        print("⚠️  Model not loaded (no_model_load=True). Set to False for actual analysis.")
        return
    
    # 1. VOCABULARY PROJECTION ANALYSIS
    print("\n" + "="*60)
    print("📊 VOCABULARY PROJECTION ANALYSIS")
    print("="*60)
    
    projector = WinoBiasVocabProjector(mt)
    
    # Analyze bias emergence across model variants
    for variant in ["baseline", "safety", "jailbreak"]:
        print(f"\n🔍 Analyzing {variant} variant...")
        
        vocab_result = projector.analyze_bias_emergence(
            example=example,
            model_variant=variant,
            k=15  # Top 15 tokens per layer
        )
        
        bias_analysis = vocab_result["bias_analysis"]
        print(f"  📈 Male terms first emerge at layer: {bias_analysis.get('male_terms_emergence_first_layer', 'N/A')}")
        print(f"  📈 Female terms first emerge at layer: {bias_analysis.get('female_terms_emergence_first_layer', 'N/A')}")
        print(f"  💼 Tech terms first emerge at layer: {bias_analysis.get('tech_terms_emergence_first_layer', 'N/A')}")
        print(f"  🏥 Care terms first emerge at layer: {bias_analysis.get('care_terms_emergence_first_layer', 'N/A')}")
    
    # 2. COMPONENT-SPECIFIC TRACING
    print("\n" + "="*60)
    print("🧩 COMPONENT-SPECIFIC TRACING")
    print("="*60)
    
    # Compare MLP vs Attention across all variants
    mlp_vs_attn_results = analyze_mlp_vs_attention_bias(
        mt=mt,
        example=example,
        model_variants=["baseline", "safety", "jailbreak"]
    )
    
    print("\n📊 MLP vs Attention Analysis Results:")
    cross_analysis = mlp_vs_attn_results["cross_variant_analysis"]
    
    for variant, analysis in cross_analysis.items():
        print(f"\n🔬 {variant.upper()} variant:")
        print(f"  🧠 MLP dominates: {analysis['mlp_dominates']}")
        print(f"  👁️  Attention dominates: {analysis['attention_dominates']}")
        print(f"  📊 MLP bias strength: {analysis['mlp_bias_strength']:.4f}")
        print(f"  📊 Attention bias strength: {analysis['attention_bias_strength']:.4f}")
        print(f"  📈 Bias ratio (MLP/Attn): {analysis['bias_ratio_mlp_to_attn']:.2f}")
    
    # Summary insights
    summary = mlp_vs_attn_results["summary"]
    print(f"\n🎯 SUMMARY INSIGHTS:")
    print(f"  🔄 Consistent MLP dominance across variants: {summary['consistent_mlp_dominance']}")
    print(f"  🔄 Consistent Attention dominance across variants: {summary['consistent_attention_dominance']}")
    
    # 3. INDIVIDUAL COMPONENT ANALYSIS
    print("\n" + "="*60)
    print("🔍 INDIVIDUAL COMPONENT ANALYSIS")
    print("="*60)
    
    # Analyze each component separately for baseline
    for component in ["mlp", "attn"]:
        print(f"\n🧩 Analyzing {component.upper()} component...")
        
        comp_result = trace_winobias_component_specific(
            mt=mt,
            example=example,
            component_type=component,
            model_variant="baseline"
        )
        
        comp_analysis = comp_result["component_analysis"]
        print(f"  📍 Max bias layer: {comp_analysis['max_bias_layer']}")
        print(f"  📊 Max bias value: {comp_analysis['max_bias_value']:.4f}")
        print(f"  📈 Bias progression: {comp_analysis['bias_progression']}")
        print(f"  🌅 Early layers bias: {comp_analysis['early_layers_bias']:.4f}")
        print(f"  🌆 Late layers bias: {comp_analysis['late_layers_bias']:.4f}")
    
    print("\n" + "="*60)
    print("✅ Analysis complete! Key research insights:")
    print("1. 📊 Vocabulary projection shows when biased concepts emerge")
    print("2. 🧩 Component tracing reveals whether MLP or attention drives bias")
    print("3. 🔄 Cross-variant analysis shows how safety/jailbreak affects components")
    print("4. 📈 Layer-wise analysis reveals bias progression patterns")
    print("="*60)

if __name__ == "__main__":
    main()
