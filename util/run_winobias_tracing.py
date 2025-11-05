import argparse
from read_and_process import read_winocoref_file
from scoring_and_patching_utils import (
    ModelAndTokenizer,
    trace_winobias_mcqa_style,
    compute_winobias_accuracy,
    WinoBiasVocabProjector,
    trace_winobias_component_specific,
    compare_component_contributions,
    analyze_mlp_vs_attention_bias,
)
import jsonlines

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
    parser.add_argument("--output_file", type=str, default="winobias_causal_tracing_results.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--llama_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=1)
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
                       choices=["causal_trace", "vocab_projection", "vocab_projection_paired", "component_trace", "mlp_vs_attention"],
                       help="Type of analysis to perform")
    parser.add_argument("--component_type", type=str, default="mlp",
                       choices=["mlp", "attn", "attn_heads"],
                       help="Component type for component-specific tracing")
    parser.add_argument("--vocab_k", type=int, default=20,
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

    # 3. Run analysis based on type
    with jsonlines.open(args.output_file, mode="w") as writer:
        for i, example in enumerate(examples):
            print(f"\n--- Processing Example {i+1}/{len(examples)} (Model: {args.model_variant}, Analysis: {args.analysis_type}) ---")
            if args.model_variant == "safety":
                print(f"Using safety prompt: '{args.safety_prompt_key}'")
            elif args.model_variant == "jailbreak":
                print(f"Using jailbreak prompt: '{args.jailbreak_prompt_key}'")
            print(example)
            
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
            elif args.analysis_type == "vocab_projection":
                projector = WinoBiasVocabProjector(mt)
                result = projector.analyze_bias_emergence(
                    example=example,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key,
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection"
            elif args.analysis_type == "vocab_projection_paired":
                projector = WinoBiasVocabProjector(mt)
                result = projector.analyze_bias_emergence_paired(
                    example=example,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key,
                    k=args.vocab_k
                )
                result["analysis_type"] = "vocab_projection_paired"
            elif args.analysis_type == "component_trace":
                result = trace_winobias_component_specific(
                    mt=mt,
                    example=example,
                    component_type=args.component_type,
                    model_variant=args.model_variant,
                    safety_prompt_key=args.safety_prompt_key,
                    jailbreak_prompt_key=args.jailbreak_prompt_key
                )
                result["analysis_type"] = "component_trace"
            elif args.analysis_type == "mlp_vs_attention":
                result = analyze_mlp_vs_attention_bias(
                    mt=mt,
                    example=example,
                    model_variants=[args.model_variant]  # Single variant for now
                )
                result["analysis_type"] = "mlp_vs_attention"
            else:
                raise ValueError(f"Unknown analysis type: {args.analysis_type}")
            
            writer.write(result)
            
            if args.analysis_type == "causal_trace":
                if result.get("correct_prediction", False):
                    print("Prediction: CORRECT")
                elif result.get("skip_reason"):
                    print("Skipped:", result["skip_reason"])
                else:
                    print("Prediction: INCORRECT")
            elif args.analysis_type == "vocab_projection":
                bias_analysis = result.get("bias_analysis", {})
                print(f"Bias emergence analysis completed. First bias layer: {bias_analysis.get('male_terms_emergence_first_layer', 'N/A')}")
            elif args.analysis_type == "vocab_projection_paired":
                comparison = result.get("paired_comparison", {})
                bias_ind = comparison.get("bias_indicators", {})
                print(f"Paired analysis completed:")
                print(f"  Stereotype type: {comparison.get('example_type', 'N/A')}")
                print(f"  Stereotypical direction: {comparison.get('stereotypical_direction', 'N/A')}")
                if "layer_difference" in bias_ind:
                    print(f"  Layer difference: {bias_ind['layer_difference']}")
                    print(f"  Stereotypical emerges earlier: {bias_ind.get('stereotypical_emerges_earlier', 'N/A')}")
                print(f"  Bias strength: {bias_ind.get('bias_strength', 'N/A')}")
            elif args.analysis_type == "component_trace":
                comp_analysis = result.get("component_analysis", {})
                print(f"Component analysis ({args.component_type}): Max bias at layer {comp_analysis.get('max_bias_layer', 'N/A')}")
            elif args.analysis_type == "mlp_vs_attention":
                summary = result.get("summary", {})
                print(f"MLP vs Attention: MLP dominance = {summary.get('consistent_mlp_dominance', False)}")

    print(f"\nDone! Results saved to {args.output_file}")