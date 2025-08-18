import argparse
from read_and_process import read_winocoref_file
from scoring_and_patching_utils import (
    ModelAndTokenizer,
    trace_winobias_mcqa_style,
    compute_winobias_accuracy,
)
import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pro_file", type=str, default="../data/pro_stereotyped_type1.txt.dev")
    parser.add_argument("--anti_file", type=str, default="../data/anti_stereotyped_type1.txt.dev")
    parser.add_argument("--output_file", type=str, default="winobias_causal_tracing_results.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--llama_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--accuracy_only", action="store_true", help="If set, only computes accuracy, no causal tracing.")
    args = parser.parse_args()

    # 1. Load WinoBias/Coref examples
    examples = read_winocoref_file(args.pro_file, args.anti_file)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]

    # 2. Initialize model and tokenizer
    mt = ModelAndTokenizer(
        model_name=args.model_name,
        llama_path=args.llama_path
    )
    print(f"Loaded model: {mt}")

    # 3. Run and write results
    with jsonlines.open(args.output_file, mode="w") as writer:
        for i, example in enumerate(examples):
            print(f"\n--- Processing Example {i+1}/{len(examples)} ---")
            if args.accuracy_only:
                result = compute_winobias_accuracy(mt, example)
            else:
                result = trace_winobias_mcqa_style(
                    mt=mt,
                    example=example,
                    kind=None,
                    include_negatives=False,
                )
            writer.write(result)
            if result.get("correct_prediction", False):
                print("Prediction: CORRECT")
            elif result.get("skip_reason"):
                print("Skipped:", result["skip_reason"])
            else:
                print("Prediction: INCORRECT")

    print(f"\nDone! Results saved to {args.output_file}")