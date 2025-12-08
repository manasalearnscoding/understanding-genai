#########################################################################################################
######################################## JAILBREAKING ANALYSIS ##########################################
#########################################################################################################
def analyze_pathway_reuse(baseline_result, jailbreak_result):
    """
    Analyze if jailbreaking reuses original bias pathways.
    Compares layer-by-layer activation patterns.
    
    Args:
        baseline_result: Results from M_B model variant
        jailbreak_result: Results from M_J model variant
        
    Returns:
        dict: Analysis of pathway reuse patterns
    """
    baseline_layers = baseline_result["probits_correct"]
    jailbreak_layers = jailbreak_result["probits_correct"]
    
    # Calculate correlation between layer patterns
    import numpy as np
    correlation = np.corrcoef(baseline_layers, jailbreak_layers)[0,1]
    
    # Calculate layer-wise differences
    differences = [abs(b - j) for b, j in zip(baseline_layers, jailbreak_layers)]
    max_diff_layer = differences.index(max(differences))
    
    return {
        "pathway_correlation": correlation,
        "likely_reuses_pathways": correlation > 0.7,
        "max_difference_layer": max_diff_layer,
        "layer_differences": differences,
        "average_difference": sum(differences) / len(differences)
    }

def detect_new_pathways(safety_result, jailbreak_result):
    """
    Detect if jailbreaking creates new bias pathways different from safety alignment.
    
    Args:
        safety_result: Results from M_S model variant
        jailbreak_result: Results from M_J model variant
        
    Returns:
        dict: Analysis of new pathway creation
    """
    safety_layers = safety_result["probits_correct"] 
    jailbreak_layers = jailbreak_result["probits_correct"]
    
    # Find layers where jailbreak differs most from safety
    differences = [abs(s - j) for s, j in zip(safety_layers, jailbreak_layers)]
    max_diff_layer = differences.index(max(differences))
    
    # Calculate divergence patterns
    early_layers_diff = sum(differences[:len(differences)//3]) / (len(differences)//3)
    late_layers_diff = sum(differences[2*len(differences)//3:]) / (len(differences)//3)
    
    return {
        "max_difference_layer": max_diff_layer,
        "creates_new_pathways": max(differences) > 0.01,  # Threshold for "new" pathway
        "layer_differences": differences,
        "early_layers_divergence": early_layers_diff,
        "late_layers_divergence": late_layers_diff,
        "divergence_pattern": "early" if early_layers_diff > late_layers_diff else "late"
    }

def analyze_jailbreak_strategy_differences(mt, example, jailbreak_strategies=["roleplay", "character", "hypothetical"]):
    """
    Compare different jailbreaking strategies to see if they use similar pathways.
    
    Args:
        mt: ModelAndTokenizer instance
        example: WinoBias example dict
        jailbreak_strategies: List of jailbreak prompt keys to compare
        
    Returns:
        dict: Comparison of different jailbreak strategies
    """
    strategy_results = {}
    
    for strategy in jailbreak_strategies:
        print(f"Testing jailbreak strategy: {strategy}")
        result = trace_winobias_mcqa_style(
            mt, example,
            model_variant="jailbreak",
            jailbreak_prompt_key=strategy
        )
        strategy_results[strategy] = result
    
    # Compare strategies pairwise
    comparisons = {}
    strategies = list(strategy_results.keys())
    
    for i, strategy1 in enumerate(strategies):
        for strategy2 in strategies[i+1:]:
            layers1 = strategy_results[strategy1]["probits_correct"]
            layers2 = strategy_results[strategy2]["probits_correct"]
            
            import numpy as np
            correlation = np.corrcoef(layers1, layers2)[0,1]
            
            comparisons[f"{strategy1}_vs_{strategy2}"] = {
                "correlation": correlation,
                "similar_pathways": correlation > 0.8,
                "bias_difference": abs(
                    strategy_results[strategy1]["base_probs_logs"][0][0] - 
                    strategy_results[strategy2]["base_probs_logs"][0][0]
                )
            }
    
    return {
        "strategy_results": strategy_results,
        "pairwise_comparisons": comparisons,
        "most_similar_strategies": max(comparisons.keys(), key=lambda k: comparisons[k]["correlation"]),
        "most_different_strategies": min(comparisons.keys(), key=lambda k: comparisons[k]["correlation"])
    }
