import argparse
import json
import math
import os
from typing import Dict, List, Tuple
from isobench import CounterfactualQuestion, PipelineState, run_pipeline


def load_dataset(filepath: str) -> List[CounterfactualQuestion]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[CounterfactualQuestion] = []
    for item in data:
        questions.append(
            CounterfactualQuestion(
                id=item["id"],
                question=item["question"],
                bridge_entity_correct=item["bridge_entity_correct"],
                bridge_entity_injected=item["bridge_entity_injected"],
                target_answer_correct=item.get("target_answer_correct", item["bridge_entity_correct"]),
                target_answer_injected=item.get("target_answer_injected", item["bridge_entity_injected"]),
                mutated_claim=item.get("mutated_claim", ""),
                evidence_hop1=item["evidence_hop1"],
                evidence_hop2_correct=item["evidence_hop2_correct"],
                evidence_hop2_injected=item["evidence_hop2_injected"],
            )
        )
    return questions


def exact_mcnemar_test(outcomes_a: List[int], outcomes_b: List[int]) -> Dict[str, float]:
    """
    Section 4.8 Paired Statistical Analysis:
    Computes exact two-sided McNemar test p-value using binomial distribution on discordant pairs:
    - n10 (b): architecture A propagated error (1), architecture B stopped error (0)
    - n01 (c): architecture A stopped error (0), architecture B propagated error (1)
    """
    n10 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 0 and b == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        return {"n10": 0, "n01": 0, "n_discordant": 0, "p_value": 1.0}

    k = min(n10, n01)
    tail_prob = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
    p_val = min(1.0, 2.0 * tail_prob)
    return {"n10": n10, "n01": n01, "n_discordant": n_disc, "p_value": p_val}


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """Applies Holm-Bonferroni step-down multiple-testing correction across confirmatory hypotheses."""
    sorted_items: List[Tuple[str, float]] = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    results: Dict[str, Dict[str, object]] = {}
    rejected_so_far = True
    prev_adj_p = 0.0
    for idx, (name, raw_p) in enumerate(sorted_items):
        multiplier = m - idx
        adj_p = min(1.0, max(prev_adj_p, raw_p * multiplier))
        prev_adj_p = adj_p
        threshold = alpha / multiplier
        if rejected_so_far and raw_p <= threshold:
            significant = True
        else:
            rejected_so_far = False
            significant = False
        results[name] = {
            "raw_p": raw_p,
            "holm_adjusted_p": adj_p,
            "alpha_threshold": threshold,
            "significant_at_0_05": significant,
        }
    return results


def summarize_runs(runs: Dict[str, Dict[str, List[PipelineState]]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"architectures": {}, "hypothesis_tests": {}}
    epr_vectors: Dict[str, List[int]] = {}

    for arch in ["B0", "B1", "B2", "B3"]:
        clean_states = runs[arch]["clean"]
        inj_states = runs[arch]["injected"]
        n = max(1, len(inj_states))

        epr_final_list = [s.epr_final for s in inj_states]
        epr_Vectors_arch = epr_final_list
        epr_vectors[arch] = epr_Vectors_arch

        epr_final = sum(epr_final_list) / n
        epr_inter = sum(s.epr_intermediate for s in inj_states) / n
        faa_clean = sum(s.faa for s in clean_states) / max(1, len(clean_states))
        rr_injected = sum(s.faa for s in inj_states) / n
        far_injected = sum(s.verifier_far for s in inj_states) / n

        all_states = clean_states + inj_states
        n_all = max(1, len(all_states))
        mean_calls = sum(s.model_calls for s in all_states) / n_all
        mean_in_tok = sum(s.input_tokens for s in all_states) / n_all
        mean_out_tok = sum(s.output_tokens for s in all_states) / n_all
        mean_latency = sum(s.latency_seconds for s in all_states) / n_all

        summary["architectures"][arch] = {
            "N": len(inj_states),
            "EPR_final": round(epr_final, 4),
            "EPR_intermediate": round(epr_inter, 4),
            "FAA_clean": round(faa_clean, 4),
            "RR_injected": round(rr_injected, 4),
            "Verifier_FAR": round(far_injected, 4),
            "mean_model_calls": round(mean_calls, 2),
            "mean_input_tokens": round(mean_in_tok, 1),
            "mean_output_tokens": round(mean_out_tok, 1),
            "mean_latency_s": round(mean_latency, 3),
        }

    # Confirmatory Paired McNemar Tests (Section 4.8: H1, H2, H3)
    m_h1 = exact_mcnemar_test(epr_vectors["B0"], epr_vectors["B2"])
    m_h2 = exact_mcnemar_test(epr_vectors["B2"], epr_vectors["B3"])
    m_h3 = exact_mcnemar_test(epr_vectors["B0"], epr_vectors["B3"])

    holm = holm_bonferroni(
        {
            "H1_B0_vs_B2": m_h1["p_value"],
            "H2_B2_vs_B3": m_h2["p_value"],
            "H3_B0_vs_B3": m_h3["p_value"],
        }
    )

    summary["hypothesis_tests"] = {
        "H1_B0_vs_B2": {**m_h1, **holm["H1_B0_vs_B2"]},
        "H2_B2_vs_B3": {**m_h2, **holm["H2_B2_vs_B3"]},
        "H3_B0_vs_B3": {**m_h3, **holm["H3_B0_vs_B3"]},
    }
    return summary


def print_summary_table(summary: Dict[str, object]) -> None:
    print("\n" + "=" * 88)
    print(" ISOBENCH EMPIRICAL SUMMARY (SECTION 4.7 & 4.8)")
    print("=" * 88)
    header = (
        f"{'Arch':<6} | {'EPR_final':<10} | {'EPR_inter':<10} | {'FAA_clean':<10} | "
        f"{'RR_inj':<8} | {'Ver_FAR':<8} | {'InTok':<7} | {'Lat(s)':<7}"
    )
    print(header)
    print("-" * 88)
    arch_data = summary["architectures"]
    for arch in ["B0", "B1", "B2", "B3"]:
        m = arch_data[arch]
        print(
            f"{arch:<6} | {m['EPR_final']:<10.3f} | {m['EPR_intermediate']:<10.3f} | "
            f"{m['FAA_clean']:<10.3f} | {m['RR_injected']:<8.3f} | {m['Verifier_FAR']:<8.3f} | "
            f"{m['mean_input_tokens']:<7.1f} | {m['mean_latency_s']:<7.2f}"
        )
    print("-" * 88)
    print("Paired Exact McNemar Tests (Holm-Bonferroni Corrected at alpha = 0.05):")
    for h_name, h_stats in summary["hypothesis_tests"].items():
        print(
            f"  {h_name:<14}: discordant (n10={h_stats['n10']}, n01={h_stats['n01']}), "
            f"raw_p={h_stats['raw_p']:.4g}, holm_p={h_stats['holm_adjusted_p']:.4g}, "
            f"sig={h_stats['significant_at_0_05']}"
        )
    print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IsoBench 4-Arm Counterfactual Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("ISOBENCH_LIMIT", "0")),
        help="Optional number of questions to evaluate (0 = all 75 questions).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results_summary.json"),
        help="Path to save JSON summary and statistical results.",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), "counterfactual_dataset.json")
    questions = load_dataset(dataset_path)
    if args.limit > 0:
        questions = questions[: args.limit]

    runs: Dict[str, Dict[str, List[PipelineState]]] = {
        arch: {"clean": [], "injected": []} for arch in ["B0", "B1", "B2", "B3"]
    }

    print(f"Testing IsoBench Harness across {len(questions)} Counterfactual Questions...")
    for i, q in enumerate(questions):
        print(f"\n========================================")
        print(f" QUESTION {i + 1}/{len(questions)} [{q.id}]: {q.question}")
        print(f"========================================")

        for arch in ["B0", "B1", "B2", "B3"]:
            clean_state = run_pipeline(q, "clean", arch)
            inj_state = run_pipeline(q, "injected", arch)
            runs[arch]["clean"].append(clean_state)
            runs[arch]["injected"].append(inj_state)

    summary = summarize_runs(runs)
    print_summary_table(summary)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved IsoBench summary metrics to {args.output}")


if __name__ == "__main__":
    main()
