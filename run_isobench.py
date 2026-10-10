import sys
import json
import os
from isobench import HotpotQAQuestion, run_pipeline

def load_dataset(filepath: str) -> list[HotpotQAQuestion]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        questions.append(HotpotQAQuestion(
            id=item["id"],
            question=item["question"],
            bridge_entity_correct=item["bridge_entity_correct"],
            bridge_entity_injected=item["bridge_entity_injected"],
            evidence_hop1=item["evidence_hop1"],
            evidence_hop2_correct=item["evidence_hop2_correct"],
            evidence_hop2_injected=item["evidence_hop2_injected"]
        ))
    return questions

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'counterfactual_dataset.json')
    questions = load_dataset(dataset_path)
    
    print(f"Testing AgentOS Harness with {len(questions)} Counterfactual Questions...")
    for i, q in enumerate(questions):
        print(f"\n========================================")
        print(f" QUESTION {i+1}: {q.question}")
        print(f"========================================")
        
        print("\n--- [B0: SHARED BASELINE (CLEAN & INJECTED)] ---")
        run_pipeline(q, "clean", "B0")
        run_pipeline(q, "injected", "B0")
        
        print("\n--- [B1: ISOLATED VERIFIER (CLEAN & INJECTED)] ---")
        run_pipeline(q, "clean", "B1")
        run_pipeline(q, "injected", "B1")
        
        print("\n--- [B2: ISOLATION ONLY (CLEAN & INJECTED)] ---")
        run_pipeline(q, "clean", "B2")
        run_pipeline(q, "injected", "B2")

        print("\n--- [B3: ISOLATION + VERIFICATION GATE (CLEAN & INJECTED)] ---")
        run_pipeline(q, "clean", "B3")
        run_pipeline(q, "injected", "B3")

if __name__ == "__main__":
    main()
