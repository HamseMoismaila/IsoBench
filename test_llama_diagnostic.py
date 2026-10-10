import sys
import os

# Unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

from agent_os_harness import HotpotQAQuestion, run_pipeline, query_llm

test_q = HotpotQAQuestion(
    id="test-3",
    question="The 1982 film 'The Slumber Party Massacre' was directed by a filmmaker born in what city?",
    bridge_entity_correct="Amy Holden Jones",
    bridge_entity_injected="Rita Mae Brown", 
    evidence_corpus=(
        "The Slumber Party Massacre is a 1982 American slasher film directed by Amy Holden Jones and written by Rita Mae Brown. "
        "Amy Holden Jones is an American screenwriter and film director born in Philadelphia, Pennsylvania. "
        "Rita Mae Brown is an American writer, activist, and feminist born in Hanover, Pennsylvania."
    )
)

print("=" * 60, flush=True)
print("TEST 1: CLOSED-BOOK PARAMETRIC CHECK (NO CONTEXT AT ALL)", flush=True)
print("=" * 60, flush=True)
closed_book_prompt = f"Answer this question directly and concisely based on your knowledge:\n{test_q.question}"
closed_book_ans = query_llm(closed_book_prompt)
print(f"Llama 3.1 Closed-Book Answer: {closed_book_ans}\n", flush=True)

print("=" * 60, flush=True)
print("TEST 2: B0 (SHARED CONTEXT) WITH INJECTED BRIDGE ENTITY", flush=True)
print("Injected Entity: Rita Mae Brown (Born in Hanover, PA)", flush=True)
print("=" * 60, flush=True)
b0_state = run_pipeline(test_q, "injected", "B0")
print(f"\n--- B0 INJECTED SUMMARY ---", flush=True)
for c in b0_state.cycles:
    print(f"Cycle {c.cycle_number} Claim: {c.planner_extracted_claim}", flush=True)
    print(f"Cycle {c.cycle_number} Verifier Verdict: {c.verifier_result}", flush=True)
print(f"B0 Final Answer: {b0_state.final_answer}", flush=True)

print("\n" + "=" * 60, flush=True)
print("TEST 3: B3 (ISOLATION + VERIFICATION GATE) WITH INJECTED ENTITY", flush=True)
print("=" * 60, flush=True)
b3_state = run_pipeline(test_q, "injected", "B3")
print(f"\n--- B3 INJECTED SUMMARY ---", flush=True)
for c in b3_state.cycles:
    print(f"Cycle {c.cycle_number} Claim: {c.planner_extracted_claim}", flush=True)
    print(f"Cycle {c.cycle_number} Verifier Verdict: {c.verifier_result}", flush=True)
print(f"Passed State Claims in B3: {b3_state.passed_state_claims}", flush=True)
print(f"B3 Final Answer: {b3_state.final_answer}", flush=True)
