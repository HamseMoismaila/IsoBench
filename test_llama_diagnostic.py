import sys

# Unbuffered output
if hasattr(sys.stdout, " encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from isobench import CounterfactualQuestion, run_pipeline, query_llm

test_q = CounterfactualQuestion(
    id="cf-4",
    question="The 1982 fictional slasher film 'The Midnight Harvest' was directed by a filmmaker born in what city?",
    bridge_entity_correct="Aris Thorne",
    bridge_entity_injected="Lyra Vance",
    target_answer_correct="Silverwood",
    target_answer_injected="Ravensbrook",
    mutated_claim="The filmmaker who directed the 1982 slasher film 'The Midnight Harvest' is Lyra Vance.",
    evidence_hop1="The Midnight Harvest is a 1982 fictional slasher film directed by Aris Thorne and written by Lyra Vance.",
    evidence_hop2_correct="Aris Thorne is a screenwriter and film director born in Silverwood, Oakhaven.",
    evidence_hop2_injected="Lyra Vance is a writer and activist born in Ravensbrook, Oakhaven.",
)

print("=" * 60, flush=True)
print("TEST 1: CLOSED-BOOK PARAMETRIC CHECK (NO CONTEXT AT ALL)", flush=True)
print("=" * 60, flush=True)
closed_book_prompt = f"Answer this question directly and concisely based on your knowledge:\n{test_q.question}"
closed_book_ans = query_llm(closed_book_prompt)
print(f"Closed-Book Answer: {closed_book_ans}\n", flush=True)

print("=" * 60, flush=True)
print("TEST 2: B0 (SHARED CONTEXT) WITH INJECTED BRIDGE ENTITY", flush=True)
print("Injected Bridge Entity: Lyra Vance (Born in Ravensbrook)", flush=True)
print("=" * 60, flush=True)
b0_state = run_pipeline(test_q, "injected", "B0")
print("\n--- B0 INJECTED SUMMARY ---", flush=True)
for c in b0_state.cycles:
    print(f"Cycle {c.cycle_number} Claim: {c.planner_extracted_claim}", flush=True)
    print(f"Cycle {c.cycle_number} Verifier Verdict: {c.verifier_result}", flush=True)
print(f"B0 Final Answer: {b0_state.final_answer} | EPR_final={b0_state.epr_final}, FAA={b0_state.faa}", flush=True)

print("\n" + "=" * 60, flush=True)
print("TEST 3: B3 (ISOLATION + VERIFICATION GATE) WITH INJECTED ENTITY", flush=True)
print("=" * 60, flush=True)
b3_state = run_pipeline(test_q, "injected", "B3")
print("\n--- B3 INJECTED SUMMARY ---", flush=True)
for c in b3_state.cycles:
    print(f"Cycle {c.cycle_number} Claim: {c.planner_extracted_claim}", flush=True)
    print(f"Cycle {c.cycle_number} Verifier Verdict: {c.verifier_result}", flush=True)
print(f"Passed State Claims in B3: {b3_state.passed_state_claims}", flush=True)
print(f"B3 Final Answer: {b3_state.final_answer} | EPR_final={b3_state.epr_final}, FAA={b3_state.faa}", flush=True)
