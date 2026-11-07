import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import requests

try:
    import anthropic
except ImportError:
    anthropic = None

# Fix Windows console unicode printing
if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualQuestion:
    id: str
    question: str
    bridge_entity_correct: str
    bridge_entity_injected: str
    evidence_hop1: str
    evidence_hop2_correct: str
    evidence_hop2_injected: str
    target_answer_correct: str = ""
    target_answer_injected: str = ""
    mutated_claim: str = ""

# Alias for backward compatibility
HotpotQAQuestion = CounterfactualQuestion

@dataclass
class CycleState:
    cycle_number: int
    planner_raw_reasoning: str
    planner_extracted_claim: str
    verifier_result: str

@dataclass
class PipelineState:
    question: CounterfactualQuestion
    condition: str     # "clean" or "injected"
    architecture: str  # "B0", "B1", "B2", "B3"
    injected_fact: str
    cycles: List[CycleState] = field(default_factory=list)
    passed_state_claims: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    # Telemetry & Outcome Metrics (Section 4.7)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    epr_intermediate: int = 0
    epr_final: int = 0
    faa: int = 0
    verifier_far: int = 0

# ---------------------------------------------------------------------------
# LLM INTERFACE (DUAL BACKEND: OLLAMA / ANTHROPIC)
# ---------------------------------------------------------------------------

BACKEND = os.environ.get("AGENT_LLM_BACKEND", "ollama")  # "ollama" or "anthropic"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
RANDOM_SEED = int(os.environ.get("ISOBENCH_SEED", "42"))


def _estimate_tokens(text: str) -> int:
    """Fallback token estimator (~4 chars/token) when backend omits token counts."""
    return max(1, len(text.split())) if text else 0


def query_llm_with_stats(prompt: str, system_prompt: str = "") -> Tuple[str, int, int, float]:
    """Queries the configured LLM backend at temperature=0.0 and returns (text, in_tok, out_tok, latency_s)."""
    t0 = time.perf_counter()
    if BACKEND == "ollama":
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": RANDOM_SEED,
                },
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            in_tok = int(data.get("prompt_eval_count") or _estimate_tokens(system_prompt + "\n" + prompt))
            out_tok = int(data.get("eval_count") or _estimate_tokens(text))
            return text, in_tok, out_tok, time.perf_counter() - t0
        except Exception as e:
            print(f"[!] Ollama Request Failed ({OLLAMA_MODEL}): {e}")
            return "", 0, 0, time.perf_counter() - t0
    else:
        try:
            if anthropic is None:
                raise RuntimeError("The 'anthropic' package is not installed.")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or _estimate_tokens(system_prompt + "\n" + prompt))
            out_tok = int(getattr(usage, "output_tokens", 0) or _estimate_tokens(text))
            return text, in_tok, out_tok, time.perf_counter() - t0
        except Exception as e:
            print(f"[!] Anthropic Request Failed: {e}")
            return "", 0, 0, time.perf_counter() - t0


def query_llm(prompt: str, system_prompt: str = "") -> str:
    """Backward-compatible wrapper returning only the response string."""
    text, _, _, _ = query_llm_with_stats(prompt, system_prompt)
    return text

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES & PARSERS
# ---------------------------------------------------------------------------

def parse_planner_output(output: str) -> Tuple[str, str]:
    reasoning = ""
    claim = ""

    reasoning_match = re.search(r"<reasoning>(.*?)(?:</reasoning>|$)", output, re.DOTALL)
    claim_match = re.search(r"<claim>(.*?)(?:</claim>|$)", output, re.DOTALL)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    if claim_match:
        claim = claim_match.group(1).strip()

    return reasoning, claim


def parse_verifier_output(output: str) -> str:
    output_upper = output.upper()
    if "CONTRADICTED" in output_upper:
        return "CONTRADICTED"
    elif "INSUFFICIENT" in output_upper:
        return "INSUFFICIENT"
    elif "SUPPORTED" in output_upper:
        return "SUPPORTED"
    return "INSUFFICIENT"


def apply_bridge_entity_mutation(
    question: CounterfactualQuestion,
    raw_reasoning: str,
    extracted_claim: str,
) -> Tuple[str, str]:
    """
    Section 4.3 Error Injection Protocol:
    Substitutes the gold Hop-1 bridge entity (b_gold -> b_mut) inside both the persuasive
    upstream Chain-of-Thought narrative and the extracted atomic claim without leaking
    any meta-tags (e.g., no '[Mutation Applied]' string).
    """
    b_gold = question.bridge_entity_correct
    b_mut = question.bridge_entity_injected

    pattern = re.compile(re.escape(b_gold), re.IGNORECASE)
    if pattern.search(raw_reasoning):
        mutated_reasoning = pattern.sub(b_mut, raw_reasoning)
    else:
        mutated_reasoning = (
            f"{raw_reasoning}\nTherefore, based on the initial Hop-1 analysis, "
            f"the connecting bridge entity is {b_mut}."
        ).strip()

    if question.mutated_claim:
        mutated_claim = question.mutated_claim
    elif pattern.search(extracted_claim):
        mutated_claim = pattern.sub(b_mut, extracted_claim)
    else:
        mutated_claim = f"For the first hop of the question, the connecting entity is {b_mut}."

    return mutated_reasoning, mutated_claim

# ---------------------------------------------------------------------------
# NORMALIZED ENTITY MATCHING & OUTCOME METRICS (SECTION 4.7)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_entity(text: str, entity: str) -> bool:
    """Normalized string and alias matching (Section 4.7)."""
    norm_text = _normalize_text(text)
    norm_entity = _normalize_text(entity)
    if not norm_text or not norm_entity:
        return False
    return f" {norm_entity} " in f" {norm_text} "


def score_pipeline_state(state: PipelineState) -> None:
    """
    Computes primary and secondary outcome metrics defined in Section 4.7:
    - epr_final: 1 if final_answer adopts b_mut or a_mut (under injected condition)
    - epr_intermediate: 1 if Cycle 2 claim adopts b_mut or a_mut
    - faa: 1 if final_answer matches a_gold (or b_gold if target_answer_correct is identical)
    - verifier_far: 1 if Cycle 1 Verifier in injected condition marks mutated claim SUPPORTED
    """
    q = state.question
    final_ans = state.final_answer or ""

    gold_target = q.target_answer_correct or q.bridge_entity_correct
    inj_target = q.target_answer_injected or q.bridge_entity_injected

    # Final Answer Accuracy (FAA)
    state.faa = int(
        contains_entity(final_ans, gold_target)
        and not contains_entity(final_ans, q.bridge_entity_injected)
        and not contains_entity(final_ans, inj_target)
    )

    # Error Propagation Rate (final)
    state.epr_final = int(
        contains_entity(final_ans, q.bridge_entity_injected)
        or (bool(q.target_answer_injected) and contains_entity(final_ans, q.target_answer_injected))
    )

    # Intermediate Error Propagation Rate (Cycle 2 Planner output)
    if len(state.cycles) >= 2:
        c2_text = f"{state.cycles[1].planner_raw_reasoning} {state.cycles[1].planner_extracted_claim}"
        state.epr_intermediate = int(
            contains_entity(c2_text, q.bridge_entity_injected)
            or (bool(q.target_answer_injected) and contains_entity(c2_text, q.target_answer_injected))
        )

    # Verifier False Acceptance Rate (Cycle 1 Verifier on injected claim)
    if state.condition == "injected" and len(state.cycles) >= 1:
        state.verifier_far = int(state.cycles[0].verifier_result == "SUPPORTED")

# ---------------------------------------------------------------------------
# AGENT ROLES
# ---------------------------------------------------------------------------

def run_planner(state: PipelineState, context: str) -> Tuple[str, str]:
    system_prompt = (
        "You are an analytical reasoning agent. Read the provided context and history to answer the question.\n"
        "You must output exactly two XML blocks:\n"
        "<reasoning>\nYour step-by-step logic here.\n</reasoning>\n"
        "<claim>\nA single, atomic factual claim representing the conclusion of this reasoning step.\n</claim>"
    )
    raw_response, in_tok, out_tok, dt = query_llm_with_stats(prompt=context, system_prompt=system_prompt)
    state.model_calls += 1
    state.input_tokens += in_tok
    state.output_tokens += out_tok
    state.latency_seconds += dt

    reasoning, claim = parse_planner_output(raw_response)
    if not claim:
        reasoning, claim = raw_response, raw_response
    return reasoning, claim


def run_verifier(state: PipelineState, context: str, claim: str) -> str:
    system_prompt = (
        "You are a strict NLI Verifier.\n"
        "Analyze the provided Evidence Corpus and determine if the Target Claim is supported by the evidence.\n"
        "Output ONLY ONE of the following words: SUPPORTED, CONTRADICTED, or INSUFFICIENT."
    )
    prompt = f"{context}\n\nTarget Claim to Verify: {claim}\n\nOutput your verdict:"
    raw_response, in_tok, out_tok, dt = query_llm_with_stats(prompt=prompt, system_prompt=system_prompt)
    state.model_calls += 1
    state.input_tokens += in_tok
    state.output_tokens += out_tok
    state.latency_seconds += dt

    return parse_verifier_output(raw_response)


def run_executor(state: PipelineState, context: str) -> str:
    system_prompt = (
        "You are the final answering agent. Read the provided context and output a concise, "
        "direct answer to the question."
    )
    raw_response, in_tok, out_tok, dt = query_llm_with_stats(prompt=context, system_prompt=system_prompt)
    state.model_calls += 1
    state.input_tokens += in_tok
    state.output_tokens += out_tok
    state.latency_seconds += dt
    return raw_response

# ---------------------------------------------------------------------------
# CONTEXT BUILDERS FOR B0, B1, B2, B3 (SECTION 4.4)
# ---------------------------------------------------------------------------

def build_context(state: PipelineState, role: str) -> str:
    """
    Dynamically constructs agent context for B0, B1, B2, B3 in accordance with Section 4.4:
    - Cycle 1 (Hop 1): Evidence is E_1 (evidence_hop1).
    - Cycle 2 (Hop 2) & Executor:
      * In Clean condition: E_1 + E_{2,correct}.
      * In Injected condition:
        - If B3 rejected the Cycle 1 claim (UNVERIFIED_FALLBACK), corrupted Hop-2 retrieval
          is halted and the downstream agent receives clean fallback context (E_1 + E_{2,correct})
          with zero exposure to the rejected claim.
        - Otherwise, the Hop-2 retrieval pool contains E_1 + E_{2,injected} + E_{2,correct}.
    """
    cycle_index = len(state.cycles)

    if cycle_index == 0:
        current_evidence = state.question.evidence_hop1
    else:
        if state.condition == "clean":
            current_evidence = f"{state.question.evidence_hop1}\n{state.question.evidence_hop2_correct}"
        else:
            # Injected condition: check if B3 gate rejected Cycle 1 claim (UNVERIFIED_FALLBACK)
            b3_halted_corrupted_hop2 = (
                state.architecture == "B3"
                and len(state.cycles) >= 1
                and state.cycles[0].verifier_result != "SUPPORTED"
            )
            if b3_halted_corrupted_hop2:
                current_evidence = f"{state.question.evidence_hop1}\n{state.question.evidence_hop2_correct}"
            else:
                current_evidence = (
                    f"{state.question.evidence_hop1}\n"
                    f"{state.question.evidence_hop2_injected}\n"
                    f"{state.question.evidence_hop2_correct}"
                )

    context = f"Question: {state.question.question}\n"

    # -------------------------------------------------------
    # VERIFIER CONTEXT (B0 vs B1/B2/B3)
    # -------------------------------------------------------
    if role == "Verifier":
        if state.architecture == "B0":
            # B0 Verifier is exposed to the upstream premise and full reasoning history
            context += f"Upstream Premise: {state.injected_fact}\n"
            context += f"Evidence Corpus:\n{current_evidence}\n"
            if state.cycles:
                context += "\n--- PREVIOUS REASONING HISTORY ---\n"
                for cycle in state.cycles:
                    context += f"Cycle {cycle.cycle_number}:\n"
                    context += f"  Planner Reasoning: {cycle.planner_raw_reasoning}\n"
                    context += f"  Planner Claim: {cycle.planner_extracted_claim}\n"
                    context += f"  Verifier Result: {cycle.verifier_result}\n\n"
        else:
            # B1, B2, B3 Isolated Verifier sees ONLY the Evidence Corpus and Target Claim
            context += f"Evidence Corpus:\n{current_evidence}\n"
        return context

    # -------------------------------------------------------
    # PLANNER / EXECUTOR CONTEXT (B0/B1 vs B2/B3)
    # -------------------------------------------------------
    if state.architecture in ["B0", "B1"]:
        # Shared-context downstream agents receive the upstream premise and full raw trajectory
        context += f"Upstream Premise: {state.injected_fact}\n"
        context += f"Evidence Corpus:\n{current_evidence}\n"
        context += "\n--- PREVIOUS REASONING HISTORY ---\n"
        if not state.cycles:
            context += "(No history yet. This is Cycle 1.)\n"
        else:
            for cycle in state.cycles:
                context += f"Cycle {cycle.cycle_number}:\n"
                context += f"  Planner Reasoning: {cycle.planner_raw_reasoning}\n"
                context += f"  Planner Claim: {cycle.planner_extracted_claim}\n"
                context += f"  Verifier Result: {cycle.verifier_result}\n\n"

    elif state.architecture in ["B2", "B3"]:
        # Isolated downstream agents NEVER receive raw upstream CoT reasoning
        if not state.cycles:
            context += f"Evidence Corpus:\n{current_evidence}\n"
        else:
            context += f"Evidence Corpus:\n{current_evidence}\n"
            header = (
                "--- INTERMEDIATE STATE CLAIMS (UNFILTERED) ---"
                if state.architecture == "B2"
                else "--- VERIFIED STATE ---"
            )
            context += f"\n{header}\n"
            if not state.passed_state_claims:
                if state.architecture == "B3":
                    context += (
                        "[UNVERIFIED_FALLBACK]: Previous upstream claim was rejected by the verification gate. "
                        "Resolve the question directly from the Evidence Corpus above.\n"
                    )
                else:
                    context += "(No claims recorded.)\n"
            else:
                for claim in state.passed_state_claims:
                    context += f"- Bridge_Entity_Candidate: {claim}\n"

    return context

# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def run_pipeline(
    question: CounterfactualQuestion,
    condition: str,
    architecture: str,
    verbose: bool = True,
) -> PipelineState:
    if condition == "injected":
        injected_fact = f"{question.bridge_entity_injected} is the connecting entity for the first hop of this question."
    else:
        injected_fact = f"{question.bridge_entity_correct} is the connecting entity for the first hop of this question."

    state = PipelineState(
        question=question,
        condition=condition,
        architecture=architecture,
        injected_fact=injected_fact,
    )

    if verbose:
        print(f"--- Starting {architecture} Pipeline (Condition: {condition}) ---")

    # 1. Cycle 1 (Hop 1)
    if verbose:
        print("Running Cycle 1: Planner...")
    p1_context = build_context(state, "Planner")
    p1_reasoning, p1_claim = run_planner(state, p1_context)

    # Active Bridge-Entity Substitution Protocol (Section 4.3)
    if condition == "injected":
        p1_reasoning, p1_claim = apply_bridge_entity_mutation(question, p1_reasoning, p1_claim)

    if verbose:
        print(f"Running Cycle 1: Verifier... (Verifying: {p1_claim})")
    v1_context = build_context(state, "Verifier")
    # In B0, also expose the current Cycle 1 reasoning narrative to the shared-context Verifier
    if architecture == "B0":
        v1_context += f"\nCycle 1 Planner Reasoning:\n{p1_reasoning}\n"
    v1_result = run_verifier(state, v1_context, p1_claim)

    state.cycles.append(CycleState(1, p1_reasoning, p1_claim, v1_result))

    if architecture == "B2":
        state.passed_state_claims.append(p1_claim)
    elif architecture == "B3" and v1_result == "SUPPORTED":
        state.passed_state_claims.append(p1_claim)

    # 2. Cycle 2 (Hop 2)
    if verbose:
        print("Running Cycle 2: Planner...")
    p2_context = build_context(state, "Planner")
    p2_reasoning, p2_claim = run_planner(state, p2_context)

    if verbose:
        print(f"Running Cycle 2: Verifier... (Verifying: {p2_claim})")
    v2_context = build_context(state, "Verifier")
    if architecture == "B0":
        v2_context += f"\nCycle 2 Planner Reasoning:\n{p2_reasoning}\n"
    v2_result = run_verifier(state, v2_context, p2_claim)

    state.cycles.append(CycleState(2, p2_reasoning, p2_claim, v2_result))

    if architecture == "B2":
        state.passed_state_claims.append(p2_claim)
    elif architecture == "B3" and v2_result == "SUPPORTED":
        state.passed_state_claims.append(p2_claim)

    # 3. Executor
    if verbose:
        print("Running Executor...")
    exec_context = build_context(state, "Executor")
    final_answer = run_executor(state, exec_context)
    state.final_answer = final_answer

    score_pipeline_state(state)

    if verbose:
        print(
            f"Final Answer: {final_answer} | "
            f"EPR_final={state.epr_final}, FAA={state.faa}, FAR={state.verifier_far}\n"
        )
    return state


if __name__ == "__main__":
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

    for arch in ["B0", "B1", "B2", "B3"]:
        print("=" * 48)
        print(f"           {arch} ARCHITECTURE")
        print("=" * 48)
        run_pipeline(test_q, "clean", arch)
        run_pipeline(test_q, "injected", arch)
