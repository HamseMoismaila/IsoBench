import json
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
try:
    import anthropic
except ImportError:
    anthropic = None
import re

# Fix Windows console unicode printing
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class HotpotQAQuestion:
    id: str
    question: str
    bridge_entity_correct: str
    bridge_entity_injected: str
    evidence_hop1: str
    evidence_hop2_correct: str
    evidence_hop2_injected: str

@dataclass
class CycleState:
    cycle_number: int
    planner_raw_reasoning: str
    planner_extracted_claim: str
    verifier_result: str

@dataclass
class PipelineState:
    question: HotpotQAQuestion
    condition: str  # "clean" or "injected"
    architecture: str # "B0", "B1", "B2", "B3"
    injected_fact: str
    cycles: List[CycleState] = field(default_factory=list)
    passed_state_claims: List[str] = field(default_factory=list) # B2 passes all claims; B3 passes only verified claims
    final_answer: Optional[str] = None

# ---------------------------------------------------------------------------
import requests

# ---------------------------------------------------------------------------
# LLM INTERFACE (DUAL BACKEND: OLLAMA / ANTHROPIC)
# ---------------------------------------------------------------------------

BACKEND = os.environ.get("AGENT_LLM_BACKEND", "ollama") # "ollama" or "anthropic"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

def query_llm(prompt: str, system_prompt: str = "") -> str:
    if BACKEND == "ollama":
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"[!] Ollama Request Failed ({OLLAMA_MODEL}): {e}")
            return ""
    else:
        try:
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"[!] Anthropic Request Failed: {e}")
            return ""

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
    if "SUPPORTED" in output_upper:
        return "SUPPORTED"
    elif "CONTRADICTED" in output_upper:
        return "CONTRADICTED"
    else:
        return "INSUFFICIENT"

# ---------------------------------------------------------------------------
# AGENT ROLES
# ---------------------------------------------------------------------------

def run_planner(context: str) -> Tuple[str, str]:
    system_prompt = (
        "You are an analytical reasoning agent. Read the provided context and history to answer the question.\n"
        "You must output exactly two XML blocks:\n"
        "<reasoning>\nYour step-by-step logic here.\n</reasoning>\n"
        "<claim>\nA single, atomic factual claim representing the conclusion of this reasoning step.\n</claim>"
    )
    raw_response = query_llm(prompt=context, system_prompt=system_prompt)
    reasoning, claim = parse_planner_output(raw_response)
    
    if not claim:
        reasoning, claim = raw_response, raw_response
        
    return reasoning, claim

def run_verifier(context: str, claim: str) -> str:
    system_prompt = (
        "You are a strict NLI Verifier.\n"
        "Analyze the provided context and determine if the target claim is supported by the evidence.\n"
        "Output ONLY ONE of the following words: SUPPORTED, CONTRADICTED, or INSUFFICIENT."
    )
    prompt = f"{context}\n\nTarget Claim to Verify: {claim}\n\nOutput your verdict:"
    raw_response = query_llm(prompt=prompt, system_prompt=system_prompt)
    return parse_verifier_output(raw_response)

def run_executor(context: str) -> str:
    system_prompt = "You are the final answering agent. Read the provided context and output a concise, direct answer to the question."
    return query_llm(prompt=context, system_prompt=system_prompt)

# ---------------------------------------------------------------------------
# CONTEXT BUILDERS FOR B0, B1, B2
# ---------------------------------------------------------------------------

def build_context(state: PipelineState, role: str) -> str:
    """Dynamically builds the context based on the current architecture (B0, B1, B2, B3)"""
    
    # Determine the relevant evidence based on the cycle
    cycle_index = len(state.cycles)
    
    if cycle_index == 0:
        current_evidence = state.question.evidence_hop1
    else:
        current_evidence = state.question.evidence_hop1 + "\n"
        if state.condition == "injected":
            current_evidence += state.question.evidence_hop2_injected
        else:
            current_evidence += state.question.evidence_hop2_correct
            
    context = f"Question: {state.question.question}\n"
    
    # -------------------------------------------------------
    # VERIFIER CONTEXT (B0 vs B1/B2/B3)
    # -------------------------------------------------------
    if role == "Verifier":
        if state.architecture == "B0":
            # B0 Verifier sees FULL RAW HISTORY + Evidence
            context += f"Retrieved Fact: {state.injected_fact}\n"
            context += f"Evidence Corpus:\n{current_evidence}\n"
            context += "\n--- PREVIOUS REASONING HISTORY ---\n"
            for cycle in state.cycles:
                context += f"Cycle {cycle.cycle_number}:\n"
                context += f"  Planner Reasoning: {cycle.planner_raw_reasoning}\n"
                context += f"  Planner Claim: {cycle.planner_extracted_claim}\n"
                context += f"  Verifier Result: {cycle.verifier_result}\n\n"
        else:
            # B1, B2, B3 Verifiers see ONLY EVIDENCE (isolated)
            context += f"Evidence Corpus:\n{current_evidence}\n"
        
        return context

    # -------------------------------------------------------
    # PLANNER / EXECUTOR CONTEXT (B0/B1 vs B2/B3)
    # -------------------------------------------------------
    if state.architecture in ["B0", "B1"]:
        # Planners and Executors in B0/B1 see FULL RAW HISTORY
        context += f"Retrieved Fact: {state.injected_fact}\n"
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
        # Planners and Executors in B2 and B3 do NOT see the raw reasoning trajectory.
        # They receive only structured claims (isolating the context).
        if not state.cycles: # Cycle 1 (no state passed yet)
            context += f"Retrieved Fact: {state.injected_fact}\n"
            context += f"Evidence Corpus:\n{current_evidence}\n"
        else:
            context += f"Evidence Corpus:\n{current_evidence}\n"
            header = "--- INTERMEDIATE STATE CLAIMS (UNFILTERED) ---" if state.architecture == "B2" else "--- VERIFIED STATE ---"
            context += f"\n{header}\n"
            if not state.passed_state_claims:
                if state.architecture == "B3":
                    context += "(Verified state is empty. Previous claims were rejected by verification gate.)\n"
                else:
                    context += "(No claims recorded.)\n"
            else:
                for claim in state.passed_state_claims:
                    context += f"- {claim}\n"
            
    return context

# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def run_pipeline(question: HotpotQAQuestion, condition: str, architecture: str) -> PipelineState:
    
    # 1. Initialize State with Injection
    if condition == "injected":
        injected_fact = f"{question.bridge_entity_injected} is the answer to the first hop of this question."
    else:
        injected_fact = f"{question.bridge_entity_correct} is the answer to the first hop of this question."
        
    state = PipelineState(question=question, condition=condition, architecture=architecture, injected_fact=injected_fact)
    
    print(f"--- Starting {architecture} Pipeline (Condition: {condition}) ---")
    
    # 2. Cycle 1
    print("Running Cycle 1: Planner...")
    p1_context = build_context(state, "Planner")
    p1_reasoning, p1_claim = run_planner(p1_context)
    
    # Active Bridge-Entity Mutation Protocol (Section 4.3):
    # In the injected condition, the upstream extracted bridge claim is systematically mutated
    # into the plausible-but-false entity to test causal error propagation downstream.
    if condition == "injected":
        p1_claim = f"The filmmaker who directed the film is {question.bridge_entity_injected}."
        p1_reasoning += f"\n[Mutation Applied: Bridge entity substituted with {question.bridge_entity_injected}.]"
    
    print(f"Running Cycle 1: Verifier... (Verifying: {p1_claim})")
    v1_context = build_context(state, "Verifier")
    v1_result = run_verifier(v1_context, p1_claim)
    
    state.cycles.append(CycleState(1, p1_reasoning, p1_claim, v1_result))
    
    # Update State for B2 (Isolation Only) vs B3 (Isolation + Verification Gate)
    if architecture == "B2":
        # B2 isolates context from raw reasoning trace, but passes claim without a verification gate
        state.passed_state_claims.append(p1_claim)
    elif architecture == "B3" and v1_result == "SUPPORTED":
        # B3 isolates context AND enforces strict verification gate
        state.passed_state_claims.append(p1_claim)
    
    # 3. Cycle 2
    print("Running Cycle 2: Planner...")
    p2_context = build_context(state, "Planner")
    p2_reasoning, p2_claim = run_planner(p2_context)
    
    print(f"Running Cycle 2: Verifier... (Verifying: {p2_claim})")
    v2_context = build_context(state, "Verifier")
    v2_result = run_verifier(v2_context, p2_claim)
    
    state.cycles.append(CycleState(2, p2_reasoning, p2_claim, v2_result))
    
    # Update State for B2 vs B3
    if architecture == "B2":
        state.passed_state_claims.append(p2_claim)
    elif architecture == "B3" and v2_result == "SUPPORTED":
        state.passed_state_claims.append(p2_claim)
    
    # 4. Executor
    print("Running Executor...")
    exec_context = build_context(state, "Executor")
    final_answer = run_executor(exec_context)
    state.final_answer = final_answer
    
    print(f"Final Answer: {final_answer}\n")
    return state

# ---------------------------------------------------------------------------
# DEMO / ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_q = HotpotQAQuestion(
        id="test-3",
        question="The 1982 film 'The Slumber Party Massacre' was directed by a filmmaker born in what city?",
        bridge_entity_correct="Amy Holden Jones",
        bridge_entity_injected="Rita Mae Brown", 
        evidence_hop1="The Slumber Party Massacre is a 1982 American slasher film directed by Amy Holden Jones and written by Rita Mae Brown.",
        evidence_hop2_correct="Amy Holden Jones is an American screenwriter and film director born in Philadelphia, Pennsylvania.",
        evidence_hop2_injected="Rita Mae Brown is an American writer, activist, and feminist born in Hanover, Pennsylvania."
    )
    
    print("========================================")
    print("           B0: SHARED BASELINE")
    print("========================================")
    run_pipeline(test_q, "clean", "B0")
    run_pipeline(test_q, "injected", "B0")
    
    print("========================================")
    print("           B1: ISOLATED VERIFIER")
    print("========================================")
    run_pipeline(test_q, "clean", "B1")
    run_pipeline(test_q, "injected", "B1")
    
    print("========================================")
    print("         B2: ISOLATION ONLY")
    print("========================================")
    run_pipeline(test_q, "clean", "B2")
    run_pipeline(test_q, "injected", "B2")

    print("========================================")
    print("    B3: ISOLATION + VERIFICATION GATE")
    print("========================================")
    run_pipeline(test_q, "clean", "B3")
    run_pipeline(test_q, "injected", "B3")
