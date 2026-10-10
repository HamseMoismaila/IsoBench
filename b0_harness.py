import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import requests
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
    evidence_corpus: str

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
    injected_fact: str
    cycles: List[CycleState] = field(default_factory=list)
    final_answer: Optional[str] = None

# ---------------------------------------------------------------------------
# LLM INTERFACE (OLLAMA)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" # Change to whatever local model you have pulled

def query_llm(prompt: str, system_prompt: str = "") -> str:
    """Queries the local Ollama instance with temperature 0 for reproducibility."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"[!] LLM Request Failed: {e}")
        return ""

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES & PARSERS
# ---------------------------------------------------------------------------

def parse_planner_output(output: str) -> Tuple[str, str]:
    """Extracts reasoning and claim from the Planner's structured output."""
    reasoning = ""
    claim = ""
    
    # Try regex first to handle missing closing tags or trailing content
    reasoning_match = re.search(r"<reasoning>(.*?)(?:</reasoning>|$)", output, re.DOTALL)
    claim_match = re.search(r"<claim>(.*?)(?:</claim>|$)", output, re.DOTALL)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    if claim_match:
        claim = claim_match.group(1).strip()
        
    return reasoning, claim

def parse_verifier_output(output: str) -> str:
    """Extracts SUPPORTED, CONTRADICTED, or INSUFFICIENT."""
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
    
    # Fallback if the LLM fails to format correctly
    if not claim:
        reasoning = raw_response
        claim = raw_response
        
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
    system_prompt = (
        "You are the final execution agent. Read the entire reasoning history and provide a concise, direct answer to the question."
    )
    
    return query_llm(prompt=context, system_prompt=system_prompt)

# ---------------------------------------------------------------------------
# B0 CONTEXT BUILDER
# ---------------------------------------------------------------------------

def build_b0_context(state: PipelineState, role: str) -> str:
    """
    In B0, ALL agents see the fully accumulating raw trajectory.
    """
    context = f"Question: {state.question.question}\n"
    context += f"Retrieved Fact: {state.injected_fact}\n"
    
    if role == "Verifier":
         context += f"Evidence Corpus:\n{state.question.evidence_corpus}\n"
    
    context += "\n--- PREVIOUS REASONING HISTORY ---\n"
    if not state.cycles:
        context += "(No history yet. This is Cycle 1.)\n"
    else:
        for cycle in state.cycles:
            context += f"Cycle {cycle.cycle_number}:\n"
            context += f"  Planner Reasoning: {cycle.planner_raw_reasoning}\n"
            context += f"  Planner Claim: {cycle.planner_extracted_claim}\n"
            context += f"  Verifier Result: {cycle.verifier_result}\n\n"
            
    return context

# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

def run_b0_pipeline(question: HotpotQAQuestion, condition: str) -> PipelineState:
    
    # 1. Initialize State with Injection
    if condition == "injected":
        injected_fact = f"{question.bridge_entity_injected} is the answer to the first hop of this question."
    else:
        # Design B: Clean condition inserts the correct fact using the exact same formatting.
        injected_fact = f"{question.bridge_entity_correct} is the answer to the first hop of this question."
        
    state = PipelineState(question=question, condition=condition, injected_fact=injected_fact)
    
    print(f"--- Starting B0 Pipeline (Condition: {condition}) ---")
    
    # 2. Cycle 1
    print("Running Cycle 1: Planner...")
    p1_context = build_b0_context(state, "Planner")
    p1_reasoning, p1_claim = run_planner(p1_context)
    
    print(f"Running Cycle 1: Verifier... (Verifying: {p1_claim})")
    v1_context = build_b0_context(state, "Verifier")
    v1_result = run_verifier(v1_context, p1_claim)
    
    state.cycles.append(CycleState(1, p1_reasoning, p1_claim, v1_result))
    
    # 3. Cycle 2
    print("Running Cycle 2: Planner...")
    p2_context = build_b0_context(state, "Planner")
    p2_reasoning, p2_claim = run_planner(p2_context)
    
    print(f"Running Cycle 2: Verifier... (Verifying: {p2_claim})")
    v2_context = build_b0_context(state, "Verifier")
    v2_result = run_verifier(v2_context, p2_claim)
    
    state.cycles.append(CycleState(2, p2_reasoning, p2_claim, v2_result))
    
    # 4. Executor
    print("Running Executor...")
    exec_context = build_b0_context(state, "Executor")
    final_answer = run_executor(exec_context)
    state.final_answer = final_answer
    
    print(f"Final Answer: {final_answer}")
    return state

# ---------------------------------------------------------------------------
# DEMO / ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A dummy HotpotQA question to test the pipeline loop
    test_q = HotpotQAQuestion(
        id="test-1",
        question="What is the birthplace of the person who discovered polonium?",
        bridge_entity_correct="Marie Curie",
        bridge_entity_injected="Irène Joliot-Curie", # Plausible but false bridge
        evidence_corpus=(
            "Marie Curie discovered polonium and radium. She was born in Warsaw, Poland. "
            "Irène Joliot-Curie was Marie's daughter. She was born in Paris, France."
        )
    )
    
    print("=== Testing B0 INJECTED Condition ===")
    injected_state = run_b0_pipeline(test_q, "injected")
    
    print("\n=== Testing B0 CLEAN Condition ===")
    clean_state = run_b0_pipeline(test_q, "clean")
