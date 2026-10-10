<div align="center">
  <h1>🛡️ IsoBench</h1>
  <p><strong>A Confound-Free Laboratory Instrument for Evaluating Error Propagation in Multi-Agent RAG Systems</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![Framework](https://img.shields.io/badge/Framework-Zero%20Dependencies-red.svg)]()
</div>

<br>

## 📖 Overview

**IsoBench** is a pure, minimal Python evaluation framework designed to empirically test **Error Propagation** and **Context Isolation** in multi-agent Retrieval-Augmented Generation (RAG) architectures. 

It was built specifically to evaluate compounding hallucinations in compound AI systems, demonstrating that without strict topological isolation, an upstream hallucination guarantees a downstream systemic failure.

By completely avoiding bulky frameworks like LangChain or AutoGen, IsoBench acts as a mathematically pure laboratory instrument.

---

## 🔬 Methodology: The Counterfactual Dataset

Standard datasets (like HotpotQA) are highly compromised. Modern LLMs have memorized Wikipedia, allowing them to bypass corrupted RAG context using pre-trained weights (the "cheat sheet" effect). 

IsoBench solves this via a **Counterfactual Dataset**:
* **Fictional Entities**: Questions use entirely fictional names (e.g., *The Aether Diaries*).
* **Parametric Blinding**: The LLM possesses exactly zero prior knowledge, guaranteeing 100% reliance on the provided context.
* **Controlled Injections**: The dataset splits evidence into `hop1`, `hop2_correct`, and `hop2_injected` to perfectly simulate a multi-hop RAG retrieval failure.

---

## 🏗️ The 4-Arm Architectural Ablation

IsoBench evaluates identical questions across four distinct agentic paradigms. This diagram explicitly highlights **Context Isolation**—showing whether downstream agents can see the "raw thoughts" of previous agents, or if they are blinded and only see extracted claims.

```mermaid
graph TD
    subgraph B0: Shared Baseline (No Isolation)
        P0[Planner] -- "Sees Planner's Raw Thoughts" --> V0[Verifier]
        V0 -- "Sees Planner & Verifier's Raw Thoughts" --> E0[Executor]
    end

    subgraph B1: Isolated Verifier
        P1[Planner] -- "Isolated: Only sees extracted claim" --> V1[Verifier]
        V1 -- "Sees Planner's Raw Thoughts" --> E1[Executor]
    end

    subgraph B2: Full Context Isolation
        P2[Planner] -- "Isolated: Only sees extracted claim" --> V2[Verifier]
        V2 -- "Isolated: Only sees verified claims" --> E2[Executor]
    end

    subgraph B3: Strict Isolation + Verification Gate
        P3[Planner] -- "Isolated: Only sees extracted claim" --> V3{Verification Gate}
        V3 -- "Claim Rejected" --> H[Halt RAG Pipeline]
        V3 -- "Isolated: Only sees verified claims" --> E3[Executor]
    end
```

### Architectural Breakdown ($B_0$ through $B_3$):

1. **$B_0$ (Shared Baseline)**: Highly vulnerable. The Verifier and Executor agents read the Planner's full, raw "thought process". If the Planner hallucinates, the downstream agents get infected via context contamination.
2. **$B_1$ (Isolated Verifier)**: Diagnostic. The Verifier is blinded from the reasoning trace to prevent sycophancy (agreeing with the Planner just because the Planner sounds confident). However, the Executor remains vulnerable to the global context.
3. **$B_2$ (Isolation Only)**: Structural cut-off. All downstream agents only see structured, extracted claims. The raw reasoning trace is completely purged from the context window, so they do not see each other's work.
4. **$B_3$ (Isolation + Verification Gate)**: The robust architecture. Claims are isolated AND must pass a strict boolean gate. *Crucially, IsoBench proves that $B_3$ only succeeds if it enforces Strict Retrieval Isolation (halting downstream RAG search upon rejection).*

---

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.10+
* `requests` (for local Ollama execution)
* `anthropic` (if using Claude API)
* A local instance of [Ollama](https://ollama.com/) running `llama3.1` OR an Anthropic API Key for **Claude 3.5 Sonnet**.

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HamseMoismaila/IsoBench.git
   cd IsoBench
   ```

2. **Option A: Run with Anthropic API (Claude 3.5 Sonnet)**
   IsoBench defaults to using Claude via the Anthropic API for maximum reasoning capability.
   ```bash
   # On Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   python run_isobench.py
   
   # On Linux/Mac
   export ANTHROPIC_API_KEY="your-api-key-here"
   python run_isobench.py
   ```

3. **Option B: Run with Local LLMs (Ollama / Llama 3.1)**
   You can easily swap out the backend to run entirely locally using Ollama.
   ```bash
   # On Windows (PowerShell)
   $env:AGENT_LLM_BACKEND="ollama"
   python run_isobench.py
   
   # On Linux/Mac
   AGENT_LLM_BACKEND=ollama python run_isobench.py
   ```

### Output Interpretation
The script outputs the exact reasoning trace, the verifier's extraction, and the Executor's Final Answer for all 4 architectures under both `clean` and `injected` conditions. Calculate the system's **Error Propagation Rate (EPR)** by observing the Final Answer under the injected conditions.
