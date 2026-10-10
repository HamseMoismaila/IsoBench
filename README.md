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

IsoBench evaluates identical questions across four distinct agentic paradigms. This ablation study proves exactly where and how errors propagate through an agentic pipeline.

```mermaid
graph TD
    subgraph B0: Shared Baseline
        direction TB
        P0[Planner] -- "Passes Raw Reasoning" --> V0[Verifier]
        V0 -- "Passes Raw Reasoning" --> E0[Executor]
        style P0 fill:#ffe6e6,stroke:#ff0000
        style V0 fill:#ffe6e6,stroke:#ff0000
        style E0 fill:#ffe6e6,stroke:#ff0000
    end

    subgraph B1: Isolated Verifier
        direction TB
        P1[Planner] -- "Passes Extracted Claim Only" --> V1[Verifier]
        V1 -- "Passes Raw Reasoning" --> E1[Executor]
        style P1 fill:#fff5e6,stroke:#ff9900
        style V1 fill:#e6ffe6,stroke:#00cc00
        style E1 fill:#ffe6e6,stroke:#ff0000
    end

    subgraph B2: Isolation Only
        direction TB
        P2[Planner] -- "Passes Extracted Claim Only" --> V2[Verifier]
        V2 -- "Passes Verified Claim Only" --> E2[Executor]
        style P2 fill:#fff5e6,stroke:#ff9900
        style V2 fill:#e6ffe6,stroke:#00cc00
        style E2 fill:#e6ffe6,stroke:#00cc00
    end

    subgraph B3: Isolation + Verification Gate
        direction TB
        P3[Planner] -- "Passes Extracted Claim Only" --> V3{Strict Boolean Gate}
        V3 -- "Rejected (Halts Retrieval)" --> H[Fail Fast]
        V3 -- "Verified Claim Only" --> E3[Executor]
        style P3 fill:#fff5e6,stroke:#ff9900
        style V3 fill:#ccffcc,stroke:#00cc00
        style E3 fill:#ccffcc,stroke:#00cc00
        style H fill:#333333,stroke:#ffffff,color:#ffffff
    end
```

### Architectural Breakdown:
1. **$B_0$ (Shared Baseline)**: Highly vulnerable. The Verifier and Executor agents read the Planner's full, raw "thought process". If the Planner hallucinates, the downstream agents get infected via context contamination.
2. **$B_1$ (Isolated Verifier)**: Diagnostic. The Verifier is blinded from the reasoning trace to prevent sycophancy (agreeing with the Planner just because the Planner sounds confident). However, the Executor remains vulnerable to the global context.
3. **$B_2$ (Isolation Only)**: Structural cut-off. All downstream agents only see structured, extracted claims. The raw reasoning trace is completely purged from the context window.
4. **$B_3$ (Isolation + Verification Gate)**: The robust architecture. Claims are isolated AND must pass a strict boolean gate. *Crucially, IsoBench proves that $B_3$ only succeeds if it enforces Strict Retrieval Isolation (halting downstream RAG search upon rejection).*

---

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.10+
* `requests` (for local Ollama execution)
* [Ollama](https://ollama.com/) running `llama3.1` (or your preferred local model).

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HamseMoismaila/IsoBench.git
   cd IsoBench
   ```

2. **Run the Benchmark (Local Llama 3.1):**
   ```bash
   # On Windows (PowerShell)
   $env:AGENT_LLM_BACKEND="ollama"
   python run_isobench.py
   
   # On Linux/Mac
   AGENT_LLM_BACKEND=ollama python run_isobench.py
   ```

### Output Interpretation
The script outputs the exact reasoning trace, the verifier's extraction, and the Executor's Final Answer for all 4 architectures under both `clean` and `injected` conditions. Calculate the system's **Error Propagation Rate (EPR)** by observing the Final Answer under the injected conditions.
