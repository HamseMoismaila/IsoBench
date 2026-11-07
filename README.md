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

IsoBench solves this via a **Counterfactual Dataset ($N = 75$)**:
* **Fictional Entities**: Questions use entirely synthetic counterfactual names across 7 domains (e.g., *The Aether Diaries*, *Kaelen Thorne*, *The Modern Sentinel*).
* **Parametric Blinding**: The LLM possesses exactly zero prior knowledge ($P(\text{knowledge} \mid \text{pretraining}) = 0$), guaranteeing 100% reliance on the provided context.
* **Controlled Two-Hop Injections**: Each instance splits evidence into `evidence_hop1` ($E_1$), `evidence_hop2_correct` ($E_{2,\text{correct}}$), and `evidence_hop2_injected` ($E_{2,\text{injected}}$), separating the Hop-1 bridge entities (`bridge_entity_correct`, `bridge_entity_injected`) from the Hop-2 target answers (`target_answer_correct`, `target_answer_injected`).

---

## 🏗️ The 4-Arm Architectural Ablation

IsoBench evaluates identical questions across four distinct agentic paradigms. This diagram explicitly visualizes the **Context Windows**. Agents inside the same "Shared Context" box can see each other's raw thoughts, while agents in "Isolated Context" boxes are blinded to everything except the specific claim passed to them.

```mermaid
graph TD
    %% B0
    subgraph B0 [B0: Shared Baseline]
        subgraph S0 [Shared Context Window]
            P0(Planner)
            V0(Verifier)
            E0(Executor)
        end
        P0 -->|Passes Raw Thoughts| V0
        V0 -->|Passes Raw Thoughts| E0
    end

    %% B1
    subgraph B1 [B1: Isolated Verifier]
        subgraph S1 [Shared Context Window]
            P1(Planner)
            E1(Executor)
        end
        subgraph I1 [Isolated Context Window]
            V1(Verifier)
        end
        P1 -.->|Passes Extracted Claim| V1
        P1 -->|Passes Raw Thoughts| E1
    end

    %% B2
    subgraph B2 [B2: Full Isolation]
        subgraph I2_P [Isolated Context Window]
            P2(Planner)
        end
        subgraph I2_V [Isolated Context Window]
            V2(Verifier)
        end
        subgraph I2_E [Isolated Context Window]
            E2(Executor)
        end
        P2 -.->|Passes Extracted Claim| V2
        V2 -.->|Passes Verified Claim| E2
    end

    %% B3
    subgraph B3 [B3: Isolation + Verification Gate]
        subgraph I3_P [Isolated Context Window]
            P3(Planner)
        end
        subgraph I3_V [Strict Verification Gate]
            V3{Verifier}
        end
        subgraph I3_E [Isolated Context Window]
            E3(Executor)
        end
        P3 -.->|Passes Extracted Claim| V3
        V3 -.->|Rejected| H3[Halt Downstream RAG]
        V3 -.->|Passes Verified Claim| E3
    end
```

### Architectural Breakdown ($B_0$ through $B_3$):

1. **$B_0$ (Shared Baseline)**: Highly vulnerable. All agents live in one Shared Context Window. The Verifier and Executor agents read the Planner's full, raw "thought process". If the Planner hallucinates, the downstream agents get infected via context contamination.
2. **$B_1$ (Isolated Verifier)**: Diagnostic. The Verifier is placed in an Isolated Context Window so it cannot be influenced by the Planner's confident thoughts (preventing sycophancy). However, the Executor still lives in the Shared Context Window and remains vulnerable.
3. **$B_2$ (Isolation Only)**: Structural cut-off. Every agent is placed in its own Isolated Context Window. Downstream agents only see structured, extracted claims. The raw reasoning trace is completely purged from the pipeline.
4. **$B_3$ (Isolation + Verification Gate)**: The robust architecture. Context is fully isolated AND claims must pass a strict boolean gate. *Crucially, IsoBench proves that $B_3$ only succeeds if it enforces Strict Retrieval Isolation (halting downstream RAG search upon rejection).*

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

### Output & Automated Statistical Analysis
`run_isobench.py` evaluates all 4 architectures ($B_0, B_1, B_2, B_3$) under both `clean` and `injected` conditions (pass `--limit 5` for a quick 5-question test or omit `--limit` to run all $N = 75$ questions) and automatically computes and saves to `results_summary.json`:
* **Primary Metric**: Final Error Propagation Rate ($EPR_{\text{final}}$)
* **Secondary Metrics**: Intermediate Error Propagation Rate ($EPR_{\text{intermediate}}$), Final Answer Accuracy ($FAA$), Recovery Rate ($RR$), Verifier False Acceptance Rate ($FAR$), and Computational Overhead (model calls, tokens, latency)
* **Hypothesis Tests ($H_1, H_2, H_3$)**: Exact paired **McNemar's test** on discordant pairs with **Holm-Bonferroni** step-down correction at $\alpha = 0.05$.
