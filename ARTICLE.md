# Does Hiding Upstream Reasoning Stop Multi-Agent Hallucinations? Testing Context Isolation with IsoBench

**Author:** Hamse Mohamed Ismail (B.CompSc Data Science, Nilai University)  
**Code & Dataset:** [github.com/HamseMoismaila/IsoBench](https://github.com/HamseMoismaila/IsoBench)

---

## 1. The Problem: Why Multi-Agent Pipelines Cascade Errors (and Why Standard Benchmarks Hide It)

When you chain multiple Large Language Models (LLMs) together—a **Planner** that decomposes a question, a **Verifier** that checks intermediate facts, and an **Executor** that synthesizes the final answer—a single upstream mistake often turns into a systemic failure. If the Planner hallucinates an entity in Step 1 and writes a confident Chain-of-Thought (CoT) justification, downstream agents reading that shared conversation history tend to adopt and rationalize the error rather than correct it.

While building a testbed to measure whether **context isolation** (restricting what downstream agents are allowed to see from upstream agents) can stop these cascades, I ran into two methodological problems in existing evaluations:

1. **The "Wikipedia Cheat-Sheet" Confound (Parametric Memory Leakage):**  
   Standard multi-hop benchmarks like **HotpotQA** are built on real Wikipedia articles. Modern LLMs have already memorized most of those facts during pre-training. If you inject a false intermediate entity into a HotpotQA pipeline and the downstream agent still outputs the right answer, you cannot tell whether your *agent architecture* stopped the error or whether the LLM simply ignored the context and answered from pre-trained memory.
2. **Conflating Context Isolation with Verification Gating:**  
   Prior work on self-correction and verification (such as Reflexion, CRITIC, or Chain-of-Verification) usually changes two things at once: *how much context* downstream components see, and *whether a verification gate filters out bad claims*. Because these two mechanisms are bundled together, it is unclear how much error reduction comes simply from **stripping persuasive upstream Chain-of-Thought narratives** versus **actively gating and discarding unsupported claims**.

To test these two mechanisms cleanly—without heavy frameworks like LangChain or AutoGen adding hidden prompt overhead—I built **IsoBench**: a zero-dependency Python evaluation harness paired with a $75$-question **Parametrically Blinded Counterfactual Dataset**.

---

## 2. How IsoBench Works

### Step 1: Parametric Blinding ($P(\text{Knowledge} \mid \text{Pretraining}) = 0$)

Instead of testing on Wikipedia entities, `counterfactual_dataset.json` ($N = 75$) uses synthetic, counterfactual entities across seven domains (cinema, literature, corporate history, music, architecture, science, and maritime history) while preserving the exact two-hop bridge structure of HotpotQA.

For example, consider Question **`cf-4`**:
* **Question:** *"The 1982 slasher film 'The Midnight Harvest' was directed by a filmmaker born in what city?"*
* **Hop-1 Evidence ($E_1$):** *"'The Midnight Harvest' is a 1982 regional slasher film directed by **Aris Thorne** from an original screenplay written by dramatist **Lyra Vance**."*
* **Hop-2 Correct Evidence ($E_{2,\text{correct}}$):** *"**Aris Thorne** is an independent film director and editor who was born in the timber port of **Silverwood**, Oakhaven."*
* **Hop-2 Injected Distractor Evidence ($E_{2,\text{injected}}$):** *"**Lyra Vance** is a playwright, screenwriter, and essayist who was born in the valley municipality of **Ravensbrook**, Oakhaven."*

Because *The Midnight Harvest*, *Aris Thorne*, and *Lyra Vance* do not exist in pre-training data, the model cannot guess `Silverwood` from parametric memory. It must rely $100\%$ on the context passed through the multi-agent pipeline.

---

### Step 2: Controlled Bridge-Entity Error Injection

Every question is evaluated under two conditions:
* **`clean` (Control):** The Hop-1 Planner identifies the correct bridge entity ($b_{\text{gold}} = \text{Aris Thorne}$), and we measure baseline **Final Answer Accuracy (`FAA_clean`)** to verify that restricting context does not break normal reasoning.
* **`injected` (Intervention):** After the Hop-1 Planner runs (cached so all architectures start from the exact same base generation), the harness substitutes the distractor entity ($b_{\text{gold}} \mapsto b_{\text{mut}}$, e.g., `Aris Thorne` $\mapsto$ `Lyra Vance`) into **both** the upstream multi-sentence Chain-of-Thought reasoning and the extracted atomic claim.

---

### Step 3: The 4-Arm Architectural Ablation ($B_0, B_1, B_2, B_3$)

To separate information-flow restriction from verification gating, `IsoBench` routes the exact same upstream state through four architectures:

| Architecture | What the Verifier Sees | What Downstream Agents See | Verification Gate | What It Tests |
| :--- | :--- | :--- | :--- | :--- |
| **`B0` (Shared Baseline)** | Full upstream CoT + claim + evidence | Full upstream CoT + claim + evidence | None | How badly errors cascade when all agents share one context window |
| **`B1` (Isolated Verifier)** | Atomic claim + evidence **only** | Full upstream CoT + claim + evidence | None | Whether isolating the Verifier stops verifier sycophancy (`Verifier_FAR`) |
| **`B2` (Isolation Only)** | Atomic claim + evidence **only** | **Bare atomic claim** + evidence (CoT purged) | None | Whether stripping persuasive upstream CoT alone stops downstream anchoring |
| **`B3` (Isolation + Gate)** | Atomic claim + evidence **only** | **Verified claims only**; clean fallback on rejection | **Strict Gate** (`SUPPORTED` only) | Whether combining context isolation with a strict gate + retrieval fallback stops cascades |

---

## 3. Empirical Pilot Results ($N = 5$, Local `llama3.1:latest`)

Running all $75$ questions across $4$ architectures $\times$ $2$ conditions ($600$ multi-agent pipeline runs, or $3,000$ sequential LLM calls) on a laptop GPU takes several hours. To validate the harness end-to-end on local hardware, I ran a deterministic $5$-question pilot (`cf-1` through `cf-5`, totaling $40$ pipeline runs / $200$ LLM calls) on **`llama3.1:latest` (8B)** via Ollama at `temperature = 0.0` (`seed = 42`).

Here is the recorded summary from [`results_summary.json`](./results_summary.json) and [`pilot_5q_execution_log.txt`](./pilot_5q_execution_log.txt):

| Architecture | `EPR_final` (↓) | `EPR_inter` (↓) | `FAA_clean` (↑) | `RR_injected` (↑) | `Verifier_FAR` (↓) | Mean Input Tokens | Mean Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0` (Shared Baseline)** | **0.600** (60%) | 1.000 (100%) | 0.800 (80%) | 0.200 (20%) | 0.200 (20%) | 1,281.8 | 84.62s |
| **`B1` (Isolated Verifier)** | **0.000** (0%) | 1.000 (100%) | 1.000 (100%) | 0.800 (80%) | 0.000 (0%) | 1,069.7 | 64.13s |
| **`B2` (Isolation Only)** | **0.400** (40%) | 1.000 (100%) | 0.800 (80%) | 0.400 (40%) | 0.000 (0%) | 860.0 | 60.69s |
| **`B3` (Isolation + Gate)** | **0.000** (0%) | **0.000** (0%) | 0.800 (80%) | **1.000** (100%) | 0.000 (0%) | **844.4** | **58.21s** |

*(Definitions: `EPR_final` = Final Error Propagation Rate under injection; `EPR_inter` = Intermediate Hop-2 Error Propagation Rate; `FAA_clean` = Final Answer Accuracy on clean control; `RR_injected` = Recovery Rate to the true gold answer under injection; `Verifier_FAR` = Verifier False Acceptance Rate).*

---

## 4. What the Traces Reveal: Three Concrete Takeaways

### Finding 1: Shared Context (`B0`) Causes Both Error Cascades and Verifier Sycophancy
In `B0`, the injected error propagated into the final output on **3 out of 5 questions (`60%`)**, and the pipeline only recovered the right answer **20%** of the time.

More importantly, look at what happened on **`cf-4` (*The Midnight Harvest*)**:
* The Hop-1 evidence explicitly states that *The Midnight Harvest* was **directed by Aris Thorne** and written by Lyra Vance.
* Under the `injected` condition, the upstream trace claims it was directed by **Lyra Vance**.
* In **`B0`**, because the Verifier was exposed to the upstream premise and reasoning history, it suffered **sycophantic anchoring** and marked the false claim as **`SUPPORTED` (`Verifier_FAR = 1`)**. The Executor then confidently answered: *"Lyra Vance, the director of 'The Midnight Harvest', was born in Ravensbrook."*
* In **`B1`, `B2`, and `B3`**, where the Verifier was isolated to see *only* the atomic claim and the Hop-1 evidence (`Verifier_FAR = 0.000`), it immediately flagged the claim as contradicted.

### Finding 2: Hiding Upstream Reasoning (`B2`) Saves 33% of Tokens, but Bare Claims Still Anchor Downstream Agents
My first hypothesis ($H_1$) was that stripping the persuasive upstream Chain-of-Thought narrative (`B2`) would stop downstream agents from being talked into the error.
* It helped partially: `EPR_final` dropped from **`0.600` (`B0`) to `0.400` (`B2`)**, recovery doubled (`0.200` $\rightarrow$ `0.400`), and mean input tokens dropped by **32.9%** (`1,281.8` $\rightarrow$ `860.0`).
* **However, `B2` still failed on `cf-1` and `cf-4` (`EPR_final = 40%`).** Why? Because in `B2`, the downstream Hop-2 agent receives an isolated context containing `Bridge_Entity_Candidate: The 1982 slasher film 'The Midnight Harvest' was directed by Lyra Vance` alongside Hop-2 evidence (`Aris Thorne -> Silverwood`, `Lyra Vance -> Ravensbrook`). Without the Hop-1 evidence in its window to refute the candidate claim, the isolated Hop-2 agent treated the bare claim as an authoritative instruction and output `Ravensbrook`.

### Finding 3: Context Isolation + Strict Verification Gate (`B3`) Eliminated Propagation and Cut Latency by 31%
In **`B3`**, when the isolated Verifier rejects the corrupted Hop-1 claim (`CONTRADICTED` or `INSUFFICIENT`), two things happen:
1. The corrupted claim is blocked from entering the `Verified State` (`EPR_inter = 0.000`).
2. Corrupted Hop-2 retrieval is halted, and the downstream agent receives a clean `[UNVERIFIED_FALLBACK]` state with clean evidence.

Across the 5-question pilot, **`B3` achieved `0.0%` error propagation (`EPR_final = 0.000`) and recovered the exact gold answer on `5/5` (`100%`) injected runs**, while using **34.1% fewer input tokens** (`844.4` vs. `1,281.8`) and running **31.2% faster** (`58.21s` vs. `84.62s`) than the shared-context baseline.

---

## 5. Limitations & Open Questions for Graduate Research

This project is an early empirical prototype, and I want to be transparent about its current scope:

1. **Sample Size & Statistical Power:**  
   While the full $N = 75$ counterfactual benchmark is generated and included in [`counterfactual_dataset.json`](./counterfactual_dataset.json), the local laptop run above covers $N = 5$ instances ($40$ pipeline runs). With $N = 5$ and $3$ discordant pairs between $B_0$ and $B_3$ ($n_{10} = 3, n_{01} = 0$), the minimum possible two-sided binomial McNemar $p$-value is $2 \times 0.5^3 = 0.25$. Running the full $N = 75$ dataset across multiple checkpoints is the immediate next step to reach formal statistical power ($\alpha = 0.05$ after Holm-Bonferroni correction).
2. **Synthetic Counterfactuals vs. Entrenched Priors:**  
   Parametric blinding ($P(\text{prior}) = 0$) gives clean laboratory control, but in production RAG systems, corrupted context often collides with strong pre-trained priors. Studying how context isolation behaves when external evidence *contradicts* pre-trained weights is a natural extension.
3. **Scaling to Deeper Graphs and Frontier Models:**  
   `IsoBench` currently tests a deterministic two-hop bridge topology on an 8B open-weight model (`Llama 3.1`). Extending this harness to $3+$ hop reasoning graphs, cyclic tool retries, and frontier models (`Claude 3.5 Sonnet`, `GPT-4o`, `Llama-3.1-70B`) will clarify whether larger models are more resistant—or more sycophantically vulnerable—to shared-context rationalization.

---

## 6. Reproducing This Work

The complete codebase, the $N = 75$ counterfactual dataset, and the exact $N = 5$ execution logs are open-source and require zero external agent frameworks:

```bash
git clone https://github.com/HamseMoismaila/IsoBench.git
cd IsoBench
pip install -r requirements.txt

# Run the 5-question evaluation on local Ollama (llama3.1:latest)
python run_isobench.py --limit 5
```
