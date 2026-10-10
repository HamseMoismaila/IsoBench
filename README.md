# IsoBench

IsoBench is a simple Python framework that tests how mistakes spread across automated systems.

When multiple programs talk to each other to solve a problem, a single mistake early on can cause the entire system to fail. IsoBench allows you to test exactly how these mistakes spread and how to stop them.

---

## How it Works

Normally, testing these systems is hard because they memorize too much information from the internet. IsoBench solves this by using **fictional stories**. 

Because the stories are completely made up (e.g., a fake movie called *The Aether Diaries*), the system cannot cheat by remembering the real answer. It is forced to read only the text we provide, allowing us to perfectly track where it makes a mistake.

---

## The 4 Test Approaches

IsoBench tests four different ways of organizing information flow to see which one is the safest:

1. **The Shared Approach ($B_0$)**: Every part of the system sees everything. If one part makes a mistake, the mistake spreads instantly to everyone else. (Highly vulnerable).
2. **The Blind Checker ($B_1$)**: The fact-checker is separated from the main process so it isn't influenced by previous mistakes, but the final decision maker is still exposed to bad info.
3. **The Cut-Off Approach ($B_2$)**: The system only passes clean, verified facts down the line, removing all messy and unverified thought processes.
4. **The Strict Gate ($B_3$)**: The most secure approach. It completely stops the program if a fact is proven false, preventing any bad information from moving forward.

---

## How to Run It

You can run this project locally on your computer.

### Requirements
* Python 3.10 or higher
* [Ollama](https://ollama.com/) (if running locally) or an Anthropic API Key (if using Claude).

### Quick Start

1. **Download the code:**
   ```bash
   git clone https://github.com/HamseMoismaila/IsoBench.git
   cd IsoBench
   ```

2. **Run it locally using Ollama:**
   ```bash
   # On Windows (PowerShell)
   $env:AGENT_LLM_BACKEND="ollama"
   python run_isobench.py
   
   # On Mac/Linux
   AGENT_LLM_BACKEND=ollama python run_isobench.py
   ```

3. **Run it using an API Key (Claude):**
   ```bash
   # On Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="your-api-key"
   python run_isobench.py
   ```
