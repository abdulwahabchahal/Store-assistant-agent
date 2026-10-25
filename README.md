# Small AI Agent (Gemini + Tools + Agent Loop)

A store assistant agent built without any framework, to show the raw agent architecture.

## Setup
    pip install -r requirements.txt
    cp .env.example .env        # Windows: copy .env.example .env
    # open .env and replace your_api_key_here with your real Gemini key

The real key lives only in `.env`, which is listed in `.gitignore` and never pushed to GitHub.

## Run
    python agent.py                      # runs all 5 test cases
    python agent.py "Price of phone?"    # single request

If Gemini returns a temporary `503 UNAVAILABLE` error, wait and retry. The SDK
automatically retries transient API errors; if the issue persists, set
`GEMINI_MODEL` in `.env` to another model available to your API key.

## Architecture
1. **Instructions** - `SYSTEM_PROMPT` defines role, tools, and the "never invent data" rule.
2. **LLM (Gemini)** - decides whether to answer or call a tool.
3. **Tools** - `get_product_info`, `check_stock`, `calculate_total` (fictional data).
4. **Agent loop** - call the model; if it requests tools, run them and append results; repeat until a final answer (max 8 steps).
5. **State** - the `contents` list is the agent's memory.

## Test cases
1. Simple question (no tool)
2. Single tool (headphones price)
3. Multiple tools (laptop price + stock + total for 2)
4. Out-of-stock handling (phones)
5. Failure case (non-existent product -> says unavailable)
