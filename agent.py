"""
Small AI agent: LLM (Gemini) + tools + agent loop (no framework).

Flow:  user task -> LLM decides -> tool call -> tool result -> LLM decides again
       -> ... -> final answer (when the LLM stops asking for tools)

Setup:
    pip install -r requirements.txt
    copy .env.example to .env and put your real key in it
Run:
    python agent.py                 # runs the 5 test cases
    python agent.py "your request"  # run a single request
"""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from google.genai import types

load_dotenv()  # reads variables from the local .env file (which is git-ignored)

API_KEY = os.getenv("GEMINI_API_KEY")  # never hard-code the key in this file
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_STEPS = 8  # safety limit so the loop can never run forever

# ---------------------------------------------------------------------------
# 1. Fictional data (stands in for a database / API)
# ---------------------------------------------------------------------------
PRODUCTS = {
    "laptop": {"price": 850.00, "description": "15-inch laptop, 16GB RAM, 512GB SSD"},
    "phone": {"price": 500.00, "description": "6.5-inch smartphone, 128GB storage"},
    "headphones": {"price": 60.00, "description": "Wireless noise-cancelling headphones"},
}
STOCK = {"laptop": 5, "phone": 0, "headphones": 20}


# ---------------------------------------------------------------------------
# 2. Tools (plain Python functions)
# ---------------------------------------------------------------------------
def get_product_info(product_name: str) -> dict:
    item = PRODUCTS.get(product_name.strip().lower())
    if item is None:
        return {"found": False, "message": f"No product named '{product_name}' exists."}
    return {"found": True, "product": product_name.lower(), **item}


def check_stock(product_name: str) -> dict:
    qty = STOCK.get(product_name.strip().lower())
    if qty is None:
        return {"found": False, "message": f"No stock record for '{product_name}'."}
    return {"found": True, "product": product_name.lower(), "units_in_stock": qty,
            "available": qty > 0}


def calculate_total(price: float, quantity: int) -> dict:
    return {"price": price, "quantity": quantity, "total": round(price * quantity, 2)}


TOOL_FUNCTIONS = {
    "get_product_info": get_product_info,
    "check_stock": check_stock,
    "calculate_total": calculate_total,
}

# Tool descriptions the LLM sees. The LLM only *chooses* tools;
# our code is what actually executes them.
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_product_info",
        description="Get the price (USD) and description of a product by name.",
        parameters={
            "type": "object",
            "properties": {"product_name": {"type": "string", "description": "e.g. 'laptop'"}},
            "required": ["product_name"],
        },
    ),
    types.FunctionDeclaration(
        name="check_stock",
        description="Check how many units of a product are in stock.",
        parameters={
            "type": "object",
            "properties": {"product_name": {"type": "string", "description": "e.g. 'laptop'"}},
            "required": ["product_name"],
        },
    ),
    types.FunctionDeclaration(
        name="calculate_total",
        description="Calculate total cost = price x quantity. Use this for any price math.",
        parameters={
            "type": "object",
            "properties": {
                "price": {"type": "number", "description": "Unit price"},
                "quantity": {"type": "integer", "description": "Number of units"},
            },
            "required": ["price", "quantity"],
        },
    ),
]

# ---------------------------------------------------------------------------
# 3. Agent instructions (system prompt)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a store assistant for a small electronics shop.

Your responsibilities:
- Answer questions about product prices, descriptions, and stock.
- Calculate totals for orders.

Tools available: get_product_info, check_stock, calculate_total.

Rules:
- Use tools to get product facts. Never guess or invent prices, stock levels, or products.
- Use calculate_total for any price calculation instead of doing the math yourself.
- If a tool says a product or information was not found, tell the user clearly that
  the information is unavailable. Do not make up an answer.
- If the question needs no tool (greetings, general chat), just answer directly.
- If a product is out of stock, say so, and do not present it as available.
- Keep final answers short and clear."""


# ---------------------------------------------------------------------------
# 4. The agent loop
# ---------------------------------------------------------------------------
def run_agent(client: genai.Client, user_request: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        # We run the loop ourselves so every step is visible
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    # conversation state / memory
    contents = [types.Content(role="user", parts=[types.Part(text=user_request)])]

    print(f"\n{'=' * 70}\nUSER: {user_request}\n{'=' * 70}")

    for step in range(1, MAX_STEPS + 1):
        # (a) LLM looks at the conversation and decides: answer, or call tool(s)
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        calls = response.function_calls

        # (b) No tool requested -> the LLM is done; return the final answer
        if not calls:
            answer = response.text or "(no response)"
            print(f"\n[Step {step}] FINAL ANSWER:\n{answer}")
            return answer

        # (c) Tool(s) requested -> remember what the LLM said, then run each tool
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            print(f"[Step {step}] TOOL CALL  -> {call.name}({json.dumps(args)})")
            func = TOOL_FUNCTIONS.get(call.name)
            try:
                result = func(**args) if func else {"error": "Unknown tool"}
            except Exception as e:  # tool errors go back to the LLM, not crash the agent
                result = {"error": str(e)}
            print(f"[Step {step}] TOOL RESULT <- {json.dumps(result)}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # (d) Feed results back so the LLM can inspect them and decide the next action
        contents.append(types.Content(role="user", parts=result_parts))

    return "Stopped: reached the maximum number of steps."


# ---------------------------------------------------------------------------
# 5. Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    ("1. Simple question (no tool)", "Hi! What can you help me with?"),
    ("2. Single tool", "How much is the headphones?"),
    ("3. Multiple tools",
     "Tell me the price of the laptop, check whether it is available and calculate the price for two units."),
    ("4. Out-of-stock handling", "I want to buy 3 phones. Can I?"),
    ("5. FAILURE case (data doesn't exist)", "What is the price of the gaming console, and is it in stock?"),
]

if __name__ == "__main__":
    if not API_KEY:
        sys.exit("GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")
    gemini = genai.Client(api_key=API_KEY)
    try:
        if len(sys.argv) > 1:
            run_agent(gemini, " ".join(sys.argv[1:]))
        else:
            for title, request in TEST_CASES:
                print(f"\n\n##### TEST {title} #####")
                run_agent(gemini, request)
    except APIError as error:
        if error.code == 503:
            sys.exit(
                "Gemini is temporarily unavailable (503). The SDK already retried "
                "the request; please try again later. If the issue persists, set "
                "GEMINI_MODEL in .env to another model available to your API key."
            )
        sys.exit(f"Gemini API request failed ({error.code}): {error.message or error}")
