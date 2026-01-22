
import os
import google.genai as genai

# Load API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # Try loading from .gemini/.env if not in environment
    try:
        with open("../.gemini/.env", "r") as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.strip().split("=")[1]
                    break
    except Exception:
        pass

if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    models = list(client.models.list())
    for m in models:
        # Filter for gemini models to keep output clean, or specific updated ones
        if "gemini" in m.name:
            print(f"Model: {m.name}")
            print(f"  Display Name: {m.display_name}")
            print(f"  Input Token Limit: {m.input_token_limit}")
            print(f"  Output Token Limit: {m.output_token_limit}")
            print(f"  Supported Actions: {m.supported_actions}")
            print("-" * 20)
except Exception as e:
    print(f"Error listing models: {e}")
