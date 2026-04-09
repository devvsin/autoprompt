import sys
import os
import groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found.")
    print("   1. Create a .env file in the project root")
    print("   2. Add: GROQ_API_KEY=your_key_here")
    print("   3. Get your key from: https://console.groq.com/keys")
    sys.exit(1)

client = groq.Groq(api_key=api_key)

try:
    models = client.models.list()
    # Extract the IDs/names from the data array
    model_names = [m.id for m in models.data]
    print(f"OK: API key valid. Found {len(model_names)} model(s) available on Groq:")
    for name in model_names:
        print(f"   - {name}")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: API call failed: {e}")
    print("   Check your network connection, API key validity, and quota.")
    sys.exit(1)