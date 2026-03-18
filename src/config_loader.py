import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

def load_secure_config(config_path: str = "config/prompt_config.yaml") -> Dict[str, Any]:
    """Load config with API key from environment"""
    # Load .env file if it exists
    load_dotenv()
    
    # Get API key (will raise error if not found)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY not found!\n"
            "   1. Create a .env file in project root\n"
            "   2. Add: GEMINI_API_KEY=your_key_here\n"
            "   3. Get your key from: https://aistudio.google.com/app/apikey"
        )
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Inject API key into config
    config["api_key"] = api_key
    
    return config