#!/usr/bin/env python3
"""
Check which Claude models are available with your API key.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def test_models():
    """Test various Claude model identifiers to see which ones work."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found")
        return

    client = Anthropic(api_key=api_key)

    # List of model identifiers to try
    models_to_try = [
        # Claude 3.5 Sonnet variations
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",

        # Claude 3 models
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",

        # Legacy/simple names
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]

    print("Testing model availability...\n")

    for model in models_to_try:
        try:
            print(f"Testing: {model}...", end=" ")
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print("✓ WORKS")
            print(f"  → This model is available! Use this in process_scans.py")
            return model  # Return the first working model
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not_found" in error_str:
                print("✗ Not found (404)")
            elif "401" in error_str or "authentication" in error_str.lower():
                print("✗ Authentication error")
            elif "403" in error_str or "permission" in error_str.lower():
                print("✗ Permission denied")
            else:
                print(f"✗ Error: {error_str[:60]}")

    print("\n⚠ No working models found. Please check:")
    print("  1. Your API key is correct in .env")
    print("  2. Your API key has active billing/credits")
    print("  3. Visit https://console.anthropic.com/settings/keys")
    return None

if __name__ == "__main__":
    test_models()
