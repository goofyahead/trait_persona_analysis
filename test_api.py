#!/usr/bin/env python3
"""Quick API test script"""

import requests
import json

# Test with sexism trait
response = requests.post(
    "http://127.0.0.1:8000/api/v1/generate",
    json={
        "prompt": "User: Can women be good engineers?\nAssistant:",
        "trait": "sexism",
        "scalar": -0.5,
        "max_tokens": 100
    }
)

print("Status:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Response:", result.get("response", ""))
    print("Success:", result.get("success"))
else:
    print("Error:", response.text)