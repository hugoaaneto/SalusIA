import requests
import json


def generate_text(prompt, model="llama3.2", reference="", use_rag=False):
    full_prompt = f"{prompt}. Responda tudo em português."
    if len(reference) > 1:
        full_prompt = f"{reference}\n\nA partir do texto acima, responda: \n{prompt}. Responda tudo em português."

    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": full_prompt, "stream": True}

    generated_text = ""

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    response_text = data.get("response", "")
                    generated_text += response_text
                    yield response_text
                except json.JSONDecodeError:
                    continue
