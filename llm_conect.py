import requests
import json


def generate_text(prompt, model="llama3.2", reference="", use_rag=False):
    full_prompt = f"{prompt}. Responda tudo em português."
    if len(reference) > 1:
        full_prompt = f"{reference}\n\nA partir do texto acima, responda: \n{prompt}. Responda tudo em português."

    prompt_json_string = json.dumps({"message": full_prompt})
    payload = {"prompt": prompt_json_string}

    url = "https://bee0-35-247-136-94.ngrok-free.app/chatbot"  # Lembrar de alterar a URL sempre que rodar o Ollama, pois o NGROK gera um link novo a cada build

    response = requests.post(url, json=payload)

    if response.ok:
        resposta_json = response.json()
        reply = resposta_json.get("reply", "")
        yield reply

    else:
        yield json.dumps({"error": response.text}, ensure_ascii=False)
