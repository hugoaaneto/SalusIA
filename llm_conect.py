import requests
import json


def generate_text_cloud(prompt, reference="", use_rag=False):
    full_prompt = f"Considerando: {prompt}. Explique qual o medico mais apropriado para lidar com essa situação e faça uma avaliação inicial do caso. Responda tudo em português."
    if len(reference) > 1:
        full_prompt = f"Considerando {reference}\n\n e \n{prompt}. Explique qual o medico mais apropriado para lidar com essa situação e faça uma avaliação inicial do caso. Responda tudo em português."

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


def generate_text_local(prompt, model, reference="", use_rag=False):
    full_prompt = f"Considerando: {prompt}. Explique qual o medico mais apropriado para lidar com essa situação e faça uma avaliação inicial do caso. Responda tudo em português."
    if len(reference) > 1:
        full_prompt = f"Considerando {reference}\n\n e \n{prompt}. Explique qual o medico mais apropriado para lidar com essa situação e faça uma avaliação inicial do caso. Responda tudo em português."

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
