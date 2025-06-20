import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from huggingface_hub import login
from Levenshtein import distance

login("xxxxxxxx")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

tokenizer_text_correc = AutoTokenizer.from_pretrained("sdadas/byt5-text-correction")
model_text_correc = AutoModelForSeq2SeqLM.from_pretrained("sdadas/byt5-text-correction")


def dist_n(tokens, n):
    # Diversidade léxica n-grama
    # Faz a avaliação dos n gramas
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    return len(set(ngrams)) / len(ngrams)


def perplexity(prompt, generation):
    # Verifica o quanto a sequencia é razoavel ou 'estranha'
    text = prompt + generation
    enc = tokenizer(text, return_tensors="pt")
    gen_len = len(tokenizer(generation).input_ids)
    labels = enc.input_ids.clone()
    labels[:, :-gen_len] = -100

    with torch.no_grad():
        loss = model(**enc, labels=labels).loss.item()

    return math.exp(loss)


def grammaticality(text):
    # Usa um corretor para validar o quanto correto esta o texto e faz a diferença entre o corrijido e o original
    input_ids = tokenizer_text_correc(text, return_tensors="pt").input_ids
    outputs = model_text_correc.generate(input_ids, max_new_tokens=100)
    corrected = tokenizer_text_correc.decode(outputs[0], skip_special_tokens=True)
    dist = distance(text, corrected)
    return 1 - (dist / max(len(text), 1))


def evaluate_generation(prompt, generated):
    tokens = generated.split()
    return {
        "perplexity": perplexity(prompt, generated),
        "dist_1": dist_n(tokens, 1),
        "dist_2": dist_n(tokens, 2),
        "dist_3": dist_n(tokens, 3),
        "grammaticality": grammaticality(generated),
    }
