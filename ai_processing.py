import faiss
import nltk
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
    AutoModel,
)

# Model for RAG
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
encoding = tiktoken.get_encoding("cl100k_base")
nltk.download("punkt")

# Model for NER
NER_MODELS = {
    "Médico": "pucpr/clinicalnerpt-medical",
    "Químico": "pucpr/clinicalnerpt-chemical",
    "Diagnóstico": "pucpr/clinicalnerpt-diagnostic",
    "Doença": "pucpr/clinicalnerpt-disease",
    "Procedimento": "pucpr/clinicalnerpt-procedure",
}

PIPELINES = {
    tipo: pipeline(
        task="token-classification",
        model=path,
        aggregation_strategy="simple",
    )
    for tipo, path in NER_MODELS.items()
}


# -----------------------------------------------------------------------------------------------------------------
# -------------------------------------- RAG (Retrieval Augmented Generation) -------------------------------------
# -----------------------------------------------------------------------------------------------------------------
def split_text_into_chunks(text, chunk_size=512):
    tokenized_text = encoding.encode(text)
    chunks = [
        tokenized_text[i : i + chunk_size]
        for i in range(0, len(tokenized_text), chunk_size)
    ]
    text_chunks = [encoding.decode(chunk) for chunk in chunks]

    return text_chunks


# Transforms text chunks into vectors
def generate_embeddings(text_chunks):
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    return embeddings


# Generates a similarity index between text vectors
def build_faiss_index(text_chunks):
    embeddings = generate_embeddings(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(query, text_chunks, index, top_k=3, distance=False):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    if distance:
        similarities = 1 / (1 + distances[0])
        return [
            (text_chunks[i], similarities[idx], distances[0][idx])
            for idx, i in enumerate(indices[0])
        ]

    return [text_chunks[i] for i in indices[0]]


def rag_personalized(text, query, top_k=3):
    text_chunks = split_text_into_chunks(text, 384)
    index, _ = build_faiss_index(text_chunks)
    return retrieve_relevant_chunks(query, text_chunks, index, top_k)


# -----------------------------------------------------------------------------------------------------------------
# ----------------------------------------- NER (Named entity recognition) ----------------------------------------
# -----------------------------------------------------------------------------------------------------------------
def agrupar_entidades(entidades):
    entidades = sorted(entidades, key=lambda x: x["start"])

    resultados, buffer_palavra, buffer_scores = [], [], []
    buffer_tipo, ultimo_end = None, None

    for token in entidades:
        tipo_atual = token["tipo"]
        palavra = token["entidade"]
        score = float(token["score"])

        novo_grupo = (
            buffer_tipo is None
            or tipo_atual != buffer_tipo
            or token["start"] > (ultimo_end or 0) + 1
        )

        if novo_grupo:
            if buffer_palavra:
                if sum(buffer_scores) / len(buffer_scores) > 0.9:
                    resultados.append(
                        {
                            "word": "".join(buffer_palavra).replace("##", ""),
                            "entity_group": buffer_tipo,
                            "score": sum(buffer_scores) / len(buffer_scores),
                        }
                    )

            buffer_palavra = [palavra]
            buffer_scores = [score]
            buffer_tipo = tipo_atual
        else:
            prefixo = "" if palavra.startswith("##") else " "
            buffer_palavra.append(prefixo + palavra)
            buffer_scores.append(score)

        ultimo_end = token["end"]

    if buffer_palavra:
        if sum(buffer_scores) / len(buffer_scores) > 0.9:
            resultados.append(
                {
                    "word": "".join(buffer_palavra).replace("##", ""),
                    "entity_group": buffer_tipo,
                    "score": sum(buffer_scores) / len(buffer_scores),
                }
            )

    return resultados


def extrair_e_agrupar(text):
    entidades_por_tipo = {}
    query = "Texto contendo nomes medicamentos, procedimentos medicos e assuntos ligados a saude."
    relevant_chunks = rag_personalized(text, query, 8)
    relevant_chunks = list(dict.fromkeys(relevant_chunks))

    for tipo, ner in PIPELINES.items():
        tokens = ner(text)

        tokens_normalizados = [
            {
                "tipo": tipo,
                "entidade": t["word"],
                "score": t["score"],
                "start": t["start"],
                "end": t["end"],
            }
            for t in tokens
        ]

        entidades_por_tipo[tipo] = agrupar_entidades(tokens_normalizados)

    return entidades_por_tipo
