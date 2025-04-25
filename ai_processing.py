import faiss
import nltk
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Model for RAG
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
encoding = tiktoken.get_encoding("cl100k_base")
nltk.download("punkt")

# Model for NER
ner_tokenizer = AutoTokenizer.from_pretrained(
    "pierreguillou/ner-bert-base-cased-pt-lenerbr"
)
ner_model = AutoModelForTokenClassification.from_pretrained(
    "pierreguillou/ner-bert-base-cased-pt-lenerbr"
)
ner_model = pipeline(
    "ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple"
)


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
def named_entity_recognition(text, top_k=16):
    query = "Texto contendo nomes de pessoas, organizações, locais e jurisprudências relevantes."
    relevant_chunks = rag_personalized(text, query, top_k)
    relevant_chunks = list(dict.fromkeys(relevant_chunks))

    entities = ner_model(relevant_chunks)

    entities_merged = [item for sentence in entities for item in sentence]
    formatted_entities = []
    temp_entity = None

    for entity in entities_merged:
        current_entity = entity["entity_group"]

        if temp_entity and (
            current_entity.startswith("I-")
            or current_entity == temp_entity["entity_group"]
        ):
            if "##" in entity["word"]:
                temp_entity["word"] += entity["word"].replace("##", "")
            else:
                temp_entity["word"] += " " + entity["word"]
            temp_entity["score"] = (temp_entity["score"] + entity["score"]) / 2
        else:
            if temp_entity:
                formatted_entities.append(temp_entity)
            temp_entity = {
                "word": entity["word"],
                "entity_group": current_entity,
                "score": entity["score"],
            }

    if temp_entity:
        formatted_entities.append(temp_entity)

    formatted_entities = [
        entity for entity in formatted_entities if not entity["word"].startswith("#")
    ]

    for entity in formatted_entities:
        entity["score"] = float(entity["score"])

    return formatted_entities
