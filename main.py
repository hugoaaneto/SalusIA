import time

import streamlit as st

import llm_conect
from pre_processing import preprocess

st.set_page_config(
    page_title="chatbot",
    page_icon="📄",
    layout="wide",
)

available_models = [
    "phi4:latest",
    "mistral:latest",
    "llama3.2:latest",
    "gemma:7b",
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:12b-it-qat",
    "gemma3:27b-it-qat",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
]

st.sidebar.title("Selecionar configurações:")
model_selected = st.sidebar.selectbox(
    "Modelo:", available_models, index=available_models.index("llama3.2:latest")
)

to_lower = st.sidebar.checkbox("Converter para minúsculas", value=True)
remove_punct = st.sidebar.checkbox("Remover pontuação", value=True)
remove_stop = st.sidebar.checkbox("Remover stopwords", value=False)
use_historic = st.sidebar.checkbox("Usar conversa", value=False)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🗨️ Chat com o Bot")

with st.container():
    for entry in st.session_state.chat_history:
        sender, msg, model, duration = entry
        prefix = "🧑" if sender == "Você" else "🤖"
        meta = (
            f" _(modelo: `{model}`, tempo: {duration:.2f}s)_" if sender == "Bot" else ""
        )
        st.markdown(f"**{prefix} {sender}:** {msg}{meta}")


response_placeholder = st.empty()

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Digite sua mensagem:", key="user_input_form")
    submitted = st.form_submit_button("Enviar")

if submitted and user_input.strip():
    reference_context = "\n".join(
        [
            f"{'Usuário' if sender == 'Você' else 'Bot'}: {msg}"
            for sender, msg, _, _ in st.session_state.chat_history
        ]
    )
    processed = preprocess(user_input, to_lower, remove_punct, remove_stop)
    st.session_state.chat_history.append(("Você", user_input, "", 0))

    full_response = ""
    start_time = time.time()
    if not use_historic:
        reference_context = ""

    for chunk in llm_conect.generate_text(
        processed, model=model_selected, reference=reference_context, use_rag=False
    ):
        full_response += chunk
        response_placeholder.markdown(f"**🤖 Bot:** {full_response}▌")

    end_time = time.time()
    duration = end_time - start_time

    response_placeholder.markdown(f"**🤖 Bot:** {full_response}")
    st.session_state.chat_history.append(
        ("Bot", full_response, model_selected, duration)
    )
    st.rerun()
