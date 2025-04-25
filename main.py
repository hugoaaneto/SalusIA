import time
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import llm_conect
from ai_processing import named_entity_recognition
from pre_processing import preprocess
import tempfile
import speech_recognition as sr
import hashlib
import pandas as pd

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


def get_audio_hash(audio_bytes):
    return hashlib.sha256(audio_bytes).hexdigest()


def format_ner_result(ner_data):
    df = pd.DataFrame(ner_data)
    df = df.rename(
        columns={
            "word": "Termo",
            "entity_group": "Tipo de Entidade",
            "score": "Score",
        }
    )
    df["Score"] = df["Score"].apply(lambda x: round(x, 3))
    return df


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "input_submitted" not in st.session_state:
    st.session_state.input_submitted = False
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

st.sidebar.title("Selecionar configurações:")
model_selected = st.sidebar.selectbox(
    "Modelo:", available_models, index=available_models.index("llama3.2:latest")
)
to_lower = st.sidebar.checkbox("Converter para minúsculas", value=True)
remove_punct = st.sidebar.checkbox("Remover pontuação", value=True)
remove_stop = st.sidebar.checkbox("Remover stopwords", value=False)
use_historic = st.sidebar.checkbox("Usar histórico", value=False)
input_mode = st.sidebar.radio("Modo de entrada:", ["Texto", "Áudio"])

tab_conversa, tab_ner = st.tabs(["💬 Conversa", "🧠 Resultado NER"])

with tab_conversa:
    st.title("🗨️ Chat com o Bot")

    with st.container():
        for entry in st.session_state.chat_history:
            sender, msg, model, duration = entry
            prefix = "🧑" if sender == "Você" else "🤖"
            meta = (
                f" _(modelo: `{model}`, tempo: {duration:.2f}s)_"
                if sender == "Bot"
                else ""
            )
            st.markdown(f"**{prefix} {sender}:** {msg}{meta}")

    response_placeholder = st.empty()

    if input_mode == "Texto":
        with st.form(key="chat_form", clear_on_submit=True):
            text_input = st.text_input("Digite sua mensagem:", key="user_input_form")
            submitted = st.form_submit_button("Enviar")
            if submitted and text_input.strip():
                st.session_state.pending_input = text_input.strip()
                st.session_state.input_submitted = True
                st.rerun()

    elif input_mode == "Áudio":
        st.info("Grave sua mensagem e aguarde o processamento.")
        audio_bytes = audio_recorder(pause_threshold=1.5, sample_rate=16000)

        if audio_bytes:
            audio_hash = get_audio_hash(audio_bytes)
            last_audio_hash = st.session_state.get("last_audio_hash")

            if audio_hash != last_audio_hash:
                st.session_state.last_audio_hash = audio_hash

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                recognizer = sr.Recognizer()
                with sr.AudioFile(tmp_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        transcript = recognizer.recognize_google(
                            audio_data, language="pt-BR"
                        )
                        st.success(f"Você disse: {transcript}")
                        st.session_state.pending_input = transcript
                        st.session_state.input_submitted = True
                        st.rerun()
                    except sr.UnknownValueError:
                        st.error("Não foi possível reconhecer a fala.")
                    except sr.RequestError:
                        st.error(
                            "Erro ao conectar ao serviço de reconhecimento de voz."
                        )

    if st.session_state.input_submitted and st.session_state.pending_input:
        user_input = st.session_state.pending_input
        reference_context = "\n".join(
            [
                f"{'Usuário' if sender == 'Você' else 'Bot'}: {msg}"
                for sender, msg, _, _ in st.session_state.chat_history
            ]
        )
        if not use_historic:
            reference_context = ""

        processed = preprocess(user_input, to_lower, remove_punct, remove_stop)
        st.session_state.chat_history.append(("Você", user_input, "", 0))

        full_response = ""
        start_time = time.time()
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

        st.session_state.pending_input = None
        st.session_state.input_submitted = False
        st.session_state.audio_processed = False
        st.rerun()

with tab_ner:
    exec_ner = st.sidebar.button("Identificar NER")

    if exec_ner:
        all_user_text = " ".join(
            msg
            for sender, msg, _, _ in st.session_state.chat_history
            if sender == "Você"
        )
        ner_result = named_entity_recognition(all_user_text)
        st.session_state.ner_result = ner_result

    if "ner_result" in st.session_state and st.session_state.ner_result:
        st.subheader("🧠 Resultado do NER")
        df_ner = format_ner_result(st.session_state.ner_result)
        st.dataframe(df_ner, use_container_width=True)
    else:
        st.info("Clique no botão na barra lateral para gerar o resultado do NER.")
