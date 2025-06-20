# SalusIA

**SalusIA** is an intelligent system developed in Python with Streamlit, which uses Artificial Intelligence (AI) and Natural Language Processing (NLP) techniques to offer health recommendations based on clinical data. The project is being developed with the goal of assisting healthcare professionals in analyzing medical records, generating diagnostic hypotheses, and structuring anamnesis, thereby reducing cognitive overload and increasing accuracy in patient care.

---

## Features

* **Automated Extraction:** Extracts structured and unstructured clinical data.
* **Medical Text Normalization:** Utilizes NLP to normalize medical text.
* **Entity Extraction:** Identifies medical entities and relationships between symptoms, exams, and diagnoses.
* **Diagnostic Hypotheses:** Suggests diagnostic hypotheses based on reference ranges and knowledge bases.
* **Clinical Decision Support:** Provides clinical decision support with final validation by professionals.

---

## Technologies Used

* **Python**: The primary language for the backend and clinical data manipulation.
* **Streamlit**: Framework used for creating the interactive interface.
* **AI and NLP Models**: For semantic data analysis and diagnostic suggestions.
* **APIs for LLMs**: Integration with intelligent agents for medical inference.

---

## Requirements

You'll need to have [Ollama](https://ollama.com/) installed and running locally, with at least a few models previously downloaded (e.g., `llama3.2`, `mistral`, or `gemma`) to ensure the system functions correctly.

It's recommended to use a virtual environment (venv) to ensure dependency isolation:


```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

Then, install the dependencies with:

```bash
pip install -r requirements.txt
```

## Execution

To start the application, run the following command:



```bash
streamlit run main.py
```

---

