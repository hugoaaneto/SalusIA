# SalusIA

**SalusIA** é um sistema inteligente desenvolvido em Python com Streamlit, que utiliza técnicas de Inteligência Artificial (IA) e Processamento de Linguagem Natural (PLN) para oferecer recomendações de saúde a partir de dados clínicos. O projeto está sendo desenvolvido com o objetivo de auxiliar profissionais da saúde na análise de prontuários, geração de hipóteses diagnósticas e estruturação da anamnese, reduzindo a sobrecarga cognitiva e aumentando a precisão no atendimento.

## Funcionalidades

- Extração automatizada de dados clínicos estruturados e não estruturados.
- Normalização de texto médico utilizando PLN.
- Extração de entidades médicas e identificação de relações entre sintomas, exames e diagnósticos.
- Sugestão de hipóteses diagnósticas com base em faixas de referência e bases de conhecimento.
- Geração automatizada de anamnese estruturada.
- Apoio à decisão clínica com validação final feita por profissionais.

## Tecnologias Utilizadas

- **Python**: Linguagem principal para backend e manipulação de dados clínicos.
- **Streamlit**: Framework utilizado para a criação da interface interativa.
- **Modelos de IA e PLN**: Para análise semântica dos dados e sugestões diagnósticas.
- **APIs para LLMs**: Integração com agentes inteligentes para inferência médica.

## Requisitos

- Ter o [Ollama](https://ollama.com/) instalado e em execução localmente, com pelo menos alguns modelos previamente baixados (por exemplo, `llama3.2`, `mistral` ou `gemma`) para garantir o funcionamento do sistema.

Recomenda-se o uso de um ambiente virtual (venv) para garantir o isolamento das dependências:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

Em seguida, instale as dependências com:

```bash
pip install -r requirements.txt
```

## Execução

Para iniciar a aplicação, execute o seguinte comando:

```bash
streamlit run main.py
```

## Autores

- Rui Pontes  
- Hugo Amorim  
- Paulo Beralson  
- João Victor  
- Allanyo Santos

---

