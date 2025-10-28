# Ask My PDFs

Upload a PDF and chat with it using RAG (LangChain + Chroma + OpenAI) in Streamlit.

## Setup
```bash
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
cp .env.example .env        # put your real key in .env
streamlit run app.py
