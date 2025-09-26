# AI-Powered HR Assistant

Upload HR policy PDF(s) and ask questions. Built with Streamlit, LangChain, Chroma, and OpenAI.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Put OPENAI_API_KEY in a .env file or export it
streamlit run app.py

