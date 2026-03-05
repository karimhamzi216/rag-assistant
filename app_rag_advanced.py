import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import docx
import os

# Configuration
st.set_page_config(
    page_title="Assistant RAG Pro",
    page_icon="🤖",
    layout="wide"
)

# Récupérer clé API (Hugging Face Secrets ou .env local)
mistral_key = os.environ.get('MISTRAL_API_KEY') or st.secrets.get('MISTRAL_API_KEY', '')

if not mistral_key:
    st.error("⚠️ Clé API Mistral non configurée. Ajoutez MISTRAL_API_KEY dans les secrets.")
    st.stop()

os.environ['MISTRAL_API_KEY'] = mistral_key
os.environ['LANGCHAIN_TRACING_V2'] = 'false'  # Désactiver LangSmith

# ... (le reste du code reste identique)
# Copier tout le code de app_rag_advanced.py ci-dessus
