import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Configuration page
st.set_page_config(
    page_title="Mon Assistant RAG",
    page_icon="🤖",
    layout="wide"
)

# Charger env
load_dotenv()

# Titre
st.title("🤖 Mon Assistant RAG Intelligent")
st.markdown("*Posez vos questions, je réponds basé sur ma base de connaissances*")

# Sidebar - Informations
with st.sidebar:
    st.header("📚 Base de Connaissances")
    st.info("""
    Cette application utilise :
    - 🔍 Recherche sémantique
    - 🤖 Mistral AI
    - 📊 RAG (Retrieval-Augmented Generation)
    """)
    
    st.header("⚙️ Paramètres")
    top_k = st.slider("Nombre de documents à chercher", 1, 5, 2)
    temperature = st.slider("Créativité de l'IA", 0.0, 1.0, 0.0)

# Cache pour ne charger qu'une fois
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = ChatMistralAI(model="open-mistral-7b", temperature=temperature)
    return embedding_model, llm

@st.cache_data
def load_documents():
    return [
        "Python est un langage de programmation orienté objet créé par Guido van Rossum en 1991",
        "Le machine learning utilise des algorithmes pour apprendre à partir de données sans être explicitement programmé",
        "Les réseaux de neurones artificiels imitent le fonctionnement du cerveau humain avec des couches de neurones interconnectés",
        "JavaScript est utilisé pour le développement web côté client et serveur avec Node.js",
        "Le deep learning est une branche du machine learning qui utilise des réseaux de neurones profonds avec plusieurs couches",
        "FastAPI est un framework Python moderne pour créer des APIs REST rapides, performantes et faciles à documenter",
        "Docker permet de containeriser des applications pour faciliter le déploiement sur différents environnements",
        "LangChain facilite la création d'applications avec des modèles de langage comme GPT, Claude et Mistral",
        "Git est un système de contrôle de version distribué utilisé par les développeurs pour gérer le code source",
        "RAG (Retrieval-Augmented Generation) combine recherche documentaire et génération de texte par IA"
    ]

# Charger
embedding_model, llm = load_models()
documents = load_documents()

# Indexer documents
@st.cache_data
def index_documents(_embedding_model, _documents):
    return _embedding_model.encode(_documents)

doc_embeddings = index_documents(embedding_model, documents)

# Prompt template
template = """Tu es un assistant intelligent et pédagogue.

Réponds à la question en te basant sur le contexte suivant.
Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte:
{context}

Question: {question}

Réponse claire et détaillée:"""

prompt = ChatPromptTemplate.from_template(template)

# Interface principale
st.markdown("---")

# Zone de question
question = st.text_input(
    "❓ Posez votre question :",
    placeholder="Ex: Qu'est-ce que le machine learning ?"
)

if question:
    with st.spinner("🔍 Recherche en cours..."):
        # Rechercher documents
        q_embedding = embedding_model.encode([question])
        similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Contexte
        context_docs = [documents[idx] for idx in top_indices]
        context = "\n\n".join(context_docs)
    
    # Afficher documents trouvés
    with st.expander("📄 Documents pertinents trouvés", expanded=False):
        for i, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            st.markdown(f"""
            **Document {i}** (Score: {score:.3f})
            > {documents[idx]}
            """)
    
    # Générer réponse
    with st.spinner("🤖 L'IA génère la réponse..."):
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
    
    # Afficher réponse
    st.markdown("### 💬 Réponse")
    st.success(response.content)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents utilisés", top_k)
    with col2:
        st.metric("Meilleur score", f"{similarities[top_indices[0]]:.3f}")
    with col3:
        st.metric("Longueur réponse", f"{len(response.content)} chars")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 Créé avec LangChain + Mistral AI + Streamlit</p>
</div>
""", unsafe_allow_html=True)
