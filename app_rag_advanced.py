import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import os

# Désactiver LangSmith
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

# Configuration page
st.set_page_config(
    page_title="Assistant RAG Pro",
    page_icon="🤖",
    layout="wide"
)

# Récupérer clé API (Streamlit Secrets ou env)
try:
    mistral_key = st.secrets["MISTRAL_API_KEY"]
except:
    mistral_key = os.getenv("MISTRAL_API_KEY", "")

if not mistral_key:
    st.error("⚠️ Clé API Mistral non configurée!")
    st.info("Ajoutez MISTRAL_API_KEY dans Settings → Secrets")
    st.stop()

os.environ['MISTRAL_API_KEY'] = mistral_key

# Titre
st.title("🤖 Assistant RAG Professionnel")
st.markdown("*Uploadez vos documents et posez vos questions*")

# Session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None
if 'uploaded_files_names' not in st.session_state:
    st.session_state.uploaded_files_names = []

# Fonctions
def chunk_text(text, chunk_size=500, overlap=50):
    """Découper texte en morceaux"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

# Sidebar
with st.sidebar:
    st.header("📁 Documents")
    
    # Upload TXT uniquement pour commencer (plus stable)
    uploaded_files = st.file_uploader(
        "Upload fichiers TXT",
        type=['txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        new_docs = []
        new_names = []
        
        with st.spinner("📄 Traitement..."):
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files_names:
                    text = file.read().decode('utf-8')
                    chunks = chunk_text(text)
                    
                    for chunk in chunks:
                        new_docs.append({
                            'content': chunk,
                            'source': file.name
                        })
                    
                    new_names.append(file.name)
        
        if new_docs:
            st.session_state.documents.extend(new_docs)
            st.session_state.uploaded_files_names.extend(new_names)
            st.session_state.doc_embeddings = None
            st.success(f"✅ {len(new_docs)} chunks ajoutés!")
    
    # Afficher fichiers
    if st.session_state.uploaded_files_names:
        st.markdown("### 📚 Fichiers chargés")
        for name in st.session_state.uploaded_files_names:
            st.markdown(f"- 📄 {name}")
        st.metric("Total chunks", len(st.session_state.documents))
        
        if st.button("🗑️ Tout supprimer"):
            st.session_state.documents = []
            st.session_state.doc_embeddings = None
            st.session_state.uploaded_files_names = []
            st.rerun()
    
    st.markdown("---")
    
    # Documents exemple
    if st.button("📚 Charger exemples"):
        default_docs = [
            {"content": "Python est un langage de programmation orienté objet créé par Guido van Rossum en 1991", "source": "default"},
            {"content": "Le machine learning utilise des algorithmes pour apprendre à partir de données sans être explicitement programmé", "source": "default"},
            {"content": "Les réseaux de neurones artificiels imitent le fonctionnement du cerveau humain", "source": "default"},
            {"content": "Docker permet de containeriser des applications pour faciliter le déploiement", "source": "default"},
            {"content": "Git est un système de contrôle de version distribué utilisé par les développeurs", "source": "default"},
            {"content": "LangChain facilite la création d'applications avec des LLMs comme GPT et Mistral", "source": "default"},
        ]
        st.session_state.documents = default_docs
        st.session_state.doc_embeddings = None
        st.session_state.uploaded_files_names = ["default"]
        st.rerun()
    
    st.markdown("---")
    st.header("⚙️ Paramètres")
    top_k = st.slider("Docs à chercher", 1, 5, 3)
    temperature = st.slider("Créativité", 0.0, 1.0, 0.0)

# Charger modèles
@st.cache_resource
def load_models():
    try:
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        llm = ChatMistralAI(model="open-mistral-7b", temperature=0.0)
        return emb_model, llm
    except Exception as e:
        st.error(f"Erreur chargement modèles: {e}")
        return None, None

embedding_model, llm = load_models()

if embedding_model is None or llm is None:
    st.stop()

# Interface principale
st.markdown("---")

if not st.session_state.documents:
    st.warning("⚠️ Chargez des documents (sidebar)")
    st.stop()

# Indexer si nécessaire
if st.session_state.doc_embeddings is None:
    with st.spinner("🔄 Indexation..."):
        try:
            doc_contents = [doc['content'] for doc in st.session_state.documents]
            st.session_state.doc_embeddings = embedding_model.encode(doc_contents)
            st.success("✅ Documents indexés!")
        except Exception as e:
            st.error(f"Erreur indexation: {e}")
            st.stop()

# Question
question = st.text_input(
    "❓ Posez votre question:",
    placeholder="Ex: Qu'est-ce que Python?"
)

if question:
    try:
        # Rechercher
        with st.spinner("🔍 Recherche..."):
            q_embedding = embedding_model.encode([question])
            similarities = cosine_similarity(q_embedding, st.session_state.doc_embeddings)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            context_docs = [st.session_state.documents[idx] for idx in top_indices]
            context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Documents trouvés
        with st.expander(f"📄 {top_k} documents trouvés", expanded=False):
            for i, idx in enumerate(top_indices, 1):
                doc = st.session_state.documents[idx]
                score = similarities[idx]
                st.markdown(f"**{i}.** Score: {score:.3f} - `{doc['source']}`")
                st.markdown(f"> {doc['content'][:150]}...")
        
        # Générer réponse
        with st.spinner("🤖 Génération..."):
            template = """Réponds à la question en te basant sur le contexte.

Contexte:
{context}

Question: {question}

Réponse:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            messages = prompt.format_messages(context=context, question=question)
            
            llm.temperature = temperature
            response = llm.invoke(messages)
        
        # Afficher
        st.markdown("### 💬 Réponse")
        st.success(response.content)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Docs", top_k)
        with col2:
            st.metric("Score", f"{similarities[top_indices[0]]:.3f}")
        with col3:
            st.metric("Total", len(st.session_state.documents))
    
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

# Footer
st.markdown("---")
st.markdown("🚀 *Assistant RAG - LangChain + Mistral AI*")
