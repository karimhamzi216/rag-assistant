import streamlit as st
import os

# Config
st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

# Désactiver tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

st.title("🤖 Assistant RAG")

# Test de base
st.write("✅ App démarrée avec succès!")

# Vérifier secret
try:
    api_key = st.secrets["MISTRAL_API_KEY"]
    st.success("✅ Clé API Mistral détectée")
    has_key = True
except:
    st.error("❌ Clé API Mistral manquante")
    st.info("Ajoutez MISTRAL_API_KEY dans Settings → Secrets")
    has_key = False

if has_key:
    # Import uniquement si la clé existe
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_mistralai import ChatMistralAI
        from sklearn.metrics.pairwise import cosine_similarity
        
        os.environ['MISTRAL_API_KEY'] = api_key
        
        st.success("✅ Modules chargés")
        
        # Modèles
        @st.cache_resource
        def load_models():
            emb = SentenceTransformer('all-MiniLM-L6-v2')
            llm = ChatMistralAI(model="open-mistral-7b", temperature=0.0)
            return emb, llm
        
        with st.spinner("Chargement des modèles..."):
            embedding_model, llm = load_models()
        
        st.success("✅ Modèles chargés")
        
        # Documents exemple
        documents = [
            "Python est un langage de programmation créé par Guido van Rossum",
            "Le machine learning utilise des algorithmes pour apprendre",
            "Docker permet de containeriser des applications",
            "Git est un système de contrôle de version",
            "Streamlit est un framework pour créer des apps web en Python"
        ]
        
        # Indexer
        @st.cache_data
        def index_documents(_model, docs):
            return _model.encode(docs)
        
        doc_embeddings = index_documents(embedding_model, documents)
        
        st.info(f"📚 {len(documents)} documents indexés")
        
        # Interface
        st.markdown("---")
        question = st.text_input("❓ Posez votre question:", placeholder="Ex: Qu'est-ce que Python?")
        
        if question:
            # Rechercher
            q_emb = embedding_model.encode([question])
            similarities = cosine_similarity(q_emb, doc_embeddings)[0]
            best_idx = similarities.argmax()
            
            context = documents[best_idx]
            
            with st.expander("📄 Document trouvé"):
                st.write(f"Score: {similarities[best_idx]:.3f}")
                st.write(context)
            
            # Générer réponse
            with st.spinner("🤖 Génération..."):
                prompt = f"""Réponds à la question en te basant sur ce contexte:

Contexte: {context}

Question: {question}

Réponse courte:"""
                
                response = llm.invoke(prompt)
            
            st.success(f"💬 {response.content}")
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.code(str(e))

st.markdown("---")
st.caption("🚀 Assistant RAG - LangChain + Mistral AI")
