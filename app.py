import streamlit as st
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'false'

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

st.title("🤖 Assistant RAG")

# Clé API
try:
    api_key = st.secrets["MISTRAL_API_KEY"]
    os.environ['MISTRAL_API_KEY'] = api_key
    st.success("✅ Clé API configurée")
except:
    st.error("❌ Clé API manquante - Ajoutez MISTRAL_API_KEY dans Settings → Secrets")
    st.stop()

# Imports
try:
    from sentence_transformers import SentenceTransformer
    from langchain_mistralai import ChatMistralAI
    from sklearn.metrics.pairwise import cosine_similarity
    
    st.success("✅ Modules chargés")
    
    # Modèles
    @st.cache_resource
    def load_models():
        emb = SentenceTransformer('all-MiniLM-L6-v2')
        llm = ChatMistralAI(model="open-mistral-7b", temperature=0.0)
        return emb, llm
    
    embedding_model, llm = load_models()
    st.success("✅ Modèles IA prêts")
    
    # Documents
    documents = [
        "Python est un langage de programmation créé par Guido van Rossum en 1991",
        "Le machine learning utilise des algorithmes pour apprendre des données",
        "Docker permet de containeriser des applications pour faciliter le déploiement",
        "Git est un système de contrôle de version distribué",
        "Streamlit est un framework Python pour créer des applications web"
    ]
    
    doc_embeddings = embedding_model.encode(documents)
    st.info(f"📚 {len(documents)} documents indexés")
    
    st.markdown("---")
    
    # Interface
    question = st.text_input("❓ Posez votre question:", placeholder="Ex: Qu'est-ce que Python?")
    
    if question:
        # Rechercher
        q_emb = embedding_model.encode([question])
        similarities = cosine_similarity(q_emb, doc_embeddings)[0]
        best_idx = similarities.argmax()
        
        context = documents[best_idx]
        
        with st.expander("📄 Document trouvé"):
            st.write(f"**Score:** {similarities[best_idx]:.3f}")
            st.write(context)
        
        # Générer
        with st.spinner("🤖 Génération de la réponse..."):
            prompt = f"Contexte: {context}\n\nQuestion: {question}\n\nRéponse:"
            response = llm.invoke(prompt)
        
        st.success(f"💬 **Réponse:**\n\n{response.content}")

except Exception as e:
    st.error(f"❌ Erreur: {e}")

st.markdown("---")
st.caption("🚀 Assistant RAG - LangChain + Mistral AI + Streamlit")
