from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

print("🤖 Premier RAG avec IA\n")

# 1. EMBEDDINGS (Recherche)
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python est un langage de programmation orienté objet créé par Guido van Rossum en 1991",
    "Le machine learning utilise des algorithmes pour apprendre à partir de données sans être explicitement programmé",
    "Les réseaux de neurones artificiels imitent le fonctionnement du cerveau humain avec des couches de neurones",
    "JavaScript est utilisé pour le développement web côté client et serveur avec Node.js",
    "Le deep learning est une branche du machine learning qui utilise des réseaux de neurones profonds",
    "FastAPI est un framework Python moderne pour créer des APIs REST rapides et performantes",
    "Docker permet de containeriser des applications pour faciliter le déploiement sur différents environnements",
    "LangChain facilite la création d'applications avec des modèles de langage comme GPT et Mistral"
]

print("📚 Indexation des documents...")
doc_embeddings = model.encode(documents)
print(f"✅ {len(documents)} documents indexés\n")

# 2. LLM (Génération)
llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.0
)

# 3. PROMPT
template = """Tu es un assistant intelligent.

Réponds à la question en te basant UNIQUEMENT sur le contexte suivant.
Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information dans mes documents".

Contexte:
{context}

Question: {question}

Réponse claire et concise:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. FONCTION RAG COMPLETE
def rag_query(question, top_k=2):
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print(f"{'='*70}\n")
    
    # a) Rechercher documents pertinents
    q_embedding = model.encode([question])
    similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # b) Récupérer contexte
    context_docs = [documents[idx] for idx in top_indices]
    context = "\n\n".join(context_docs)
    
    print("📄 Documents trouvés:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. (Score: {similarities[idx]:.3f}) {documents[idx][:80]}...")
    
    # c) Générer réponse avec l'IA
    print("\n🤔 L'IA réfléchit...\n")
    
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    
    print(f"💬 Réponse de l'IA:\n{response.content}\n")
    
    return response.content

# 5. TESTS
rag_query("Qu'est-ce que Python et qui l'a créé ?")
rag_query("Comment fonctionne le machine learning ?")
rag_query("Quelle est la capitale de la France ?")  # Pas dans les docs
rag_query("Qu'est-ce que Docker et à quoi ça sert ?")

print("="*70)
print("✅ RAG terminé!")
print("="*70)
