from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("🔍 Moteur de Recherche Sémantique\n")

# Charger modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Base de documents (votre "base de connaissances")
documents = [
    "Python est un langage de programmation orienté objet créé par Guido van Rossum",
    "Le machine learning utilise des algorithmes pour apprendre à partir de données",
    "Les réseaux de neurones artificiels imitent le fonctionnement du cerveau humain",
    "JavaScript est utilisé pour le développement web côté client et serveur",
    "Le deep learning est une branche du machine learning qui utilise des réseaux profonds",
    "HTML et CSS sont des technologies fondamentales pour créer des sites web",
    "FastAPI est un framework Python moderne pour créer des APIs REST rapides",
    "Docker permet de containeriser des applications pour faciliter le déploiement",
    "Git est un système de contrôle de version utilisé par les développeurs",
    "LangChain facilite la création d'applications avec des modèles de langage"
]

# Indexer (créer embeddings)
print("📚 Indexation de", len(documents), "documents...")
doc_embeddings = model.encode(documents)
print("✅ Indexation terminée!\n")

# Fonction de recherche
def rechercher(question, top_k=3):
    # Embedding de la question
    q_embedding = model.encode([question])
    
    # Calculer similarités
    similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
    
    # Top K résultats
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    print(f"\n{'='*70}")
    print(f"🔍 Question: {question}")
    print(f"{'='*70}\n")
    
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        print(f"{rank}. {emoji} Score: {score:.3f}")
        print(f"   📄 {documents[idx]}\n")

# Tests
rechercher("Qu'est-ce que l'intelligence artificielle ?", top_k=3)
rechercher("Comment faire un site web ?", top_k=3)
rechercher("Outil pour créer des APIs en Python", top_k=3)
rechercher("Qu'est-ce que Docker ?", top_k=2)

print("\n" + "="*70)
print("✅ Recherches terminées!")
print("="*70)
