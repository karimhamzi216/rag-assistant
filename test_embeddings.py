from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 Test des Embeddings\n")

# Charger modèle
print("📥 Chargement du modèle...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modèle chargé!\n")

# Phrases de test
phrases = [
    "Le chat dort sur le canapé",
    "Un chat se repose sur le sofa",
    "Le chien joue dans le jardin",
    "Python est un langage de programmation",
    "Je code en Python"
]

# Créer embeddings
embeddings = model.encode(phrases)
print(f"✅ {len(embeddings)} embeddings créés (dimension: {embeddings.shape[1]})\n")

# Calculer similarités
sim = cosine_similarity(embeddings)

# Afficher résultats
print("📊 Similarités entre phrases:\n")
for i in range(len(phrases)):
    print(f"{i+1}. {phrases[i]}")
    for j in range(len(phrases)):
        if i != j:
            score = sim[i][j]
            emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
            print(f"   {emoji} vs {j+1}: {score:.3f}")
    print()

print("✅ Test terminé!")
