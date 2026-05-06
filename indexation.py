"""
indexation.py
==============
Phase 1 du pipeline RAG : indexation du Code du Travail.

Ce script :
  1. Charge le corpus JSON contenant les articles du Code du travail.
  2. Enrichit chaque article (article + titre + section + texte).
  3. Découpe les contenus en chunks avec chevauchement (recursive splitter).
  4. Génère les embeddings avec un modèle multilingue.
  5. Crée un index FAISS (produit scalaire = cosinus si vecteurs normalisés).
  6. Persiste l'index et les métadonnées sur disque.

Usage : python indexation.py
À exécuter une seule fois (ou à chaque mise à jour du corpus).
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

CORPUS_PATH = Path("corpus/code_travail.json")
INDEX_DIR = Path("index_faiss")
INDEX_PATH = INDEX_DIR / "code_travail.index"
META_PATH = INDEX_DIR / "code_travail_meta.pkl"

# Modèle multilingue recommandé pour le français.
# - Dimension : 768
# - Limite : 128 tokens (utile pour des articles courts)
# - Source : https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Paramètres de chunking.
# Les articles du Code du travail sont courts mais denses : on choisit une
# taille suffisante pour qu'un article tienne en un seul chunk, avec un
# overlap modéré pour les rares articles longs.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


# =============================================================================
# CHUNKING
# =============================================================================

def chunker(texte: str, taille_max: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Stratégie : recursive character splitter. On essaie d'abord de couper
    sur les paragraphes (\\n\\n), puis sur les sauts de ligne, puis sur les
    phrases (. ), puis sur les espaces. Cela évite de couper en plein milieu
    d'une phrase, ce qui dégraderait la qualité de l'embedding.

    Args:
        texte: texte à découper.
        taille_max: nombre maximum de caractères par chunk.
        overlap: nombre de caractères répétés entre deux chunks consécutifs.

    Returns:
        Liste de chunks (chaînes de caractères non vides).
    """
    texte = texte.strip()
    if len(texte) <= taille_max:
        return [texte]

    separateurs = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    debut = 0

    while debut < len(texte):
        fin = min(debut + taille_max, len(texte))

        if fin == len(texte):
            chunks.append(texte[debut:fin].strip())
            break

        # On cherche le meilleur séparateur (le plus haut dans la hiérarchie)
        # situé après le milieu du chunk courant.
        coupure = -1
        for sep in separateurs:
            pos = texte.rfind(sep, debut, fin)
            if pos != -1 and pos > debut + taille_max // 2:
                coupure = pos + len(sep)
                break

        if coupure == -1:
            coupure = fin  # Fallback : coupure brute si aucun séparateur trouvé.

        chunks.append(texte[debut:coupure].strip())
        debut = max(coupure - overlap, debut + 1)

    return [c for c in chunks if c]


# =============================================================================
# PRÉPARATION DES DOCUMENTS
# =============================================================================

def preparer_documents(chemin_corpus: Path) -> list[dict]:
    """
    Charge le corpus JSON et construit une liste de documents enrichis.

    Pour maximiser la pertinence de la recherche sémantique, on inclut
    le numéro d'article, le titre et la section dans le texte qui sera
    embedded (et pas seulement en métadonnée). Cela permet à la recherche
    de capter des requêtes du type « quelle est l'indemnité de licenciement »
    même si le mot-clé apparaît dans le titre et non dans le corps.
    """
    if not chemin_corpus.exists():
        raise FileNotFoundError(f"Corpus introuvable : {chemin_corpus}")

    with open(chemin_corpus, "r", encoding="utf-8") as f:
        articles = json.load(f)

    documents: list[dict] = []
    for i, art in enumerate(articles):
        contenu = (
            f"Article {art['article']} - {art['titre']}\n"
            f"Section : {art['section']}\n\n"
            f"{art['texte']}"
        )
        documents.append({
            "id": f"doc_{i:03d}",
            "contenu": contenu,
            "metadata": {
                "article": art["article"],
                "titre": art["titre"],
                "section": art["section"],
                "source": "Code du travail",
            },
        })
    return documents


def decouper_documents(documents: list[dict]) -> list[dict]:
    """
    Applique le chunker à chaque document. Chaque chunk hérite des
    métadonnées de son document parent pour garantir la traçabilité.
    """
    chunks_avec_meta: list[dict] = []
    for doc in documents:
        morceaux = chunker(doc["contenu"])
        for j, morceau in enumerate(morceaux):
            chunks_avec_meta.append({
                "doc_id": doc["id"],
                "chunk_id": f"{doc['id']}_chunk_{j:02d}",
                "contenu": morceau,
                "metadata": doc["metadata"],
            })
    return chunks_avec_meta


# =============================================================================
# EMBEDDINGS
# =============================================================================

def embedder_chunks(chunks_avec_meta: list[dict], modele: SentenceTransformer) -> np.ndarray:
    """
    Encode tous les chunks en vecteurs.

    Les vecteurs sont normalisés en L2 (norme unitaire) afin que le produit
    scalaire utilisé par FAISS (IndexFlatIP) corresponde exactement à la
    similarité cosinus.
    """
    textes = [c["contenu"] for c in chunks_avec_meta]
    print(f"  Encodage de {len(textes)} chunks...")
    vecteurs = modele.encode(
        textes,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecteurs.astype(np.float32)


# =============================================================================
# INDEX FAISS
# =============================================================================

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS basé sur le produit scalaire (Inner Product).

    Comme les vecteurs sont L2-normalisés, IP équivaut à la similarité
    cosinus. Le score retourné est donc compris entre -1 et 1, plus il est
    proche de 1, plus le chunk est sémantiquement pertinent.

    Pour un petit corpus (quelques centaines/milliers de chunks), un index
    plat est largement suffisant. Pour des millions de vecteurs, on
    utiliserait IndexHNSWFlat ou IndexIVFFlat.
    """
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)
    return index


def sauvegarder(index: faiss.Index, chunks_avec_meta: list[dict]) -> None:
    """
    Sauvegarde l'index FAISS et les métadonnées sur disque.

    L'index FAISS associe chaque vecteur à un entier (son rang). Il est
    donc impératif de sauvegarder la liste des chunks dans le même ordre
    pour pouvoir retrouver le texte correspondant à un résultat de recherche.
    """
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks_avec_meta, f)
    print(f"  Index sauvegardé : {INDEX_PATH}")
    print(f"  Métadonnées sauvegardées : {META_PATH}")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    print("=" * 64)
    print("  INDEXATION DU CODE DU TRAVAIL")
    print("=" * 64)

    print("\n[1/5] Chargement du corpus...")
    documents = preparer_documents(CORPUS_PATH)
    print(f"  {len(documents)} articles chargés.")

    print("\n[2/5] Découpage en chunks...")
    chunks_avec_meta = decouper_documents(documents)
    print(f"  {len(chunks_avec_meta)} chunks générés.")
    tailles = [len(c["contenu"]) for c in chunks_avec_meta]
    print(
        f"  Tailles : moyenne={sum(tailles) // len(tailles)} car., "
        f"min={min(tailles)}, max={max(tailles)}"
    )

    print(f"\n[3/5] Chargement du modèle '{EMBEDDING_MODEL}'...")
    print("  (Premier lancement : téléchargement ~470 Mo, ensuite cache local)")
    modele = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Dimension des vecteurs : {modele.get_sentence_embedding_dimension()}")

    print("\n[4/5] Génération des embeddings...")
    vecteurs = embedder_chunks(chunks_avec_meta, modele)
    print(f"  Shape de la matrice : {vecteurs.shape}")

    print("\n[5/5] Création et sauvegarde de l'index FAISS...")
    index = creer_index_faiss(vecteurs)
    print(f"  {index.ntotal} vecteurs indexés.")
    sauvegarder(index, chunks_avec_meta)

    print("\n" + "=" * 64)
    print("  INDEXATION TERMINÉE AVEC SUCCÈS")
    print("=" * 64)
    print("\n  Lance maintenant : python rag.py\n")


if __name__ == "__main__":
    main()
