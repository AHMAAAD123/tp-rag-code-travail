"""
rag.py
=======
Phase 2 du pipeline RAG : interrogation de la base de connaissance.

Ce script :
  1. Charge l'index FAISS et les métadonnées (jamais de réindexation).
  2. Embedde la question utilisateur avec le MÊME modèle que l'indexation.
  3. Recherche les k chunks les plus proches (similarité cosinus).
  4. Filtre par score de confiance pour éviter les hallucinations.
  5. Construit un prompt avec contexte enrichi et historique.
  6. Appelle l'API Groq pour générer une réponse citant ses sources.

Bonus implémentés :
  A. Historique de conversation (questions de suivi).
  B. Score de confiance (refus si aucun chunk pertinent).
  C. Reformulation automatique de la question avant recherche.

Usage : python rag.py
"""

import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit(
        "Erreur : variable GROQ_API_KEY introuvable.\n"
        "  Crée un fichier .env à la racine du projet avec :\n"
        "  GROQ_API_KEY=ta_cle_api_groq\n"
        "  Tu peux générer une clé gratuite sur https://console.groq.com/"
    )

INDEX_DIR = Path("index_faiss")
INDEX_PATH = INDEX_DIR / "code_travail.index"
META_PATH = INDEX_DIR / "code_travail_meta.pkl"

# IMPORTANT : doit être strictement le même que dans indexation.py.
# Changer de modèle après indexation casserait la cohérence de l'espace
# vectoriel et rendrait la recherche impossible.
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Modèles Groq actifs (vérifiés mai 2026).
# - llama-3.3-70b-versatile : qualité supérieure pour la synthèse finale
# - llama-3.1-8b-instant : rapide, suffisant pour la reformulation
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL_RAPIDE = "llama-3.1-8b-instant"

K_CHUNKS = 4              # nombre de chunks à passer au LLM
SCORE_MINIMUM = 0.35      # seuil de confiance (similarité cosinus)
TEMPERATURE = 0.1         # basse température = fidélité au contexte
MAX_TOKENS = 1024         # longueur maximale de la réponse
MAX_HISTORIQUE = 12       # nombre de messages conservés (6 tours)

client_groq = Groq(api_key=GROQ_API_KEY)


# =============================================================================
# CHARGEMENT DE L'INDEX
# =============================================================================

def charger_index() -> tuple[faiss.Index, list[dict]]:
    """Recharge l'index FAISS et les métadonnées depuis le disque."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit(
            f"Erreur : index introuvable.\n"
            f"  Lance d'abord : python indexation.py"
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        chunks_avec_meta = pickle.load(f)
    return index, chunks_avec_meta


# =============================================================================
# RECHERCHE VECTORIELLE
# =============================================================================

def rechercher(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict],
    k: int = K_CHUNKS,
) -> list[dict]:
    """
    Recherche les k chunks les plus pertinents pour une question.

    On utilise la même normalisation L2 qu'à l'indexation. Avec un
    IndexFlatIP, le score retourné est la similarité cosinus, comprise
    entre -1 et 1. Plus le score est proche de 1, plus le chunk est
    sémantiquement pertinent.
    """
    vec_question = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(vec_question, k)

    resultats: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS retourne -1 quand il y a moins de k résultats
            continue
        chunk = chunks_avec_meta[idx]
        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": chunk["metadata"],
            "score": float(score),
        })
    return resultats


# =============================================================================
# BONUS C — REFORMULATION DE LA QUESTION
# =============================================================================

def reformuler_question(question: str) -> str:
    """
    Reformule la question utilisateur en une formulation juridique précise
    pour améliorer la pertinence de la recherche vectorielle.

    Une question familière (« j'ai été viré, qu'est-ce que je peux faire ? »)
    produit un vecteur très éloigné des articles de loi. Reformuler en
    « Quels sont les droits du salarié en cas de licenciement ? » rapproche
    le vecteur de la question des vecteurs des articles indexés.

    En cas d'erreur ou de réponse aberrante, on retombe sur la question
    originale (fail-safe).
    """
    try:
        completion = client_groq.chat.completions.create(
            model=LLM_MODEL_RAPIDE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu reformules la question d'un utilisateur en une "
                        "formulation juridique précise pour une recherche "
                        "documentaire dans le Code du travail français. "
                        "Tu réponds UNIQUEMENT par la reformulation, sans "
                        "préambule ni guillemets, en une seule phrase courte."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        reformulation = completion.choices[0].message.content.strip()
        if not reformulation or len(reformulation) > 300:
            return question
        return reformulation
    except Exception:
        return question


# =============================================================================
# CONSTRUCTION DU PROMPT
# =============================================================================

PROMPT_SYSTEME = """Tu es un assistant juridique spécialisé dans le Code du travail français.

RÈGLES STRICTES À RESPECTER :
1. Tu réponds UNIQUEMENT en t'appuyant sur les articles fournis dans la section CONTEXTE.
2. Tu cites systématiquement le numéro d'article (par exemple : « selon l'article L3141-3 ») sur lequel tu te bases.
3. Si l'information demandée n'est pas présente dans le contexte fourni, tu réponds clairement : « Je ne trouve pas cette information dans ma base de connaissances. »
4. Tu n'inventes JAMAIS de numéro d'article ni de disposition légale.
5. Tu termines TOUJOURS ta réponse par cet avertissement exact : « Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle. »
6. Quand la réponse dépend de la convention collective, du secteur d'activité ou de la taille de l'entreprise, tu le précises explicitement.
7. Tu rappelles que le droit du travail évolue régulièrement et qu'il est nécessaire de vérifier la version en vigueur.

FORMAT DE RÉPONSE ATTENDU :
- Réponse claire et structurée (listes à puces si nécessaire).
- Citation des articles entre parenthèses dans le corps de la réponse.
- Section « Sources : » en fin de réponse listant les articles utilisés.
- Avertissement final imposé par la règle 5.
"""


def construire_prompt_utilisateur(question: str, chunks_pertinents: list[dict]) -> str:
    """Construit le prompt utilisateur avec contexte numéroté et traçable."""
    blocs_contexte = []
    for i, chunk in enumerate(chunks_pertinents, 1):
        meta = chunk["metadata"]
        blocs_contexte.append(
            f"[Source {i} | Article {meta['article']} - {meta['titre']} "
            f"| Section : {meta['section']}]\n"
            f"{chunk['contenu']}"
        )
    contexte = "\n\n---\n\n".join(blocs_contexte)

    return (
        f"CONTEXTE (extraits du Code du travail) :\n\n"
        f"{contexte}\n\n"
        f"---\n\n"
        f"QUESTION DE L'UTILISATEUR : {question}\n\n"
        f"Réponds en respectant strictement les 7 règles données dans tes instructions."
    )


# =============================================================================
# GÉNÉRATION DE LA RÉPONSE
# =============================================================================

def generer_reponse(
    question: str,
    chunks_pertinents: list[dict],
    historique: list[dict] | None = None,
) -> str:
    """Appelle l'API Groq avec prompt système + historique + question contextualisée."""
    messages: list[dict] = [{"role": "system", "content": PROMPT_SYSTEME}]
    if historique:
        messages.extend(historique)
    messages.append({
        "role": "user",
        "content": construire_prompt_utilisateur(question, chunks_pertinents),
    })

    completion = client_groq.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# AFFICHAGE DEBUG
# =============================================================================

def afficher_chunks(chunks: list[dict]) -> None:
    """Mode debug : affiche les chunks récupérés et leurs scores."""
    print("\n  Chunks récupérés :")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        print(
            f"    [{i}] score={c['score']:.3f} | "
            f"art. {meta['article']} - {meta['titre']}"
        )


# =============================================================================
# BOUCLE INTERACTIVE
# =============================================================================

def main() -> None:
    print("=" * 64)
    print("  RAG - ASSISTANT CODE DU TRAVAIL")
    print("=" * 64)
    print("\nChargement de la base de connaissances...")

    index, chunks_avec_meta = charger_index()
    print(f"  {index.ntotal} chunks indexés.")

    print(f"Chargement du modèle d'embedding '{EMBEDDING_MODEL}'...")
    modele = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nSystème RAG prêt. (LLM : {LLM_MODEL})")
    print("Commandes disponibles :")
    print("  - 'quit', 'exit' ou 'q' : quitter")
    print("  - 'debug on' / 'debug off' : afficher/masquer les chunks récupérés")
    print("  - 'reset' : effacer l'historique de la conversation")
    print()

    historique: list[dict] = []
    debug = False

    while True:
        try:
            question = input("Question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not question:
            continue

        commande = question.lower()
        if commande in {"quit", "exit", "q"}:
            print("Au revoir !")
            break
        if commande == "debug on":
            debug = True
            print("  [debug activé]\n")
            continue
        if commande == "debug off":
            debug = False
            print("  [debug désactivé]\n")
            continue
        if commande == "reset":
            historique = []
            print("  [historique effacé]\n")
            continue

        # 1. Bonus C : reformulation pour améliorer la recherche
        question_reformulee = reformuler_question(question)
        if debug and question_reformulee != question:
            print(f"  Reformulation : {question_reformulee}")

        # 2. Recherche vectorielle
        chunks = rechercher(question_reformulee, modele, index, chunks_avec_meta)
        if debug:
            afficher_chunks(chunks)

        # 3. Bonus B : filtre par score de confiance
        if not chunks or chunks[0]["score"] < SCORE_MINIMUM:
            meilleur_score = chunks[0]["score"] if chunks else 0.0
            print(
                f"\nJe n'ai pas trouvé d'information directement pertinente "
                f"dans ma base (meilleur score : {meilleur_score:.2f}).\n"
                f"Reformule ta question ou demande quelque chose en lien avec "
                f"les thèmes couverts (durée du travail, congés payés, contrat, "
                f"licenciement, rupture conventionnelle, harcèlement, CSE, "
                f"parentalité, SMIC).\n"
            )
            continue

        # 4. Génération de la réponse par le LLM
        try:
            print("\nRéponse :\n")
            reponse = generer_reponse(question, chunks, historique=historique)
            print(reponse)
            print()
        except Exception as e:
            print(f"\nErreur lors de l'appel à l'API Groq : {e}\n")
            continue

        # 5. Bonus A : mise à jour de l'historique (questions de suivi)
        # On ne stocke que la question originale et la réponse, pas le contexte,
        # pour ne pas saturer la fenêtre de contexte du LLM.
        historique.append({"role": "user", "content": question})
        historique.append({"role": "assistant", "content": reponse})
        if len(historique) > MAX_HISTORIQUE:
            historique = historique[-MAX_HISTORIQUE:]


if __name__ == "__main__":
    main()
