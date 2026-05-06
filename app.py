"""
app.py
=======
Interface web Streamlit pour le RAG Assistant Code du Travail.

Lancement : streamlit run app.py

L'application :
  - Charge l'index FAISS au démarrage (mis en cache)
  - Charge le modèle d'embedding au démarrage (mis en cache)
  - Permet à l'utilisateur de poser des questions via une interface chat
  - Affiche la réponse du LLM, les sources citées et les scores de confiance
  - Conserve l'historique de la conversation pour les questions de suivi
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

INDEX_DIR = Path("index_faiss")
INDEX_PATH = INDEX_DIR / "code_travail.index"
META_PATH = INDEX_DIR / "code_travail_meta.pkl"

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL_RAPIDE = "llama-3.1-8b-instant"

K_CHUNKS = 4
SCORE_MINIMUM = 0.35
TEMPERATURE = 0.1
MAX_TOKENS = 1024
MAX_HISTORIQUE = 12


# =============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Assistant Code du Travail",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS PERSONNALISÉ
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .source-card {
        background-color: #f7fafc;
        border-left: 4px solid #2c5282;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .source-article {
        font-weight: 600;
        color: #2c5282;
    }
    .source-section {
        font-style: italic;
        color: #718096;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #fef5e7;
        border-left: 4px solid #d69e2e;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .stats-box {
        background-color: #edf2f7;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CHARGEMENT DES RESSOURCES (mis en cache)
# =============================================================================

@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def charger_modele() -> SentenceTransformer:
    """Charge le modèle d'embedding. Mis en cache pour éviter les rechargements."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Chargement de l'index FAISS...")
def charger_index() -> tuple[faiss.Index, list[dict]]:
    """Charge l'index FAISS et les métadonnées. Mis en cache."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        st.error(
            "Index introuvable. Lance d'abord la commande `python indexation.py` "
            "depuis le terminal pour générer l'index."
        )
        st.stop()
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        chunks_avec_meta = pickle.load(f)
    return index, chunks_avec_meta


@st.cache_resource
def get_groq_client() -> Groq:
    """Crée le client Groq. Mis en cache."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "Variable GROQ_API_KEY introuvable. Crée un fichier `.env` avec ta clé "
            "ou définis la variable d'environnement GROQ_API_KEY."
        )
        st.stop()
    return Groq(api_key=api_key)


# =============================================================================
# LOGIQUE RAG
# =============================================================================

def rechercher(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict],
    k: int = K_CHUNKS,
) -> list[dict]:
    """Recherche les k chunks les plus pertinents pour une question."""
    vec = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, indices = index.search(vec, k)
    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks_avec_meta[idx]
        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": chunk["metadata"],
            "score": float(score),
        })
    return resultats


def reformuler_question(question: str, client: Groq) -> str:
    """Bonus C : reformule la question pour améliorer la recherche."""
    try:
        completion = client.chat.completions.create(
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


def construire_prompt_utilisateur(question: str, chunks: list[dict]) -> str:
    """Construit le prompt utilisateur avec contexte numéroté."""
    blocs = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        blocs.append(
            f"[Source {i} | Article {m['article']} - {m['titre']} "
            f"| Section : {m['section']}]\n{chunk['contenu']}"
        )
    contexte = "\n\n---\n\n".join(blocs)
    return (
        f"CONTEXTE (extraits du Code du travail) :\n\n{contexte}\n\n---\n\n"
        f"QUESTION DE L'UTILISATEUR : {question}\n\n"
        f"Réponds en respectant strictement les 7 règles données dans tes instructions."
    )


def generer_reponse(
    question: str,
    chunks: list[dict],
    client: Groq,
    historique: list[dict] | None = None,
) -> str:
    """Appelle l'API Groq et retourne la réponse générée."""
    messages = [{"role": "system", "content": PROMPT_SYSTEME}]
    if historique:
        messages.extend(historique)
    messages.append({
        "role": "user",
        "content": construire_prompt_utilisateur(question, chunks),
    })
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# INITIALISATION DE LA SESSION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "historique_llm" not in st.session_state:
    st.session_state.historique_llm = []


# =============================================================================
# EN-TÊTE
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>⚖️ Assistant Code du Travail</h1>
    <p>RAG basé sur 34 articles réels du Code du travail français • Powered by Groq + FAISS</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CHARGEMENT DES RESSOURCES
# =============================================================================

modele = charger_modele()
index, chunks_avec_meta = charger_index()
client = get_groq_client()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    nb_chunks = st.slider(
        "Nombre de chunks récupérés (k)",
        min_value=2, max_value=10, value=K_CHUNKS,
        help="Plus de chunks = plus de contexte, mais aussi plus de bruit.",
    )

    seuil_confiance = st.slider(
        "Seuil de confiance minimal",
        min_value=0.0, max_value=1.0, value=SCORE_MINIMUM, step=0.05,
        help="Si le meilleur chunk a un score inférieur, le système refuse de répondre.",
    )

    use_reformulation = st.checkbox(
        "Activer la reformulation (Bonus C)",
        value=True,
        help="Reformule la question en formulation juridique avant la recherche.",
    )

    show_sources = st.checkbox(
        "Afficher les sources et scores",
        value=True,
        help="Affiche les chunks récupérés et leur score de confiance.",
    )

    st.divider()

    st.markdown("### 📊 Statistiques")
    st.markdown(f"""
    <div class="stats-box">
        <b>{index.ntotal}</b> chunks indexés<br>
        <b>{len(set(c['metadata']['section'] for c in chunks_avec_meta))}</b> sections couvertes<br>
        Modèle : <code>{LLM_MODEL.split('-versatile')[0]}</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 💡 Exemples de questions")
    exemples = [
        "Combien de jours de congés payés par an ?",
        "Quelle est la durée légale du préavis pour un CDI ?",
        "Comment fonctionne la rupture conventionnelle ?",
        "Quels sont mes droits en cas de licenciement économique ?",
        "Qu'est-ce que le harcèlement moral ?",
        "Combien de jours de congé paternité ?",
    ]
    for i, ex in enumerate(exemples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.question_suggeree = ex
            st.rerun()

    st.divider()

    if st.button("🔄 Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.historique_llm = []
        st.rerun()

    st.markdown("""
    <div class="warning-box">
        <b>⚠️ Avertissement</b><br>
        Cet assistant ne fournit pas de conseil juridique. Consultez un avocat
        ou l'inspection du travail pour votre situation personnelle.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# AFFICHAGE DE L'HISTORIQUE
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    m = src["metadata"]
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-article">[{i}] Article {m['article']} - {m['titre']}</span><br>
                        <span class="source-section">Section : {m['section']} • Score : {src['score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# ENTRÉE UTILISATEUR
# =============================================================================

# Question pré-remplie via les exemples
question_par_defaut = st.session_state.pop("question_suggeree", None)
question = st.chat_input("Pose ta question sur le Code du travail...")
if question_par_defaut and not question:
    question = question_par_defaut


# =============================================================================
# TRAITEMENT D'UNE NOUVELLE QUESTION
# =============================================================================

if question:
    # Affichage immédiat de la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1. Reformulation (Bonus C)
        with st.spinner("Reformulation de la question..."):
            if use_reformulation:
                question_reformulee = reformuler_question(question, client)
            else:
                question_reformulee = question

        if use_reformulation and question_reformulee != question:
            st.caption(f"🔄 Reformulée : *{question_reformulee}*")

        # 2. Recherche vectorielle
        with st.spinner("Recherche dans le Code du travail..."):
            chunks = rechercher(
                question_reformulee, modele, index, chunks_avec_meta, k=nb_chunks
            )

        # 3. Bonus B : filtre par score de confiance
        meilleur_score = chunks[0]["score"] if chunks else 0.0
        if not chunks or meilleur_score < seuil_confiance:
            reponse = (
                f"Je n'ai pas trouvé d'information directement pertinente dans ma base "
                f"(meilleur score : **{meilleur_score:.2f}**, seuil : {seuil_confiance:.2f}).\n\n"
                f"Reformule ta question ou demande quelque chose en lien avec les thèmes "
                f"couverts : durée du travail, congés payés, contrat, licenciement, rupture "
                f"conventionnelle, harcèlement, CSE, parentalité, SMIC."
            )
            st.warning(reponse)
            st.session_state.messages.append({
                "role": "assistant", "content": reponse, "sources": []
            })
        else:
            # 4. Génération de la réponse
            try:
                with st.spinner(f"Génération de la réponse avec {LLM_MODEL}..."):
                    reponse = generer_reponse(
                        question, chunks, client,
                        historique=st.session_state.historique_llm,
                    )
                st.markdown(reponse)

                # Affichage des sources
                if show_sources:
                    with st.expander(f"📚 Sources ({len(chunks)} chunks)"):
                        for i, src in enumerate(chunks, 1):
                            m = src["metadata"]
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-article">[{i}] Article {m['article']} - {m['titre']}</span><br>
                                <span class="source-section">Section : {m['section']} • Score : {src['score']:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # Sauvegarde dans l'historique
                st.session_state.messages.append({
                    "role": "assistant", "content": reponse, "sources": chunks
                })

                # Mise à jour de l'historique pour le LLM (Bonus A)
                st.session_state.historique_llm.append({"role": "user", "content": question})
                st.session_state.historique_llm.append({"role": "assistant", "content": reponse})
                if len(st.session_state.historique_llm) > MAX_HISTORIQUE:
                    st.session_state.historique_llm = st.session_state.historique_llm[-MAX_HISTORIQUE:]

            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API Groq : {e}")
