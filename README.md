# TP RAG — Assistant Code du Travail

Système RAG (Retrieval-Augmented Generation) construit **from scratch** (sans LangChain ni LlamaIndex), capable de répondre à des questions de droit du travail français en s'appuyant sur des articles réels du Code du travail et en citant systématiquement leurs numéros.

**Sujet choisi : C — Assistant Code du Travail**

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 1 — INDEXATION (indexation.py)                    │
│                                                          │
│  corpus/code_travail.json                                │
│        ↓                                                 │
│  Préparation (texte enrichi : article + titre + texte)   │
│        ↓                                                 │
│  Chunking récursif (taille=500, overlap=80)              │
│        ↓                                                 │
│  Embeddings normalisés (paraphrase-multilingual-mpnet)   │
│        ↓                                                 │
│  Index FAISS IndexFlatIP + métadonnées (.pkl)            │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  PHASE 2 — INTERROGATION (rag.py)                        │
│                                                          │
│  Question utilisateur                                    │
│        ↓                                                 │
│  Reformulation par LLM (Bonus C)                         │
│        ↓                                                 │
│  Embedding de la question (même modèle)                  │
│        ↓                                                 │
│  Recherche vectorielle FAISS (top-k = 4)                 │
│        ↓                                                 │
│  Filtre par score de confiance (Bonus B)                 │
│        ↓                                                 │
│  Construction du prompt (système + historique + ctx)     │
│        ↓                                                 │
│  Appel API Groq (llama-3.3-70b-versatile)                │
│        ↓                                                 │
│  Réponse avec sources et avertissement juridique         │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Installation

### 2.1. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
```

### 2.2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2.3. Configurer la clé API Groq

1. Crée un compte gratuit sur https://console.groq.com/
2. Génère une clé dans la section **API Keys**
3. Copie `.env.example` vers `.env` puis colle ta clé :

```bash
cp .env.example .env
# puis édite le fichier .env
# GROQ_API_KEY=gsk_...
```

> Le fichier `.env` est déjà dans `.gitignore`. Il ne doit jamais être commité.

---

## 3. Lancement

### 3.1. Indexation (à exécuter une seule fois)

```bash
python indexation.py
```

Sortie attendue :
```
[1/5] Chargement du corpus...
  34 articles chargés.
[2/5] Découpage en chunks...
  42 chunks générés.
[3/5] Chargement du modèle...
[4/5] Génération des embeddings...
[5/5] Création et sauvegarde de l'index FAISS...
INDEXATION TERMINÉE AVEC SUCCÈS
```

> Le premier lancement télécharge le modèle d'embedding (~470 Mo). Les lancements suivants utilisent le cache local.

### 3.2. Interrogation interactive (CLI)

```bash
python rag.py
```

Commandes disponibles dans la boucle :
- `quit` / `exit` / `q` : quitter
- `debug on` / `debug off` : afficher ou masquer les chunks récupérés et leur score
- `reset` : effacer l'historique de la conversation

### 3.3. Interface web Streamlit (recommandé)

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans ton navigateur à l'adresse `http://localhost:8501`.

L'interface web propose :
- Un **chat conversationnel** avec historique
- L'**affichage des sources** (articles cités avec score de confiance)
- Des **paramètres ajustables** dans la sidebar (k chunks, seuil de confiance, reformulation)
- Des **questions d'exemple** cliquables
- Un **bouton de reset** pour effacer la conversation

Pour le déploiement sur Streamlit Community Cloud, voir le fichier [`DEPLOIEMENT.md`](DEPLOIEMENT.md).

---

## 4. Choix techniques justifiés

### 4.1. Source des données

J'ai retenu l'**option corpus manuel** au format JSON plutôt que l'API Légifrance ou les fichiers XML LEGI :
- L'API Légifrance demande une inscription PISTE avec OAuth2 dont l'accès n'est pas immédiat.
- Les fichiers XML LEGI sont volumineux et nécessitent un parsing complexe pour un gain pédagogique faible.
- Un corpus JSON propre et structuré, couvrant **8 thèmes prioritaires** avec **34 articles réels du Code du travail**, est suffisant pour démontrer le RAG et garantit l'**exactitude des numéros d'articles cités**, point critique pour un assistant juridique.

Thèmes couverts : durée du travail et heures supplémentaires, congés payés, contrat de travail (CDI, CDD, période d'essai), licenciement (préavis, cause réelle, économique, indemnité), rupture conventionnelle, salaire minimum, harcèlement et discrimination, représentation du personnel (CSE), maternité et parentalité.

### 4.2. Stratégie de chunking

J'ai implémenté un **chunker récursif** (équivalent simplifié de `RecursiveCharacterTextSplitter`) :
- Hiérarchie de séparateurs : `\n\n` → `\n` → `. ` → ` `
- `taille_max = 500` caractères
- `overlap = 80` caractères

**Pourquoi cette taille ?** Les articles du Code du travail sont courts mais denses. Un article tient généralement sur un seul chunk, ce qui préserve la cohérence juridique. Les rares articles longs (par exemple `L1132-1` sur la discrimination ou `L1234-1` sur le préavis) sont découpés proprement avec overlap pour ne pas perdre de sens aux jointures.

**Pourquoi pas un chunking par section ?** Parce que chaque article est déjà une unité sémantique autonome — c'est l'unité juridique de référence. Indexer chaque article séparément, avec son numéro en métadonnée, garantit la traçabilité parfaite des citations.

### 4.3. Modèle d'embedding

**`paraphrase-multilingual-mpnet-base-v2`** :
- Le contenu est en **français**, un modèle multilingue est obligatoire.
- Dimension 768, bon équilibre qualité/performance.
- Modèle recommandé dans le cours pour le français.

Les vecteurs sont **normalisés en L2** (`normalize_embeddings=True`) afin que le **produit scalaire** soit équivalent à la **similarité cosinus**.

### 4.4. Index FAISS

**`IndexFlatIP`** (Inner Product) plutôt que `IndexFlatL2` :
- Avec des vecteurs L2-normalisés, IP équivaut directement à la similarité cosinus.
- Le score retourné est compris entre -1 et 1, plus il est proche de **1**, plus le chunk est pertinent.
- Pour un corpus de 42 chunks, un index plat est largement suffisant. Pour des millions de vecteurs, on utiliserait HNSW ou IVF.

### 4.5. Modèles Groq utilisés

- **`llama-3.3-70b-versatile`** pour la génération finale : qualité supérieure pour la synthèse juridique, accessible gratuitement avec une latence très faible.
- **`llama-3.1-8b-instant`** pour la reformulation (Bonus C) : modèle léger et rapide, suffisant pour cette tâche.

`temperature=0.1` pour maximiser la fidélité au contexte fourni et minimiser la créativité.

### 4.6. Prompt système

Le prompt système impose 7 règles strictes :
1. Répondre **uniquement** sur la base du contexte fourni
2. **Citer** systématiquement le numéro d'article
3. Dire « Je ne trouve pas cette information » si le contexte est insuffisant
4. **Ne jamais inventer** d'article ou de disposition légale
5. Avertissement juridique **obligatoire** en fin de réponse
6. Préciser quand la réponse dépend de la convention collective ou de la taille de l'entreprise
7. Rappeler que le droit évolue et qu'il faut vérifier la version en vigueur

### 4.7. Bonus implémentés

- **Bonus A — Historique de conversation** : les 6 derniers tours sont conservés et passés au LLM, ce qui permet des questions de suivi. La taille est plafonnée pour ne pas saturer la fenêtre de contexte.
- **Bonus B — Score de confiance** : si le meilleur chunk a un score inférieur à 0.35, le système refuse de répondre et invite à reformuler. Cela évite que le LLM s'appuie sur des chunks faiblement pertinents.
- **Bonus C — Reformulation de la question** : avant la recherche vectorielle, un appel rapide à `llama-3.1-8b-instant` reformule la question utilisateur en formulation juridique précise. Une question familière comme « j'ai été viré, qu'est-ce que je peux faire ? » est transformée en « Quels sont les droits du salarié en cas de licenciement ? », ce qui rapproche le vecteur de la question des vecteurs des articles indexés.

---

## 5. Réponses aux questions de réflexion

**Q1. Indexer chaque article séparément ou les regrouper par section ?**
J'ai choisi d'indexer **article par article**. Les articles sont des unités juridiques autonomes ; les regrouper par section noierait les références précises et empêcherait la citation exacte. Pour gagner en pertinence sémantique, le texte enrichi `Article XXX - Titre + Section + Texte` est embedded dans son ensemble.

**Q2. Comment intégrer le numéro d'article ?**
Le numéro est stocké en métadonnée (`metadata.article`) puis ré-injecté dans le prompt au format `[Source N | Article L3141-3 - Titre | Section : ...]`. Le prompt système exige explicitement la citation au format « selon l'article L3141-3 ».

**Q3. Comment indiquer que l'information peut être obsolète ?**
La règle 7 du prompt système oblige le LLM à rappeler systématiquement que le droit du travail évolue régulièrement et qu'il faut vérifier la version en vigueur sur Légifrance.

**Q4. Et si la réponse dépend de la taille de l'entreprise ou du secteur d'activité ?**
La règle 6 du prompt système impose au LLM de préciser explicitement quand la réponse dépend de la convention collective, du secteur ou de la taille de l'entreprise (par exemple : seuils CSE à 11 et 50 salariés).

**Q5. Comment distinguer une question factuelle d'une question d'interprétation juridique ?**
Le score de confiance (Bonus B) sert de premier filtre : si aucun chunk n'a un score supérieur à 0.35, c'est probablement une question hors corpus et le système oriente vers un juriste. Pour les questions factuelles couvertes, l'avertissement juridique en fin de réponse rappelle dans tous les cas la nécessité de consulter un professionnel.

---

## 6. Tests recommandés

```
Question : Combien de jours de congés payés par an ?
Question : Quelle est la durée légale du préavis pour un CDI ?
Question : Comment fonctionne la rupture conventionnelle ?
Question : Combien d'heures supplémentaires peut-on faire par semaine ?
Question : Quels sont mes droits en cas de licenciement économique ?
Question : Qu'est-ce que le harcèlement moral ?
Question : À partir de combien de salariés faut-il un CSE ?
Question : Combien de jours de congé paternité ?
Question : Quel est le délai de rétractation après une rupture conventionnelle ?

# Test du score de confiance (question hors corpus) :
Question : Quel est le prix d'un café à Paris ?
→ doit retourner « Je n'ai pas trouvé d'information directement pertinente »
```

---

## 7. Structure du projet

```
tp_rag/
├── corpus/
│   └── code_travail.json          # 34 articles du Code du travail
├── index_faiss/
│   ├── code_travail.index         # index FAISS persistant (généré)
│   └── code_travail_meta.pkl      # métadonnées des chunks (généré)
├── indexation.py                  # script d'indexation (Phase 1)
├── rag.py                         # interface CLI (Phase 2)
├── app.py                         # interface web Streamlit
├── requirements.txt               # dépendances Python
├── .env.example                   # modèle pour la clé Groq
├── .env                           # ta clé Groq (ignoré par Git)
├── .gitignore                     # exclusions Git
├── compte_rendu.pdf               # compte-rendu détaillé
├── DEPLOIEMENT.md                 # guide de déploiement Streamlit
└── README.md                      # ce fichier
```

---

## 8. Pour aller plus loin (production)

Pour passer en production, j'ajouterais :
- Un **reranker** (Cohere Rerank ou un cross-encoder local) pour passer de top-20 à top-4 avec une qualité supérieure.
- Une **base vectorielle persistante** (Qdrant ou ChromaDB) pour gérer un corpus complet du Code du travail (environ 10 000 articles).
- Une **API REST FastAPI** à la place de la CLI.
- Un **monitoring** des requêtes et des scores pour identifier les questions hors couverture.
- Un **système d'évaluation automatique** avec Ragas pour mesurer faithfulness, answer relevance et context precision.
