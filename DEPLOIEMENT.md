# Guide de déploiement — Assistant Code du Travail

Ce document explique comment lancer l'application Streamlit en local et comment la déployer sur **Streamlit Community Cloud** (gratuit).

---

## 1. Lancement en local

### Prérequis

- Python 3.10 ou supérieur
- Une clé API Groq gratuite ([console.groq.com](https://console.groq.com/))

### Étapes

```bash
# 1. Création de l'environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Configuration de la clé API
cp .env.example .env
# puis édite .env et colle ta clé : GROQ_API_KEY=gsk_...

# 4. Génération de l'index FAISS (une seule fois)
python indexation.py

# 5. Lancement de l'interface web
streamlit run app.py
```

L'application s'ouvre automatiquement dans ton navigateur à l'adresse `http://localhost:8501`.

---

## 2. Fonctionnalités de l'interface

### Zone principale
- **Chat conversationnel** : pose tes questions, l'historique est conservé pour les questions de suivi.
- **Affichage des sources** : chaque réponse est accompagnée des articles cités avec leur numéro, leur titre, leur section et leur score de confiance.
- **Reformulation visible** : si activée, la question reformulée est affichée sous la question originale.

### Sidebar (panneau latéral)
- **Slider k** : nombre de chunks récupérés (entre 2 et 10)
- **Slider seuil de confiance** : score minimum pour qu'une réponse soit générée
- **Switch reformulation (Bonus C)** : activer/désactiver la reformulation automatique
- **Switch sources** : afficher/masquer les chunks récupérés
- **Statistiques** : nombre de chunks indexés, sections couvertes, modèle utilisé
- **Boutons d'exemples** : 6 questions pré-remplies pour tester rapidement
- **Bouton reset** : effacer la conversation

---

## 3. Déploiement sur Streamlit Community Cloud (gratuit)

Streamlit Community Cloud permet d'héberger gratuitement une application Streamlit avec une URL publique du type `https://ton-app.streamlit.app`.

### Étape 1 — Pousser le code sur GitHub

```bash
git init
git add .
git commit -m "Initial commit - RAG Code du Travail"
git branch -M main
git remote add origin https://github.com/TON_USER/tp-rag-code-travail.git
git push -u origin main
```

> ⚠️ **Important** : vérifie que `.env` est bien dans `.gitignore` avant le premier push. Le fichier `.env.example` est commité, mais jamais `.env`.

### Étape 2 — Créer un compte Streamlit Cloud

1. Va sur [share.streamlit.io](https://share.streamlit.io/)
2. Connecte-toi avec ton compte GitHub
3. Autorise Streamlit à lire ton dépôt

### Étape 3 — Déployer

1. Clique sur **New app**
2. Sélectionne le dépôt GitHub
3. Branche : `main`
4. Fichier principal : `app.py`
5. Avant de déployer, clique sur **Advanced settings**

### Étape 4 — Ajouter la clé API Groq

Dans la section **Secrets** des paramètres avancés, colle :

```toml
GROQ_API_KEY = "gsk_ta_cle_ici"
```

> Streamlit Cloud lit automatiquement les secrets via `os.environ.get("GROQ_API_KEY")`. Le code de `app.py` est compatible sans aucune modification.

### Étape 5 — Lancer le déploiement

Clique sur **Deploy**. Le premier déploiement prend 5 à 10 minutes (téléchargement du modèle d'embedding + construction de l'environnement). Les démarrages suivants sont quasi instantanés.

---

## 4. Particularités du déploiement cloud

### Génération de l'index

L'index FAISS (`index_faiss/code_travail.index` et `code_travail_meta.pkl`) doit être présent dans le dépôt. **Deux solutions** :

**Solution A — Commit l'index dans le dépôt (recommandé pour ce TP)** :
```bash
# Modifier .gitignore pour autoriser les fichiers d'index
# Retirer ces lignes :
# index_faiss/*.index
# index_faiss/*.pkl

git add index_faiss/
git commit -m "Add FAISS index"
git push
```

**Solution B — Générer l'index au démarrage** :
Ajoute en haut de `app.py` :
```python
if not INDEX_PATH.exists():
    import subprocess
    subprocess.run(["python", "indexation.py"])
```
> Cette solution rallonge le premier démarrage de plusieurs minutes.

### Limites de Streamlit Cloud (tier gratuit)

- **1 Go de RAM** : suffisant pour ce projet (modèle d'embedding ~470 Mo + index ~150 Ko + librairies).
- **Mise en veille après inactivité** : l'app redémarre en quelques secondes au prochain accès.
- **Domaines autorisés** : tout est autorisé pour les API publiques (Groq fonctionne sans configuration spéciale).

---

## 5. Alternatives de déploiement

| Plateforme | Gratuit | Avantages | Inconvénients |
|---|---|---|---|
| **Streamlit Cloud** | ✅ | Plus simple, intégration GitHub native | Limité en RAM/CPU |
| **Hugging Face Spaces** | ✅ | Excellent pour les apps ML | Configuration légèrement plus complexe |
| **Render** | ✅ (limité) | Plus de RAM disponible | Mise en veille agressive |
| **Railway** | ⚠️ ($5 crédits) | Très flexible | Devient payant après les crédits |
| **VPS (Contabo, OVH)** | ❌ | Contrôle total, performance | Configuration manuelle, maintenance |

Pour ce TP, **Streamlit Community Cloud** est la solution la plus rapide et la plus pédagogique.

---

## 6. Captures d'écran à inclure dans le rendu

Pour valoriser ton TP, tu peux ajouter quelques captures d'écran de l'interface :

1. La page d'accueil avec la sidebar visible
2. Une question avec sa réponse et les sources affichées
3. Le filtre par score de confiance qui refuse une question hors corpus
4. La reformulation automatique d'une question familière

Place-les dans un dossier `screenshots/` à la racine du projet.

---

## 7. Dépannage

### « Index introuvable »
Lance `python indexation.py` avant `streamlit run app.py`.

### « GROQ_API_KEY introuvable »
Vérifie que le fichier `.env` existe et contient bien la clé. En déploiement cloud, vérifie la section Secrets.

### Premier lancement très lent
Le modèle d'embedding (~470 Mo) est téléchargé depuis Hugging Face. Ce téléchargement n'a lieu qu'une seule fois.

### Erreur « Model not found » sur Groq
Les modèles Groq évoluent. Vérifie sur [console.groq.com/docs/models](https://console.groq.com/docs/models) que `llama-3.3-70b-versatile` est toujours actif.
