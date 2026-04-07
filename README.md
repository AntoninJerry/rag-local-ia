# Assistant de Recherche Intelligent - RAG Local

Assistant de recherche local base sur une architecture RAG (Retrieval-Augmented Generation). Le projet permet d'indexer des documents locaux, de retrouver les extraits les plus pertinents pour une question, puis de produire une reponse structuree avec les sources utilisees.

Le code est pense comme une base MVP propre : modulaire, testable, locale par defaut et extensible vers une interface desktop ou web.

## Objectif

Construire une application locale capable de scanner un dossier de documents, extraire leur texte, decouper le contenu, generer des embeddings, stocker les vecteurs localement et repondre a des questions uniquement a partir des sources indexees.

## Fonctionnalites MVP

- Ingestion documentaire locale : PDF, TXT, MD.
- Extraction PDF page par page quand possible.
- Chunking configurable avec overlap et metadonnees de provenance.
- Provider d'embeddings abstrait avec implementation `sentence-transformers`.
- Vector store local persistant base sur JSON + `numpy`.
- Retriever testable et independant du LLM.
- Prompt builder RAG avec consignes anti-hallucination.
- Client LLM abstrait avec provider MVP `local_stub`.
- Pipeline RAG principal : retrieval -> prompt -> LLM -> reponse structuree.
- Service d'indexation reutilisable par CLI et API.
- Service de question/reponse reutilisable par CLI et API.
- CLI simple avec `index`, `ask` et `info`.
- API FastAPI avec `GET /health`, `POST /index`, `POST /ask`.
- Tests unitaires et tests d'integration MVP.

## Architecture Du Projet

```text
app/
  api/              Routes FastAPI, schemas et dependances
  chunking/         Decoupage du texte en chunks
  core/             Configuration et logging
  embeddings/       Abstraction et provider sentence-transformers
  indexing/         Orchestration de la phase d'indexation
  ingestion/        Scan de fichiers et loaders PDF/TXT/MD
  llm/              Prompt builder et clients LLM
  querying/         Service question/reponse reutilisable
  rag/              Pipeline RAG principal
  retrieval/        Retrieval via embeddings + vector store
  vector_store/     Stockage vectoriel local persistant
  cli.py            Interface CLI
  models.py         Modeles Pydantic partages

data/
  documents/        Documents sources locaux
  processed/        Artefacts intermediaires futurs
  vector_store/     Index vectoriel local

tests/              Tests unitaires et integration MVP
```

Principe important : les routes API et les commandes CLI ne contiennent pas de logique metier RAG. Elles deleguent aux services reutilisables.

## Stack Technique

- Python 3.11+
- FastAPI
- Typer
- Pydantic v2
- pydantic-settings
- pypdf
- numpy
- sentence-transformers en extra optionnel
- pytest

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Pour utiliser l'indexation reelle avec embeddings locaux :

```powershell
python -m pip install -e ".[embeddings-local]"
```

Note Windows : ChromaDB est garde en extra optionnel car son backend natif peut demander Microsoft C++ Build Tools. Le MVP utilise actuellement un vector store local JSON + `numpy`, qui evite cette contrainte.

```powershell
python -m pip install -e ".[vector-chroma]"
```

## Configuration

Copier le fichier d'exemple si une configuration locale est necessaire :

```powershell
Copy-Item .env.example .env
```

Variables principales :

```env
APP_NAME="RAG Local IA"
APP_ENV=local
DATA_DIR=data
DOCUMENTS_DIR=data/documents
VECTOR_STORE_DIR=data/vector_store
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=local_stub
LLM_MODEL_NAME=local-stub
LLM_TIMEOUT_SECONDS=30
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
LOG_LEVEL=INFO
```

## Lancement En CLI

Afficher la configuration active :

```powershell
rag-local info
```

Indexer les documents :

```powershell
rag-local index data/documents
```

Poser une question :

```powershell
rag-local ask "Que disent les documents sur le projet ?" --top-k 5
```

La commande `index` affiche un resume : fichiers lus, fichiers ignores ou en erreur, documents extraits, chunks crees et duree.

La commande `ask` affiche la reponse, puis les sources utilisees avec fichier, page si disponible, `chunk_id`, score et extrait.

## Lancement API

Demarrer l'API :

```powershell
uvicorn app.api.main:app --reload
```

Verifier l'etat :

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Indexer un dossier :

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/index" `
  -ContentType "application/json" `
  -Body '{"source_path":"data/documents"}'
```

Poser une question :

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/ask" `
  -ContentType "application/json" `
  -Body '{"question":"Que disent les documents sur le projet ?","top_k":5}'
```

## Tests

```powershell
python -m pytest --basetemp .pytest_tmp
```

La base de tests vise les points critiques du MVP :

- loaders documentaires TXT/MD/PDF ;
- text splitter avec metadonnees ;
- embeddings via faux modele ;
- vector store local ;
- retriever avec composants factices ;
- prompt builder RAG ;
- pipeline RAG avec composants mockes ;
- services d'indexation et de question/reponse ;
- API et CLI avec services remplaces par des fakes.

`--basetemp .pytest_tmp` evite les problemes de permissions observes sur certains dossiers temporaires Windows.

## Limites Connues

- Le provider LLM par defaut est `local_stub` : il ne contacte pas encore un vrai modele.
- L'index vectoriel MVP est un store JSON + `numpy`, adapte a un usage local mono-utilisateur et a des volumes modestes.
- `sentence-transformers` est optionnel et doit etre installe via l'extra `embeddings-local` pour une indexation reelle.
- Le support documentaire initial est limite a PDF, TXT et MD.
- Pas encore d'interface graphique desktop/web.
- Pas encore de streaming de reponse LLM.
- Pas encore de gestion multi-utilisateur ni d'authentification.

## Roadmap V2

- Brancher un vrai backend LLM local, par exemple Ollama.
- Ajouter un backend distant optionnel, par exemple OpenRouter.
- Ajouter un vector store avance optionnel : ChromaDB ou FAISS.
- Ajouter le support DOCX.
- Ajouter une UI web ou desktop.
- Ajouter une page de gestion des documents indexes.
- Ajouter la suppression/reindexation selective de documents.
- Ajouter des citations plus riches : page, extrait, score, chemin relatif.
- Ajouter une configuration de profils de modeles.
- Preparer le packaging desktop.

## Etat Du Projet

Le projet est au stade MVP technique : les briques principales sont separees et testees, la CLI/API sont branchables, et l'architecture est prete pour remplacer les providers embeddings, vector store et LLM sans modifier le pipeline principal.
