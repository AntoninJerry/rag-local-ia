# Assistant de Recherche Intelligent - RAG Local

Application locale pour indexer des documents et repondre a des questions en s'appuyant uniquement sur les sources indexees.

## Objectif MVP

- Scanner un dossier de documents locaux.
- Extraire le texte de fichiers PDF, TXT et MD.
- Decouper le contenu en chunks.
- Generer des embeddings localement.
- Stocker les embeddings dans une base vectorielle locale.
- Recuperer les chunks pertinents pour une question.
- Produire une reponse avec les sources utilisees.
- Exposer une CLI simple et une API FastAPI locale.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copier ensuite `.env.example` vers `.env` si une configuration locale est necessaire.

Note Windows : ChromaDB est garde en dependance optionnelle, car son backend `chroma-hnswlib` peut demander Microsoft C++ Build Tools. Le socle MVP s'installe sans compilateur C++.

Le backend `sentence-transformers` est aussi optionnel pour garder l'installation initiale legere. Il sera utile quand le provider d'embeddings local sera branche.

```powershell
python -m pip install -e ".[embeddings-local]"
```

```powershell
python -m pip install -e ".[vector-chroma]"
```

## Utilisation CLI

```powershell
rag-local --help
rag-local index data/documents
rag-local ask "Quelle est l'information principale ?"
```

## Utilisation API

```powershell
uvicorn app.api.main:app --reload
```

Endpoints initiaux :

- `GET /health`
- `POST /index`
- `POST /ask`

## Etat actuel

Ce depot contient la structure initiale du projet. Les modules sont separes pour faciliter l'evolution du MVP sans melanger ingestion, chunking, embeddings, stockage vectoriel, retrieval, generation LLM, API et CLI.

## TODO

- Implementer les loaders PDF/TXT/MD.
- Implementer le chunking avec overlap.
- Brancher un provider d'embeddings local.
- TODO: Activer `sentence-transformers` via l'extra `embeddings-local` quand le provider sera implemente.
- Ajouter un vector store local persistant sans dependance native obligatoire.
- TODO: Brancher ChromaDB comme option avancee si l'environnement dispose des dependances natives.
- Implementer le retrieval et la generation de reponse avec citations.
- Ajouter des tests unitaires et d'integration.
