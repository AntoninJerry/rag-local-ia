# AGENTS.md

Instructions de travail pour les agents de codage intervenant sur ce depot.

## Objectif Du Projet

Construire un assistant RAG local en Python, capable d'indexer des documents locaux et de repondre aux questions en s'appuyant uniquement sur les sources indexees.

Le projet doit rester simple pour le MVP, mais structure pour evoluer vers une API plus complete et une future interface desktop.

## Regles Generales

- Privilegier une implementation MVP claire plutot qu'une architecture complexe.
- Ne pas introduire de dependance lourde sans justification concrete.
- Garder tout le fonctionnement local par defaut.
- Ne jamais hardcoder de chemins sensibles, absolus ou propres a une machine.
- Lire les chemins depuis la configuration, les arguments CLI ou les requetes API.
- Ajouter des TODO explicites quand une amelioration est volontairement repoussee.
- Eviter le code jetable : meme un stub doit avoir une place claire dans l'architecture.

## Style Python

- Ecrire du Python lisible, direct et idiomatique.
- Utiliser des noms explicites pour les modules, fonctions, classes et variables.
- Ajouter du typage quand il clarifie les contrats ou protege les composants critiques.
- Eviter les fonctions trop longues : extraire des services ou helpers quand la responsabilite devient floue.
- Ajouter des commentaires uniquement quand ils expliquent une decision ou une logique non evidente.
- Preferer les modeles Pydantic pour les donnees qui traversent l'API ou les frontieres de modules.

## Separation Des Responsabilites

Conserver une separation stricte entre :

- `app/ingestion` : detection de fichiers et extraction documentaire.
- `app/chunking` : decoupage du texte en chunks.
- `app/embeddings` : generation d'embeddings.
- `app/vector_store` : stockage et recherche vectorielle locale.
- `app/retrieval` : orchestration de la recherche de contexte.
- `app/llm` : generation de reponse a partir du contexte.
- `app/api` : routes HTTP, schemas et adaptation entree/sortie.
- `app/cli.py` : interface CLI et adaptation entree/sortie.
- `app/core` : configuration et services partages.

Une route API ou une commande CLI ne doit pas contenir de logique metier RAG. Elle doit valider l'entree, appeler un service reutilisable, puis retourner une reponse.

## Services

- Mettre la logique reutilisable dans des services ou classes dediees.
- Concevoir les services pour etre appelables depuis l'API, la CLI et plus tard une interface desktop.
- Eviter les dependances directes a FastAPI dans la logique metier.
- Eviter les dependances directes a Typer dans la logique metier.
- Injecter les chemins, providers et parametres plutot que les lire partout depuis l'environnement.

## API FastAPI

- Garder les routes fines et previsibles.
- Utiliser des schemas Pydantic explicites pour les requetes et reponses.
- Ne pas mettre d'extraction PDF, de chunking, d'embeddings ou de recherche vectorielle directement dans `app/api/main.py`.
- Retourner des erreurs comprehensibles pour les cas utilisateur courants.
- Ne pas exposer d'information sensible dans les messages d'erreur.

## CLI

- Garder les commandes CLI simples et orientees utilisateur.
- La CLI doit appeler les memes services que l'API.
- Les sorties CLI doivent etre lisibles et utiles, sans bruit inutile.
- Ne pas dupliquer la logique metier dans les commandes.

## Logs

- Ajouter des logs simples mais utiles autour des operations importantes :
  - debut et fin d'indexation ;
  - nombre de documents detectes ;
  - nombre de chunks generes ;
  - erreurs d'extraction ;
  - requetes de retrieval ;
  - absence de contexte suffisant.
- Ne pas logger le contenu complet des documents.
- Ne pas logger de chemins sensibles si une version relative ou anonymisee suffit.

## Tests

- Ajouter des tests minimaux pour chaque composant critique.
- Tester au minimum :
  - scan de documents ;
  - loaders TXT, MD et PDF quand ils existent ;
  - chunking et overlap ;
  - providers d'embeddings via fakes quand possible ;
  - vector store via implementation locale ou fake ;
  - retrieval ;
  - endpoints API principaux ;
  - commandes CLI principales.
- Preferer les tests rapides et deterministes.
- Eviter les tests qui dependent d'un modele ML telecharge ou d'un service reseau.

## Donnees Et Chemins

- Garder les documents locaux dans `data/documents` par defaut.
- Garder les artefacts intermediaires dans `data/processed`.
- Garder les index vectoriels dans `data/vector_store`.
- Ne pas versionner les documents utilisateur, index vectoriels, caches ou fichiers `.env`.
- Versionner uniquement les `.gitkeep` necessaires pour conserver l'arborescence.

## Dependances

- Les dependances de base doivent permettre de lancer le socle du projet sans compilateur natif.
- Garder les dependances lourdes en extras optionnels quand elles ne sont pas indispensables au scaffold.
- Avant d'ajouter une dependance, verifier si une implementation simple suffit pour le MVP.

## Compatibilite Future Desktop

- La logique RAG doit rester independante de FastAPI et de Typer.
- Les services doivent pouvoir etre appeles depuis une UI desktop ou web sans reecriture majeure.
- Eviter les variables globales mutables pour l'etat applicatif critique.
- Preferer des contrats clairs entre ingestion, retrieval et generation.

## Avant De Terminer Une Tache

- Verifier l'etat Git pour ne pas inclure d'artefacts generes.
- Lancer les tests pertinents si les dependances sont disponibles.
- Signaler clairement les tests non executes et pourquoi.
- Documenter les fichiers crees ou modifies.
