# Training Recommendation System (RAG)

Ce projet implémente un système de Retrieval-Augmented Generation (RAG) pour aider Martin (DRH) à fournir des recommandations de formation personnalisées à ses équipes. Le système enrichit les évaluations sommaires des employés en récupérant des informations complémentaires depuis un corpus documentaire de formations disponibles.

## System Architecture

The system consists of 4 main modules:

1. **Preprocessor** (preprocessor.py): Processes employee evaluations, extracts keywords, and assigns priorities.
2. **Corpus Indexer** (corpus_indexer.py): Creates a vector index of available training programs.
3. **Retriever** (retriever.py): Searches for relevant training programs for each employee.
4. **Generator** (generator.py): Produces personalized recommendations.

A **Pipeline** (pipeline.py) orchestrates the entire process.

## Prerequisites

- Python 3.12+
- An OpenAI API key

## Installation

1. Clone this repository:


2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add a .env with you OPENAI_API_KEY

## Data Preparation

Place your data files in the data directory:

- employe.json: List of employee evaluations in JSON format
- formation.json: Documentary corpus of available training programs

## Usage

### Run individual modules

```bash
python pipeline.py
```

You can also run each module separately:

```bash
python src/preprocessor.py
python src/corpus_indexer.py
python src/retriever.py
python src/generator.py
```

```

## Tests

You can test each module separately using:

```bash
python -m unittest tests.[test_file_name]
```
