# Text Classification and Retrieval System

A comprehensive text classification and information retrieval system that combines machine learning classification with full-text search capabilities.

## System Overview

This system integrates two core components:
1. **Text Classification**: Automatically categorizes text using TF-IDF + Naive Bayes
2. **Information Retrieval**: Searches and retrieves relevant documents using BM25 algorithm

### Architecture

```
User Query → Text Classifier → Predicted Category →
                                      ↓
              Filtered Search in Category → Ranked Results
```

## Features

- **Smart Search**: Automatically classify queries and search within predicted categories
- **Direct Search**: Full-text search with optional category filtering
- **Category Browsing**: Browse documents by specific categories
- **Interactive UI**: User-friendly Streamlit web interface
- **Evaluation Tools**: Comprehensive metrics for classification and retrieval

## Tech Stack

| Component | Technology |
|-----------|------------|
| Classification | TF-IDF + Naive Bayes (scikit-learn) |
| Retrieval | Whoosh (BM25 algorithm) |
| Web Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Dataset | 20 Newsgroups (~18,000 documents, 20 categories) |

## Project Structure

```
Text-Classification-Retrieval-System/
├── app/
│   ├── main.py              # Streamlit web application
│   └── utils.py             # Utility functions
├── classification/
│   ├── models/              # Trained model files
│   │   ├── classifier.pkl   # Trained classifier
│   │   ├── vectorizer.pkl   # TF-IDF vectorizer
│   │   └── category_mapping.pkl
│   ├── classifier_model.py  # TextClassifier class
│   └── train_classifier.py  # Training script
├── data/
│   ├── processed/           # Cached dataset
│   └── load_data.py         # DataLoader class
├── retrieval/
│   ├── indexdir/            # Whoosh search index
│   ├── index_builder.py     # IndexBuilder class
│   └── searcher.py          # DocumentSearcher class
├── evaluation/
│   ├── experiments.py       # System evaluation
│   └── metrics.py           # Evaluation metrics
├── tests/
│   ├── test_classification.py
│   └── test_retrieval.py
├── config.py                # System configuration
├── requirements.txt         # Dependencies
├── setup.py                 # Setup and initialization script
└── run.py                   # Main entry point
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize the System

Run the setup script to download data, train the classifier, and build the search index:

```bash
python setup.py
```

This will:
- Download the 20 Newsgroups dataset
- Train the text classifier
- Build the search index
- Verify the installation

### 3. Run the Application

```bash
streamlit run app/main.py
```

Or use the run script:

```bash
python run.py
```

The application will be available at `http://localhost:8501`

## Usage Guide

### Search Modes

#### 1. Smart Search (Recommended)
- Enter your query text
- System automatically classifies the query
- Searches within the predicted category
- Returns ranked results

#### 2. Direct Search
- Enter your query text
- Optionally select a category filter
- Searches across all or filtered documents

#### 3. Category Browsing
- Select a category from the sidebar
- Browse all documents in that category

### Example Queries

| Category | Example Queries |
|----------|-----------------|
| comp.graphics | "3D rendering techniques", "image processing" |
| rec.sport.hockey | "NHL playoffs", "hockey equipment" |
| sci.space | "NASA Mars mission", "black hole" |
| talk.politics | "Middle East peace", "government policy" |

## Configuration

Edit `config.py` to customize system settings:

```python
# Dataset configuration
DATASET_CONFIG = {
    'name': '20newsgroups',
    'categories': [...],  # List of categories to use
    'subset': 'all',
    'remove': ('headers', 'footers', 'quotes')
}

# Classifier configuration
CLASSIFIER_CONFIG = {
    'vectorizer': 'tfidf',
    'classifier': 'naive_bayes',  # Options: naive_bayes, svm, random_forest
    'test_size': 0.2,
    'random_state': 42
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    'index_dir': 'retrieval/indexdir',
    'max_results': 20,
    'score_type': 'BM25F'
}
```

## Manual Setup (Step by Step)

If you prefer to run each step manually:

### Step 1: Load and Process Data

```bash
cd data
python load_data.py
```

### Step 2: Train Classifier

```bash
cd classification
python train_classifier.py
```

### Step 3: Build Search Index

```bash
cd retrieval
python index_builder.py
```

### Step 4: Run Application

```bash
streamlit run app/main.py
```

## API Usage

### Classification

```python
from classification.classifier_model import TextClassifier

# Load trained model
classifier = TextClassifier()
classifier.load_model('classification/models')

# Predict category
text = "computer graphics and image processing"
predictions, probabilities = classifier.predict(text)
```

### Retrieval

```python
from retrieval.searcher import DocumentSearcher

# Initialize searcher
searcher = DocumentSearcher()
searcher.open_index()

# Search documents
results = searcher.search("computer graphics", category_filter=None, max_results=10)

# Search with classification
results = searcher.search_with_classification("hockey game", classifier)
```

## Evaluation

Run the evaluation script to assess system performance:

```bash
cd evaluation
python experiments.py
```

### Metrics

**Classification Metrics:**
- Accuracy
- Precision, Recall, F1-score (per class and weighted average)
- Confusion Matrix

**Retrieval Metrics:**
- Precision@K
- Recall@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

## Categories

The system supports 20 newsgroup categories:

| Computer | Recreation | Science | Talk |
|----------|------------|---------|------|
| comp.graphics | rec.autos | sci.crypt | talk.politics.guns |
| comp.os.ms-windows.misc | rec.motorcycles | sci.electronics | talk.politics.mideast |
| comp.sys.ibm.pc.hardware | rec.sport.baseball | sci.med | talk.politics.misc |
| comp.sys.mac.hardware | rec.sport.hockey | sci.space | talk.religion.misc |
| comp.windows.x | | | soc.religion.christian |
| | | | alt.atheism |
| | | | misc.forsale |

## Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   python setup.py  # Re-run setup
   ```

2. **Index not found**
   ```bash
   cd retrieval
   python index_builder.py
   ```

3. **Import errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

4. **Memory issues**
   - Reduce `max_features` in classifier configuration
   - Use a subset of categories

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. **New classifier type**: Add to `classifier_model.py`
2. **New retrieval method**: Extend `searcher.py`
3. **UI modifications**: Edit `app/main.py`

## Performance

Typical performance on 20 Newsgroups dataset:

| Metric | Value |
|--------|-------|
| Classification Accuracy | ~85% |
| Retrieval P@10 | ~70% |
| Average Response Time | <500ms |

## License

MIT License

## Acknowledgments

- 20 Newsgroups dataset: [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
- Whoosh search engine: [Whoosh](https://whoosh.readthedocs.io/)
- Streamlit: [Streamlit](https://streamlit.io/)
