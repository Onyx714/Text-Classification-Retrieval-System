#!/usr/bin/env python
# setup.py - System initialization and setup script

"""
Text Classification and Retrieval System - Setup Script

This script initializes the entire system:
1. Downloads and processes the 20 Newsgroups dataset
2. Trains the text classifier
3. Builds the search index
4. Verifies the installation

Usage:
    python setup.py [--force]

Options:
    --force    Force re-initialization even if files exist
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step_num, total, description):
    """Print step progress"""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 40)


def check_dependencies():
    """Check if all required packages are installed"""
    print_step(1, 5, "Checking dependencies...")

    required_packages = [
        'streamlit',
        'sklearn',
        'pandas',
        'numpy',
        'whoosh',
        'joblib',
        'tqdm',
        'matplotlib',
        'seaborn'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("\n[OK] All dependencies installed")
    return True


def load_dataset(force=False):
    """Load and cache the dataset"""
    print_step(2, 5, "Loading dataset...")

    cache_file = os.path.join(PROJECT_ROOT, 'data', 'processed', 'dataset.pkl')

    if os.path.exists(cache_file) and not force:
        print(f"  Dataset cache found: {cache_file}")
        print("  Use --force to re-download")

        # Still load to get stats
        from data.load_data import DataLoader
        loader = DataLoader(use_cached=True)
        df = loader.load_20newsgroups()
        print(f"  [OK] Loaded {len(df)} documents")
        return df

    from data.load_data import DataLoader

    loader = DataLoader(use_cached=False)
    df = loader.load_20newsgroups()

    print(f"  [OK] Downloaded and processed {len(df)} documents")
    print(f"  [OK] Categories: {len(df['target_name'].unique())}")

    return df


def train_classifier(df, force=False):
    """Train the text classifier"""
    print_step(3, 5, "Training classifier...")

    model_dir = os.path.join(PROJECT_ROOT, 'classification', 'models')
    model_file = os.path.join(model_dir, 'classifier.pkl')

    if os.path.exists(model_file) and not force:
        print(f"  Classifier model found: {model_file}")
        print("  Use --force to re-train")

        # Load and test
        from classification.classifier_model import TextClassifier
        classifier = TextClassifier()
        classifier.load_model(model_dir)
        print("  [OK] Model loaded successfully")
        return classifier

    from classification.classifier_model import TextClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    # Split dataset
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['target']
    )

    print(f"  Training set: {len(train_df)} documents")
    print(f"  Test set: {len(test_df)} documents")

    # Create and train classifier
    classifier = TextClassifier(model_type='naive_bayes')
    classifier.create_vectorizer(max_features=5000)
    classifier.create_classifier()

    classifier.train(
        train_df['text_clean'].tolist(),
        train_df['target'].tolist()
    )

    # Evaluate
    accuracy, report = classifier.evaluate(
        test_df['text_clean'].tolist(),
        test_df['target'].tolist(),
        save_report=True
    )

    # Save model
    classifier.save_model(model_dir)

    # Save category mapping
    category_mapping = {
        idx: name for idx, name in enumerate(df['target_name'].unique())
    }
    joblib.dump(category_mapping, os.path.join(model_dir, 'category_mapping.pkl'))

    print(f"  [OK] Classifier trained with accuracy: {accuracy:.2%}")

    return classifier


def build_index(df, force=False):
    """Build the search index"""
    print_step(4, 5, "Building search index...")

    index_dir = os.path.join(PROJECT_ROOT, 'retrieval', 'indexdir')

    if os.path.exists(index_dir) and os.listdir(index_dir) and not force:
        print(f"  Index found: {index_dir}")
        print("  Use --force to rebuild")

        # Get stats
        from retrieval.index_builder import IndexBuilder
        builder = IndexBuilder(index_dir)
        try:
            stats = builder.get_index_stats()
            print(f"  [OK] Index contains {stats['total_documents']} documents")
        except:
            print("  [OK] Index exists")
        return

    from retrieval.index_builder import IndexBuilder

    # Clear existing index if force
    if force and os.path.exists(index_dir):
        import shutil
        shutil.rmtree(index_dir)
        print("  Cleared existing index")

    builder = IndexBuilder(index_dir)
    builder.define_schema()
    builder.build_index(df)

    stats = builder.get_index_stats()
    print(f"  [OK] Indexed {stats['total_documents']} documents")
    print(f"  [OK] Categories: {len(stats['categories'])}")


def verify_installation():
    """Verify the installation by running a test query"""
    print_step(5, 5, "Verifying installation...")

    try:
        # Load classifier
        from classification.classifier_model import TextClassifier
        classifier = TextClassifier()
        model_dir = os.path.join(PROJECT_ROOT, 'classification', 'models')
        classifier.load_model(model_dir)

        # Load searcher
        from retrieval.searcher import DocumentSearcher
        searcher = DocumentSearcher()
        index_dir = os.path.join(PROJECT_ROOT, 'retrieval', 'indexdir')
        searcher.index_dir = index_dir
        searcher.open_index()

        # Test query
        test_query = "stock market trading"
        predictions, probs = classifier.predict(test_query)
        results = searcher.search(test_query, max_results=5)

        print(f"  Test query: '{test_query}'")
        print(f"  [OK] Classification working: predicted category {predictions[0]}")
        print(f"  [OK] Retrieval working: found {len(results)} documents")

        return True

    except Exception as e:
        print(f"  [FAIL] Verification failed: {str(e)}")
        return False


def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete!")

    print("Next steps:")
    print("-" * 40)
    print("\n1. Run the web application:")
    print("   streamlit run app/main.py")
    print("\n2. Or use the run script:")
    print("   python run.py")
    print("\n3. Open in browser:")
    print("   http://localhost:8501")
    print("\n4. Run evaluation (optional):")
    print("   python evaluation/experiments.py")
    print("\n" + "=" * 60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description='Initialize Text Classification and Retrieval System'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-initialization even if files exist'
    )
    args = parser.parse_args()

    print_header("Text Classification & Retrieval System Setup")

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n[FAIL] Setup failed: missing dependencies")
        sys.exit(1)

    # Step 2: Load dataset
    df = load_dataset(force=args.force)

    # Step 3: Train classifier
    train_classifier(df, force=args.force)

    # Step 4: Build index
    build_index(df, force=args.force)

    # Step 5: Verify installation
    if verify_installation():
        print_next_steps()
    else:
        print("\n[WARNING] Setup completed with warnings. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
