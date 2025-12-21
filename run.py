#!/usr/bin/env python
# run.py - Main entry point for the Text Classification and Retrieval System

"""
Text Classification and Retrieval System - Runner Script

This script provides multiple ways to run the system:
1. Web interface (Streamlit) - default
2. Command-line interface
3. Quick test mode

Usage:
    python run.py              # Run Streamlit web app
    python run.py --cli        # Run command-line interface
    python run.py --test       # Run quick test
    python run.py --setup      # Run setup first, then start app
"""

import os
import sys
import argparse
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_setup():
    """Check if the system is properly set up"""
    model_file = os.path.join(PROJECT_ROOT, 'classification', 'models', 'classifier.pkl')
    index_dir = os.path.join(PROJECT_ROOT, 'retrieval', 'indexdir')

    if not os.path.exists(model_file):
        print("Warning: Classifier model not found.")
        return False

    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print("Warning: Search index not found.")
        return False

    return True


def run_setup():
    """Run the setup script"""
    print("Running setup...")
    setup_script = os.path.join(PROJECT_ROOT, 'setup.py')
    subprocess.run([sys.executable, setup_script], check=True)


def run_streamlit():
    """Run the Streamlit web application"""
    app_path = os.path.join(PROJECT_ROOT, 'app', 'main.py')

    print("\n" + "=" * 50)
    print("  Starting Text Classification & Retrieval System")
    print("=" * 50)
    print("\n  Web interface will be available at:")
    print("  http://localhost:8501")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    subprocess.run(['streamlit', 'run', app_path])


def run_cli():
    """Run command-line interface"""
    print("\n" + "=" * 50)
    print("  Text Classification & Retrieval System - CLI")
    print("=" * 50 + "\n")

    # Load components
    from classification.classifier_model import TextClassifier
    from retrieval.searcher import DocumentSearcher
    import joblib

    model_dir = os.path.join(PROJECT_ROOT, 'classification', 'models')
    index_dir = os.path.join(PROJECT_ROOT, 'retrieval', 'indexdir')

    print("Loading classifier...")
    classifier = TextClassifier()
    classifier.load_model(model_dir)

    print("Loading search index...")
    searcher = DocumentSearcher()
    searcher.open_index(index_dir)

    # Load category mapping
    category_mapping = joblib.load(os.path.join(model_dir, 'category_mapping.pkl'))

    print("\nSystem ready! Enter your queries (type 'quit' to exit)\n")
    print("-" * 50)

    while True:
        try:
            query = input("\nQuery: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Classify
            predictions, probs = classifier.predict(query)
            predicted_category = category_mapping.get(predictions[0], "Unknown")

            print(f"\n  Predicted Category: {predicted_category}")
            if probs is not None:
                confidence = probs[0][predictions[0]]
                print(f"  Confidence: {confidence:.1%}")

            # Search
            results = searcher.search(query, predicted_category, max_results=5)

            print(f"\n  Found {len(results)} results:")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] Document {result['doc_id']} (Score: {result['score']:.4f})")
                print(f"      Category: {result['category']}")
                preview = result.get('content_preview', '')[:150]
                if preview:
                    print(f"      Preview: {preview}...")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test():
    """Run a quick test of the system"""
    print("\n" + "=" * 50)
    print("  Quick System Test")
    print("=" * 50 + "\n")

    from classification.classifier_model import TextClassifier
    from retrieval.searcher import DocumentSearcher
    import joblib

    model_dir = os.path.join(PROJECT_ROOT, 'classification', 'models')
    index_dir = os.path.join(PROJECT_ROOT, 'retrieval', 'indexdir')

    # Load components
    print("Loading classifier...")
    classifier = TextClassifier()
    classifier.load_model(model_dir)

    print("Loading search index...")
    searcher = DocumentSearcher()
    searcher.open_index(index_dir)

    category_mapping = joblib.load(os.path.join(model_dir, 'category_mapping.pkl'))

    # Test queries
    test_queries = [
        ("computer graphics rendering", "comp.graphics"),
        ("hockey game playoff", "rec.sport.hockey"),
        ("NASA space mission", "sci.space"),
        ("car engine performance", "rec.autos"),
        ("cryptography encryption", "sci.crypt")
    ]

    print("\nRunning test queries...")
    print("-" * 50)

    correct = 0
    total = len(test_queries)

    for query, expected_category in test_queries:
        predictions, probs = classifier.predict(query)
        predicted_category = category_mapping.get(predictions[0], "Unknown")

        is_correct = predicted_category == expected_category
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        confidence = probs[0][predictions[0]] if probs is not None else 0

        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected_category}")
        print(f"  Predicted: {predicted_category} ({confidence:.1%}) {status}")

        # Test retrieval
        results = searcher.search(query, predicted_category, max_results=3)
        print(f"  Retrieved: {len(results)} documents")

    print("\n" + "-" * 50)
    print(f"\nClassification Accuracy: {correct}/{total} ({correct/total:.0%})")
    print("Test completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Text Classification and Retrieval System'
    )
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run command-line interface instead of web app'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick system test'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup before starting the application'
    )
    args = parser.parse_args()

    # Check if setup is needed
    if args.setup or not check_setup():
        if not check_setup():
            print("\nSystem not initialized. Running setup first...")
        run_setup()

    # Run appropriate mode
    if args.test:
        run_test()
    elif args.cli:
        run_cli()
    else:
        run_streamlit()


if __name__ == "__main__":
    main()
