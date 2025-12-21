# evaluation/experiments.py - Comprehensive Experiment Suite

import sys
import os

# Setup path
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
import joblib
import json
import time
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from data.load_data import DataLoader
from classification.classifier_model import TextClassifier
from retrieval.searcher import DocumentSearcher
from evaluation.metrics import RetrievalMetrics, ClassificationMetrics

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


class ExperimentRunner:
    """Comprehensive Experiment Runner"""

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(_project_root, 'results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.loader = DataLoader()
        self.classifier = None
        self.searcher = None
        self.df = None
        self.category_mapping = None

        self.results = {
            'classification': {},
            'retrieval': {},
            'baselines': {},
            'ablation': {},
            'case_studies': {},
            'query_types': {}
        }

    def setup(self):
        """Initialize all components"""
        print("=" * 70)
        print("EXPERIMENT SETUP")
        print("=" * 70)

        model_dir = os.path.join(_project_root, 'classification', 'models')
        index_dir = os.path.join(_project_root, 'retrieval', 'indexdir')

        print("\n[1/4] Loading classifier...")
        self.classifier = TextClassifier()
        self.classifier.load_model(model_dir)

        print("[2/4] Loading retriever...")
        self.searcher = DocumentSearcher()
        self.searcher.open_index(index_dir)

        print("[3/4] Loading dataset...")
        self.df = self.loader.load_20newsgroups()

        print("[4/4] Loading category mapping...")
        mapping_path = os.path.join(model_dir, 'category_mapping.pkl')
        self.category_mapping = joblib.load(mapping_path)

        print(f"\nSetup complete: {len(self.df):,} documents, {len(self.category_mapping)} categories")

    # =========================================================================
    # EXPERIMENT 1: Classification Evaluation
    # =========================================================================

    def experiment_classification(self):
        """Evaluate classification model"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: CLASSIFICATION MODEL EVALUATION")
        print("=" * 70)

        train_size = int(len(self.df) * 0.8)
        test_df = self.df.iloc[train_size:]

        X_test = test_df['text_clean'].tolist()
        y_test = np.array(test_df['target'].tolist())

        print(f"\nTest set size: {len(X_test)}")

        start_time = time.time()
        y_pred, y_probs = self.classifier.predict(X_test)
        inference_time = time.time() - start_time

        y_pred = np.array(y_pred)

        metrics = ClassificationMetrics.compute_metrics(y_test, y_pred)
        metrics['inference_time_ms'] = inference_time / len(X_test) * 1000

        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Latency:   {metrics['inference_time_ms']:.2f} ms/sample")

        # Per-class report
        from sklearn.metrics import classification_report, confusion_matrix

        class_names = [self.category_mapping.get(i, f"Class_{i}")
                       for i in range(len(self.category_mapping))]
        short_names = [n.split('.')[-1] for n in class_names]

        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        self.results['classification'] = {
            'overall': metrics,
            'per_class': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }

        # Visualizations
        self._plot_confusion_matrix(cm, short_names)
        self._plot_per_class_metrics(report, class_names)

        return metrics

    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Counts
        sns.heatmap(cm, ax=axes[0], cmap='Blues', fmt='d', annot=True,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 8})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)

        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, ax=axes[1], cmap='Blues', fmt='.2f', annot=True,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 8})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    def _plot_per_class_metrics(self, report, class_names):
        """Plot per-class performance"""
        short_names = [n.split('.')[-1] for n in class_names]

        metrics_df = pd.DataFrame({
            'Category': short_names,
            'Precision': [report[n]['precision'] for n in class_names],
            'Recall': [report[n]['recall'] for n in class_names],
            'F1-Score': [report[n]['f1-score'] for n in class_names],
            'Support': [report[n]['support'] for n in class_names]
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b']

        for ax, metric, color in zip(axes.flat, ['Precision', 'Recall', 'F1-Score', 'Support'], colors):
            bars = ax.barh(metrics_df['Category'], metrics_df[metric], color=color, edgecolor='white')
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} by Category', fontweight='bold')
            if metric != 'Support':
                ax.set_xlim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'per_class_performance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # EXPERIMENT 2: Baseline Comparisons
    # =========================================================================

    def experiment_baselines(self, num_queries=100):
        """Compare against baselines"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: BASELINE COMPARISONS")
        print("=" * 70)

        test_queries = self._generate_queries(num_queries)
        print(f"\nRunning {len(test_queries)} queries...")

        systems = {
            'Baseline (Pure Retrieval)': {'results': defaultdict(list), 'timing': []},
            'Baseline (Post-Filter)': {'results': defaultdict(list), 'timing': []},
            'Proposed System': {'results': defaultdict(list), 'timing': []},
            'Oracle (Perfect Class.)': {'results': defaultdict(list), 'timing': []}
        }

        for query, true_category in tqdm(test_queries, desc="Evaluating"):
            relevant_docs = set(self.df[self.df['target_name'] == true_category]['id'].tolist())

            # Baseline 1: Pure Retrieval
            start = time.time()
            results = self.searcher.search(query, None, max_results=20)
            systems['Baseline (Pure Retrieval)']['timing'].append(time.time() - start)
            docs = [r['doc_id'] for r in results]
            self._calc_metrics(systems['Baseline (Pure Retrieval)']['results'], relevant_docs, docs)

            # Baseline 2: Post-Filter
            start = time.time()
            all_results = self.searcher.search(query, None, max_results=50)
            pred_cat, _ = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            filtered = [r['doc_id'] for r in all_results if r['category'] == pred_name][:20]
            systems['Baseline (Post-Filter)']['timing'].append(time.time() - start)
            self._calc_metrics(systems['Baseline (Post-Filter)']['results'], relevant_docs, filtered)

            # Proposed: Classification-Guided
            start = time.time()
            pred_cat, _ = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            results = self.searcher.search(query, pred_name, max_results=20)
            systems['Proposed System']['timing'].append(time.time() - start)
            docs = [r['doc_id'] for r in results]
            self._calc_metrics(systems['Proposed System']['results'], relevant_docs, docs)

            # Oracle
            start = time.time()
            results = self.searcher.search(query, true_category, max_results=20)
            systems['Oracle (Perfect Class.)']['timing'].append(time.time() - start)
            docs = [r['doc_id'] for r in results]
            self._calc_metrics(systems['Oracle (Perfect Class.)']['results'], relevant_docs, docs)

        # Aggregate
        summary = {}
        for name, data in systems.items():
            summary[name] = {
                metric: {'mean': np.mean(vals), 'std': np.std(vals)}
                for metric, vals in data['results'].items()
            }
            summary[name]['latency_ms'] = np.mean(data['timing']) * 1000

        self._print_baseline_table(summary)
        self.results['baselines'] = summary
        self._plot_baseline_comparison(summary)

        return summary

    def _calc_metrics(self, results_dict, relevant, retrieved):
        """Calculate retrieval metrics"""
        results_dict['P@5'].append(RetrievalMetrics.precision_at_k(relevant, retrieved, 5))
        results_dict['P@10'].append(RetrievalMetrics.precision_at_k(relevant, retrieved, 10))
        results_dict['R@10'].append(RetrievalMetrics.recall_at_k(relevant, retrieved, 10))
        results_dict['MAP'].append(RetrievalMetrics.average_precision(relevant, retrieved))
        results_dict['NDCG@10'].append(RetrievalMetrics.normalized_dcg(relevant, retrieved, 10))

    def _print_baseline_table(self, summary):
        """Print baseline comparison table"""
        print("\n" + "-" * 95)
        print(f"{'System':<28} {'P@5':<10} {'P@10':<10} {'R@10':<10} {'MAP':<10} {'NDCG@10':<10} {'Latency':<10}")
        print("-" * 95)

        for name in ['Baseline (Pure Retrieval)', 'Baseline (Post-Filter)', 'Proposed System', 'Oracle (Perfect Class.)']:
            s = summary[name]
            print(f"{name:<28} "
                  f"{s['P@5']['mean']:.4f}     "
                  f"{s['P@10']['mean']:.4f}     "
                  f"{s['R@10']['mean']:.4f}     "
                  f"{s['MAP']['mean']:.4f}     "
                  f"{s['NDCG@10']['mean']:.4f}     "
                  f"{s['latency_ms']:.1f}ms")

        print("-" * 95)

        # Improvement
        base = summary['Baseline (Pure Retrieval)']['P@10']['mean']
        prop = summary['Proposed System']['P@10']['mean']
        if base > 0:
            print(f"\nImprovement (P@10): {(prop-base)/base*100:+.1f}%")

    def _plot_baseline_comparison(self, summary):
        """Plot baseline comparison"""
        systems = ['Baseline (Pure Retrieval)', 'Baseline (Post-Filter)', 'Proposed System', 'Oracle (Perfect Class.)']
        short_names = ['Pure Retrieval', 'Post-Filter', 'Proposed', 'Oracle']
        metrics = ['P@5', 'P@10', 'R@10', 'MAP', 'NDCG@10']
        colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        x = np.arange(len(metrics))
        width = 0.2

        for i, (sys, label, color) in enumerate(zip(systems, short_names, colors)):
            values = [summary[sys][m]['mean'] for m in metrics]
            axes[0].bar(x + i * width, values, width, label=label, color=color, edgecolor='white')

        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Retrieval Performance Comparison', fontweight='bold')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(metrics)
        axes[0].legend(loc='upper right')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)

        # Latency
        latencies = [summary[s]['latency_ms'] for s in systems]
        bars = axes[1].bar(short_names, latencies, color=colors, edgecolor='white')
        axes[1].set_xlabel('System')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('Average Query Latency', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        for bar, lat in zip(bars, latencies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{lat:.1f}', ha='center', fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'baseline_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # EXPERIMENT 3: Ablation Study
    # =========================================================================

    def experiment_ablation(self, num_queries=50):
        """Study impact of classification confidence threshold"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: ABLATION STUDY (Confidence Threshold)")
        print("=" * 70)

        test_queries = self._generate_queries(num_queries)
        thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
        threshold_results = {t: defaultdict(list) for t in thresholds}

        for query, true_category in tqdm(test_queries, desc="Ablation"):
            relevant_docs = set(self.df[self.df['target_name'] == true_category]['id'].tolist())

            pred_cat, probs = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            confidence = probs[0][pred_cat[0]] if probs is not None else 0

            for threshold in thresholds:
                if confidence >= threshold:
                    results = self.searcher.search(query, pred_name, max_results=20)
                else:
                    results = self.searcher.search(query, None, max_results=20)

                docs = [r['doc_id'] for r in results]
                threshold_results[threshold]['P@10'].append(
                    RetrievalMetrics.precision_at_k(relevant_docs, docs, 10))
                threshold_results[threshold]['MAP'].append(
                    RetrievalMetrics.average_precision(relevant_docs, docs))
                threshold_results[threshold]['correct'].append(
                    1 if pred_name == true_category else 0)

        # Aggregate
        ablation = {}
        for t in thresholds:
            ablation[t] = {
                'P@10': np.mean(threshold_results[t]['P@10']),
                'MAP': np.mean(threshold_results[t]['MAP']),
                'Classification_Accuracy': np.mean(threshold_results[t]['correct'])
            }

        print("\n" + "-" * 55)
        print(f"{'Threshold':<12} {'P@10':<12} {'MAP':<12} {'Class. Acc.':<15}")
        print("-" * 55)
        for t in thresholds:
            print(f"{t:<12.2f} {ablation[t]['P@10']:<12.4f} {ablation[t]['MAP']:<12.4f} {ablation[t]['Classification_Accuracy']:<15.4f}")
        print("-" * 55)

        self.results['ablation'] = ablation
        self._plot_ablation(ablation, thresholds)

        return ablation

    def _plot_ablation(self, ablation, thresholds):
        """Plot ablation study"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        p10 = [ablation[t]['P@10'] for t in thresholds]
        map_scores = [ablation[t]['MAP'] for t in thresholds]
        class_acc = [ablation[t]['Classification_Accuracy'] for t in thresholds]

        axes[0].plot(thresholds, p10, 'o-', label='P@10', color='#3b82f6', linewidth=2, markersize=8)
        axes[0].plot(thresholds, map_scores, 's-', label='MAP', color='#10b981', linewidth=2, markersize=8)
        axes[0].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Retrieval vs Confidence Threshold', fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(-0.05, 1.0)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresholds, class_acc, 'o-', color='#8b5cf6', linewidth=2, markersize=8)
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('Classification Accuracy')
        axes[1].set_title('Classification Accuracy vs Threshold', fontweight='bold')
        axes[1].set_xlim(-0.05, 1.0)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'ablation_study.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # EXPERIMENT 4: Query Type Analysis
    # =========================================================================

    def experiment_query_types(self, num_queries=100):
        """Analyze performance by query type"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: QUERY TYPE ANALYSIS")
        print("=" * 70)

        # AG News category groups
        category_groups = {
            'World': ['World'],
            'Sports': ['Sports'],
            'Business': ['Business'],
            'Sci/Tech': ['Sci/Tech']
        }

        test_queries = self._generate_queries(num_queries)
        group_results = {g: defaultdict(list) for g in category_groups}

        for query, true_category in tqdm(test_queries, desc="Query Types"):
            group = None
            for g, cats in category_groups.items():
                if true_category in cats:
                    group = g
                    break
            if group is None:
                continue

            relevant = set(self.df[self.df['target_name'] == true_category]['id'].tolist())

            pred_cat, _ = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            results = self.searcher.search(query, pred_name, max_results=20)
            docs = [r['doc_id'] for r in results]

            group_results[group]['P@10'].append(RetrievalMetrics.precision_at_k(relevant, docs, 10))
            group_results[group]['correct'].append(1 if pred_name == true_category else 0)

        # Aggregate
        query_types = {}
        for g in category_groups:
            if group_results[g]['P@10']:
                query_types[g] = {
                    'P@10': np.mean(group_results[g]['P@10']),
                    'Classification_Accuracy': np.mean(group_results[g]['correct']),
                    'Count': len(group_results[g]['P@10'])
                }

        print("\n" + "-" * 55)
        print(f"{'Query Type':<15} {'P@10':<12} {'Class. Acc.':<15} {'Count':<10}")
        print("-" * 55)
        for g, s in query_types.items():
            print(f"{g:<15} {s['P@10']:<12.4f} {s['Classification_Accuracy']:<15.4f} {s['Count']:<10}")
        print("-" * 55)

        self.results['query_types'] = query_types
        self._plot_query_types(query_types)

        return query_types

    def _plot_query_types(self, query_types):
        """Plot query type analysis"""
        groups = list(query_types.keys())
        p10 = [query_types[g]['P@10'] for g in groups]
        acc = [query_types[g]['Classification_Accuracy'] for g in groups]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(groups))
        width = 0.35

        ax.bar(x - width/2, p10, width, label='P@10', color='#3b82f6', edgecolor='white')
        ax.bar(x + width/2, acc, width, label='Classification Accuracy', color='#10b981', edgecolor='white')

        ax.set_xlabel('Query Type')
        ax.set_ylabel('Score')
        ax.set_title('Performance by Query Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'query_type_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # EXPERIMENT 5: Case Studies
    # =========================================================================

    def experiment_case_studies(self, num_cases=15):
        """Detailed case studies"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: CASE STUDIES")
        print("=" * 70)

        cases = {'success': [], 'failure': [], 'boundary': []}
        test_queries = self._generate_queries(num_cases * 5)

        for query, true_category in tqdm(test_queries, desc="Case Studies"):
            pred_cat, probs = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            confidence = probs[0][pred_cat[0]] if probs is not None else 0

            relevant = set(self.df[self.df['target_name'] == true_category]['id'].tolist())

            proposed = self.searcher.search(query, pred_name, max_results=10)
            baseline = self.searcher.search(query, None, max_results=10)

            p_p10 = RetrievalMetrics.precision_at_k(relevant, [r['doc_id'] for r in proposed], 10)
            b_p10 = RetrievalMetrics.precision_at_k(relevant, [r['doc_id'] for r in baseline], 10)

            case = {
                'query': query[:80],
                'true': true_category.split('.')[-1],
                'predicted': pred_name.split('.')[-1] if pred_name else 'Unknown',
                'confidence': confidence,
                'correct': pred_name == true_category,
                'proposed_p10': p_p10,
                'baseline_p10': b_p10,
                'improvement': p_p10 - b_p10
            }

            if case['correct'] and case['improvement'] > 0.05 and len(cases['success']) < num_cases:
                cases['success'].append(case)
            elif not case['correct'] and case['improvement'] < -0.05 and len(cases['failure']) < num_cases:
                cases['failure'].append(case)
            elif 0.35 <= confidence <= 0.65 and len(cases['boundary']) < num_cases:
                cases['boundary'].append(case)

        self._print_cases(cases)
        self.results['case_studies'] = cases

        return cases

    def _print_cases(self, cases):
        """Print case studies"""
        for case_type, label in [('success', 'SUCCESS'), ('failure', 'FAILURE'), ('boundary', 'BOUNDARY')]:
            print(f"\n--- {label} CASES ---")
            for i, c in enumerate(cases[case_type][:3], 1):
                print(f"[{i}] Query: {c['query'][:50]}...")
                print(f"    True: {c['true']} | Pred: {c['predicted']} ({c['confidence']:.1%})")
                print(f"    P@10: {c['baseline_p10']:.2f} -> {c['proposed_p10']:.2f} ({c['improvement']:+.2f})")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_queries(self, num_queries):
        """Generate diverse test queries"""
        queries = []
        categories = list(self.category_mapping.values())
        per_cat = max(1, num_queries // len(categories))

        for cat in categories:
            docs = self.df[self.df['target_name'] == cat]
            if len(docs) > 0:
                samples = docs.sample(min(per_cat, len(docs)))
                for _, doc in samples.iterrows():
                    words = doc['text_clean'].split()[:6]
                    query = ' '.join(words)
                    if len(query) > 5:
                        queries.append((query, cat))

        np.random.shuffle(queries)
        return queries[:num_queries]

    # =========================================================================
    # Main Runner
    # =========================================================================

    def run_all(self):
        """Run all experiments"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EXPERIMENT SUITE")
        print(f"Output: {self.output_dir}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        self.setup()

        self.experiment_classification()
        self.experiment_baselines(num_queries=100)
        self.experiment_ablation(num_queries=50)
        self.experiment_query_types(num_queries=100)
        self.experiment_case_studies(num_cases=10)

        self._save_report()
        self._plot_summary()

        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def _save_report(self):
        """Save JSON report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': {'documents': len(self.df), 'categories': len(self.category_mapping)},
            'results': {}
        }

        # Convert numpy types
        for key, val in self.results.items():
            report['results'][key] = json.loads(json.dumps(val, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))

        path = os.path.join(self.output_dir, 'experiment_report.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {path}")

    def _plot_summary(self):
        """Plot summary dashboard"""
        fig = plt.figure(figsize=(16, 12))

        # 1. Classification metrics
        ax1 = fig.add_subplot(2, 3, 1)
        metrics = self.results['classification']['overall']
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        bars = ax1.bar(names, values, color=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b'], edgecolor='white')
        ax1.set_ylim(0, 1)
        ax1.set_title('Classification Performance', fontweight='bold')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)

        # 2. Baseline comparison
        ax2 = fig.add_subplot(2, 3, 2)
        systems = ['Pure Retrieval', 'Post-Filter', 'Proposed', 'Oracle']
        p10_values = [
            self.results['baselines']['Baseline (Pure Retrieval)']['P@10']['mean'],
            self.results['baselines']['Baseline (Post-Filter)']['P@10']['mean'],
            self.results['baselines']['Proposed System']['P@10']['mean'],
            self.results['baselines']['Oracle (Perfect Class.)']['P@10']['mean']
        ]
        bars = ax2.bar(systems, p10_values, color=['#94a3b8', '#64748b', '#3b82f6', '#10b981'], edgecolor='white')
        ax2.set_ylim(0, 1)
        ax2.set_title('P@10 Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)

        # 3. Ablation
        ax3 = fig.add_subplot(2, 3, 3)
        thresholds = sorted(self.results['ablation'].keys())
        p10_abl = [self.results['ablation'][t]['P@10'] for t in thresholds]
        ax3.plot(thresholds, p10_abl, 'o-', color='#3b82f6', linewidth=2, markersize=8)
        ax3.set_xlabel('Confidence Threshold')
        ax3.set_ylabel('P@10')
        ax3.set_title('Ablation: P@10 vs Threshold', fontweight='bold')
        ax3.set_xlim(-0.05, 1.0)
        ax3.grid(True, alpha=0.3)

        # 4. Query types
        ax4 = fig.add_subplot(2, 3, 4)
        if self.results['query_types']:
            groups = list(self.results['query_types'].keys())
            qt_p10 = [self.results['query_types'][g]['P@10'] for g in groups]
            ax4.bar(groups, qt_p10, color='#3b82f6', edgecolor='white')
            ax4.set_ylim(0, 1)
            ax4.set_title('P@10 by Query Type', fontweight='bold')
            ax4.tick_params(axis='x', rotation=15)

        # 5. Improvement summary
        ax5 = fig.add_subplot(2, 3, 5)
        base_p10 = self.results['baselines']['Baseline (Pure Retrieval)']['P@10']['mean']
        prop_p10 = self.results['baselines']['Proposed System']['P@10']['mean']
        orac_p10 = self.results['baselines']['Oracle (Perfect Class.)']['P@10']['mean']

        improvement = (prop_p10 - base_p10) / base_p10 * 100 if base_p10 > 0 else 0
        potential = (orac_p10 - prop_p10) / prop_p10 * 100 if prop_p10 > 0 else 0

        bars = ax5.bar(['Achieved\nImprovement', 'Remaining\nPotential'], [improvement, potential],
                       color=['#10b981', '#f59e0b'], edgecolor='white')
        ax5.set_ylabel('Percentage (%)')
        ax5.set_title('Improvement Analysis', fontweight='bold')
        for bar, val in zip(bars, [improvement, potential]):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=10)

        # 6. Key findings text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        findings = f"""
KEY FINDINGS

Classification:
  - Accuracy: {metrics['accuracy']:.2%}
  - F1 Score: {metrics['f1']:.4f}

Retrieval:
  - Baseline P@10: {base_p10:.4f}
  - Proposed P@10: {prop_p10:.4f}
  - Improvement: {improvement:+.1f}%

Oracle P@10: {orac_p10:.4f}
(Upper bound with perfect classification)
        """
        ax6.text(0.1, 0.5, findings, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#e2e8f0'))

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'experiment_summary.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def main():
    runner = ExperimentRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
