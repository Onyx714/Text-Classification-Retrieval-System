# evaluation/comprehensive_experiments.py - 完整实验系统 v2
# Comprehensive Experiment Suite with User Simulation & Personalization

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
import random
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from data.load_data import DataLoader
from classification.classifier_model import TextClassifier
from retrieval.searcher import DocumentSearcher

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.unicode_minus'] = False

# Random seed for reproducibility
np.random.seed(42)
random.seed(42)


class UserSimulator:
    """用户行为模拟器 - 模拟不同类型的用户"""

    def __init__(self, df: pd.DataFrame, num_users: int = 100):
        self.df = df
        self.num_users = num_users
        self.users = {}
        self.user_histories = {}
        self.categories = df['target_name'].unique().tolist()

    def generate_users(self):
        """生成模拟用户"""
        user_types = ['focused', 'diverse', 'new', 'active']

        for user_id in range(self.num_users):
            user_type = random.choice(user_types)

            if user_type == 'focused':
                primary_cats = random.sample(self.categories, k=random.randint(1, 2))
                preferences = {cat: 0.8 if cat in primary_cats else 0.1 for cat in self.categories}
                history_size = random.randint(20, 50)
            elif user_type == 'diverse':
                preferences = {cat: random.uniform(0.2, 0.8) for cat in self.categories}
                history_size = random.randint(30, 80)
            elif user_type == 'new':
                primary_cats = random.sample(self.categories, k=1)
                preferences = {cat: 0.6 if cat in primary_cats else 0.3 for cat in self.categories}
                history_size = random.randint(1, 5)
            else:
                preferences = {cat: random.uniform(0.3, 0.7) for cat in self.categories}
                history_size = random.randint(80, 150)

            total = sum(preferences.values())
            preferences = {k: v/total for k, v in preferences.items()}

            self.users[user_id] = {
                'type': user_type,
                'preferences': preferences,
                'history_size': history_size
            }
            self._generate_user_history(user_id)

        return self.users

    def _generate_user_history(self, user_id: int):
        """为用户生成浏览历史"""
        user = self.users[user_id]
        preferences = user['preferences']
        history_size = user['history_size']

        history = []
        for _ in range(history_size):
            cat = random.choices(list(preferences.keys()), weights=list(preferences.values()))[0]
            cat_docs = self.df[self.df['target_name'] == cat]
            if len(cat_docs) > 0:
                doc = cat_docs.sample(1).iloc[0]
                history.append({'doc_id': doc['id'], 'category': cat, 'timestamp': random.randint(1, 1000)})

        history.sort(key=lambda x: x['timestamp'])
        self.user_histories[user_id] = history

    def get_user_profile(self, user_id: int) -> Dict[str, float]:
        """基于用户历史构建用户画像"""
        history = self.user_histories.get(user_id, [])
        if not history:
            return {cat: 1.0/len(self.categories) for cat in self.categories}

        cat_counts = defaultdict(int)
        for item in history:
            cat_counts[item['category']] += 1

        total = sum(cat_counts.values())
        return {cat: cat_counts.get(cat, 0) / total for cat in self.categories}

    def generate_query_with_doc(self, user_id: int) -> Tuple[str, str, int]:
        """为用户生成查询，返回(查询, 类别, 源文档ID)"""
        user = self.users[user_id]
        preferences = user['preferences']
        target_cat = random.choices(list(preferences.keys()), weights=list(preferences.values()))[0]

        cat_docs = self.df[self.df['target_name'] == target_cat]
        doc = cat_docs.sample(1).iloc[0]
        words = doc['text_clean'].split()[:random.randint(4, 8)]
        query = ' '.join(words)

        return query, target_cat, doc['id']


class PersonalizedRetriever:
    """个性化检索系统"""

    def __init__(self, searcher: DocumentSearcher, classifier: TextClassifier, category_mapping: Dict):
        self.searcher = searcher
        self.classifier = classifier
        self.category_mapping = category_mapping

    def search_baseline(self, query: str, max_results: int = 20) -> List[Dict]:
        """基线检索: 纯BM25，无个性化"""
        return self.searcher.search(query, None, max_results=max_results)

    def search_classification_only(self, query: str, max_results: int = 20) -> Tuple[List[Dict], str, float]:
        """分类引导的检索，返回结果、预测类别、置信度"""
        pred_cat, probs = self.classifier.predict(query)
        pred_name = self.category_mapping.get(pred_cat[0], "")
        conf = probs[0][pred_cat[0]] if probs is not None else 0.5
        results = self.searcher.search(query, pred_name, max_results=max_results)
        return results, pred_name, conf

    def search_personalized(self, query: str, user_profile: Dict[str, float],
                           max_results: int = 20, alpha: float = 0.5) -> List[Dict]:
        """个性化检索"""
        pred_cat, probs = self.classifier.predict(query)
        pred_name = self.category_mapping.get(pred_cat[0], "")
        conf = probs[0][pred_cat[0]] if probs is not None else 0.5
        user_pref = user_profile.get(pred_name, 0.25)

        if conf > 0.6 and user_pref > 0.2:
            results = self.searcher.search(query, pred_name, max_results=max_results * 2)
        elif conf < 0.5:
            top_user_cat = max(user_profile, key=user_profile.get)
            if user_profile[top_user_cat] > 0.4:
                results = self.searcher.search(query, top_user_cat, max_results=max_results * 2)
            else:
                results = self.searcher.search(query, None, max_results=max_results * 2)
        else:
            results = self.searcher.search(query, pred_name, max_results=max_results * 2)

        for r in results:
            doc_cat = r.get('category', '')
            user_boost = user_profile.get(doc_cat, 0.25)
            r['personalized_score'] = r['score'] * (1 + alpha * user_boost)

        results.sort(key=lambda x: x['personalized_score'], reverse=True)
        return results[:max_results]

    def search_hybrid(self, query: str, user_profile: Dict[str, float],
                     max_results: int = 20, conf_threshold: float = 0.7) -> List[Dict]:
        """混合检索: 自适应策略"""
        pred_cat, probs = self.classifier.predict(query)
        pred_name = self.category_mapping.get(pred_cat[0], "")
        conf = probs[0][pred_cat[0]] if probs is not None else 0.5

        if conf >= conf_threshold:
            return self.searcher.search(query, pred_name, max_results=max_results)

        top_user_cat = max(user_profile, key=user_profile.get)
        if user_profile[top_user_cat] > 0.35:
            return self.searcher.search(query, top_user_cat, max_results=max_results)

        return self.searcher.search(query, None, max_results=max_results)


class ComprehensiveMetrics:
    """综合评估指标"""

    @staticmethod
    def precision_at_k(relevant: set, retrieved: List, k: int) -> float:
        if not retrieved or k == 0:
            return 0.0
        top_k = retrieved[:k]
        hits = len([d for d in top_k if d in relevant])
        return hits / len(top_k)

    @staticmethod
    def recall_at_k(relevant: set, retrieved: List, k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        hits = len([d for d in top_k if d in relevant])
        return hits / len(relevant)

    @staticmethod
    def average_precision(relevant: set, retrieved: List) -> float:
        if not relevant:
            return 0.0
        hits = 0
        sum_prec = 0.0
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                hits += 1
                sum_prec += hits / i
        return sum_prec / len(relevant) if relevant else 0.0

    @staticmethod
    def reciprocal_rank(relevant: set, retrieved: List) -> float:
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(relevant: set, retrieved: List, k: int) -> float:
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], 1):
            rel = 1 if doc in relevant else 0
            dcg += rel / np.log2(i + 1)

        ideal_rels = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_rels + 1))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def category_precision(retrieved_categories: List[str], true_category: str) -> float:
        """类别精确率：返回结果中正确类别的比例"""
        if not retrieved_categories:
            return 0.0
        correct = sum(1 for c in retrieved_categories if c == true_category)
        return correct / len(retrieved_categories)

    @staticmethod
    def hit_rate_at_k(source_doc_id: int, retrieved: List, k: int) -> float:
        """源文档是否在前K个结果中"""
        return 1.0 if source_doc_id in retrieved[:k] else 0.0


class ComprehensiveExperiments:
    """综合实验系统"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(_project_root, 'results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.loader = DataLoader()
        self.classifier = None
        self.searcher = None
        self.df = None
        self.category_mapping = None
        self.user_simulator = None
        self.personalized_retriever = None

        self.results = {
            'dataset_info': {},
            'classification': {},
            'retrieval_baselines': {},
            'personalization': {},
            'cold_start': {},
            'user_satisfaction': {},
            'ablation': {},
            'error_analysis': {},
            'category_analysis': {}
        }

    def setup(self):
        """初始化系统"""
        print("=" * 70)
        print("COMPREHENSIVE EXPERIMENT SYSTEM SETUP")
        print("=" * 70)

        model_dir = os.path.join(_project_root, 'classification', 'models')
        index_dir = os.path.join(_project_root, 'retrieval', 'indexdir')

        print("\n[1/5] Loading classifier...")
        self.classifier = TextClassifier()
        self.classifier.load_model(model_dir)

        print("[2/5] Loading search index...")
        self.searcher = DocumentSearcher()
        self.searcher.open_index(index_dir)

        print("[3/5] Loading dataset...")
        self.df = self.loader.load_ag_news()

        print("[4/5] Loading category mapping...")
        mapping_path = os.path.join(model_dir, 'category_mapping.pkl')
        self.category_mapping = joblib.load(mapping_path)

        print("[5/5] Initializing user simulator...")
        self.user_simulator = UserSimulator(self.df, num_users=200)
        self.user_simulator.generate_users()

        self.personalized_retriever = PersonalizedRetriever(
            self.searcher, self.classifier, self.category_mapping
        )

        self.results['dataset_info'] = {
            'name': 'AG News',
            'total_documents': len(self.df),
            'categories': list(self.category_mapping.values()),
            'num_categories': len(self.category_mapping),
            'category_distribution': self.df['target_name'].value_counts().to_dict(),
            'simulated_users': len(self.user_simulator.users),
            'user_type_distribution': self._count_user_types()
        }

        print(f"\nSetup complete:")
        print(f"  - Documents: {len(self.df):,}")
        print(f"  - Categories: {len(self.category_mapping)}")
        print(f"  - Simulated users: {len(self.user_simulator.users)}")

    def _count_user_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for user in self.user_simulator.users.values():
            counts[user['type']] += 1
        return dict(counts)

    # =========================================================================
    # EXPERIMENT 1: Classification Evaluation
    # =========================================================================

    def exp1_classification(self):
        """实验1: 分类模型评估"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: CLASSIFICATION MODEL EVALUATION")
        print("=" * 70)

        train_size = int(len(self.df) * 0.8)
        test_df = self.df.iloc[train_size:].copy()

        X_test = test_df['text_clean'].tolist()
        y_test = np.array(test_df['target'].tolist())

        print(f"\nTest set size: {len(X_test)}")

        start_time = time.time()
        y_pred, y_probs = self.classifier.predict(X_test)
        inference_time = time.time() - start_time

        y_pred = np.array(y_pred)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'inference_time_total_s': inference_time,
            'inference_time_per_sample_ms': inference_time / len(X_test) * 1000
        }

        class_names = [self.category_mapping.get(i, f"Class_{i}") for i in range(len(self.category_mapping))]
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        self.results['classification'] = {
            'overall_metrics': metrics,
            'per_class_metrics': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }

        print(f"\n{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Accuracy':<25} {metrics['accuracy']:.4f}")
        print(f"{'Precision (macro)':<25} {metrics['precision_macro']:.4f}")
        print(f"{'Recall (macro)':<25} {metrics['recall_macro']:.4f}")
        print(f"{'F1-Score (macro)':<25} {metrics['f1_macro']:.4f}")
        print(f"{'Inference (ms/sample)':<25} {metrics['inference_time_per_sample_ms']:.3f}")

        print(f"\nPer-class Performance:")
        print("-" * 60)
        for name in class_names:
            p = report[name]['precision']
            r = report[name]['recall']
            f = report[name]['f1-score']
            s = report[name]['support']
            print(f"{name:<15} P={p:.4f}  R={r:.4f}  F1={f:.4f}  Support={int(s)}")

        self._plot_classification_results(cm, class_names, metrics)
        return metrics

    def _plot_classification_results(self, cm, class_names, metrics):
        """绘制分类结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, ax=axes[0], cmap='Blues', fmt='.2%', annot=True,
                   xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('Normalized Confusion Matrix', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        f1_scores = [self.results['classification']['per_class_metrics'][n]['f1-score'] for n in class_names]
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        bars = axes[1].bar(class_names, f1_scores, color=colors, edgecolor='white')
        axes[1].set_ylim(0, 1)
        axes[1].set_title('F1-Score by Category', fontweight='bold')
        axes[1].set_ylabel('F1-Score')
        for bar, score in zip(bars, f1_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', fontsize=10)

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision_macro'],
                        metrics['recall_macro'], metrics['f1_macro']]
        bars = axes[2].bar(metric_names, metric_values, color='#3b82f6', edgecolor='white')
        axes[2].set_ylim(0, 1)
        axes[2].set_title('Overall Classification Metrics', fontweight='bold')
        for bar, val in zip(bars, metric_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp1_classification.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 2: Retrieval Baseline Comparison
    # =========================================================================

    def exp2_retrieval_baselines(self, num_queries: int = 200):
        """实验2: 检索基线对比 - 使用类别精确率和源文档命中率"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: RETRIEVAL BASELINE COMPARISON")
        print("=" * 70)

        test_queries = self._generate_test_queries(num_queries)
        print(f"\nEvaluating {len(test_queries)} queries across 4 systems...")

        systems = {
            'BM25 (No Filter)': defaultdict(list),
            'BM25 + Classification': defaultdict(list),
            'BM25 + Personalization': defaultdict(list),
            'Oracle (Perfect)': defaultdict(list)
        }
        latencies = {name: [] for name in systems}

        for query, true_category, source_doc_id, user_id in tqdm(test_queries, desc="Evaluating"):
            user_profile = self.user_simulator.get_user_profile(user_id)

            # System 1: BM25 baseline
            start = time.time()
            results = self.personalized_retriever.search_baseline(query, max_results=20)
            latencies['BM25 (No Filter)'].append(time.time() - start)
            self._compute_retrieval_metrics(systems['BM25 (No Filter)'], results, true_category, source_doc_id)

            # System 2: BM25 + Classification
            start = time.time()
            results, pred_cat, conf = self.personalized_retriever.search_classification_only(query, max_results=20)
            latencies['BM25 + Classification'].append(time.time() - start)
            self._compute_retrieval_metrics(systems['BM25 + Classification'], results, true_category, source_doc_id)
            systems['BM25 + Classification']['classification_accuracy'].append(1 if pred_cat == true_category else 0)
            systems['BM25 + Classification']['confidence'].append(conf)

            # System 3: BM25 + Personalization
            start = time.time()
            results = self.personalized_retriever.search_personalized(query, user_profile, max_results=20)
            latencies['BM25 + Personalization'].append(time.time() - start)
            self._compute_retrieval_metrics(systems['BM25 + Personalization'], results, true_category, source_doc_id)

            # System 4: Oracle
            start = time.time()
            results = self.searcher.search(query, true_category, max_results=20)
            latencies['Oracle (Perfect)'].append(time.time() - start)
            self._compute_retrieval_metrics(systems['Oracle (Perfect)'], results, true_category, source_doc_id)

        # 汇总结果
        summary = {}
        for name, data in systems.items():
            summary[name] = {
                metric: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                for metric, vals in data.items()
            }
            summary[name]['latency_ms'] = {
                'mean': float(np.mean(latencies[name]) * 1000),
                'std': float(np.std(latencies[name]) * 1000)
            }

        self.results['retrieval_baselines'] = summary
        self._print_retrieval_table(summary)
        self._plot_retrieval_baselines(summary)

        return summary

    def _compute_retrieval_metrics(self, results_dict: Dict, results: List[Dict],
                                   true_category: str, source_doc_id: int):
        """计算检索指标"""
        retrieved_ids = [r['doc_id'] for r in results]
        retrieved_cats = [r['category'] for r in results]

        # 类别精确率 (最重要的指标)
        results_dict['Category_P@5'].append(
            ComprehensiveMetrics.category_precision(retrieved_cats[:5], true_category))
        results_dict['Category_P@10'].append(
            ComprehensiveMetrics.category_precision(retrieved_cats[:10], true_category))
        results_dict['Category_P@20'].append(
            ComprehensiveMetrics.category_precision(retrieved_cats[:20], true_category))

        # 源文档命中率
        results_dict['Hit@5'].append(
            ComprehensiveMetrics.hit_rate_at_k(source_doc_id, retrieved_ids, 5))
        results_dict['Hit@10'].append(
            ComprehensiveMetrics.hit_rate_at_k(source_doc_id, retrieved_ids, 10))
        results_dict['Hit@20'].append(
            ComprehensiveMetrics.hit_rate_at_k(source_doc_id, retrieved_ids, 20))

        # MRR for source document
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == source_doc_id:
                mrr = 1.0 / i
                break
        results_dict['MRR'].append(mrr)

    def _print_retrieval_table(self, summary: Dict):
        """打印检索结果表格"""
        print("\n" + "-" * 115)
        print(f"{'System':<25} {'Cat_P@5':<10} {'Cat_P@10':<10} {'Hit@5':<10} {'Hit@10':<10} {'MRR':<10} {'Latency':<10}")
        print("-" * 115)

        for name in ['BM25 (No Filter)', 'BM25 + Classification', 'BM25 + Personalization', 'Oracle (Perfect)']:
            s = summary[name]
            print(f"{name:<25} "
                  f"{s['Category_P@5']['mean']:.4f}    "
                  f"{s['Category_P@10']['mean']:.4f}    "
                  f"{s['Hit@5']['mean']:.4f}    "
                  f"{s['Hit@10']['mean']:.4f}    "
                  f"{s['MRR']['mean']:.4f}    "
                  f"{s['latency_ms']['mean']:.1f}ms")
        print("-" * 115)

        # 计算改进
        baseline_cp = summary['BM25 (No Filter)']['Category_P@10']['mean']
        proposed_cp = summary['BM25 + Personalization']['Category_P@10']['mean']
        oracle_cp = summary['Oracle (Perfect)']['Category_P@10']['mean']

        if baseline_cp > 0:
            improvement = (proposed_cp - baseline_cp) / baseline_cp * 100
            print(f"\n*** Category P@10 Improvement over baseline: {improvement:+.2f}%")

        class_acc = summary['BM25 + Classification'].get('classification_accuracy', {}).get('mean', 0)
        print(f"*** Classification Accuracy on queries: {class_acc:.2%}")

    def _plot_retrieval_baselines(self, summary: Dict):
        """绘制检索基线对比"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        systems = ['BM25 (No Filter)', 'BM25 + Classification', 'BM25 + Personalization', 'Oracle (Perfect)']
        short_names = ['BM25', 'BM25+Class', 'Personalized', 'Oracle']
        colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']

        # 1. 类别精确率对比
        metrics = ['Category_P@5', 'Category_P@10', 'Category_P@20']
        x = np.arange(len(metrics))
        width = 0.2

        for i, (sys_name, short, color) in enumerate(zip(systems, short_names, colors)):
            values = [summary[sys_name][m]['mean'] for m in metrics]
            axes[0].bar(x + i * width, values, width, label=short, color=color, edgecolor='white')

        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Category Precision Comparison', fontweight='bold')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(['P@5', 'P@10', 'P@20'])
        axes[0].legend(loc='lower right')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(axis='y', alpha=0.3)

        # 2. Hit Rate对比
        hit_metrics = ['Hit@5', 'Hit@10', 'Hit@20']
        x = np.arange(len(hit_metrics))

        for i, (sys_name, short, color) in enumerate(zip(systems, short_names, colors)):
            values = [summary[sys_name][m]['mean'] for m in hit_metrics]
            axes[1].bar(x + i * width, values, width, label=short, color=color, edgecolor='white')

        axes[1].set_xlabel('Metric')
        axes[1].set_ylabel('Hit Rate')
        axes[1].set_title('Source Document Hit Rate', fontweight='bold')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(['Hit@5', 'Hit@10', 'Hit@20'])
        axes[1].legend(loc='lower right')
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis='y', alpha=0.3)

        # 3. MRR和延迟对比
        mrr_values = [summary[s]['MRR']['mean'] for s in systems]
        bars = axes[2].bar(short_names, mrr_values, color=colors, edgecolor='white')
        axes[2].set_ylabel('MRR')
        axes[2].set_title('Mean Reciprocal Rank', fontweight='bold')
        axes[2].set_ylim(0, 1.1)
        for bar, val in zip(bars, mrr_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp2_retrieval_baselines.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 3: Personalization Analysis by User Type
    # =========================================================================

    def exp3_personalization_analysis(self, num_queries: int = 200):
        """实验3: 按用户类型的个性化效果分析"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: PERSONALIZATION EFFECT BY USER TYPE")
        print("=" * 70)

        user_type_results = {ut: defaultdict(list) for ut in ['focused', 'diverse', 'new', 'active']}

        for user_type in ['focused', 'diverse', 'new', 'active']:
            users_of_type = [uid for uid, u in self.user_simulator.users.items() if u['type'] == user_type]
            queries_per_type = num_queries // 4

            for _ in range(queries_per_type):
                user_id = random.choice(users_of_type)
                query, true_cat, source_doc_id = self.user_simulator.generate_query_with_doc(user_id)
                user_profile = self.user_simulator.get_user_profile(user_id)

                # Baseline
                baseline_results = self.personalized_retriever.search_baseline(query, 20)
                baseline_cats = [r['category'] for r in baseline_results]
                baseline_cp = ComprehensiveMetrics.category_precision(baseline_cats[:10], true_cat)

                # Classification
                class_results, pred_cat, conf = self.personalized_retriever.search_classification_only(query, 20)
                class_cats = [r['category'] for r in class_results]
                class_cp = ComprehensiveMetrics.category_precision(class_cats[:10], true_cat)

                # Personalized
                pers_results = self.personalized_retriever.search_personalized(query, user_profile, 20)
                pers_cats = [r['category'] for r in pers_results]
                pers_cp = ComprehensiveMetrics.category_precision(pers_cats[:10], true_cat)

                user_type_results[user_type]['baseline_CP'].append(baseline_cp)
                user_type_results[user_type]['classification_CP'].append(class_cp)
                user_type_results[user_type]['personalized_CP'].append(pers_cp)
                user_type_results[user_type]['classification_correct'].append(1 if pred_cat == true_cat else 0)

        # 汇总
        summary = {}
        for user_type, data in user_type_results.items():
            summary[user_type] = {
                'baseline_CP': {'mean': float(np.mean(data['baseline_CP'])), 'std': float(np.std(data['baseline_CP']))},
                'classification_CP': {'mean': float(np.mean(data['classification_CP'])), 'std': float(np.std(data['classification_CP']))},
                'personalized_CP': {'mean': float(np.mean(data['personalized_CP'])), 'std': float(np.std(data['personalized_CP']))},
                'classification_accuracy': float(np.mean(data['classification_correct'])),
                'num_queries': len(data['baseline_CP'])
            }

        self.results['personalization'] = summary

        print(f"\n{'User Type':<12} {'Baseline':<12} {'Class.':<12} {'Personal.':<12} {'Class.Acc':<12}")
        print("-" * 60)
        for ut in ['focused', 'diverse', 'new', 'active']:
            s = summary[ut]
            print(f"{ut:<12} {s['baseline_CP']['mean']:<12.4f} {s['classification_CP']['mean']:<12.4f} "
                  f"{s['personalized_CP']['mean']:<12.4f} {s['classification_accuracy']:<12.2%}")

        self._plot_personalization(summary)
        return summary

    def _plot_personalization(self, summary: Dict):
        """绘制个性化分析结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        user_types = ['focused', 'diverse', 'new', 'active']
        x = np.arange(len(user_types))
        width = 0.25

        baseline = [summary[ut]['baseline_CP']['mean'] for ut in user_types]
        classification = [summary[ut]['classification_CP']['mean'] for ut in user_types]
        personalized = [summary[ut]['personalized_CP']['mean'] for ut in user_types]

        axes[0].bar(x - width, baseline, width, label='Baseline (BM25)', color='#94a3b8', edgecolor='white')
        axes[0].bar(x, classification, width, label='Classification', color='#64748b', edgecolor='white')
        axes[0].bar(x + width, personalized, width, label='Personalized', color='#3b82f6', edgecolor='white')
        axes[0].set_xlabel('User Type')
        axes[0].set_ylabel('Category Precision@10')
        axes[0].set_title('Category Precision by User Type', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([t.capitalize() for t in user_types])
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(axis='y', alpha=0.3)

        # Classification accuracy by user type
        class_acc = [summary[ut]['classification_accuracy'] for ut in user_types]
        bars = axes[1].bar(user_types, class_acc, color='#10b981', edgecolor='white')
        axes[1].set_xlabel('User Type')
        axes[1].set_ylabel('Classification Accuracy')
        axes[1].set_title('Classification Accuracy by User Type', fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        for bar, acc in zip(bars, class_acc):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp3_personalization.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 4: Cold Start Analysis
    # =========================================================================

    def exp4_cold_start(self, num_queries: int = 150):
        """实验4: 冷启动问题分析"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: COLD START ANALYSIS")
        print("=" * 70)

        history_bins = [(0, 3), (4, 10), (11, 30), (31, 80), (81, 200)]
        bin_labels = ['0-3', '4-10', '11-30', '31-80', '80+']

        cold_start_results = {label: defaultdict(list) for label in bin_labels}

        for user_id, user in self.user_simulator.users.items():
            history_size = len(self.user_simulator.user_histories.get(user_id, []))

            bin_label = None
            for (low, high), label in zip(history_bins, bin_labels):
                if low <= history_size <= high:
                    bin_label = label
                    break

            if bin_label is None:
                continue

            for _ in range(2):
                query, true_cat, source_doc_id = self.user_simulator.generate_query_with_doc(user_id)
                user_profile = self.user_simulator.get_user_profile(user_id)

                # Test all methods
                baseline_results = self.personalized_retriever.search_baseline(query, 20)
                class_results, _, _ = self.personalized_retriever.search_classification_only(query, 20)
                pers_results = self.personalized_retriever.search_personalized(query, user_profile, 20)
                hybrid_results = self.personalized_retriever.search_hybrid(query, user_profile, 20)

                cold_start_results[bin_label]['baseline'].append(
                    ComprehensiveMetrics.category_precision([r['category'] for r in baseline_results][:10], true_cat))
                cold_start_results[bin_label]['classification'].append(
                    ComprehensiveMetrics.category_precision([r['category'] for r in class_results][:10], true_cat))
                cold_start_results[bin_label]['personalized'].append(
                    ComprehensiveMetrics.category_precision([r['category'] for r in pers_results][:10], true_cat))
                cold_start_results[bin_label]['hybrid'].append(
                    ComprehensiveMetrics.category_precision([r['category'] for r in hybrid_results][:10], true_cat))

        summary = {}
        for label in bin_labels:
            data = cold_start_results[label]
            if data['baseline']:
                summary[label] = {
                    method: {'mean': float(np.mean(data[method])), 'std': float(np.std(data[method]))}
                    for method in ['baseline', 'classification', 'personalized', 'hybrid']
                }
                summary[label]['num_queries'] = len(data['baseline'])

        self.results['cold_start'] = summary

        print(f"\n{'History':<12} {'Baseline':<12} {'Class.':<12} {'Personal.':<12} {'Hybrid':<12}")
        print("-" * 60)
        for label in bin_labels:
            if label in summary:
                s = summary[label]
                print(f"{label:<12} {s['baseline']['mean']:<12.4f} {s['classification']['mean']:<12.4f} "
                      f"{s['personalized']['mean']:<12.4f} {s['hybrid']['mean']:<12.4f}")

        self._plot_cold_start(summary, bin_labels)
        return summary

    def _plot_cold_start(self, summary: Dict, bin_labels: List[str]):
        """绘制冷启动分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        valid_labels = [l for l in bin_labels if l in summary]
        x = np.arange(len(valid_labels))
        width = 0.2

        methods = ['baseline', 'classification', 'personalized', 'hybrid']
        colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']
        labels = ['Baseline', 'Classification', 'Personalized', 'Hybrid']

        for i, (method, color, label) in enumerate(zip(methods, colors, labels)):
            values = [summary[l][method]['mean'] for l in valid_labels]
            axes[0].bar(x + i * width, values, width, label=label, color=color, edgecolor='white')

        axes[0].set_xlabel('User History Size')
        axes[0].set_ylabel('Category Precision@10')
        axes[0].set_title('Performance vs User History (Cold Start)', fontweight='bold')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(valid_labels)
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(axis='y', alpha=0.3)

        # Line plot
        for method, color, mlabel in zip(methods, colors, labels):
            values = [summary[l][method]['mean'] for l in valid_labels]
            axes[1].plot(valid_labels, values, 'o-', color=color, label=mlabel, linewidth=2, markersize=8)

        axes[1].set_xlabel('User History Size')
        axes[1].set_ylabel('Category Precision@10')
        axes[1].set_title('Cold Start Effect: Performance Trend', fontweight='bold')
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp4_cold_start.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 5: User Satisfaction Simulation
    # =========================================================================

    def exp5_user_satisfaction(self, num_sessions: int = 100):
        """实验5: 用户满意度模拟"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: USER SATISFACTION SIMULATION")
        print("=" * 70)

        satisfaction_results = {'baseline': [], 'personalized': []}
        session_metrics = []

        for session_id in tqdm(range(num_sessions), desc="Simulating sessions"):
            user_id = random.choice(list(self.user_simulator.users.keys()))
            user_profile = self.user_simulator.get_user_profile(user_id)

            session_baseline_scores = []
            session_pers_scores = []

            for _ in range(5):
                query, true_cat, source_doc_id = self.user_simulator.generate_query_with_doc(user_id)

                baseline_results = self.personalized_retriever.search_baseline(query, 10)
                baseline_cats = [r['category'] for r in baseline_results]
                baseline_cp = ComprehensiveMetrics.category_precision(baseline_cats, true_cat)
                baseline_hit = ComprehensiveMetrics.hit_rate_at_k(source_doc_id, [r['doc_id'] for r in baseline_results], 10)

                pers_results = self.personalized_retriever.search_personalized(query, user_profile, 10)
                pers_cats = [r['category'] for r in pers_results]
                pers_cp = ComprehensiveMetrics.category_precision(pers_cats, true_cat)
                pers_hit = ComprehensiveMetrics.hit_rate_at_k(source_doc_id, [r['doc_id'] for r in pers_results], 10)

                # 满意度 = 类别精确率 * 0.7 + 命中率 * 0.3
                baseline_sat = 0.7 * baseline_cp + 0.3 * baseline_hit
                pers_sat = 0.7 * pers_cp + 0.3 * pers_hit

                session_baseline_scores.append(baseline_sat)
                session_pers_scores.append(pers_sat)

            satisfaction_results['baseline'].append(np.mean(session_baseline_scores))
            satisfaction_results['personalized'].append(np.mean(session_pers_scores))

            session_metrics.append({
                'session_id': session_id,
                'user_id': user_id,
                'user_type': self.user_simulator.users[user_id]['type'],
                'baseline_satisfaction': np.mean(session_baseline_scores),
                'personalized_satisfaction': np.mean(session_pers_scores)
            })

        summary = {
            'overall': {
                'baseline': {'mean': float(np.mean(satisfaction_results['baseline'])),
                            'std': float(np.std(satisfaction_results['baseline']))},
                'personalized': {'mean': float(np.mean(satisfaction_results['personalized'])),
                                'std': float(np.std(satisfaction_results['personalized']))}
            },
            'improvement': float(np.mean(satisfaction_results['personalized']) - np.mean(satisfaction_results['baseline'])),
            'num_sessions': num_sessions
        }

        session_df = pd.DataFrame(session_metrics)
        by_type = {}
        for ut in session_df['user_type'].unique():
            type_data = session_df[session_df['user_type'] == ut]
            by_type[ut] = {
                'baseline': float(type_data['baseline_satisfaction'].mean()),
                'personalized': float(type_data['personalized_satisfaction'].mean()),
                'count': len(type_data)
            }
        summary['by_user_type'] = by_type

        self.results['user_satisfaction'] = summary

        print(f"\nOverall Satisfaction Scores:")
        print(f"  Baseline:     {summary['overall']['baseline']['mean']:.4f} (+/- {summary['overall']['baseline']['std']:.4f})")
        print(f"  Personalized: {summary['overall']['personalized']['mean']:.4f} (+/- {summary['overall']['personalized']['std']:.4f})")

        imp_pct = summary['improvement'] / summary['overall']['baseline']['mean'] * 100 if summary['overall']['baseline']['mean'] > 0 else 0
        print(f"  Improvement:  {summary['improvement']:.4f} ({imp_pct:+.2f}%)")

        self._plot_user_satisfaction(summary, satisfaction_results)
        return summary

    def _plot_user_satisfaction(self, summary: Dict, raw_results: Dict):
        """绘制用户满意度"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        means = [summary['overall']['baseline']['mean'], summary['overall']['personalized']['mean']]
        stds = [summary['overall']['baseline']['std'], summary['overall']['personalized']['std']]
        bars = axes[0].bar(['Baseline', 'Personalized'], means, yerr=stds,
                          color=['#94a3b8', '#3b82f6'], edgecolor='white', capsize=5)
        axes[0].set_ylabel('Satisfaction Score')
        axes[0].set_title('Overall User Satisfaction', fontweight='bold')
        axes[0].set_ylim(0, 1)
        for bar, mean in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{mean:.3f}', ha='center', fontsize=11, fontweight='bold')

        axes[1].hist(raw_results['baseline'], bins=20, alpha=0.7, label='Baseline', color='#94a3b8')
        axes[1].hist(raw_results['personalized'], bins=20, alpha=0.7, label='Personalized', color='#3b82f6')
        axes[1].set_xlabel('Satisfaction Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Satisfaction Score Distribution', fontweight='bold')
        axes[1].legend()

        by_type = summary['by_user_type']
        user_types = list(by_type.keys())
        x = np.arange(len(user_types))
        width = 0.35

        baseline_vals = [by_type[ut]['baseline'] for ut in user_types]
        pers_vals = [by_type[ut]['personalized'] for ut in user_types]

        axes[2].bar(x - width/2, baseline_vals, width, label='Baseline', color='#94a3b8', edgecolor='white')
        axes[2].bar(x + width/2, pers_vals, width, label='Personalized', color='#3b82f6', edgecolor='white')
        axes[2].set_xlabel('User Type')
        axes[2].set_ylabel('Satisfaction Score')
        axes[2].set_title('Satisfaction by User Type', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([t.capitalize() for t in user_types])
        axes[2].legend()
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp5_user_satisfaction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 6: Ablation Study
    # =========================================================================

    def exp6_ablation(self, num_queries: int = 100):
        """实验6: 消融实验"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 6: ABLATION STUDY")
        print("=" * 70)

        test_queries = self._generate_test_queries(num_queries)

        # 1. 置信度阈值
        print("\n1. Confidence Threshold Ablation:")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        threshold_results = {}

        for threshold in thresholds:
            cps = []
            for query, true_cat, source_doc_id, user_id in test_queries:
                user_profile = self.user_simulator.get_user_profile(user_id)
                results = self.personalized_retriever.search_hybrid(query, user_profile, 20, conf_threshold=threshold)
                cp = ComprehensiveMetrics.category_precision([r['category'] for r in results][:10], true_cat)
                cps.append(cp)
            threshold_results[threshold] = {'mean': float(np.mean(cps)), 'std': float(np.std(cps))}
            print(f"  Threshold {threshold}: Category P@10 = {np.mean(cps):.4f}")

        # 2. Alpha参数
        print("\n2. Personalization Alpha Ablation:")
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        alpha_results = {}

        for alpha in alphas:
            cps = []
            for query, true_cat, source_doc_id, user_id in test_queries:
                user_profile = self.user_simulator.get_user_profile(user_id)
                results = self.personalized_retriever.search_personalized(query, user_profile, 20, alpha=alpha)
                cp = ComprehensiveMetrics.category_precision([r['category'] for r in results][:10], true_cat)
                cps.append(cp)
            alpha_results[alpha] = {'mean': float(np.mean(cps)), 'std': float(np.std(cps))}
            print(f"  Alpha {alpha}: Category P@10 = {np.mean(cps):.4f}")

        self.results['ablation'] = {
            'confidence_threshold': threshold_results,
            'personalization_alpha': alpha_results
        }

        self._plot_ablation(threshold_results, alpha_results, thresholds, alphas)
        return self.results['ablation']

    def _plot_ablation(self, threshold_results, alpha_results, thresholds, alphas):
        """绘制消融实验"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        maps = [threshold_results[t]['mean'] for t in thresholds]
        stds = [threshold_results[t]['std'] for t in thresholds]
        axes[0].errorbar(thresholds, maps, yerr=stds, fmt='o-', color='#3b82f6',
                        linewidth=2, markersize=10, capsize=5)
        axes[0].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('Category Precision@10')
        axes[0].set_title('Ablation: Confidence Threshold', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        best_idx = np.argmax(maps)
        axes[0].scatter([thresholds[best_idx]], [maps[best_idx]], color='#ef4444', s=200, zorder=5, marker='*')

        maps = [alpha_results[a]['mean'] for a in alphas]
        stds = [alpha_results[a]['std'] for a in alphas]
        axes[1].errorbar(alphas, maps, yerr=stds, fmt='s-', color='#10b981',
                        linewidth=2, markersize=10, capsize=5)
        axes[1].set_xlabel('Personalization Alpha')
        axes[1].set_ylabel('Category Precision@10')
        axes[1].set_title('Ablation: Personalization Weight', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        best_idx = np.argmax(maps)
        axes[1].scatter([alphas[best_idx]], [maps[best_idx]], color='#ef4444', s=200, zorder=5, marker='*')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp6_ablation.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # EXPERIMENT 7: Error Analysis
    # =========================================================================

    def exp7_error_analysis(self, num_cases: int = 100):
        """实验7: 错误分析"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 7: ERROR ANALYSIS")
        print("=" * 70)

        success_cases = []
        failure_cases = []
        all_cases = []

        test_queries = self._generate_test_queries(num_cases)

        for query, true_cat, source_doc_id, user_id in test_queries:
            user_profile = self.user_simulator.get_user_profile(user_id)

            # 分类
            pred_cat, probs = self.classifier.predict(query)
            pred_name = self.category_mapping.get(pred_cat[0], "")
            conf = probs[0][pred_cat[0]] if probs is not None else 0

            # Baseline
            baseline_results = self.personalized_retriever.search_baseline(query, 10)
            baseline_cp = ComprehensiveMetrics.category_precision([r['category'] for r in baseline_results], true_cat)

            # Personalized
            pers_results = self.personalized_retriever.search_personalized(query, user_profile, 10)
            pers_cp = ComprehensiveMetrics.category_precision([r['category'] for r in pers_results], true_cat)

            case = {
                'query': query[:60],
                'true_category': true_cat,
                'predicted_category': pred_name,
                'confidence': float(conf),
                'classification_correct': pred_name == true_cat,
                'baseline_cp': float(baseline_cp),
                'personalized_cp': float(pers_cp),
                'improvement': float(pers_cp - baseline_cp),
                'user_type': self.user_simulator.users[user_id]['type']
            }
            all_cases.append(case)

            if case['classification_correct'] and case['personalized_cp'] > case['baseline_cp']:
                success_cases.append(case)
            elif not case['classification_correct'] or case['personalized_cp'] < case['baseline_cp']:
                failure_cases.append(case)

        # 分析错误模式
        error_patterns = defaultdict(int)
        for case in failure_cases:
            if not case['classification_correct']:
                error_patterns['Classification Error'] += 1
            if case['confidence'] < 0.5:
                error_patterns['Low Confidence'] += 1
            if case['user_type'] == 'new':
                error_patterns['Cold Start User'] += 1
            if case['improvement'] < -0.3:
                error_patterns['Severe Degradation'] += 1

        # 按类别分析
        category_analysis = defaultdict(lambda: {'correct': 0, 'total': 0, 'baseline_cp': [], 'pers_cp': []})
        for case in all_cases:
            cat = case['true_category']
            category_analysis[cat]['total'] += 1
            if case['classification_correct']:
                category_analysis[cat]['correct'] += 1
            category_analysis[cat]['baseline_cp'].append(case['baseline_cp'])
            category_analysis[cat]['pers_cp'].append(case['personalized_cp'])

        cat_summary = {}
        for cat, data in category_analysis.items():
            cat_summary[cat] = {
                'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                'baseline_cp': float(np.mean(data['baseline_cp'])),
                'personalized_cp': float(np.mean(data['pers_cp'])),
                'count': data['total']
            }

        self.results['error_analysis'] = {
            'success_cases': success_cases[:10],
            'failure_cases': failure_cases[:10],
            'error_patterns': dict(error_patterns),
            'category_analysis': cat_summary,
            'stats': {
                'success_count': len(success_cases),
                'failure_count': len(failure_cases),
                'overall_classification_accuracy': sum(1 for c in all_cases if c['classification_correct']) / len(all_cases)
            }
        }

        print(f"\nError Analysis Summary:")
        print(f"  Total cases: {len(all_cases)}")
        print(f"  Success cases: {len(success_cases)}")
        print(f"  Failure cases: {len(failure_cases)}")
        print(f"  Classification accuracy: {self.results['error_analysis']['stats']['overall_classification_accuracy']:.2%}")

        print(f"\nError Patterns:")
        for pattern, count in error_patterns.items():
            print(f"  {pattern}: {count}")

        print(f"\nCategory Analysis:")
        for cat, data in cat_summary.items():
            print(f"  {cat}: Acc={data['accuracy']:.2%}, Baseline_CP={data['baseline_cp']:.3f}, Pers_CP={data['personalized_cp']:.3f}")

        self._plot_error_analysis(error_patterns, cat_summary)
        return self.results['error_analysis']

    def _plot_error_analysis(self, error_patterns: Dict, category_analysis: Dict):
        """绘制错误分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if error_patterns:
            patterns = list(error_patterns.keys())
            counts = list(error_patterns.values())
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(patterns)))
            bars = axes[0].barh(patterns, counts, color=colors, edgecolor='white')
            axes[0].set_xlabel('Count')
            axes[0].set_title('Error Pattern Distribution', fontweight='bold')
            for bar, count in zip(bars, counts):
                axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(count), va='center')

        # Category accuracy
        categories = list(category_analysis.keys())
        accuracies = [category_analysis[c]['accuracy'] for c in categories]
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        bars = axes[1].bar(categories, accuracies, color=colors, edgecolor='white')
        axes[1].set_ylabel('Classification Accuracy')
        axes[1].set_title('Classification Accuracy by Category', fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        for bar, acc in zip(bars, accuracies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'exp7_error_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_test_queries(self, num_queries: int) -> List[Tuple[str, str, int, int]]:
        """生成测试查询: (query, true_category, source_doc_id, user_id)"""
        queries = []
        user_ids = list(self.user_simulator.users.keys())

        for _ in range(num_queries):
            user_id = random.choice(user_ids)
            query, true_cat, source_doc_id = self.user_simulator.generate_query_with_doc(user_id)
            queries.append((query, true_cat, source_doc_id, user_id))

        return queries

    # =========================================================================
    # Main Runner
    # =========================================================================

    def run_all_experiments(self):
        """运行所有实验"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EXPERIMENT SUITE v2.0")
        print(f"Output Directory: {self.output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        self.setup()

        self.exp1_classification()
        self.exp2_retrieval_baselines(num_queries=200)
        self.exp3_personalization_analysis(num_queries=200)
        self.exp4_cold_start(num_queries=150)
        self.exp5_user_satisfaction(num_sessions=100)
        self.exp6_ablation(num_queries=100)
        self.exp7_error_analysis(num_cases=100)

        self._save_comprehensive_report()
        self._plot_summary_dashboard()

        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def _save_comprehensive_report(self):
        """保存综合报告"""
        report = {
            'experiment_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'system': 'Text Classification & Personalized Retrieval System',
                'version': '2.0'
            },
            'dataset': self.results['dataset_info'],
            'experiments': {
                'exp1_classification': self.results['classification'],
                'exp2_retrieval_baselines': self.results['retrieval_baselines'],
                'exp3_personalization': self.results['personalization'],
                'exp4_cold_start': self.results['cold_start'],
                'exp5_user_satisfaction': self.results['user_satisfaction'],
                'exp6_ablation': self.results['ablation'],
                'exp7_error_analysis': self.results['error_analysis']
            }
        }

        path = os.path.join(self.output_dir, 'comprehensive_experiment_report.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nReport saved: {path}")

    def _plot_summary_dashboard(self):
        """绘制总结仪表板"""
        fig = plt.figure(figsize=(20, 12))

        # 1. 分类性能
        ax1 = fig.add_subplot(2, 3, 1)
        class_metrics = self.results['classification']['overall_metrics']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [class_metrics['accuracy'], class_metrics['precision_macro'],
                 class_metrics['recall_macro'], class_metrics['f1_macro']]
        bars = ax1.bar(metrics, values, color=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'], edgecolor='white')
        ax1.set_ylim(0, 1)
        ax1.set_title('Classification Performance', fontweight='bold', fontsize=12)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)

        # 2. 检索对比
        ax2 = fig.add_subplot(2, 3, 2)
        baselines = self.results['retrieval_baselines']
        systems = ['BM25 (No Filter)', 'BM25 + Classification', 'BM25 + Personalization', 'Oracle (Perfect)']
        short_names = ['BM25', 'BM25+Class', 'Personal.', 'Oracle']
        cp_values = [baselines[s]['Category_P@10']['mean'] for s in systems]
        colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']
        bars = ax2.bar(short_names, cp_values, color=colors, edgecolor='white')
        ax2.set_ylabel('Category P@10')
        ax2.set_title('Retrieval: Category Precision', fontweight='bold', fontsize=12)
        ax2.set_ylim(0, 1.1)
        for bar, val in zip(bars, cp_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)

        # 3. 用户类型对比
        ax3 = fig.add_subplot(2, 3, 3)
        pers = self.results['personalization']
        user_types = list(pers.keys())
        pers_cps = [pers[ut]['personalized_CP']['mean'] for ut in user_types]
        bars = ax3.bar(user_types, pers_cps, color='#3b82f6', edgecolor='white')
        ax3.set_ylabel('Category P@10')
        ax3.set_title('Personalization by User Type', fontweight='bold', fontsize=12)
        ax3.set_ylim(0, 1.1)

        # 4. 冷启动
        ax4 = fig.add_subplot(2, 3, 4)
        cold = self.results['cold_start']
        history_labels = list(cold.keys())
        pers_cold = [cold[l]['personalized']['mean'] for l in history_labels]
        ax4.plot(range(len(history_labels)), pers_cold, 'o-', color='#3b82f6', linewidth=2, markersize=8)
        ax4.set_xticks(range(len(history_labels)))
        ax4.set_xticklabels(history_labels)
        ax4.set_ylabel('Category P@10')
        ax4.set_title('Cold Start Effect', fontweight='bold', fontsize=12)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)

        # 5. 用户满意度
        ax5 = fig.add_subplot(2, 3, 5)
        sat = self.results['user_satisfaction']
        sat_data = [sat['overall']['baseline']['mean'], sat['overall']['personalized']['mean']]
        bars = ax5.bar(['Baseline', 'Personalized'], sat_data, color=['#94a3b8', '#3b82f6'], edgecolor='white')
        ax5.set_ylabel('Satisfaction Score')
        ax5.set_title('User Satisfaction', fontweight='bold', fontsize=12)
        ax5.set_ylim(0, 1)
        for bar, val in zip(bars, sat_data):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=10)

        # 6. 关键发现
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')

        baseline_cp = baselines['BM25 (No Filter)']['Category_P@10']['mean']
        pers_cp = baselines['BM25 + Personalization']['Category_P@10']['mean']
        oracle_cp = baselines['Oracle (Perfect)']['Category_P@10']['mean']
        improvement = (pers_cp - baseline_cp) / baseline_cp * 100 if baseline_cp > 0 else 0

        findings = f"""
KEY FINDINGS SUMMARY
{'='*40}

CLASSIFICATION:
  Accuracy:     {class_metrics['accuracy']:.2%}
  F1-Score:     {class_metrics['f1_macro']:.4f}

RETRIEVAL (Category Precision@10):
  Baseline:      {baseline_cp:.4f}
  Personalized:  {pers_cp:.4f}
  Improvement:   {improvement:+.2f}%
  Oracle:        {oracle_cp:.4f}

USER SATISFACTION:
  Baseline:     {sat['overall']['baseline']['mean']:.4f}
  Personalized: {sat['overall']['personalized']['mean']:.4f}

DATASET:
  Documents: {self.results['dataset_info']['total_documents']:,}
  Categories: {self.results['dataset_info']['num_categories']}
  Simulated Users: {self.results['dataset_info']['simulated_users']}
        """
        ax6.text(0.05, 0.95, findings, fontsize=10, verticalalignment='top',
                fontfamily='monospace', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#e2e8f0', pad=0.5))

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'experiment_dashboard.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dashboard saved: {path}")


def main():
    runner = ComprehensiveExperiments()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
