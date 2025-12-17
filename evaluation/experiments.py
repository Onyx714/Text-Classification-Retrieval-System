# evaluation/experiments.py - 实验脚本

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import joblib
import json
from tqdm import tqdm
from datetime import datetime

from data.load_data import DataLoader
from classification.classifier_model import TextClassifier
from retrieval.searcher import DocumentSearcher
from evaluation.metrics import RetrievalMetrics, ClassificationMetrics

class SystemEvaluator:
    """系统评估器"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.classifier = None
        self.searcher = None
        
    def load_components(self):
        """加载所有组件"""
        # 加载分类器
        self.classifier = TextClassifier()
        self.classifier.load_model()
        
        # 加载检索器
        self.searcher = DocumentSearcher()
        self.searcher.open_index()
        
        # 加载数据
        self.df = self.loader.load_20newsgroups()
        
        # 加载类别映射
        self.category_mapping = joblib.load('classification/models/category_mapping.pkl')
        
    def evaluate_classification(self):
        """评估分类器"""
        print("Evaluating classifier...")
        
        # 获取测试数据（后20%）
        train_size = int(len(self.df) * 0.8)
        test_df = self.df.iloc[train_size:]
        
        X_test = test_df['text_clean'].tolist()
        y_test = test_df['target'].tolist()
        
        # 预测
        y_pred, _ = self.classifier.predict(X_test)
        
        # 计算指标
        metrics = ClassificationMetrics.compute_metrics(y_test, y_pred)
        
        print("\nClassification Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 绘制混淆矩阵
        self.classifier.plot_confusion_matrix(
            y_test, y_pred, 
            class_names=list(self.category_mapping.values())
        )
        
        return metrics
    
    def evaluate_retrieval(self, num_queries=50):
        """评估检索系统"""
        print(f"\nEvaluating retrieval with {num_queries} queries...")
        
        # 生成测试查询
        test_queries = self._generate_test_queries(num_queries)
        
        results = {
            'baseline': {'precision@5': [], 'precision@10': [], 'recall@10': [], 'map': []},
            'proposed': {'precision@5': [], 'precision@10': [], 'recall@10': [], 'map': []}
        }
        
        # 对每个查询进行评估
        for i, (query, true_category) in enumerate(tqdm(test_queries)):
            # 获取相关文档（同一类别的所有文档）
            relevant_docs = self.df[self.df['target_name'] == true_category]['id'].tolist()
            
            # 1. Baseline: 无分类过滤
            baseline_results = self.searcher.search(query, max_results=20)
            baseline_docs = [r['doc_id'] for r in baseline_results]
            
            # 2. Proposed: 有分类过滤
            proposed_results = self.searcher.search(query, true_category, max_results=20)
            proposed_docs = [r['doc_id'] for r in proposed_results]
            
            # 计算指标
            # Baseline
            results['baseline']['precision@5'].append(
                RetrievalMetrics.precision_at_k(relevant_docs, baseline_docs, k=5)
            )
            results['baseline']['precision@10'].append(
                RetrievalMetrics.precision_at_k(relevant_docs, baseline_docs, k=10)
            )
            results['baseline']['recall@10'].append(
                RetrievalMetrics.recall_at_k(relevant_docs, baseline_docs, k=10)
            )
            results['baseline']['map'].append(
                RetrievalMetrics.average_precision(relevant_docs, baseline_docs)
            )
            
            # Proposed
            results['proposed']['precision@5'].append(
                RetrievalMetrics.precision_at_k(relevant_docs, proposed_docs, k=5)
            )
            results['proposed']['precision@10'].append(
                RetrievalMetrics.precision_at_k(relevant_docs, proposed_docs, k=10)
            )
            results['proposed']['recall@10'].append(
                RetrievalMetrics.recall_at_k(relevant_docs, proposed_docs, k=10)
            )
            results['proposed']['map'].append(
                RetrievalMetrics.average_precision(relevant_docs, proposed_docs)
            )
        
        # 计算平均指标
        summary = {}
        for system in ['baseline', 'proposed']:
            summary[system] = {
                metric: np.mean(values) 
                for metric, values in results[system].items()
            }
        
        # 打印结果
        print("\nRetrieval Evaluation Results:")
        print("=" * 60)
        print(f"{'Metric':<15} {'Baseline':<12} {'Proposed':<12} {'Improvement':<12}")
        print("-" * 60)
        
        for metric in ['precision@5', 'precision@10', 'recall@10', 'map']:
            baseline_val = summary['baseline'][metric]
            proposed_val = summary['proposed'][metric]
            improvement = ((proposed_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            print(f"{metric:<15} {baseline_val:.4f}{'':<5} {proposed_val:.4f}{'':<5} {improvement:+.1f}%")
        
        print("=" * 60)
        
        # 保存结果
        self._save_results(summary, results)
        
        return summary, results
    
    def _generate_test_queries(self, num_queries):
        """生成测试查询"""
        queries = []
        
        # 从每个类别生成一些查询
        categories = self.df['target_name'].unique()
        queries_per_category = max(1, num_queries // len(categories))
        
        for category in categories:
            # 获取该类别的文档
            category_docs = self.df[self.df['target_name'] == category]
            
            if len(category_docs) > 0:
                # 从文档中提取关键词作为查询
                sample_docs = category_docs.sample(min(queries_per_category, len(category_docs)))
                
                for _, doc in sample_docs.iterrows():
                    # 提取文档中的前几个词作为查询
                    words = doc['text_clean'].split()[:5]
                    query = ' '.join(words)
                    
                    if len(query) > 3:  # 确保查询有意义
                        queries.append((query, category))
        
        # 如果不够，随机生成一些
        if len(queries) < num_queries:
            additional = num_queries - len(queries)
            for _ in range(additional):
                doc = self.df.sample(1).iloc[0]
                words = doc['text_clean'].split()[:5]
                query = ' '.join(words)
                queries.append((query, doc['target_name']))
        
        return queries[:num_queries]
    
    def _save_results(self, summary, detailed_results):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存摘要
        summary_file = f'evaluation/results_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存详细结果（只保存部分以减少文件大小）
        results_file = f'evaluation/detailed_results_{timestamp}.pkl'
        joblib.dump({
            'summary': summary,
            'sample_queries': detailed_results.get('sample_queries', []),
            'timestamp': timestamp
        }, results_file)
        
        print(f"\nResults saved to:\n  {summary_file}\n  {results_file}")
    
    def run_complete_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 60)
        
        # 加载组件
        self.load_components()
        
        # 1. 评估分类器
        print("\n[1/3] CLASSIFICATION EVALUATION")
        class_metrics = self.evaluate_classification()
        
        # 2. 评估检索系统
        print("\n[2/3] RETRIEVAL EVALUATION")
        retrieval_summary, _ = self.evaluate_retrieval(num_queries=100)
        
        # 3. 端到端评估
        print("\n[3/3] END-TO-END EVALUATION")
        end_to_end_results = self._evaluate_end_to_end()
        
        # 生成报告
        report = self._generate_report(class_metrics, retrieval_summary, end_to_end_results)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED!")
        print("=" * 60)
        
        return report
    
    def _evaluate_end_to_end(self, num_cases=20):
        """端到端评估"""
        print(f"Testing end-to-end system with {num_cases} queries...")
        
        cases = []
        
        # 从每个类别选择一些文档作为查询来源
        for category in list(self.category_mapping.values())[:5]:  # 测试前5个类别
            category_docs = self.df[self.df['target_name'] == category]
            
            if len(category_docs) > 0:
                for _, doc in category_docs.sample(min(2, len(category_docs))).iterrows():
                    # 从文档中提取查询
                    query_words = doc['text_clean'].split()[:5]
                    query = ' '.join(query_words)
                    
                    # 使用系统处理
                    predicted_category, _ = self.classifier.predict(query)
                    predicted_name = self.category_mapping.get(predicted_category[0], "Unknown")
                    
                    # 搜索
                    results = self.searcher.search(query, predicted_name, max_results=5)
                    
                    cases.append({
                        'query': query,
                        'true_category': category,
                        'predicted_category': predicted_name,
                        'correct_prediction': (category == predicted_name),
                        'results_count': len(results),
                        'results': results[:3]  # 只保存前3个结果
                    })
        
        # 计算准确率
        correct_predictions = sum(1 for case in cases if case['correct_prediction'])
        accuracy = correct_predictions / len(cases) if cases else 0
        
        print(f"  Category prediction accuracy: {accuracy:.2%}")
        print(f"  Average results per query: {np.mean([c['results_count'] for c in cases]):.1f}")
        
        return {
            'cases': cases[:10],  # 只返回前10个案例
            'prediction_accuracy': accuracy,
            'total_cases': len(cases)
        }
    
    def _generate_report(self, class_metrics, retrieval_summary, end_to_end_results):
        """生成评估报告"""
        report = {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'classification': class_metrics,
            'retrieval': retrieval_summary,
            'end_to_end': end_to_end_results,
            'improvement': {}
        }
        
        # 计算改进百分比
        for metric in ['precision@5', 'precision@10', 'recall@10', 'map']:
            baseline = retrieval_summary['baseline'][metric]
            proposed = retrieval_summary['proposed'][metric]
            if baseline > 0:
                report['improvement'][metric] = {
                    'absolute': proposed - baseline,
                    'percentage': (proposed - baseline) / baseline * 100
                }
        
        # 保存报告
        report_file = f'evaluation/full_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nFull report saved to: {report_file}")
        
        return report

def main():
    """主评估函数"""
    evaluator = SystemEvaluator()
    report = evaluator.run_complete_evaluation()
    
    # 打印关键发现
    print("\nKEY FINDINGS:")
    print("-" * 40)
    print(f"1. Classification Accuracy: {report['classification']['accuracy']:.2%}")
    print(f"2. Retrieval Improvement (P@10): {report['improvement'].get('precision@10', {}).get('percentage', 0):+.1f}%")
    print(f"3. End-to-end Prediction Accuracy: {report['end_to_end']['prediction_accuracy']:.2%}")

if __name__ == "__main__":
    main()