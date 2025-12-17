# evaluation/metrics.py - 评估指标

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

class RetrievalMetrics:
    """检索评估指标"""
    
    @staticmethod
    def precision_at_k(relevant_docs, retrieved_docs, k=10):
        """计算Precision@K"""
        if len(retrieved_docs) == 0:
            return 0.0
        
        # 取前K个结果
        top_k = retrieved_docs[:k]
        
        # 计算相关文档数
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
        
        return relevant_in_top_k / len(top_k)
    
    @staticmethod
    def recall_at_k(relevant_docs, retrieved_docs, k=10):
        """计算Recall@K"""
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
        
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def average_precision(relevant_docs, retrieved_docs):
        """计算平均精确率 (Average Precision)"""
        if len(relevant_docs) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                relevant_count += 1
                precision_sum += relevant_count / i
        
        return precision_sum / len(relevant_docs)
    
    @staticmethod
    def mean_average_precision(query_results):
        """计算平均精确率均值 (MAP)"""
        if not query_results:
            return 0.0
        
        ap_sum = 0.0
        for relevant_docs, retrieved_docs in query_results:
            ap_sum += RetrievalMetrics.average_precision(relevant_docs, retrieved_docs)
        
        return ap_sum / len(query_results)
    
    @staticmethod
    def normalized_dcg(relevant_docs, retrieved_docs, k=10, relevancy_scores=None):
        """计算NDCG@K"""
        # 默认相关度分数：相关文档=1，不相关=0
        if relevancy_scores is None:
            relevancy_scores = {doc: 1 for doc in relevant_docs}
        
        # 实际DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], 1):
            rel = relevancy_scores.get(doc, 0)
            dcg += rel / np.log2(i + 1)
        
        # 理想DCG
        ideal_relevancies = sorted(relevancy_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevancies, 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def plot_precision_recall_curve(relevant_docs, retrieved_docs, save_path=None):
        """绘制精确率-召回率曲线"""
        precisions = []
        recalls = []
        
        relevant_set = set(relevant_docs)
        
        for k in range(1, len(retrieved_docs) + 1):
            retrieved_k = retrieved_docs[:k]
            relevant_in_k = len([d for d in retrieved_k if d in relevant_set])
            
            precision = relevant_in_k / k if k > 0 else 0
            recall = relevant_in_k / len(relevant_docs) if len(relevant_docs) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return recalls, precisions

class ClassificationMetrics:
    """分类评估指标"""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, average='weighted'):
        """计算分类指标"""
        metrics = {
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average),
            'accuracy': np.mean(y_true == y_pred)
        }
        return metrics
    
    @staticmethod
    def plot_class_distribution(y_true, class_names, save_path=None):
        """绘制类别分布图"""
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([class_names[i] for i in unique], counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()