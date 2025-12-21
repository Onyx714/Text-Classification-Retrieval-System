# classification/classifier_model.py - 文本分类器

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLASSIFIER_CONFIG

class TextClassifier:
    """文本分类器类"""
    
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = None
        self.classifier = None
        self.classes_ = None
        
    def create_vectorizer(self, max_features=10000):
        """创建TF-IDF向量化器"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # 使用1-2个词
            min_df=2,
            max_df=0.8
        )
        return self.vectorizer
    
    def create_classifier(self):
        """创建分类器"""
        if self.model_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=0.1)
        elif self.model_type == 'svm':
            self.classifier = LinearSVC(C=1.0, random_state=42, max_iter=1000)
        elif self.model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.classifier
    
    def train(self, X_train, y_train):
        """训练分类器"""
        # 文本向量化
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # 训练分类器
        print(f"Training {self.model_type} classifier...")
        self.classifier.fit(X_train_vec, y_train)
        
        self.classes_ = self.classifier.classes_
        
        # 在训练集上评估
        train_pred = self.classifier.predict(X_train_vec)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, texts):
        """预测文本类别"""
        if not isinstance(texts, list):
            texts = [texts]
        
        texts_vec = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(texts_vec)
        probabilities = self.classifier.predict_proba(texts_vec) if hasattr(self.classifier, "predict_proba") else None
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test, save_report=True):
        """评估分类器性能"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vec)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # 保存评估结果
        if save_report:
            import os
            model_dir = 'classification/models'
            os.makedirs(model_dir, exist_ok=True)
            eval_results = {
                'accuracy': accuracy,
                'report': report,
                'y_true': y_test,
                'y_pred': y_pred
            }
            joblib.dump(eval_results, os.path.join(model_dir, 'evaluation.pkl'))
        
        return accuracy, report
    
    def save_model(self, model_dir='classification/models'):
        """保存模型"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.pkl')
        joblib.dump(self.classifier, f'{model_dir}/classifier.pkl')
        joblib.dump(self.classes_, f'{model_dir}/classes.pkl')
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='classification/models'):
        """加载模型"""
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.pkl')
        self.classifier = joblib.load(f'{model_dir}/classifier.pkl')
        self.classes_ = joblib.load(f'{model_dir}/classes.pkl')
        
        print(f"Model loaded from {model_dir}")
        return self
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else self.classes_,
                   yticklabels=class_names if class_names else self.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('classification/confusion_matrix.png', dpi=300)
        plt.close()