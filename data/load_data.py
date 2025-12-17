# data/load_data.py - 数据加载和预处理

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import joblib
import os
from config import DATASET_CONFIG

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, use_cached=True):
        self.use_cached = use_cached
        self.cache_file = os.path.join('data', 'processed', 'dataset.pkl')
        
    def load_20newsgroups(self):
        """加载20 Newsgroups数据集"""
        if self.use_cached and os.path.exists(self.cache_file):
            print("Loading cached dataset...")
            return joblib.load(self.cache_file)
        
        print("Downloading 20 Newsgroups dataset...")
        # 加载完整数据集
        newsgroups = fetch_20newsgroups(
            subset='all',
            categories=DATASET_CONFIG['categories'],
            remove=DATASET_CONFIG['remove'],
            shuffle=True,
            random_state=42
        )
        
        # 转换为DataFrame
        df = pd.DataFrame({
            'text': newsgroups.data,
            'target': newsgroups.target,
            'target_name': [newsgroups.target_names[t] for t in newsgroups.target],
            'id': range(len(newsgroups.data))
        })
        
        # 清理文本
        df['text_clean'] = df['text'].apply(self._clean_text)
        
        # 保存缓存
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        joblib.dump(df, self.cache_file)
        
        print(f"Dataset loaded: {len(df)} documents")
        return df
    
    def _clean_text(self, text):
        """清理文本"""
        import re
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_dataset(self, df, test_size=0.2):
        """划分训练集和测试集"""
        from sklearn.model_selection import train_test_split
        
        # 为检索系统保留所有数据
        retrieval_df = df.copy()
        
        # 为分类器划分训练/测试集
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['target']  # 保持类别比例
        )
        
        print(f"Training set: {len(train_df)} documents")
        print(f"Testing set: {len(test_df)} documents")
        
        return retrieval_df, train_df, test_df
    
    def load_sample_queries(self):
        """加载示例查询"""
        queries = {
            'comp.graphics': [
                "3D graphics rendering techniques",
                "image processing algorithms",
                "computer animation software"
            ],
            'rec.sport.hockey': [
                "NHL playoff results",
                "hockey equipment reviews",
                "ice hockey training techniques"
            ],
            'sci.space': [
                "NASA missions to Mars",
                "black hole theories",
                "international space station"
            ],
            'talk.politics.mideast': [
                "Middle East peace process",
                "Israel Palestine conflict",
                "oil politics in Gulf region"
            ]
        }
        return queries

# 使用示例
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_20newsgroups()
    print(df.head())
    print(f"\nCategories: {df['target_name'].unique()[:5]}...")