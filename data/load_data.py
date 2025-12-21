# data/load_data.py - 数据加载和预处理

import pandas as pd
import numpy as np
import joblib
import os
import csv
from config import DATASET_CONFIG


class DataLoader:
    """数据加载器类 - AG News Dataset"""

    def __init__(self, use_cached=True):
        self.use_cached = use_cached
        self.cache_file = os.path.join('data', 'processed', 'dataset.pkl')
        self.data_dir = os.path.join('data', 'ag_news')

    def load_ag_news(self):
        """加载AG News数据集"""
        if self.use_cached and os.path.exists(self.cache_file):
            print("Loading cached dataset...")
            return joblib.load(self.cache_file)

        print("Loading AG News dataset...")

        # 检查本地数据文件
        train_file = os.path.join(self.data_dir, 'train.csv')
        test_file = os.path.join(self.data_dir, 'test.csv')

        if not os.path.exists(train_file):
            print("Downloading AG News dataset...")
            self._download_ag_news()

        # 类别映射 (AG News: 1=World, 2=Sports, 3=Business, 4=Sci/Tech)
        category_map = {
            1: 'World',
            2: 'Sports',
            3: 'Business',
            4: 'Sci/Tech'
        }

        # 加载训练和测试数据
        train_df = self._load_csv(train_file, category_map)
        test_df = self._load_csv(test_file, category_map)

        # 合并数据
        df = pd.concat([train_df, test_df], ignore_index=True)

        # 采样（如果配置了num_samples）
        num_samples = DATASET_CONFIG.get('num_samples')
        if num_samples:
            sampled_dfs = []
            for cat in category_map.values():
                cat_df = df[df['target_name'] == cat]
                n = min(num_samples, len(cat_df))
                sampled_dfs.append(cat_df.sample(n=n, random_state=42))
            df = pd.concat(sampled_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 重新分配ID
        df['id'] = range(len(df))

        # 创建target数值列
        target_to_id = {name: idx for idx, name in enumerate(category_map.values())}
        df['target'] = df['target_name'].map(target_to_id)

        # 清理文本
        df['text_clean'] = df['text'].apply(self._clean_text)

        # 保存缓存
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        joblib.dump(df, self.cache_file)

        print(f"Dataset loaded: {len(df)} documents")
        print(f"Categories: {df['target_name'].unique().tolist()}")
        return df

    def _load_csv(self, filepath, category_map):
        """加载CSV文件"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    label = int(row[0])
                    title = row[1]
                    description = row[2]
                    text = f"{title} {description}"
                    data.append({
                        'text': text,
                        'target_name': category_map.get(label, 'Unknown')
                    })
        return pd.DataFrame(data)

    def _download_ag_news(self):
        """下载AG News数据集"""
        import urllib.request
        import tarfile

        os.makedirs(self.data_dir, exist_ok=True)

        # AG News数据集URL
        urls = {
            'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
            'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
        }

        for name, url in urls.items():
            filepath = os.path.join(self.data_dir, f'{name}.csv')
            print(f"Downloading {name}.csv...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  Saved to {filepath}")
            except Exception as e:
                print(f"  Download failed: {e}")
                print("  Creating sample data instead...")
                self._create_sample_data()
                return

    def _create_sample_data(self):
        """创建示例数据（如果下载失败）"""
        os.makedirs(self.data_dir, exist_ok=True)

        # 示例数据
        sample_data = [
            # World
            (1, "Global Summit", "World leaders meet to discuss climate change policies and international cooperation"),
            (1, "Peace Treaty", "Historic agreement signed between nations ending decades of conflict"),
            (1, "UN Resolution", "United Nations passes new resolution on humanitarian aid"),
            (1, "Election Results", "Country announces new government after democratic elections"),
            (1, "Diplomatic Relations", "Two countries restore diplomatic ties after years of tension"),
            # Sports
            (2, "Championship Finals", "Team wins national championship in thrilling overtime victory"),
            (2, "Olympic Games", "Athletes prepare for upcoming summer Olympic games"),
            (2, "Soccer Match", "Football team advances to semifinals after penalty shootout"),
            (2, "Tennis Tournament", "Player wins grand slam title in straight sets"),
            (2, "Basketball League", "NBA team secures playoff spot with crucial win"),
            # Business
            (3, "Stock Market", "Markets reach record highs amid positive economic data"),
            (3, "Company Merger", "Tech giants announce billion dollar acquisition deal"),
            (3, "Quarterly Earnings", "Corporation reports strong profits exceeding expectations"),
            (3, "Trade Agreement", "New trade deal opens markets for international business"),
            (3, "Startup Funding", "Venture capital investment reaches new heights this year"),
            # Sci/Tech
            (4, "AI Breakthrough", "Researchers develop new artificial intelligence system"),
            (4, "Space Mission", "NASA launches new satellite for deep space exploration"),
            (4, "Medical Discovery", "Scientists find promising treatment for disease"),
            (4, "Tech Innovation", "Company unveils revolutionary new smartphone technology"),
            (4, "Climate Research", "Study reveals new findings about global warming effects"),
        ]

        # 扩展数据
        expanded_data = []
        for i in range(50):  # 创建更多样本
            for item in sample_data:
                expanded_data.append((item[0], f"{item[1]} {i}", f"{item[2]} Additional context {i}"))

        # 写入训练文件
        train_path = os.path.join(self.data_dir, 'train.csv')
        with open(train_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in expanded_data[:800]:
                writer.writerow(item)

        # 写入测试文件
        test_path = os.path.join(self.data_dir, 'test.csv')
        with open(test_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in expanded_data[800:]:
                writer.writerow(item)

        print(f"Sample data created in {self.data_dir}")

    def _clean_text(self, text):
        """清理文本"""
        import re

        if not isinstance(text, str):
            return ""

        # 转换为小写
        text = text.lower()

        # 移除特殊字符但保留基本标点
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
            stratify=df['target']
        )

        print(f"Training set: {len(train_df)} documents")
        print(f"Testing set: {len(test_df)} documents")

        return retrieval_df, train_df, test_df

    def load_sample_queries(self):
        """加载示例查询 - AG News categories"""
        queries = {
            'World': [
                "international diplomacy summit",
                "global peace negotiations",
                "United Nations resolution"
            ],
            'Sports': [
                "championship game results",
                "Olympic athletes training",
                "football soccer match"
            ],
            'Business': [
                "stock market trading",
                "company merger acquisition",
                "quarterly earnings report"
            ],
            'Sci/Tech': [
                "artificial intelligence research",
                "space exploration mission",
                "technology innovation startup"
            ]
        }
        return queries

    # 兼容旧接口
    def load_20newsgroups(self):
        """兼容旧接口，实际加载AG News"""
        return self.load_ag_news()


# 使用示例
if __name__ == "__main__":
    loader = DataLoader(use_cached=False)
    df = loader.load_ag_news()
    print(df.head())
    print(f"\nCategories: {df['target_name'].unique()}")
    print(f"\nCategory distribution:")
    print(df['target_name'].value_counts())
