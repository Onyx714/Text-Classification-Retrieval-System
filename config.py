# config.py - 系统配置

# 数据集配置 - AG News Dataset
DATASET_CONFIG = {
    'name': 'ag_news',
    'categories': ['World', 'Sports', 'Business', 'Sci/Tech'],
    'num_samples': 10000,  # 每个类别的样本数，设置为None使用全部数据
}

# 分类器配置
CLASSIFIER_CONFIG = {
    'vectorizer': 'tfidf',
    'classifier': 'naive_bayes',
    'test_size': 0.2,
    'random_state': 42
}

# 检索系统配置
RETRIEVAL_CONFIG = {
    'index_dir': 'retrieval/indexdir',
    'max_results': 20,
    'score_type': 'BM25F'
}

# 系统配置
SYSTEM_CONFIG = {
    'debug': True,
    'log_level': 'INFO',
    'cache_dir': '.cache'
}