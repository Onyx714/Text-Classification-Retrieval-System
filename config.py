# config.py - 系统配置

# 数据集配置
DATASET_CONFIG = {
    'name': '20newsgroups',
    'categories': ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
                   'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
                   'comp.windows.x', 'misc.forsale', 'rec.autos', 
                   'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 
                   'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
                   'soc.religion.christian', 'talk.politics.guns', 
                   'talk.politics.mideast', 'talk.politics.misc', 
                   'talk.religion.misc'],
    'subset': 'all',  # 'train', 'test', 'all'
    'remove': ('headers', 'footers', 'quotes')  # 移除邮件头尾
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