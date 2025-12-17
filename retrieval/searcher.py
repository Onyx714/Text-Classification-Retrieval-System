# retrieval/searcher.py - 检索器

import os
import sys
sys.path.append('..')

from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup, MultifieldParser
from whoosh import scoring
from whoosh.query import Term, And, Or
import joblib
from config import RETRIEVAL_CONFIG

class DocumentSearcher:
    """文档检索器"""
    
    def __init__(self, index_dir=None):
        self.index_dir = index_dir or RETRIEVAL_CONFIG['index_dir']
        self.ix = None
        self.document_mapping = None
        
    def open_index(self):
        """打开索引"""
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
        
        self.ix = open_dir(self.index_dir)
        
        # 加载文档映射
        mapping_file = os.path.join(self.index_dir, 'document_mapping.pkl')
        if os.path.exists(mapping_file):
            self.document_mapping = joblib.load(mapping_file)
        
        return self.ix
    
    def search(self, query_text, category_filter=None, max_results=20):
        """执行搜索"""
        if not self.ix:
            self.open_index()
        
        with self.ix.searcher() as searcher:
            # 构建查询
            if category_filter and category_filter != "所有类别":
                # 组合查询：内容匹配 AND 类别匹配
                content_query = QueryParser("content", self.ix.schema).parse(query_text)
                category_query = Term("category", category_filter)
                final_query = And([content_query, category_query])
            else:
                # 仅内容查询
                final_query = QueryParser("content", self.ix.schema).parse(query_text)
            
            # 执行搜索
            results = searcher.search(
                final_query, 
                limit=max_results,
                scored=True,
                terms=True
            )
            
            # 格式化结果
            formatted_results = []
            for hit in results:
                doc_id = int(hit['doc_id'])
                doc_info = self.document_mapping.get(doc_id, {})
                
                formatted_results.append({
                    'doc_id': doc_id,
                    'score': hit.score,
                    'rank': hit.rank + 1,
                    'content_preview': doc_info.get('content', ''),
                    'category': hit['category'],
                    'length': hit.get('length', 0),
                    'highlight': self._highlight_text(hit, query_text)
                })
            
            return formatted_results
    
    def search_by_category(self, category, max_results=50):
        """按类别搜索"""
        if not self.ix:
            self.open_index()
        
        with self.ix.searcher() as searcher:
            query = Term("category", category)
            results = searcher.search(query, limit=max_results)
            
            return [{
                'doc_id': int(r['doc_id']),
                'content_preview': self.document_mapping.get(int(r['doc_id']), {}).get('content', ''),
                'category': r['category']
            } for r in results]
    
    def _highlight_text(self, hit, query_text):
        """高亮显示匹配文本"""
        content = self.document_mapping.get(int(hit['doc_id']), {}).get('content', '')
        if not content:
            return ""
        
        # 简单的高亮实现
        words = query_text.lower().split()
        for word in words:
            if word in content.lower():
                start = content.lower().find(word)
                end = start + len(word)
                # 提取上下文
                context_start = max(0, start - 50)
                context_end = min(len(content), end + 50)
                context = content[context_start:context_end]
                
                # 高亮显示
                highlighted = context.replace(
                    content[start:end],
                    f"**{content[start:end]}**"
                )
                
                return f"...{highlighted}..."
        
        # 如果没有匹配，返回开头部分
        return content[:100] + "..." if len(content) > 100 else content
    
    def get_category_stats(self):
        """获取类别统计"""
        if not self.ix:
            self.open_index()
        
        with self.ix.searcher() as searcher:
            from whoosh.query import Every
            results = searcher.search(Every("category"), limit=0)
            
            categories = {}
            for facet in results.groupby("category"):
                categories[facet] = results.groups()[facet]
            
            return categories
    
    def search_with_classification(self, query_text, classifier, max_results=20):
        """结合分类的搜索"""
        # 1. 先分类
        predicted_categories, probs = classifier.predict(query_text)
        
        # 2. 在预测类别中搜索
        category_name = classifier.classes_[predicted_categories[0]]
        
        # 3. 执行搜索
        results = self.search(query_text, category_name, max_results)
        
        return {
            'predicted_category': category_name,
            'category_confidence': probs[0][predicted_categories[0]] if probs is not None else None,
            'results': results,
            'query': query_text
        }

# 测试检索器
def test_searcher():
    """测试检索器"""
    searcher = DocumentSearcher()
    searcher.open_index()
    
    # 测试查询
    test_queries = [
        ("computer graphics", "comp.graphics"),
        ("hockey game", "rec.sport.hockey"),
        ("space exploration", "sci.space")
    ]
    
    for query, expected_category in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected category: {expected_category}")
        
        # 不带过滤的搜索
        results = searcher.search(query, max_results=5)
        print(f"Results (no filter): {len(results)}")
        
        # 带过滤的搜索
        filtered_results = searcher.search(query, expected_category, max_results=5)
        print(f"Results (with filter): {len(filtered_results)}")
        
        if filtered_results:
            print(f"Top result category: {filtered_results[0]['category']}")

if __name__ == "__main__":
    test_searcher()