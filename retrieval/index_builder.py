# retrieval/index_builder.py - 构建Whoosh索引

import os
import sys
sys.path.append('..')

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, NUMERIC, KEYWORD
from whoosh.analysis import StemmingAnalyzer
import pandas as pd
import joblib
from tqdm import tqdm
from config import RETRIEVAL_CONFIG

class IndexBuilder:
    """索引构建器"""

    def __init__(self, index_dir=None):
        self.index_dir = index_dir or RETRIEVAL_CONFIG['index_dir']
        self.schema = None
        self.ix = None

    def define_schema(self):
        """定义索引schema"""
        # 使用词干分析器（英文）
        analyzer = StemmingAnalyzer()

        self.schema = Schema(
            doc_id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=analyzer),
            category=ID(stored=True),  # 使用ID类型，不进行分词
            category_id=NUMERIC(stored=True),
            length=NUMERIC(stored=True)
        )
        return self.schema
    
    def build_index(self, documents_df):
        """构建索引"""
        print("Building search index...")
        
        # 确保索引目录存在
        os.makedirs(self.index_dir, exist_ok=True)
        
        # 创建索引
        if not os.path.exists(self.index_dir) or len(os.listdir(self.index_dir)) == 0:
            self.ix = create_in(self.index_dir, self.schema)
        else:
            self.ix = open_dir(self.index_dir)
        
        # 写入文档
        writer = self.ix.writer()
        
        for _, row in tqdm(documents_df.iterrows(), total=len(documents_df)):
            writer.add_document(
                doc_id=str(row['id']),
                content=row['text_clean'],
                category=row['target_name'],
                category_id=int(row['target']),
                length=len(row['text_clean'].split())
            )
        
        writer.commit()
        print(f"Index built: {len(documents_df)} documents indexed")
        
        # 保存文档映射
        self.save_document_mapping(documents_df)
        
        return self.ix
    
    def save_document_mapping(self, df):
        """保存文档ID到内容的映射"""
        mapping = {
            row['id']: {
                'content': row['text'][:500],  # 只保存前500字符
                'category': row['target_name'],
                'full_content': row['text']
            }
            for _, row in df.iterrows()
        }
        
        mapping_file = os.path.join(self.index_dir, 'document_mapping.pkl')
        joblib.dump(mapping, mapping_file)
        print(f"Document mapping saved to {mapping_file}")
    
    def get_index_stats(self):
        """获取索引统计信息"""
        if not self.ix:
            self.ix = open_dir(self.index_dir)
        
        with self.ix.searcher() as searcher:
            doc_count = searcher.doc_count()
            
        categories = {}
        mapping_file = os.path.join(self.index_dir, 'document_mapping.pkl')
        if os.path.exists(mapping_file):
            mapping = joblib.load(mapping_file)
            for doc_info in mapping.values():
                cat = doc_info['category']
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_documents': doc_count,
            'categories': categories,
            'index_dir': self.index_dir
        }

def main():
    """主函数：构建索引"""
    print("=" * 60)
    print("BUILDING SEARCH INDEX")
    print("=" * 60)
    
    # 加载数据
    from data.load_data import DataLoader
    
    loader = DataLoader()
    df = loader.load_20newsgroups()
    
    # 构建索引
    builder = IndexBuilder()
    builder.define_schema()
    builder.build_index(df)
    
    # 显示统计信息
    stats = builder.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Categories: {len(stats['categories'])}")
    
    print("\nDocuments per category:")
    for cat, count in list(stats['categories'].items())[:5]:
        print(f"  {cat}: {count}")
    if len(stats['categories']) > 5:
        print(f"  ... and {len(stats['categories']) - 5} more categories")
    
    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()