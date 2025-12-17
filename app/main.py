# app/main.py - Streamlit主应用

import streamlit as st
import sys
import os

# ==== 核心修复：动态设置项目根目录路径 ====
# 获取当前文件（main.py）的绝对路径，然后向上追溯两级得到项目根目录
# 例如：/mount/src/your-project-name/app/main.py -> /mount/src/your-project-name
_current_file_path = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file_path))

# 将项目根目录添加到Python模块搜索路径的最前面
sys.path.insert(0, _project_root)

# 页面配置
st.set_page_config(
    page_title="文本分类+检索系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        background-color: #f8f9fa;
    }
    .highlight {
        background-color: #FFF9C4;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .category-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        background-color: #E3F2FD;
        color: #1565C0;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 导入项目模块 (现在sys.path已正确设置)
from classification.classifier_model import TextClassifier
from retrieval.searcher import DocumentSearcher
from data.load_data import DataLoader
import joblib
import pandas as pd

class TextRetrievalApp:
    """文本分类检索应用"""
    
    def __init__(self):
        self.load_components()
        self.setup_sidebar()
    
    def load_components(self):
        """加载系统组件"""
        with st.spinner("正在加载系统组件..."):
            try:
                # 注意：在Streamlit Cloud上，所有路径都应基于项目根目录
                # 构建模型文件的绝对路径
                model_dir = os.path.join(_project_root, 'classification', 'models')
                
                # 加载分类器
                self.classifier = TextClassifier()
                self.classifier.load_model(model_dir) # 需修改classifier_model.py的load_model方法以接受路径参数
                
                # 加载检索器 (修改searcher.py，使其能接收基于_project_root的索引路径)
                self.searcher = DocumentSearcher()
                index_dir = os.path.join(_project_root, 'retrieval', 'indexdir')
                self.searcher.open_index(index_dir)
                
                # 加载类别映射 (使用绝对路径)
                mapping_path = os.path.join(model_dir, 'category_mapping.pkl')
                self.category_mapping = joblib.load(mapping_path)
                
                # 反转映射：名称 -> ID
                self.category_name_to_id = {v: k for k, v in self.category_mapping.items()}
                
                # 加载数据加载器
                self.loader = DataLoader()
                
                # 获取类别统计
                self.category_stats = self.searcher.get_category_stats()
                
                st.success("系统加载完成！")
                
            except Exception as e:
                st.error(f"加载失败: {str(e)}")
                st.stop()
    
    def setup_sidebar(self):
        """设置侧边栏"""
        with st.sidebar:
            st.title("⚙️ 系统设置")
            
            # 选择搜索模式
            self.search_mode = st.radio(
                "搜索模式",
                ["智能搜索（先分类后检索）", "直接检索", "按类别浏览"]
            )
            
            # 类别过滤器（如果选择直接检索）
            if self.search_mode == "直接检索":
                all_categories = ["所有类别"] + list(self.category_mapping.values())
                self.selected_category = st.selectbox(
                    "筛选类别",
                    all_categories
                )
            elif self.search_mode == "按类别浏览":
                self.selected_browse_category = st.selectbox(
                    "选择浏览类别",
                    list(self.category_mapping.values())
                )
            
            # 结果数量
            self.max_results = st.slider(
                "最大结果数量",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            
            # 显示统计信息
            with st.expander("📊 系统统计"):
                if hasattr(self, 'category_stats'):
                    # 注意：首次运行document_mapping可能为空，需在searcher中初始化
                    doc_count = len(self.searcher.document_mapping) if hasattr(self.searcher, 'document_mapping') else 0
                    st.write(f"**文档总数:** {doc_count}")
                    st.write(f"**类别数量:** {len(self.category_mapping)}")
                    
                    # 显示前几个类别的文档数
                    if self.category_stats:
                        st.write("**文档分布:**")
                        for cat, count in list(self.category_stats.items())[:10]:
                            st.write(f"  - {cat}: {count}篇")
            
            # 示例查询
            with st.expander("💡 示例查询"):
                examples = self.loader.load_sample_queries()
                for category, queries in examples.items():
                    st.write(f"**{category}**:")
                    for query in queries[:2]:
                        if st.button(f"🔍 {query}", key=f"example_{category}_{query}"):
                            st.session_state['query'] = query
                            st.rerun()
            
            # 关于信息
            with st.expander("ℹ️ 关于系统"):
                st.write("""
                **文本分类+检索系统**
                
                本系统结合了文本分类和信息检索技术，实现：
                1. **文本分类**: 自动识别查询的类别
                2. **智能检索**: 在相关类别中搜索文档
                3. **结果排序**: 按相关性排序
                
                **技术栈**:
                - 分类: TF-IDF + 朴素贝叶斯
                - 检索: Whoosh (BM25算法)
                - 界面: Streamlit
                
                **数据集**: 20 Newsgroups (20个类别，约18,000篇文档)
                """)
    
    def display_header(self):
        """显示页头"""
        st.markdown('<h1 class="main-header">📚 文本分类+检索系统</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            st.markdown("""
            <div style="text-align: center; color: #666; margin-bottom: 2rem;">
                输入查询，系统会自动分类并在相关类别中检索最相关的文档
            </div>
            """, unsafe_allow_html=True)
    
    def search_interface(self):
        """搜索界面"""
        # 搜索框
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # 初始化session_state中的query
            if 'query' not in st.session_state:
                st.session_state['query'] = ''
                
            query = st.text_input(
                "🔍 输入搜索查询",
                value=st.session_state['query'],
                placeholder="例如：computer graphics, hockey game, space exploration...",
                key="search_input"
            )
            
            search_button = st.button("搜索", type="primary", use_container_width=True)
        
        # 处理搜索
        if search_button and query:
            with st.spinner("正在处理..."):
                if self.search_mode == "智能搜索（先分类后检索）":
                    self.smart_search(query)
                elif self.search_mode == "直接检索":
                    self.direct_search(query)
                elif self.search_mode == "按类别浏览":
                    self.browse_category()
    
    def smart_search(self, query):
        """智能搜索：先分类后检索"""
        st.markdown(f'<h3 class="sub-header">查询: "{query}"</h3>', unsafe_allow_html=True)
        
        # 1. 分类
        with st.expander("📊 分类结果", expanded=True):
            predicted_categories, probabilities = self.classifier.predict(query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 显示预测类别
                predicted_name = self.category_mapping.get(predicted_categories[0], "Unknown")
                st.metric("预测类别", predicted_name)
                
                # 显示置信度
                if probabilities is not None:
                    confidence = probabilities[0][predicted_categories[0]]
                    st.metric("置信度", f"{confidence:.1%}")
            
            with col2:
                # 显示其他可能的类别
                if probabilities is not None:
                    top_indices = probabilities[0].argsort()[-3:][::-1]
                    st.write("其他可能类别:")
                    for idx in top_indices[1:]:  # 跳过最高的
                        prob = probabilities[0][idx]
                        name = self.category_mapping.get(idx, "Unknown")
                        st.progress(float(prob), text=f"{name}: {prob:.1%}")
        
        # 2. 检索
        st.markdown('<h3 class="sub-header">📄 检索结果</h3>', unsafe_allow_html=True)
        
        # 在预测类别中搜索
        results = self.searcher.search(query, predicted_name, self.max_results)
        
        if results:
            self.display_results(results, query)
        else:
            st.warning("未找到相关文档，正在扩大搜索范围...")
            # 尝试无过滤搜索
            results = self.searcher.search(query, None, self.max_results)
            if results:
                self.display_results(results, query)
            else:
                st.error("未找到任何相关文档。")
    
    def direct_search(self, query):
        """直接检索"""
        st.markdown(f'<h3 class="sub-header">查询: "{query}"</h3>', unsafe_allow_html=True)
        
        # 显示搜索设置
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"搜索模式: 直接检索")
        with col2:
            if self.selected_category != "所有类别":
                st.info(f"筛选类别: {self.selected_category}")
        
        # 执行搜索
        category_filter = None if self.selected_category == "所有类别" else self.selected_category
        results = self.searcher.search(query, category_filter, self.max_results)
        
        if results:
            self.display_results(results, query)
        else:
            st.error("未找到相关文档。")
    
    def browse_category(self):
        """按类别浏览"""
        st.markdown(f'<h3 class="sub-header">浏览类别: {self.selected_browse_category}</h3>', unsafe_allow_html=True)
        
        # 显示类别信息
        category_id = self.category_name_to_id.get(self.selected_browse_category)
        if category_id is not None:
            st.info(f"类别ID: {category_id}")
        
        # 获取该类别的文档
        results = self.searcher.search_by_category(self.selected_browse_category, self.max_results)
        
        if results:
            # 重新格式化结果以匹配显示函数
            formatted_results = []
            for i, r in enumerate(results):
                formatted_results.append({
                    'doc_id': r['doc_id'],
                    'score': 1.0,  # 浏览模式没有分数
                    'rank': i + 1,
                    'content_preview': r['content_preview'],
                    'category': r['category'],
                    'length': len(r['content_preview'].split()),
                    'highlight': r['content_preview'][:200] + "..."
                })
            
            self.display_results(formatted_results, query="")
        else:
            st.warning("该类别暂无文档。")
    
    def display_results(self, results, query):
        """显示检索结果"""
        # 结果统计
        st.success(f"找到 {len(results)} 个相关文档")
        
        # 结果列表
        for i, result in enumerate(results):
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1E88E5;">#{result['rank']} 文档 {result['doc_id']}</h4>
                        <span class="category-tag">{result['category']}</span>
                    </div>
                    <div style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                        相关性得分: <strong>{result['score']:.4f}</strong> | 
                        长度: {result['length']} 词
                    </div>
                    <div style="margin-top: 1rem; line-height: 1.6;">
                        {self.highlight_query_in_text(result['highlight'], query)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示更多选项
                with st.expander("查看详情", key=f"details_{i}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # 显示完整内容预览
                        st.write("**内容预览:**")
                        st.write(result.get('content_preview', '')[:500] + "...")
                    
                    with col2:
                        # 操作按钮
                        if st.button("📋 复制ID", key=f"copy_{i}"):
                            st.code(str(result['doc_id']))
                        
                        if st.button("📊 分析", key=f"analyze_{i}"):
                            # 这里可以添加文档分析功能
                            st.write("文档分析功能开发中...")
    
    def highlight_query_in_text(self, text, query):
        """在文本中高亮显示查询词"""
        if not query:
            return text
        
        highlighted = text
        for word in query.lower().split():
            if len(word) > 2:  # 只高亮长度大于2的词
                highlighted = highlighted.replace(
                    word,
                    f'<span class="highlight">{word}</span>'
                )
        
        return highlighted
    
    def run(self):
        """运行应用"""
        self.display_header()
        self.search_interface()
        
        # 在底部添加一些功能
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 重新加载系统"):
                st.rerun()
        with col2:
            if st.button("📊 查看系统评估"):
                self.show_evaluation()
        with col3:
            if st.button("ℹ️ 系统帮助"):
                self.show_help()
    
    def show_evaluation(self):
        """显示系统评估"""
        st.markdown('<h3 class="sub-header">系统评估结果</h3>', unsafe_allow_html=True)
        
        # 这里可以加载之前保存的评估结果
        try:
            # 使用绝对路径加载分类器评估
            eval_path = os.path.join(_project_root, 'classification', 'models', 'evaluation.pkl')
            eval_data = joblib.load(eval_path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("分类准确率", f"{eval_data['accuracy']:.2%}")
            with col2:
                st.metric("精确率", f"{eval_data['report']['weighted avg']['precision']:.2%}")
            with col3:
                st.metric("召回率", f"{eval_data['report']['weighted avg']['recall']:.2%}")
            with col4:
                st.metric("F1分数", f"{eval_data['report']['weighted avg']['f1-score']:.2%}")
            
            # 显示混淆矩阵图片 (使用绝对路径)
            st.markdown("#### 混淆矩阵")
            try:
                conf_matrix_path = os.path.join(_project_root, 'classification', 'confusion_matrix.png')
                st.image(conf_matrix_path)
            except:
                st.info("混淆矩阵图片未生成")
                
        except Exception as e:
            st.warning(f"评估数据未找到或加载失败: {str(e)}")
    
    def show_help(self):
        """显示帮助信息"""
        with st.expander("🆘 使用帮助", expanded=True):
            st.markdown("""
            ### 如何使用本系统
            
            1. **智能搜索模式**（推荐）
               - 输入查询文本
               - 系统自动分类
               - 在预测类别中检索
            
            2. **直接检索模式**
               - 输入查询文本
               - 可选择特定类别筛选
               - 直接检索相关文档
            
            3. **按类别浏览模式**
               - 选择感兴趣的类别
               - 浏览该类别下的所有文档
            
            ### 查询建议
            
            - 使用具体的关键词而不是句子
            - 避免过于宽泛的查询
            - 示例：
              - ✅ "computer graphics rendering"
              - ✅ "hockey game results"
              - ❌ "sports"（过于宽泛）
            
            ### 技术说明
            
            - **分类器**: TF-IDF + 朴素贝叶斯，在20 Newsgroups数据集上训练
            - **检索器**: Whoosh搜索引擎，使用BM25排序算法
            - **数据集**: 20个类别，约18,000篇文档
            
            ### 遇到问题？
            
            1. 尝试重新加载系统（点击"重新加载系统"按钮）
            2. 确保查询包含有效关键词
            3. 如果某个类别没有结果，尝试使用"所有类别"筛选
            """)

def main():
    """主函数"""
    app = TextRetrievalApp()
    app.run()

if __name__ == "__main__":
    main()