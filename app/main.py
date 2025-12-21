# app/main.py - Streamlit Main Application

import streamlit as st
import sys
import os

# Setup project root path
_current_file_path = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file_path))
sys.path.insert(0, _project_root)

# Page configuration
st.set_page_config(
    page_title="Text Classification & Retrieval",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Minimalist Swiss Design
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --primary: #1a1a1a;
        --secondary: #6b7280;
        --accent: #2563eb;
        --background: #fafafa;
        --surface: #ffffff;
        --border: #e5e7eb;
        --text-primary: #111827;
        --text-secondary: #6b7280;
        --success: #059669;
        --spacing-xs: 0.5rem;
        --spacing-sm: 1rem;
        --spacing-md: 1.5rem;
        --spacing-lg: 2rem;
        --spacing-xl: 3rem;
        --radius: 8px;
    }

    /* Global styles */
    .stApp {
        background-color: var(--background);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Keep header for sidebar toggle */
    header[data-testid="stHeader"] {
        background: transparent;
        backdrop-filter: none;
    }

    /* Main container */
    .main .block-container {
        padding: var(--spacing-lg) var(--spacing-xl);
        max-width: 1200px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    p, span, div {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: var(--spacing-xl) 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: var(--spacing-lg);
    }

    .app-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--spacing-xs) 0;
        letter-spacing: -0.03em;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin: 0;
    }

    /* Search container */
    .search-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-lg);
    }

    /* Result card */
    .result-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-sm);
        transition: border-color 0.2s ease;
    }

    .result-card:hover {
        border-color: var(--accent);
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: var(--spacing-sm);
    }

    .result-rank {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-id {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-top: 0.25rem;
    }

    .result-category {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .result-meta {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-sm);
        padding-bottom: var(--spacing-sm);
        border-bottom: 1px solid var(--border);
    }

    .result-content {
        font-size: 0.9rem;
        line-height: 1.7;
        color: var(--text-primary);
    }

    /* Highlight */
    .highlight {
        background: #fef3c7;
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
    }

    /* Classification box */
    .classification-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-lg);
    }

    .classification-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: var(--spacing-xs);
    }

    .classification-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .confidence-container {
        margin-top: var(--spacing-sm);
    }

    .confidence-bar-bg {
        height: 4px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 2px;
        transition: width 0.3s ease;
    }

    .confidence-text {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* Section header */
    .section-header {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: var(--spacing-lg) 0 var(--spacing-sm) 0;
        padding-bottom: var(--spacing-xs);
        border-bottom: 1px solid var(--border);
    }

    /* Stats */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: var(--spacing-sm);
        margin-bottom: var(--spacing-md);
    }

    .stat-item {
        text-align: center;
        padding: var(--spacing-sm);
        background: var(--background);
        border-radius: var(--radius);
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .stat-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Success message */
    .success-message {
        font-size: 0.85rem;
        color: var(--success);
        font-weight: 500;
        margin-bottom: var(--spacing-md);
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding: var(--spacing-md);
    }

    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: var(--spacing-sm);
    }

    /* Form elements */
    .stTextInput > div > div > input {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        padding: 0.75rem 1rem;
        transition: border-color 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-secondary);
    }

    /* Button */
    .stButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: #374151 !important;
        color: white !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stButton > button p {
        color: white !important;
    }

    /* Radio buttons */
    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio > div > label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-primary);
    }

    /* Slider */
    .stSlider > div > div {
        color: var(--text-primary);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-primary);
        background: transparent;
        border: none;
    }

    .streamlit-expanderContent {
        border: none;
        background: transparent;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        font-family: 'Inter', sans-serif;
        border-radius: var(--radius);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: var(--spacing-lg) 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding: var(--spacing-sm);
        }

        .app-title {
            font-size: 1.5rem;
        }

        .result-header {
            flex-direction: column;
            gap: var(--spacing-xs);
        }

        .stats-container {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Import project modules
from classification.classifier_model import TextClassifier
from retrieval.searcher import DocumentSearcher
from data.load_data import DataLoader
import joblib
import pandas as pd


class TextRetrievalApp:
    """Text Classification and Retrieval Application"""

    def __init__(self):
        self.load_components()
        self.setup_sidebar()

    def load_components(self):
        """Load system components"""
        with st.spinner("Initializing..."):
            try:
                model_dir = os.path.join(_project_root, 'classification', 'models')

                self.classifier = TextClassifier()
                self.classifier.load_model(model_dir)

                self.searcher = DocumentSearcher()
                index_dir = os.path.join(_project_root, 'retrieval', 'indexdir')
                self.searcher.open_index(index_dir)

                mapping_path = os.path.join(model_dir, 'category_mapping.pkl')
                self.category_mapping = joblib.load(mapping_path)
                self.category_name_to_id = {v: k for k, v in self.category_mapping.items()}

                self.loader = DataLoader()
                self.category_stats = self.searcher.get_category_stats()

            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                st.stop()

    def setup_sidebar(self):
        """Setup sidebar"""
        with st.sidebar:
            st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)

            self.search_mode = st.radio(
                "Mode",
                ["Smart Search", "Direct Search", "Browse"],
                label_visibility="collapsed"
            )

            if self.search_mode == "Direct Search":
                all_categories = ["All"] + list(self.category_mapping.values())
                self.selected_category = st.selectbox(
                    "Category Filter",
                    all_categories,
                    label_visibility="collapsed"
                )
            elif self.search_mode == "Browse":
                self.selected_browse_category = st.selectbox(
                    "Select Category",
                    list(self.category_mapping.values()),
                    label_visibility="collapsed"
                )

            st.markdown("---")

            self.max_results = st.slider(
                "Results",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )

            st.markdown("---")

            # Stats
            if hasattr(self, 'category_stats'):
                doc_count = len(self.searcher.document_mapping) if self.searcher.document_mapping else 0

                st.markdown(f"""
                <div class="stats-container">
                    <div class="stat-item">
                        <div class="stat-value">{doc_count:,}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(self.category_mapping)}</div>
                        <div class="stat-label">Categories</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def display_header(self):
        """Display page header"""
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Text Classification & Retrieval</h1>
            <p class="app-subtitle">Intelligent document search powered by machine learning</p>
        </div>
        """, unsafe_allow_html=True)

    def search_interface(self):
        """Search interface"""
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            query = st.text_input(
                "Search",
                placeholder="Enter your query...",
                key="search_input",
                label_visibility="collapsed"
            )

            search_clicked = st.button("Search", use_container_width=True)

        if search_clicked and query:
            if self.search_mode == "Smart Search":
                self.smart_search(query)
            elif self.search_mode == "Direct Search":
                self.direct_search(query)
            elif self.search_mode == "Browse":
                self.browse_category()

    def smart_search(self, query):
        """Smart search: classify then retrieve"""
        predicted_categories, probabilities = self.classifier.predict(query)
        predicted_name = self.category_mapping.get(predicted_categories[0], "Unknown")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<p class="section-header">Classification</p>', unsafe_allow_html=True)

            if probabilities is not None:
                confidence = probabilities[0][predicted_categories[0]]
                short_name = predicted_name.split('.')[-1] if '.' in predicted_name else predicted_name

                st.markdown(f"""
                <div class="classification-container">
                    <div class="classification-label">Predicted Category</div>
                    <div class="classification-value">{short_name}</div>
                    <div class="confidence-container">
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {confidence*100}%"></div>
                        </div>
                        <div class="confidence-text">{confidence:.1%} confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-header">Alternatives</p>', unsafe_allow_html=True)

            if probabilities is not None:
                top_indices = probabilities[0].argsort()[-4:][::-1]

                for idx in top_indices[1:4]:
                    prob = probabilities[0][idx]
                    name = self.category_mapping.get(idx, "Unknown")
                    short_name = name.split('.')[-1] if '.' in name else name
                    st.progress(float(prob), text=f"{short_name}: {prob:.1%}")

        st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)

        results = self.searcher.search(query, predicted_name, self.max_results)

        if results:
            self.display_results(results, query)
        else:
            results = self.searcher.search(query, None, self.max_results)
            if results:
                self.display_results(results, query)
            else:
                st.info("No documents found matching your query.")

    def direct_search(self, query):
        """Direct search"""
        st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)

        category_filter = None if self.selected_category == "All" else self.selected_category
        results = self.searcher.search(query, category_filter, self.max_results)

        if results:
            self.display_results(results, query)
        else:
            st.info("No documents found matching your query.")

    def browse_category(self):
        """Browse by category"""
        short_cat = self.selected_browse_category.split('.')[-1] if '.' in self.selected_browse_category else self.selected_browse_category
        st.markdown(f'<p class="section-header">Browsing: {short_cat}</p>', unsafe_allow_html=True)

        results = self.searcher.search_by_category(self.selected_browse_category, self.max_results)

        if results:
            formatted_results = []
            for i, r in enumerate(results):
                formatted_results.append({
                    'doc_id': r['doc_id'],
                    'score': 1.0,
                    'rank': i + 1,
                    'content_preview': r['content_preview'],
                    'category': r['category'],
                    'length': len(r['content_preview'].split()),
                    'highlight': r['content_preview'][:200] + "..."
                })
            self.display_results(formatted_results, query="")
        else:
            st.info("No documents in this category.")

    def display_results(self, results, query):
        """Display search results"""
        st.markdown(f'<p class="success-message">{len(results)} documents found</p>', unsafe_allow_html=True)

        for i, result in enumerate(results):
            short_category = result['category'].split('.')[-1] if '.' in result['category'] else result['category']

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <div class="result-rank">Result {result['rank']}</div>
                        <div class="result-id">Document {result['doc_id']}</div>
                    </div>
                    <span class="result-category">{short_category}</span>
                </div>
                <div class="result-meta">
                    Score: {result['score']:.4f}  ·  {result['length']} words
                </div>
                <div class="result-content">
                    {self.highlight_query_in_text(result['highlight'], query)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"Full preview"):
                st.text(result.get('content_preview', '')[:1000])

    def highlight_query_in_text(self, text, query):
        """Highlight query terms in text"""
        if not query:
            return text

        highlighted = text
        for word in query.lower().split():
            if len(word) > 2:
                highlighted = highlighted.replace(
                    word,
                    f'<span class="highlight">{word}</span>'
                )
        return highlighted

    def run(self):
        """Run application"""
        self.display_header()
        self.search_interface()


def main():
    """Main function"""
    app = TextRetrievalApp()
    app.run()


if __name__ == "__main__":
    main()
