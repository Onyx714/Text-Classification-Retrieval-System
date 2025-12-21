# 使用说明书 - User Manual

## 目录

1. [系统概述](#系统概述)
2. [功能介绍](#功能介绍)
3. [使用指南](#使用指南)
4. [高级功能](#高级功能)
5. [配置说明](#配置说明)
6. [API文档](#api文档)

---

## 系统概述

### 什么是文本分类与检索系统？

这是一个智能化的文本信息管理系统，结合了两项核心技术：

1. **文本分类**：自动识别文本所属的类别
   - 使用 TF-IDF 特征提取
   - 朴素贝叶斯分类器
   - 支持 4 个类别（AG News数据集）

2. **信息检索**：在大量文档中快速查找相关内容
   - Whoosh 全文搜索引擎
   - BM25 相关性排序算法
   - 支持类别过滤

### 数据集说明

本系统使用 **AG News** 数据集，包含4个新闻类别：

| 类别 | 描述 | 示例主题 |
|------|------|----------|
| World | 世界新闻 | 国际外交、政治事件、全球事务 |
| Sports | 体育新闻 | 足球、篮球、奥运会、比赛结果 |
| Business | 商业新闻 | 股市、公司并购、财报、经济 |
| Sci/Tech | 科技新闻 | AI、太空探索、技术创新、科学发现 |

### 系统优势

- **智能化**：自动分类，提高检索精度
- **快速**：毫秒级响应，即时返回结果
- **准确**：90%+ 分类准确率
- **易用**：直观的Web界面
- **灵活**：支持多种检索模式

---

## 功能介绍

### 1. 智能搜索（推荐）

**工作流程**：
```
输入查询 → 自动分类 → 在预测类别中检索 → 返回排序结果
```

**适用场景**：
- 不确定文档所属类别
- 希望获得最精准的结果
- 日常搜索使用

**示例**：
```
查询："computer graphics algorithms"
↓
分类：comp.graphics (置信度: 92%)
↓
检索：在 comp.graphics 类别中搜索
↓
结果：15个相关文档
```

### 2. 直接检索

**工作流程**：
```
输入查询 → [可选：选择类别] → 全文检索 → 返回排序结果
```

**适用场景**：
- 已知文档所属类别
- 需要跨类别搜索
- 探索性搜索

**特点**：
- 可选类别过滤器
- 支持"所有类别"搜索
- 灵活的搜索范围

### 3. 按类别浏览

**工作流程**：
```
选择类别 → 列出该类别所有文档 → 浏览查看
```

**适用场景**：
- 浏览特定主题的文档
- 了解类别内容分布
- 学习和研究

---

## 使用指南

### Web界面操作

#### 启动应用

```bash
python run.py
```

浏览器打开：http://localhost:8501

#### 界面布局

```
┌─────────────────────────────────────────┐
│        文本分类+检索系统                    │
├──────────┬──────────────────────────────┤
│  侧边栏   │        主界面                  │
│          │                              │
│ 搜索模式  │   🔍 搜索框                   │
│ 类别选择  │                              │
│ 结果数量  │   📊 分类结果                 │
│          │                              │
│ 系统统计  │   📄 检索结果                 │
│ 示例查询  │      - 文档1                  │
│ 关于系统  │      - 文档2                  │
│          │      - ...                   │
└──────────┴──────────────────────────────┘
```

#### 基本操作步骤

**步骤1：选择搜索模式**

在侧边栏选择：
- 智能搜索（先分类后检索）
- 直接检索
- 按类别浏览

**步骤2：输入查询**

在搜索框输入您的查询文本：
```
computer graphics rendering techniques
```

**步骤3：查看结果**

系统显示：
1. 分类结果（智能搜索模式）
   - 预测类别
   - 置信度
   - 其他可能类别

2. 检索结果
   - 文档排名
   - 相关性得分
   - 文档预览
   - 类别标签

**步骤4：查看详情**

点击"查看详情"展开：
- 完整文档预览
- 文档元数据
- 操作选项

### 命令行界面操作

#### 启动CLI

```bash
python run.py --cli
```

#### 交互式查询

```bash
Query: hockey game playoff

  Predicted Category: rec.sport.hockey
  Confidence: 91.5%

  Found 12 results:
  [1] Document 5678 (Score: 0.9123)
      Category: rec.sport.hockey
      Preview: The NHL playoffs are underway with...
```

#### 退出

输入 `quit`、`exit` 或 `q`

---

## 高级功能

### 1. 自定义类别过滤

在直接检索模式下，可以指定类别：

1. 选择"直接检索"模式
2. 在侧边栏选择类别过滤器
3. 输入查询并搜索

### 2. 调整结果数量

使用侧边栏的滑块调整：
- 最小：5个结果
- 最大：50个结果
- 默认：20个结果

### 3. 使用示例查询

侧边栏提供示例查询：
- 点击任意示例
- 自动填充到搜索框
- 快速测试系统

### 4. 查看系统统计

在侧边栏展开"系统统计"：
- 文档总数
- 类别数量
- 各类别文档分布

### 5. 系统评估

点击"查看系统评估"按钮：
- 分类准确率
- 精确率、召回率
- F1分数
- 混淆矩阵

---

## 配置说明

### 编辑配置文件

打开 `config.py` 进行配置：

#### 1. 数据集配置

```python
DATASET_CONFIG = {
    'name': '20newsgroups',
    'categories': [
        'comp.graphics',
        'rec.sport.hockey',
        # 添加或删除类别
    ],
    'subset': 'all',  # 'train', 'test', 'all'
    'remove': ('headers', 'footers', 'quotes')
}
```

**可调整项**：
- `categories`：选择要使用的类别
- `subset`：使用完整数据集或仅训练/测试集
- `remove`：预处理选项

#### 2. 分类器配置

```python
CLASSIFIER_CONFIG = {
    'vectorizer': 'tfidf',
    'classifier': 'naive_bayes',  # 可选：svm, random_forest
    'test_size': 0.2,
    'random_state': 42
}
```

**分类器类型**：
- `naive_bayes`：快速，适合大数据集
- `svm`：高准确率，训练较慢
- `random_forest`：平衡性能和准确率

#### 3. 检索系统配置

```python
RETRIEVAL_CONFIG = {
    'index_dir': 'retrieval/indexdir',
    'max_results': 20,
    'score_type': 'BM25F'
}
```

### 重新训练模型

修改配置后，重新训练：

```bash
python setup.py --force
```

---

## API文档

### Python API使用

#### 分类器API

```python
from classification.classifier_model import TextClassifier

# 加载模型
classifier = TextClassifier()
classifier.load_model('classification/models')

# 预测单个文本
text = "computer graphics rendering"
predictions, probabilities = classifier.predict(text)

# 预测多个文本
texts = ["text 1", "text 2", "text 3"]
predictions, probabilities = classifier.predict(texts)

# 获取类别名称
category_id = predictions[0]
confidence = probabilities[0][category_id]
```

#### 检索器API

```python
from retrieval.searcher import DocumentSearcher

# 初始化检索器
searcher = DocumentSearcher()
searcher.open_index()

# 基本搜索
results = searcher.search(
    query_text="computer graphics",
    category_filter=None,  # 或指定类别
    max_results=10
)

# 按类别搜索
results = searcher.search_by_category(
    category="comp.graphics",
    max_results=20
)

# 结合分类的搜索
result_dict = searcher.search_with_classification(
    query_text="graphics algorithms",
    classifier=classifier,
    max_results=15
)
```

#### 结果格式

```python
result = {
    'doc_id': 1234,
    'score': 0.8756,
    'rank': 1,
    'content_preview': '文档内容...',
    'category': 'comp.graphics',
    'length': 234,
    'highlight': '...matching text...'
}
```

---

## 查询技巧

### 1. 有效的查询

**使用具体关键词**
```
Good: "stock market trading earnings"
Bad: "business" (太宽泛)
```

**使用多个相关词**
```
Good: "Olympic athletes championship medal"
Bad: "sports" (单个词)
```

**使用专业术语**
```
Good: "artificial intelligence machine learning"
Bad: "tech stuff" (太笼统)
```

### 2. 查询优化

**问题**：结果太多
- 解决：添加更多具体关键词
- 解决：使用类别过滤

**问题**：结果太少
- 解决：减少关键词
- 解决：使用"所有类别"搜索

**问题**：结果不相关
- 解决：检查查询拼写
- 解决：尝试同义词

### 3. 类别参考

| 类别 | 主题 | 示例查询 |
|------|------|----------|
| World | 世界新闻 | diplomacy, summit, UN, elections |
| Sports | 体育新闻 | championship, Olympic, football, NBA |
| Business | 商业新闻 | stock market, merger, earnings, trade |
| Sci/Tech | 科技新闻 | AI, space, innovation, research |

---

## 性能优化

### 1. 提高搜索速度

- 减少 `max_results`
- 使用类别过滤
- 定期重建索引

### 2. 提高分类准确率

- 增加训练数据
- 调整 `max_features`
- 尝试不同分类器

### 3. 减少内存使用

```python
# config.py
CLASSIFIER_CONFIG = {
    'max_features': 3000  # 减少特征数
}
```

---

## 常见问题

### Q: 系统支持中文吗？

A: 当前版本主要针对英文设计。中文支持需要：
- 修改分词器
- 调整停用词列表
- 使用中文数据集

### Q: 可以添加自己的文档吗？

A: 可以。步骤：
1. 准备CSV格式数据
2. 修改 `load_data.py`
3. 重新训练模型

### Q: 如何导出搜索结果？

A: 目前通过API使用：
```python
import pandas as pd
results = searcher.search(query, max_results=100)
df = pd.DataFrame(results)
df.to_csv('results.csv')
```

### Q: 支持哪些文件格式？

A: 当前支持纯文本。扩展支持：
- PDF：使用 `PyPDF2`
- Word：使用 `python-docx`
- HTML：使用 `BeautifulSoup`

---

## 联系与支持

- **问题反馈**：GitHub Issues
- **功能建议**：Pull Requests
- **技术讨论**：Discussions

---

最后更新：2025-12-20
