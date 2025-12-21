# Quick Start Guide - 快速入门指南

## 30秒快速启动

如果您已经克隆了仓库并且有Python环境：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 初始化系统
python setup.py

# 3. 运行应用
python run.py
```

然后打开浏览器访问：http://localhost:8501

---

## 详细步骤

### 第一步：环境准备

**需求**：
- Python 3.7 或更高版本
- 2GB 可用内存
- 1GB 可用磁盘空间

**验证 Python 版本**：
```bash
python --version
```

### 第二步：获取代码

```bash
git clone https://github.com/yourusername/Text-Classification-Retrieval-System.git
cd Text-Classification-Retrieval-System
```

### 第三步：创建虚拟环境（推荐）

**Windows**：
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**：
```bash
python -m venv venv
source venv/bin/activate
```

### 第四步：安装依赖包

```bash
pip install -r requirements.txt
```

这将安装：
- Streamlit（Web界面）
- scikit-learn（机器学习）
- Whoosh（搜索引擎）
- pandas、numpy（数据处理）
- 其他依赖包

**预计时间**：2-5分钟

### 第五步：初始化系统

```bash
python setup.py
```

这个脚本会自动：
1. ✓ 检查依赖包是否安装完整
2. ✓ 下载 20 Newsgroups 数据集（约18,000篇文档）
3. ✓ 训练文本分类器（TF-IDF + 朴素贝叶斯）
4. ✓ 构建搜索索引（Whoosh BM25）
5. ✓ 验证安装是否成功

**预计时间**：5-10分钟（首次运行）

**注意**：如果已经初始化过，使用 `python setup.py --force` 强制重新初始化

### 第六步：启动应用

```bash
python run.py
```

或者直接使用 Streamlit：
```bash
streamlit run app/main.py
```

**成功标志**：
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 使用系统

### Web 界面

1. **打开浏览器**，访问 http://localhost:8501
2. **输入查询**，例如："computer graphics rendering"
3. **选择搜索模式**：
   - 智能搜索（推荐）：自动分类后检索
   - 直接检索：全文搜索
   - 按类别浏览：浏览特定类别

### 命令行界面

```bash
python run.py --cli
```

然后输入查询：
```
Query: computer graphics

  Predicted Category: comp.graphics
  Confidence: 89.2%

  Found 15 results:
  [1] Document 1234 (Score: 0.8752)
      Category: comp.graphics
      Preview: Computer graphics rendering involves...
```

### 快速测试

```bash
python run.py --test
```

运行预定义的测试查询，验证系统功能。

---

## 常见问题排查

### 问题1：找不到模型文件

**错误**：`FileNotFoundError: classifier.pkl not found`

**解决**：
```bash
python setup.py --force
```

### 问题2：找不到索引

**错误**：`Index directory not found`

**解决**：
```bash
cd retrieval
python index_builder.py
```

### 问题3：导入错误

**错误**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：
```bash
pip install -r requirements.txt
```

### 问题4：内存不足

**错误**：`MemoryError`

**解决**：在 `config.py` 中减小 `max_features`：
```python
CLASSIFIER_CONFIG = {
    'max_features': 3000  # 从 5000 减少到 3000
}
```

### 问题5：端口已被占用

**错误**：`Port 8501 is already in use`

**解决**：
```bash
streamlit run app/main.py --server.port 8502
```

---

## 示例查询

### 计算机类

```
computer graphics rendering
windows operating system
mac hardware problems
```

### 体育类

```
hockey playoff game
baseball world series
motorcycle racing
```

### 科学类

```
space exploration mars
medical research cancer
cryptography encryption
```

### 政治/社会类

```
middle east conflict
gun control debate
religious beliefs
```

---

## 下一步

1. **查看评估结果**：
   ```bash
   python evaluation/experiments.py
   ```

2. **自定义配置**：编辑 `config.py`

3. **阅读完整文档**：[README.md](README.md)

4. **查看代码示例**：[API使用文档](docs/api_usage.md)

---

## 系统架构

```
用户查询 "computer graphics"
    ↓
[文本分类器] → 预测：comp.graphics (89%)
    ↓
[检索引擎] → 在 comp.graphics 类别中搜索
    ↓
[结果排序] → 按 BM25 分数排序
    ↓
[显示结果] → 前20个最相关的文档
```

---

## 获取帮助

- **问题反馈**：[GitHub Issues](https://github.com/yourusername/project/issues)
- **查看日志**：检查终端输出
- **详细文档**：README.md

---

## 性能基准

| 指标 | 典型值 |
|------|--------|
| 分类准确率 | 85-90% |
| 检索准确率（P@10） | 70-75% |
| 平均响应时间 | <500ms |
| 数据集大小 | 18,000 文档 |
| 内存使用 | ~500MB |

---

Happy searching! 🎉
