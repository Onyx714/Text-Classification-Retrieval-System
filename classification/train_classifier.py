# classification/train_classifier.py - 训练分类器主程序
import joblib
import sys
import os
sys.path.append('..')

from data.load_data import DataLoader
from classifier_model import TextClassifier
import joblib

def main():
    """训练分类器主函数"""
    print("=" * 60)
    print("TEXT CLASSIFIER TRAINING")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. Loading dataset...")
    loader = DataLoader(use_cached=True)
    df = loader.load_20newsgroups()
    
    # 2. 划分数据集
    print("\n2. Splitting dataset...")
    retrieval_df, train_df, test_df = loader.split_dataset(df, test_size=0.2)
    
    # 3. 训练分类器
    print("\n3. Training classifier...")
    classifier = TextClassifier(model_type='naive_bayes')
    
    # 创建向量化器
    classifier.create_vectorizer(max_features=5000)
    
    # 创建分类器
    classifier.create_classifier()
    
    # 训练
    classifier.train(train_df['text_clean'].tolist(), train_df['target'].tolist())
    
    # 4. 评估
    print("\n4. Evaluating classifier...")
    accuracy, report = classifier.evaluate(
        test_df['text_clean'].tolist(),
        test_df['target'].tolist()
    )
    
    # 5. 保存模型
    print("\n5. Saving model...")
    classifier.save_model()
    
    # 6. 保存类别名称映射
    category_mapping = {
        idx: name for idx, name in enumerate(df['target_name'].unique())
    }
    joblib.dump(category_mapping, 'classification/models/category_mapping.pkl')
    
    print("\n" + "=" * 60)
    print(f"Training completed! Accuracy: {accuracy:.4f}")
    print("=" * 60)
    
    # 测试预测
    print("\nSample predictions:")
    sample_texts = [
        "computer graphics and image processing",
        "hockey game results and player statistics",
        "space exploration missions to Mars"
    ]
    
    for text in sample_texts:
        pred, prob = classifier.predict(text)
        print(f"Text: '{text}'")
        print(f"  Predicted: {category_mapping.get(pred[0], 'Unknown')}")
        print()

if __name__ == "__main__":
    main()