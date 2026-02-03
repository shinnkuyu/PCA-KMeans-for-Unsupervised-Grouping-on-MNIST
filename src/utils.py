"""
工具函数模块
包含数据加载、预处理、模型保存等工具函数
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import logging

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 数据加载函数
def load_mnist_data(n_samples=20000, flatten=True):
    """加载MNIST数据子集"""
    import torch
    import torchvision
    
    # 设置数据转换
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载训练数据
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 提取前n_samples个样本
    data_list = []
    labels_list = []
    
    for i in range(min(n_samples, len(train_dataset))):
        img, label = train_dataset[i]
        if flatten:
            # 展平为1D向量 (784维)
            data_list.append(img.view(-1).numpy())
        else:
            # 保持原始形状 (1, 28, 28)
            data_list.append(img.numpy())
        labels_list.append(label)
    
    # 转换为numpy数组
    X = np.array(data_list)
    y = np.array(labels_list)
    
    return X, y

# 数据预处理
def preprocess_data(X):
    """数据标准化"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# 模型保存和加载
def save_model(model, filename):
    """保存模型到文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """从文件加载模型"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 结果保存
def save_results(results, filename):
    """保存实验结果到文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if filename.endswith('.csv'):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results[0].keys())  # 写入标题行
            for result in results:
                writer.writerow(result.values())
    else:
        # 保存为文本文件
        with open(filename, 'w') as f:
            for result in results:
                f.write(f"实验 {result['experiment_id']}:\n")
                for key, value in result.items():
                    if key != 'experiment_id':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

# 数据可视化辅助函数
def create_cluster_sample_grid(X, labels, n_clusters=10, n_samples_per_cluster=16, title="Cluster Samples"):
    """
    创建每个簇的代表样本网格图
    返回matplotlib图形对象
    """
    fig, axes = plt.subplots(n_clusters, n_samples_per_cluster, 
                           figsize=(n_samples_per_cluster, n_clusters))
    
    # 如果只有一行，调整axes形状
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_idx in range(n_clusters):
        # 获取属于当前簇的样本索引
        cluster_indices = np.where(labels == cluster_idx)[0]
        
        # 如果簇中样本不足，使用所有样本
        n_display = min(len(cluster_indices), n_samples_per_cluster)
        
        if len(cluster_indices) > 0:
            # 随机选择样本
            selected_indices = np.random.choice(cluster_indices, n_display, replace=False)
            
            for j, idx in enumerate(selected_indices):
                if n_clusters > 1:
                    ax = axes[cluster_idx, j]
                else:
                    ax = axes[j]
                
                # 显示图像
                img = X[idx]
                if img.ndim == 1:
                    # 展平的图像，重塑为28x28
                    img = img.reshape(28, 28)
                elif img.ndim == 3:
                    # 通道维度在第一个位置，调整为(28, 28)
                    img = img.squeeze()
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                # 在第一列添加簇标签
                if j == 0:
                    cluster_size = len(cluster_indices)
                    ax.set_title(f'Cluster {cluster_idx}\nn={cluster_size}', 
                                fontsize=8, pad=2, loc='left')
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig

def plot_comparison_bar(results, metric_name, x_labels, title, ylabel, color='skyblue'):
    """创建对比条形图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 提取数据
    x_pos = np.arange(len(x_labels))
    metric_values = [result[metric_name] for result in results]
    
    # 绘制条形图
    bars = ax.bar(x_pos, metric_values, color=color, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 设置图表属性
    ax.set_xlabel('PCA维度')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig