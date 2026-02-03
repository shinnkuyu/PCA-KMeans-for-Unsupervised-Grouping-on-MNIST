"""
主程序入口
运行MNIST PCA降维与KMeans聚类实验
"""
import sys
import os

# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 修改导入方式
# from experiments import MNISTExperiment  # 原来的相对导入

# 改为直接导入
try:
    from src.experiments import MNISTExperiment
except ImportError:
    # 如果上述方式失败，尝试当前目录导入
    try:
        from experiments import MNISTExperiment
    except ImportError as e:
        print(f"导入错误: {e}")
        print("正在尝试其他导入方式...")
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from experiments import MNISTExperiment

def main():
    """主函数"""
    print("="*80)
    print("MNIST数据集PCA降维与KMeans聚类实验")
    print("="*80)
    
    # 实验参数
    n_samples = 20000      # 样本数量
    n_init = 20            # KMeans初始化次数
    pca_dims = [8, 16, 32] # PCA降维维度
    n_clusters = 10        # 聚类数量
    
    print(f"实验参数:")
    print(f"  样本数量: {n_samples}")
    print(f"  KMeans初始化次数: {n_init}")
    print(f"  聚类数量: {n_clusters}")
    print(f"  PCA维度: {pca_dims}")
    print()
    
    # 创建实验实例
    experiment = MNISTExperiment(
        n_samples=n_samples,
        n_init=n_init,
        n_clusters=n_clusters
    )
    
    # 运行对比实验
    print("开始运行对比实验...")
    results = experiment.run_comparison_experiments(pca_dims)
    
    print("\n" + "="*80)
    print("实验完成!")
    print("结果已保存到以下目录:")
    print("  models/: PCA和KMeans模型文件")
    print("  results/: 所有实验结果和可视化")
    print("="*80)

if __name__ == "__main__":
    main()
