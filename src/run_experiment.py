import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import os
import pickle
import csv
from datetime import datetime
import logging

class MNISTExperiment:
    """MNIST PCA降维与KMeans聚类实验类"""
    
    def __init__(self, n_samples=20000, n_init=20, n_clusters=10):
        """
        初始化实验参数
        
        参数:
            n_samples: 样本数量
            n_init: KMeans初始化次数
            n_clusters: 聚类数量
        """
        self.n_samples = n_samples
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.results = []
        
        # 创建所有必要的目录
        self._create_directories()
        
        # 设置日志
        self.logger = self._setup_logger()
    
    def _create_directories(self):
        """创建所有必要的目录"""
        directories = [
            'models',
            'results',
            'results/logs',
            'results/tables',
            'results/plots',
            'results/data',
            'data'  # 用于存储MNIST数据
        ]
        
        print("创建项目目录结构...")
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  ✓ {directory}")
            except Exception as e:
                print(f"  ✗ 创建目录失败 {directory}: {e}")
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('MNISTExperiment')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = 'results/logs/experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_mnist_data(self):
        """加载MNIST数据子集"""
        self.logger.info(f"加载MNIST数据集，取前{self.n_samples}个样本...")
        
        try:
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
            
            # 提取样本
            data_list = []
            labels_list = []
            
            for i in range(min(self.n_samples, len(train_dataset))):
                img, label = train_dataset[i]
                # 展平为1D向量 (784维)
                data_list.append(img.view(-1).numpy())
                labels_list.append(label)
            
            # 转换为numpy数组
            X = np.array(data_list)
            y = np.array(labels_list)
            
            self.logger.info(f"数据加载完成，形状: {X.shape}")
            print(f"数据加载完成: {X.shape[0]}个样本, {X.shape[1]}个特征")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise
    
    def run_single_experiment(self, pca_dim, experiment_id=1):
        """
        运行单个实验
        
        参数:
            pca_dim: PCA降维维度
            experiment_id: 实验ID
            
        返回:
            实验结果的字典
        """
        self.logger.info(f"开始实验 {experiment_id}: PCA降维到{pca_dim}维")
        print(f"\n{'='*60}")
        print(f"实验 {experiment_id}: PCA降维到{pca_dim}维")
        print(f"{'='*60}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 1. 加载数据
            X, y = self.load_mnist_data()
            
            # 2. 数据预处理
            print("数据预处理...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 3. PCA降维
            print("PCA降维...")
            pca_start = time.time()
            pca = PCA(n_components=pca_dim, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            pca_time = time.time() - pca_start
            
            explained_variance = pca.explained_variance_ratio_.sum()
            self.logger.info(f"PCA降维完成，累计解释方差比: {explained_variance:.3f}")
            print(f"PCA降维完成，累计解释方差比: {explained_variance:.3f}")
            
            # 4. KMeans聚类
            print(f"KMeans聚类 (k={self.n_clusters}, n_init={self.n_init})...")
            kmeans_start = time.time()
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=42,
                verbose=0
            )
            labels = kmeans.fit_predict(X_pca)
            kmeans_time = time.time() - kmeans_start
            
            # 计算指标
            inertia = kmeans.inertia_
            cluster_counts = np.bincount(labels)
            cluster_distribution = cluster_counts.tolist()
            
            total_time = time.time() - start_time
            
            # 5. 记录结果
            experiment_result = {
                'experiment_id': experiment_id,
                'pca_dim': pca_dim,
                'n_samples': self.n_samples,
                'n_init': self.n_init,
                'total_time': total_time,
                'pca_time': pca_time,
                'kmeans_time': kmeans_time,
                'inertia': inertia,
                'explained_variance': explained_variance,
                'cluster_distribution': cluster_distribution,
                'pca_model': pca,
                'kmeans_model': kmeans,
                'scaler': scaler,
                'X_pca': X_pca,
                'labels': labels,
                'X_original': X,
                'y_true': y
            }
            
            self.results.append(experiment_result)
            
            # 6. 保存模型和中间数据
            self._save_experiment_data(experiment_result)
            
            # 7. 记录日志
            self.logger.info(f"实验 {experiment_id} 完成:")
            self.logger.info(f"  总耗时: {total_time:.2f}秒")
            self.logger.info(f"  簇内误差: {inertia:.2f}")
            self.logger.info(f"  簇分布: {cluster_distribution}")
            
            # 8. 打印摘要
            print(f"\n实验摘要:")
            print(f"  PCA维度: {pca_dim}")
            print(f"  样本数: {self.n_samples}")
            print(f"  KMeans初始化次数: {self.n_init}")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  簇内误差(inertia): {inertia:.2f}")
            print(f"  累计解释方差: {explained_variance:.3f}")
            print(f"  簇分布: {cluster_distribution}")
            
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"实验 {experiment_id} 失败: {e}")
            print(f"实验失败: {e}")
            raise
    
    def _save_experiment_data(self, result):
        """保存实验数据"""
        exp_id = result['experiment_id']
        pca_dim = result['pca_dim']
        
        print(f"保存实验 {exp_id} 的结果...")
        
        # 保存PCA模型
        with open(f'models/pca_{pca_dim}d_{exp_id}.pkl', 'wb') as f:
            pickle.dump(result['pca_model'], f)
        
        # 保存KMeans模型
        with open(f'models/kmeans_{pca_dim}d_{exp_id}.pkl', 'wb') as f:
            pickle.dump(result['kmeans_model'], f)
        
        # 保存降维后的数据
        np.save(f'results/data/X_pca_{pca_dim}d_{exp_id}.npy', result['X_pca'])
        np.save(f'results/data/labels_{pca_dim}d_{exp_id}.npy', result['labels'])
        
        print(f"  ✓ 模型和数据已保存")
    
    def _visualize_clusters(self, result):
        """可视化聚类结果"""
        X = result['X_original']
        labels = result['labels']
        pca_dim = result['pca_dim']
        exp_id = result['experiment_id']
        
        print(f"可视化聚类结果...")
        
        n_clusters = self.n_clusters
        n_samples_per_cluster = 16
        
        # 创建图形
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
                    img = X[idx].reshape(28, 28)
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    # 在第一列添加簇标签
                    if j == 0:
                        cluster_size = len(cluster_indices)
                        ax.set_title(f'Cluster {cluster_idx}\nn={cluster_size}',
                                    fontsize=8, pad=2, loc='left')
        
        plt.suptitle(f'PCA维度: {pca_dim} - 每个簇的代表样本 (实验{exp_id})', fontsize=14, y=0.98)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'results/plots/cluster_samples_pca{pca_dim}_{exp_id}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 聚类可视化已保存: results/plots/cluster_samples_pca{pca_dim}_{exp_id}.png")
    
    def _plot_pca_variance(self, result):
        """绘制PCA解释方差图"""
        pca_model = result['pca_model']
        pca_dim = result['pca_dim']
        exp_id = result['experiment_id']
        
        explained_variance = pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 各主成分解释方差
        ax1.bar(range(1, pca_dim + 1), explained_variance, alpha=0.7)
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比')
        ax1.set_title(f'PCA {pca_dim}维 - 各成分解释方差')
        ax1.grid(True, alpha=0.3)
        
        # 累计解释方差
        ax2.plot(range(1, pca_dim + 1), cumulative_variance, 'r-', marker='o')
        ax2.fill_between(range(1, pca_dim + 1), 0, cumulative_variance, alpha=0.3)
        ax2.set_xlabel('主成分数量')
        ax2.set_ylabel('累计解释方差比')
        ax2.set_title(f'PCA {pca_dim}维 - 累计解释方差: {cumulative_variance[-1]:.3f}')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # 添加数值标签
        for i, v in enumerate(cumulative_variance):
            if i % max(1, pca_dim // 4) == 0 or i == pca_dim - 1:
                ax2.text(i + 1, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'PCA {pca_dim}维解释方差分析 (实验{exp_id})', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'results/plots/pca_variance_{pca_dim}d_{exp_id}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ PCA方差分析已保存: results/plots/pca_variance_{pca_dim}d_{exp_id}.png")
    
    def run_comparison_experiments(self, pca_dims=[8, 16, 32]):
        """
        运行多组对比实验
        
        参数:
            pca_dims: PCA维度列表
            
        返回:
            所有实验结果的列表
        """
        self.logger.info(f"开始对比实验，PCA维度: {pca_dims}")
        print(f"\n开始对比实验，PCA维度: {pca_dims}")
        
        all_results = []
        
        for i, pca_dim in enumerate(pca_dims, 1):
            # 运行实验
            result = self.run_single_experiment(pca_dim, i)
            all_results.append(result)
            
            # 可视化当前实验的结果
            self._visualize_clusters(result)
            self._plot_pca_variance(result)
        
        # 生成对比分析
        self._generate_comparison_analysis(all_results)
        
        return all_results
    
    def _generate_comparison_analysis(self, results):
        """生成对比分析报告和图表"""
        print("\n生成对比分析报告...")
        
        # 创建对比数据表格
        comparison_table = []
        
        for result in results:
            comparison_table.append({
                '实验ID': result['experiment_id'],
                'PCA维度': result['pca_dim'],
                '样本数量': result['n_samples'],
                '初始化次数': result['n_init'],
                '总耗时(秒)': round(result['total_time'], 2),
                'PCA耗时(秒)': round(result['pca_time'], 2),
                'KMeans耗时(秒)': round(result['kmeans_time'], 2),
                '簇内误差': round(result['inertia'], 2),
                '累计解释方差': round(result['explained_variance'], 3)
            })
        
        # 保存对比表格为CSV
        csv_file = 'results/tables/comparison_table.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_table[0].keys())
            writer.writeheader()
            writer.writerows(comparison_table)
        
        print(f"  ✓ 对比表格已保存: {csv_file}")
        
        # 生成对比可视化
        self._create_comparison_visualizations(results)
        
        # 生成文本报告
        self._generate_text_report(results)
        
        # 显示对比表格
        print("\n" + "="*80)
        print("实验对比结果")
        print("="*80)
        print(f"{'实验':<4} {'PCA维':<6} {'总耗时':<8} {'PCA耗时':<8} {'KMeans耗时':<10} "
              f"{'簇内误差':<12} {'解释方差':<12}")
        print("-"*80)
        
        for row in comparison_table:
            print(f"{row['实验ID']:<4} {row['PCA维度']:<6} {row['总耗时(秒)']:<8} "
                  f"{row['PCA耗时(秒)']:<8} {row['KMeans耗时(秒)']:<10} "
                  f"{row['簇内误差']:<12} {row['累计解释方差']:<12}")
        
        print("="*80)
    
    def _create_comparison_visualizations(self, results):
        """创建对比可视化图表"""
        print("生成对比可视化图表...")
        
        pca_dims = [r['pca_dim'] for r in results]
        
        # 1. 运行时间对比
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 运行时间
        times = [r['total_time'] for r in results]
        bars1 = axes[0].bar(pca_dims, times, color='skyblue', alpha=0.7)
        axes[0].set_xlabel('PCA维度')
        axes[0].set_ylabel('总耗时 (秒)')
        axes[0].set_title('运行时间对比')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # 簇内误差
        inertias = [r['inertia'] for r in results]
        bars2 = axes[1].bar(pca_dims, inertias, color='lightcoral', alpha=0.7)
        axes[1].set_xlabel('PCA维度')
        axes[1].set_ylabel('簇内误差')
        axes[1].set_title('簇内误差对比')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1000,
                       f'{height:.0f}', ha='center', va='bottom')
        
        # 解释方差
        variances = [r['explained_variance'] for r in results]
        bars3 = axes[2].bar(pca_dims, variances, color='lightgreen', alpha=0.7)
        axes[2].set_xlabel('PCA维度')
        axes[2].set_ylabel('累计解释方差比')
        axes[2].set_title('累计解释方差对比')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim([0, 1.1])
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('MNIST PCA降维与KMeans聚类对比实验', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/plots/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 综合对比图已保存: results/plots/comprehensive_comparison.png")
    
    def _generate_text_report(self, results):
        """生成文本报告"""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("MNIST PCA降维与KMeans聚类实验报告")
        report_lines.append("="*80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 实验设置
        report_lines.append("一、实验设置")
        report_lines.append("-"*40)
        report_lines.append(f"数据集: MNIST (手写数字)")
        report_lines.append(f"样本数量: {self.n_samples}")
        report_lines.append(f"KMeans聚类数: {self.n_clusters}")
        report_lines.append(f"KMeans初始化次数: {self.n_init}")
        report_lines.append(f"PCA降维维度: {[r['pca_dim'] for r in results]}")
        report_lines.append("")
        
        # 实验结果
        report_lines.append("二、实验结果")
        report_lines.append("-"*40)
        
        for result in results:
            report_lines.append(f"实验 {result['experiment_id']} (PCA {result['pca_dim']}维):")
            report_lines.append(f"  总耗时: {result['total_time']:.2f}秒")
            report_lines.append(f"  簇内误差: {result['inertia']:.2f}")
            report_lines.append(f"  累计解释方差: {result['explained_variance']:.3f}")
            report_lines.append(f"  簇分布: {result['cluster_distribution']}")
            report_lines.append("")
        
        # 对比分析
        report_lines.append("三、对比分析")
        report_lines.append("-"*40)
        
        # 找出最佳结果
        min_inertia_idx = np.argmin([r['inertia'] for r in results])
        max_variance_idx = np.argmax([r['explained_variance'] for r in results])
        min_time_idx = np.argmin([r['total_time'] for r in results])
        
        report_lines.append(f"最低簇内误差: PCA {results[min_inertia_idx]['pca_dim']}维 "
                          f"({results[min_inertia_idx]['inertia']:.2f})")
        report_lines.append(f"最高解释方差: PCA {results[max_variance_idx]['pca_dim']}维 "
                          f"({results[max_variance_idx]['explained_variance']:.3f})")
        report_lines.append(f"最短运行时间: PCA {results[min_time_idx]['pca_dim']}维 "
                          f"({results[min_time_idx]['total_time']:.2f}秒)")
        report_lines.append("")
        
        # 结论
        report_lines.append("四、结论")
        report_lines.append("-"*40)
        report_lines.append("1. PCA维度越高，保留的原始信息越多（解释方差越大）")
        report_lines.append("2. 更高的PCA维度通常会降低簇内误差，但会增加计算时间")
        report_lines.append("3. 需要根据具体应用权衡计算成本和信息保留")
        report_lines.append("4. 对于MNIST数据集，16维可能是一个较好的平衡点")
        report_lines.append("")
        
        report_lines.append("五、文件说明")
        report_lines.append("-"*40)
        report_lines.append("models/: 保存的PCA和KMeans模型文件")
        report_lines.append("results/data/: 降维后的数据和聚类标签")
        report_lines.append("results/logs/: 实验日志")
        report_lines.append("results/tables/: 数据表格")
        report_lines.append("results/plots/: 可视化图表")
        report_lines.append("="*80)
        
        # 保存报告
        report_path = 'results/logs/experiment_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告
        print('\n'.join(report_lines))
        print(f"\n  ✓ 实验报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("MNIST数据集PCA降维与KMeans聚类实验 - 完整解决方案")
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
    try:
        results = experiment.run_comparison_experiments(pca_dims)
        
        print("\n" + "="*80)
        print("实验完成!")
        print("="*80)
        
        # 显示生成的文件
        print("\n生成的文件统计:")
        print("-"*40)
        print(f"模型文件数量: {len([f for f in os.listdir('models') if f.endswith('.pkl')])}")
        print(f"日志文件数量: {len([f for f in os.listdir('results/logs')])}")
        print(f"图表文件数量: {len([f for f in os.listdir('results/plots') if f.endswith('.png')])}")
        print(f"数据文件数量: {len([f for f in os.listdir('results/data') if f.endswith('.npy')])}")
        print(f"表格文件数量: {len([f for f in os.listdir('results/tables')])}")
        
        print("\n重要文件:")
        print("  - 实验日志: results/logs/experiment.log")
        print("  - 实验报告: results/logs/experiment_report.txt")
        print("  - 对比表格: results/tables/comparison_table.csv")
        print("  - 聚类样本图: results/plots/cluster_samples_*.png")
        print("  - 对比分析图: results/plots/comprehensive_comparison.png")
        
        print("\n实验成功完成!")
        
    except Exception as e:
        print(f"\n实验过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # 检查必要的库
    print("检查必要的Python库...")
    try:
        import numpy
        import torch
        import torchvision
        import sklearn
        import matplotlib
        print("✓ 所有必要的库都已安装")
    except ImportError as e:
        print(f"✗ 缺少库: {e}")
        print("请运行: pip install numpy torch torchvision scikit-learn matplotlib")
        exit(1)
    
    # 运行主函数
    exit_code = main()
    exit(exit_code)