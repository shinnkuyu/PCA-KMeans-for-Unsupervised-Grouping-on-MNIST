"""
实验模块
包含主要的实验类和方法
"""
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .utils import *

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
        self.logger = setup_logger('MNISTExperiment', 'results/logs/experiment.log')
        
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
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 加载数据
        X, y = load_mnist_data(self.n_samples, flatten=True)
        self.logger.info(f"加载数据完成，形状: {X.shape}")
        
        # 2. 数据预处理
        X_scaled, scaler = preprocess_data(X)
        
        # 3. PCA降维
        pca_start = time.time()
        pca = PCA(n_components=pca_dim, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_time = time.time() - pca_start
        
        explained_variance = pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA降维完成，累计解释方差比: {explained_variance:.3f}")
        
        # 4. KMeans聚类
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
        
        return experiment_result
    
    def _save_experiment_data(self, result):
        """保存实验数据"""
        exp_id = result['experiment_id']
        pca_dim = result['pca_dim']
        
        # 保存PCA模型
        save_model(result['pca_model'], f'models/pca_{pca_dim}d_{exp_id}.pkl')
        
        # 保存KMeans模型
        save_model(result['kmeans_model'], f'models/kmeans_{pca_dim}d_{exp_id}.pkl')
        
        # 保存scaler
        save_model(result['scaler'], f'models/scaler_{pca_dim}d_{exp_id}.pkl')
        
        # 保存降维后的数据
        np.save(f'results/data/X_pca_{pca_dim}d_{exp_id}.npy', result['X_pca'])
        np.save(f'results/data/labels_{pca_dim}d_{exp_id}.npy', result['labels'])
    
    def run_comparison_experiments(self, pca_dims=[8, 16, 32]):
        """
        运行多组对比实验
        
        参数:
            pca_dims: PCA维度列表
            
        返回:
            所有实验结果的列表
        """
        self.logger.info(f"开始对比实验，PCA维度: {pca_dims}")
        
        all_results = []
        
        for i, pca_dim in enumerate(pca_dims, 1):
            result = self.run_single_experiment(pca_dim, i)
            all_results.append(result)
            
            # 可视化当前实验的聚类结果
            self._visualize_experiment_results(result)
        
        # 生成对比分析
        self._generate_comparison_analysis(all_results)
        
        return all_results
    
    def _visualize_experiment_results(self, result):
        """可视化单个实验的结果"""
        pca_dim = result['pca_dim']
        exp_id = result['experiment_id']
        
        # 创建目录
        os.makedirs('results/plots', exist_ok=True)
        
        # 1. 可视化聚类样本
        fig1 = create_cluster_sample_grid(
            result['X_original'], 
            result['labels'],
            n_clusters=self.n_clusters,
            n_samples_per_cluster=16,
            title=f'PCA维度: {pca_dim} - 每个簇的代表样本 (实验{exp_id})'
        )
        fig1.savefig(f'results/plots/cluster_samples_pca{pca_dim}_{exp_id}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 可视化PCA解释方差
        fig2 = self._plot_pca_variance(result['pca_model'], pca_dim, exp_id)
        fig2.savefig(f'results/plots/pca_variance_{pca_dim}d_{exp_id}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 可视化簇分布
        fig3 = self._plot_cluster_distribution(result, pca_dim, exp_id)
        fig3.savefig(f'results/plots/cluster_dist_{pca_dim}d_{exp_id}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig3)
    
    def _plot_pca_variance(self, pca_model, pca_dim, exp_id):
        """绘制PCA解释方差图"""
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
        ax2.set_title(f'PCA {pca_dim}维 - 累计解释方差')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # 添加数值标签
        for i, v in enumerate(cumulative_variance):
            ax2.text(i + 1, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'PCA {pca_dim}维解释方差分析 (实验{exp_id})', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _plot_cluster_distribution(self, result, pca_dim, exp_id):
        """绘制簇分布图"""
        cluster_dist = result['cluster_distribution']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 条形图
        ax1.bar(range(len(cluster_dist)), cluster_dist, alpha=0.7, color='skyblue')
        ax1.set_xlabel('簇标签')
        ax1.set_ylabel('样本数量')
        ax1.set_title(f'PCA {pca_dim}维 - 簇分布')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(cluster_dist):
            ax1.text(i, v + 50, str(v), ha='center', va='bottom')
        
        # 饼图（显示比例）
        labels = [f'Cluster {i}' for i in range(len(cluster_dist))]
        ax2.pie(cluster_dist, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'PCA {pca_dim}维 - 簇比例分布')
        ax2.axis('equal')  # 确保饼图是圆的
        
        plt.suptitle(f'聚类分布可视化 (实验{exp_id})', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _generate_comparison_analysis(self, results):
        """生成对比分析报告和图表"""
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
        
        # 保存对比表格
        save_results(comparison_table, 'results/tables/comparison_table.csv')
        
        # 生成对比可视化
        self._create_comparison_visualizations(results)
        
        # 生成文本报告
        self._generate_text_report(results)
    
    def _create_comparison_visualizations(self, results):
        """创建对比可视化图表"""
        pca_dims = [r['pca_dim'] for r in results]
        
        # 1. 运行时间对比
        fig1 = plot_comparison_bar(
            results, 'total_time', pca_dims,
            '不同PCA维度的运行时间对比',
            '总耗时(秒)', 'skyblue'
        )
        fig1.savefig('results/plots/comparison_time.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 簇内误差对比
        fig2 = plot_comparison_bar(
            results, 'inertia', pca_dims,
            '不同PCA维度的簇内误差对比',
            '簇内误差', 'lightcoral'
        )
        fig2.savefig('results/plots/comparison_inertia.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 解释方差对比
        fig3 = plot_comparison_bar(
            results, 'explained_variance', pca_dims,
            '不同PCA维度的累计解释方差对比',
            '累计解释方差比', 'lightgreen'
        )
        fig3.savefig('results/plots/comparison_variance.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 综合对比图
        fig4, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 运行时间
        times = [r['total_time'] for r in results]
        ax1.bar(pca_dims, times, color='skyblue', alpha=0.7)
        ax1.set_xlabel('PCA维度')
        ax1.set_ylabel('总耗时(秒)')
        ax1.set_title('运行时间对比')
        ax1.grid(True, alpha=0.3)
        
        # 簇内误差
        inertias = [r['inertia'] for r in results]
        ax2.bar(pca_dims, inertias, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('PCA维度')
        ax2.set_ylabel('簇内误差')
        ax2.set_title('聚类效果对比')
        ax2.grid(True, alpha=0.3)
        
        # 解释方差
        variances = [r['explained_variance'] for r in results]
        ax3.bar(pca_dims, variances, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('PCA维度')
        ax3.set_ylabel('累计解释方差比')
        ax3.set_title('信息保留对比')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.1])
        
        plt.suptitle('MNIST PCA降维与KMeans聚类对比实验', fontsize=16)
        plt.tight_layout()
        fig4.savefig('results/plots/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
    
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
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告
        print('\n'.join(report_lines))