set -e  # 遇到错误时退出

echo "=========================================="
echo "MNIST PCA降维与KMeans聚类实验"
echo "开始时间: $(date)"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "run_experiment.py" ]; then
    echo "错误: 请在包含 run_experiment.py 的目录中运行此脚本"
    exit 1
fi

# 检查Python环境
echo "检查Python环境..."
python --version || {
    echo "错误: Python未安装"
    exit 1
}

# 检查必要的Python库
echo "检查必要的Python库..."
python -c "
try:
    import numpy
    import torch
    import torchvision
    import sklearn
    import matplotlib
    print('✓ 所有必要的库都已安装')
except ImportError as e:
    print(f'✗ 缺少库: {e}')
    print('请运行: pip install numpy torch torchvision scikit-learn matplotlib')
    exit(1)
" || exit 1

# 运行实验
echo "运行实验..."
python run_experiment.py

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "实验成功完成!"
    echo "结束时间: $(date)"
    echo "=========================================="
    
    # 显示结果摘要
    echo ""
    echo "结果摘要:"
    echo "---------"
    
    if [ -f "results/tables/comparison_table.csv" ]; then
        echo "对比表格内容:"
        echo "--------------"
        column -t -s, results/tables/comparison_table.csv
    fi
    
    echo ""
    echo "生成的文件:"
    echo "-----------"
    echo "模型文件: $(find models -name "*.pkl" 2>/dev/null | wc -l)个"
    echo "日志文件: $(find results/logs -name "*.txt" -o -name "*.log" 2>/dev/null | wc -l)个"
    echo "图表文件: $(find results/plots -name "*.png" 2>/dev/null | wc -l)个"
    echo "数据文件: $(find results/data -name "*.npy" 2>/dev/null | wc -l)个"
    echo "表格文件: $(find results/tables -name "*.csv" -o -name "*.txt" 2>/dev/null | wc -l)个"
    
    echo ""
    echo "查看结果:"
    echo "  - 实验日志: cat results/logs/experiment.log"
    echo "  - 对比表格: cat results/tables/comparison_table.csv"
    echo "  - 查看图表: ls results/plots/*.png"
    
else
    echo "=========================================="
    echo "实验失败!"
    echo "请查看上面的错误信息"
    echo "=========================================="
    exit 1
fi