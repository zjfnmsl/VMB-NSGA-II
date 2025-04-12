import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot_comparison_results(results: dict, metric_name: str = "IGD"):
    """
    可视化算法对比实验结果
    
    Args:
        results: 各算法的实验结果字典
        metric_name: 性能指标名称
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制箱线图
    plt.boxplot([results[alg] for alg in results.keys()],
                labels=list(results.keys()),
                notch=True)
    
    plt.title(f'算法性能对比 ({metric_name})')
    plt.ylabel(metric_name)
    plt.grid(True)
    
    # 保存结果
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'comparison_results.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_sensitivity_analysis(results: dict):
    """
    可视化参数敏感性分析结果
    
    Args:
        results: 参数敏感性分析结果字典
    """
    n_params = len(results)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    
    for i, (param, values) in enumerate(results.items()):
        ax = axes[i] if n_params > 1 else axes
        
        # 绘制箱线图
        param_values = list(values.keys())
        param_results = [values[v] for v in param_values]
        ax.boxplot(param_results, labels=[f'{v:.2f}' for v in param_values])
        
        ax.set_title(f'参数 {param} 的敏感性分析')
        ax.set_xlabel(param)
        ax.set_ylabel('IGD')
        ax.grid(True)
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(pareto_front: np.ndarray, pareto_objectives: np.ndarray, n_var: int, compare_with_file: str = None):
    """
    可视化NSGA-II算法的结果，支持与原始结果进行对比分析
    
    参数:
        pareto_front: Pareto最优解集
        pareto_objectives: 对应的目标函数值
        n_var: 决策变量个数
        compare_with_file: 用于比较的原始结果图片文件路径
    
    参数:
        pareto_front: Pareto最优解集
        pareto_objectives: 对应的目标函数值
        n_var: 决策变量个数
    """
    try:
        # 创建一个包含多个子图的画布
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)

        # 1. Pareto前沿
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='b', label='当前结果', s=10)  # 设置点大小为10
        
        # 如果提供了比较文件，加载并显示
        if compare_with_file and os.path.exists(compare_with_file):
            ax1_compare = fig.add_subplot(gs[0, 1])
            ax1_compare.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='r', label='优化前结果', s=10)  # 设置点大小为10
            ax1_compare.set_xlabel('f1')
            ax1_compare.set_ylabel('f2')
            ax1_compare.set_title('优化前结果')
            ax1_compare.legend()
            ax1_compare.grid(True)
            
            ax1.set_title('优化后结果 (加入精英抵抗力策略)')
        else:
            ax1.set_title('优化结果')
            
        ax1.set_xlabel('f1')
        ax1.set_ylabel('f2')
        ax1.legend()
        ax1.grid(True)

        # 2. 决策变量分布对比
        ax3 = fig.add_subplot(gs[1, :])
        colors = ['b', 'g', 'r', 'c', 'm']
        for i in range(min(5, n_var)):  # 展示前5个决策变量
            ax3.hist(pareto_front[:, i], bins=20, alpha=0.5, color=colors[i], label=f'x{i+1}')
        ax3.set_xlabel('变量取值')
        ax3.set_ylabel('频次')
        ax3.set_title('决策变量分布')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        
        # 保存图片到output文件夹
        import os
        import glob
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取已有的结果文件列表并确定新的编号
        existing_files = glob.glob(os.path.join(output_dir, 'nsga2_results_*.png'))
        if not existing_files:
            next_number = 1
        else:
            numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            next_number = max(numbers) + 1
        
        # 保存新的结果
        output_filename = f'nsga2_results_{next_number}.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
        
        # 显示图片
        plt.show()

    except ImportError:
        print("\n请安装matplotlib以查看可视化结果")


def plot_dynamic_results(objectives_history: list, population_history: list, generation_interval: int = 1):
    """
    可视化动态优化过程中的目标函数值和种群分布变化
    
    Args:
        objectives_history: 每代种群的目标函数值历史记录，格式为[[f1, f2], ...]
        population_history: 每代种群的决策变量历史记录，格式为[[x1, x2, ...], ...]
        generation_interval: 显示结果的代际间隔，默认为1
    """
    # 创建动态图表
    plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2)
    
    # 1. 目标函数值随时间的变化
    ax1 = plt.subplot(gs[0, 0])
    generations = range(0, len(objectives_history), generation_interval)
    objectives_data = np.array(objectives_history)[generations]
    
    # 绘制目标函数值的变化趋势
    ax1.plot(generations, objectives_data[:, 0], 'b-', label='f1')
    ax1.plot(generations, objectives_data[:, 1], 'r-', label='f2')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('目标函数值')
    ax1.set_title('目标函数值随时间的变化')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 种群分布的动态变化
    ax2 = plt.subplot(gs[0, 1])
    population_data = np.array(population_history)[generations]
    
    # 使用散点图展示种群分布
    scatter = ax2.scatter(population_data[-1][:, 0], 
                         population_data[-1][:, 1],
                         c='b', alpha=0.6)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title(f'第{len(objectives_history)}代种群分布')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取已有的结果文件列表并确定新的编号
    existing_files = glob.glob(os.path.join(output_dir, 'dynamic_results_*.png'))
    if not existing_files:
        next_number = 1
    else:
        numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
        next_number = max(numbers) + 1
    
    # 保存新的结果
    output_filename = f'dynamic_results_{next_number}.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    plt.show()