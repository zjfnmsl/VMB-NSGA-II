import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict

def run_statistical_tests() -> Dict[str, List[float]]:
    """运行统计测试并返回结果
    
    Returns:
        Dict[str, List[float]]: 各算法的统计检验结果字典，格式为{算法名: [检验值列表]}
    """
    # 示例数据：两个算法在10次独立运行中的性能指标
    algorithm1_results = np.array([0.85, 0.82, 0.88, 0.87, 0.84, 0.86, 0.83, 0.85, 0.87, 0.86])
    algorithm2_results = np.array([0.78, 0.75, 0.77, 0.76, 0.79, 0.77, 0.76, 0.78, 0.75, 0.77])
    
    # 执行统计检验
    stat, p_val = wilcoxon_test(algorithm1_results, algorithm2_results)
    
    # 计算效应量
    effect = effect_size_analysis(algorithm1_results, algorithm2_results)
    
    # 进行统计分析
    statistical_analysis([algorithm1_results, algorithm2_results], 
                        ["Algorithm 1", "Algorithm 2"])
    
    # 返回结果字典
    return {
        "Algorithm 1": algorithm1_results.tolist(),
        "Algorithm 2": algorithm2_results.tolist()
    }

def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """
    执行Wilcoxon秩和检验，比较两组数据的显著性差异
    
    Args:
        data1: 第一组数据
        data2: 第二组数据
        
    Returns:
        Tuple[float, float]: (统计量, p值)
    """
    statistic, p_value = stats.wilcoxon(data1, data2)
    return statistic, p_value

def statistical_analysis(results_list: List[np.ndarray], 
                       algorithm_names: List[str],
                       metric_name: str = "IGD") -> None:
    """
    对多个算法的实验结果进行统计分析
    
    Args:
        results_list: 各算法的实验结果列表
        algorithm_names: 算法名称列表
        metric_name: 性能指标名称
    """
    n_algorithms = len(results_list)
    print(f"\n{metric_name}指标的统计分析结果:")
    print("-" * 50)
    
    # 计算基本统计量
    for i, (results, name) in enumerate(zip(results_list, algorithm_names)):
        mean_val = np.mean(results)
        std_val = np.std(results)
        print(f"{name}:\n  均值: {mean_val:.4f}\n  标准差: {std_val:.4f}")
    
    # 执行Wilcoxon检验
    print("\nWilcoxon检验结果 (p值):")
    print("-" * 50)
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            stat, p_val = wilcoxon_test(results_list[i], results_list[j])
            print(f"{algorithm_names[i]} vs {algorithm_names[j]}: {p_val:.4f}")
            if p_val < 0.05:
                print("  * 存在显著差异 (p < 0.05)")

def effect_size_analysis(data1: np.ndarray, 
                        data2: np.ndarray, 
                        method: str = "cohen") -> float:
    """
    计算效应量，评估算法改进的实际效果大小
    
    Args:
        data1: 第一组数据
        data2: 第二组数据
        method: 效应量计算方法 ('cohen' 或 'hedges')
        
    Returns:
        float: 效应量值
    """
    # 预计算共同的统计量
    n1, n2 = len(data1), len(data2)
    mean_diff = np.mean(data1) - np.mean(data2)
    
    if method == "cohen":
        # 使用向量化操作计算Cohen's d
        pooled_var = ((n1 - 1) * np.var(data1, ddof=1) + 
                     (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2)
        return mean_diff / np.sqrt(pooled_var)
    elif method == "hedges":
        # 优化的Hedges' g计算
        cohen_d = mean_diff / np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                                     (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        return cohen_d * (1 - 3 / (4 * (n1 + n2) - 9))
        return d * correction
    else:
        raise ValueError("不支持的效应量计算方法")

if __name__ == "__main__":
    try:
        # 运行统计测试模块
        print("\n=== 运行统计测试模块 ===")
        statistical_results = run_statistical_tests()
        
        # 导入可视化模块并绘制结果
        from visualization import plot_comparison_results
        if statistical_results:
            plot_comparison_results(statistical_results, metric_name="统计检验值")
            print("\n统计分析完成，可视化结果已保存到output文件夹")
    except Exception as e:
        print(f"\n运行过程中出现错误: {str(e)}")