import numpy as np
from typing import List, Tuple, Dict, Any
from nsga2 import NSGA2
from statistical_tests import statistical_analysis, effect_size_analysis
from visualization import plot_comparison_results, plot_sensitivity_analysis

class MOEADIRA:
    """MOEA/D-IRA算法的简化实现，用于对比实验"""
    def __init__(self, pop_size: int, n_gen: int, n_obj: int, n_var: int,
                 var_bounds: np.ndarray):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_obj = n_obj
        self.n_var = n_var
        self.var_bounds = var_bounds
        
    def evolve(self, obj_func):
        # 优化的MOEA/D-IRA实现
        population = np.random.uniform(
            self.var_bounds[:, 0],
            self.var_bounds[:, 1],
            size=(self.pop_size, self.n_var)
        )
        
        # 预分配内存以存储目标值
        objectives = np.zeros((self.pop_size, 2))
        
        for _ in range(self.n_gen):
            # 向量化的进化操作
            noise = np.random.normal(0, 0.1, population.shape)
            offspring = np.clip(population + noise, self.var_bounds[:, 0], self.var_bounds[:, 1])
            
            # 向量化的目标函数评估
            objectives = np.array([obj_func(x) for x in population])
            offspring_objectives = np.array([obj_func(x) for x in offspring])
            
            # 高效的种群更新
            combined = np.vstack([population, offspring])
            combined_objectives = np.vstack([objectives, offspring_objectives])
            
            # 使用numpy的高效索引操作
            indices = np.argsort(combined_objectives[:, 0])[:self.pop_size]
            population = combined[indices]
            objectives = combined_objectives[indices]
            
        return population, objectives

def run_comparison_experiment(obj_func, n_runs: int = 30) -> Dict[str, List[float]]:
    """运行算法对比实验
    
    Args:
        obj_func: 目标函数
        n_runs: 重复运行次数
        
    Returns:
        Dict[str, List[float]]: 各算法的性能指标结果
    """
    # 实验参数设置
    n_var = 30
    pop_size = 300
    n_gen = 500
    n_obj = 2
    var_bounds = np.array([[0, 1] for _ in range(n_var)])
    
    results = {"NSGA2-VM": [], "MOEA/D-IRA": []}
    
    for _ in range(n_runs):
        # 运行NSGA2-VM
        nsga2 = NSGA2(
            pop_size=pop_size,
            n_gen=n_gen,
            n_obj=n_obj,
            n_var=n_var,
            var_bounds=var_bounds,
            crossover_rate=0.98,
            mutation_rate=0.1,
            virus_rate=0.1
        )
        _, nsga2_objectives = nsga2.evolve(obj_func)
        
        # 运行MOEA/D-IRA
        moead = MOEADIRA(
            pop_size=pop_size,
            n_gen=n_gen,
            n_obj=n_obj,
            n_var=n_var,
            var_bounds=var_bounds
        )
        _, moead_objectives = moead.evolve(obj_func)
        
        # 计算性能指标（如IGD）
        results["NSGA2-VM"].append(calculate_igd(nsga2_objectives))
        results["MOEA/D-IRA"].append(calculate_igd(moead_objectives))
    
    return results

def parameter_sensitivity_analysis(
    obj_func,
    param_ranges: Dict[str, List[float]],
    n_runs: int = 10
) -> Dict[str, Dict[float, List[float]]]:
    """参数敏感性分析
    
    Args:
        obj_func: 目标函数
        param_ranges: 参数及其取值范围
        n_runs: 每个参数配置的重复运行次数
        
    Returns:
        Dict[str, Dict[float, List[float]]]: 参数敏感性分析结果
    """
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    base_params = {
        "pop_size": 300,
        "n_gen": 500,
        "n_obj": 2,
        "n_var": 30,
        "var_bounds": np.array([[0, 1] for _ in range(30)]),
        "crossover_rate": 0.98,
        "mutation_rate": 0.1,
        "virus_rate": 0.1
    }
    
    def run_single_experiment(param, value, base_params, obj_func):
        current_params = base_params.copy()
        current_params[param] = value
        nsga2 = NSGA2(**current_params)
        _, objectives = nsga2.evolve(obj_func)
        return calculate_igd(objectives)
    
    results = {param: {} for param in param_ranges}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for param, values in param_ranges.items():
            for value in values:
                # 并行执行多次实验
                run_experiment = partial(run_single_experiment, param, value, base_params, obj_func)
                param_results = list(executor.map(lambda _: run_experiment(), range(n_runs)))
                results[param][value] = param_results
    
    return results

def calculate_igd(objectives: np.ndarray, reference_front: np.ndarray = None) -> float:
    """计算IGD (Inverted Generational Distance)指标
    
    Args:
        objectives: 目标函数值
        reference_front: 参考前沿（如果为None则使用真实Pareto前沿）
        
    Returns:
        float: IGD值
    """
    if reference_front is None:
        # 使用真实Pareto前沿（对于ZDT1问题）
        x = np.linspace(0, 1, 100)
        reference_front = np.column_stack((x, 1 - np.sqrt(x)))
    
    # 使用广播和向量化操作计算距离
    diff = objectives[:, np.newaxis] - reference_front
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    min_distances = np.min(distances, axis=0)
    
    return np.mean(min_distances)


def run_experiments():
    """运行实验模块，包括算法对比实验和参数敏感性分析
    
    Returns:
        Tuple[Dict[str, List[float]], Dict[str, Dict[float, List[float]]]]: 
            - 算法对比实验结果
            - 参数敏感性分析结果
    """
    # 定义测试问题（ZDT1）
    def zdt1(x):
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return [f1, f2]
    
    # 运行算法对比实验
    print("\n执行算法对比实验...")
    experiment_results = run_comparison_experiment(zdt1)
    
    # 运行参数敏感性分析
    print("\n执行参数敏感性分析...")
    param_ranges = {
        "crossover_rate": [0.7, 0.8, 0.9, 1.0],
        "mutation_rate": [0.05, 0.1, 0.15, 0.2],
        "virus_rate": [0.05, 0.1, 0.15, 0.2]
    }
    sensitivity_results = parameter_sensitivity_analysis(zdt1, param_ranges)
    
    return experiment_results, sensitivity_results

if __name__ == "__main__":
    try:
        # 运行实验模块
        print("\n=== 运行实验模块 ===")
        experiment_results, sensitivity_results = run_experiments()
        
        # 导入可视化模块并绘制结果
        from visualization import plot_comparison_results, plot_sensitivity_analysis
        if experiment_results:
            plot_comparison_results(experiment_results, metric_name="IGD")
        if sensitivity_results:
            plot_sensitivity_analysis(sensitivity_results)
            
        print("\n实验分析完成，可视化结果已保存到output文件夹")
    except Exception as e:
        print(f"\n运行过程中出现错误: {str(e)}")