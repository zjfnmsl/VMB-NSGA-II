import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import entropy
from scipy.linalg import eig

class MarkovConvergenceAnalyzer:
    """基于马尔可夫链模型的算法收敛性分析"""
    
    def __init__(self, n_states: int):
        """初始化马尔可夫链分析器
        
        Args:
            n_states: 状态空间大小（如种群中的非支配层数）
        """
        self.n_states = n_states
        self.transition_matrix = np.zeros((n_states, n_states))
        self.state_history: List[int] = []
        
    def update_transition_matrix(self, from_state: int, to_state: int) -> None:
        """更新转移矩阵
        
        Args:
            from_state: 起始状态
            to_state: 目标状态
        """
        self.state_history.append(to_state)
        # 统计转移次数
        counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(self.state_history) - 1):
            counts[self.state_history[i], self.state_history[i + 1]] += 1
            
        # 计算转移概率
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        self.transition_matrix = counts / row_sums
        
    def analyze_convergence(self) -> Tuple[bool, float, np.ndarray]:
        """分析马尔可夫链的收敛性
        
        Returns:
            Tuple[bool, float, np.ndarray]:
                - 是否收敛
                - 收敛速率
                - 稳态分布
        """
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eig(self.transition_matrix.T)
        
        # 找到最大特征值对应的特征向量（稳态分布）
        max_idx = np.argmax(np.real(eigenvalues))
        stationary_dist = np.real(eigenvectors[:, max_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        
        # 计算收敛速率（第二大特征值的模）
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        convergence_rate = sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0
        
        # 判断是否收敛（第二大特征值是否小于1）
        is_convergent = convergence_rate < 1
        
        return is_convergent, convergence_rate, stationary_dist

class PopulationEntropyAnalyzer:
    """种群熵分析器，用于评估种群多样性"""
    
    def __init__(self):
        self.entropy_history: List[float] = []
        
    def calculate_population_entropy(self, 
                                    population: np.ndarray,
                                    n_bins: int = 10) -> float:
        """计算种群的熵值
        
        Args:
            population: 种群个体
            n_bins: 直方图分箱数
            
        Returns:
            float: 种群熵值
        """
        # 计算每个维度的熵
        dimension_entropies = []
        for dim in range(population.shape[1]):
            hist, _ = np.histogram(population[:, dim], bins=n_bins, density=True)
            hist = hist[hist > 0]  # 移除零概率事件
            dim_entropy = entropy(hist)
            dimension_entropies.append(dim_entropy)
        
        # 返回平均熵
        mean_entropy = np.mean(dimension_entropies)
        self.entropy_history.append(mean_entropy)
        return mean_entropy
    
    def analyze_virus_impact(self,
                           population_before: np.ndarray,
                           population_after: np.ndarray,
                           n_bins: int = 10) -> Tuple[float, float, float]:
        """分析病毒注入对种群熵的影响
        
        Args:
            population_before: 病毒注入前的种群
            population_after: 病毒注入后的种群
            n_bins: 直方图分箱数
            
        Returns:
            Tuple[float, float, float]:
                - 注入前熵值
                - 注入后熵值
                - 熵变化率
        """
        entropy_before = self.calculate_population_entropy(population_before, n_bins)
        entropy_after = self.calculate_population_entropy(population_after, n_bins)
        entropy_change_rate = (entropy_after - entropy_before) / entropy_before
        
        return entropy_before, entropy_after, entropy_change_rate
    
    def get_entropy_trend(self) -> Tuple[List[float], Optional[float]]:
        """获取熵值变化趋势
        
        Returns:
            Tuple[List[float], Optional[float]]:
                - 熵值历史记录
                - 熵值变化率（如果有足够数据）
        """
        if len(self.entropy_history) < 2:
            return self.entropy_history, None
            
        # 计算熵值变化率（使用线性回归的斜率）
        x = np.arange(len(self.entropy_history))
        y = np.array(self.entropy_history)
        slope = np.polyfit(x, y, 1)[0]
        
        return self.entropy_history, slope


def run_theoretical_analysis() -> Tuple[np.ndarray, np.ndarray, int]:
    """运行理论分析模块
    
    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
            - Pareto前沿解集
            - 目标函数值
            - 决策变量维度
    """
    # 示例：生成一个简单的测试问题和种群
    n_vars = 30  # 决策变量维度
    population_size = 100
    n_states = 5  # 非支配分层数
    
    # 生成随机种群
    population = np.random.random((population_size, n_vars))
    
    # 1. 马尔可夫链收敛性分析
    print("\n执行马尔可夫链收敛性分析...")
    markov_analyzer = MarkovConvergenceAnalyzer(n_states)
    
    # 模拟状态转移
    for i in range(20):
        from_state = np.random.randint(n_states)
        to_state = np.random.randint(n_states)
        markov_analyzer.update_transition_matrix(from_state, to_state)
    
    # 分析收敛性
    is_convergent, conv_rate, stationary_dist = markov_analyzer.analyze_convergence()
    print(f"收敛性分析结果:\n  是否收敛: {is_convergent}\n  收敛速率: {conv_rate:.4f}")
    
    # 2. 种群熵分析
    print("\n执行种群熵分析...")
    entropy_analyzer = PopulationEntropyAnalyzer()
    
    # 计算初始种群熵
    initial_entropy = entropy_analyzer.calculate_population_entropy(population)
    print(f"初始种群熵: {initial_entropy:.4f}")
    
    # 模拟种群进化
    evolved_population = population + 0.1 * np.random.randn(population_size, n_vars)
    evolved_population = np.clip(evolved_population, 0, 1)
    
    # 分析病毒注入影响
    entropy_before, entropy_after, change_rate = entropy_analyzer.analyze_virus_impact(
        population, evolved_population)
    print(f"病毒注入影响分析:\n  注入前熵值: {entropy_before:.4f}\n  注入后熵值: {entropy_after:.4f}\n  熵变化率: {change_rate:.4f}")
    
    # 生成Pareto前沿（示例）
    pareto_front = np.random.random((50, n_vars))  # 生成与决策变量维度匹配的Pareto前沿
    objectives = np.random.random((population_size, 2))  # 双目标优化问题
    
    return pareto_front, objectives, n_vars

if __name__ == "__main__":
    try:
        # 运行理论分析
        print("\n=== 运行理论分析模块 ===")
        pareto_front, objectives, n_vars = run_theoretical_analysis()
        
        # 导入可视化模块并绘制结果
        from visualization import plot_results
        if pareto_front is not None and objectives is not None:
            plot_results(pareto_front, objectives, n_vars)
            print("\n理论分析完成，可视化结果已保存到output文件夹")
    except Exception as e:
        print(f"\n运行过程中出现错误: {str(e)}")