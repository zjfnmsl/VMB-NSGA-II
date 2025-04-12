import numpy as np
from typing import List, Tuple, Callable
from nsga2 import NSGA2
from visualization import plot_dynamic_results

class HELPProblem:
    """HELP (Hyper-Ellipsoid Linear Problem) 动态测试函数"""
    
    def __init__(self, n_var: int = 30, t_change: int = 50):
        """初始化HELP问题
        
        Args:
            n_var: 决策变量维度
            t_change: 环境变化周期
        """
        self.n_var = n_var
        self.t_change = t_change
        self.t = 0  # 当前时间步
        self.weights = np.ones(n_var)  # 初始权重
        
    def update_environment(self) -> None:
        """更新环境（改变目标函数）"""
        self.t += 1
        if self.t % self.t_change == 0:
            # 更新权重
            self.weights = np.random.uniform(0.5, 1.5, self.n_var)
            
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """评估解的目标函数值
        
        Args:
            x: 决策变量
            
        Returns:
            np.ndarray: 目标函数值
        """
        f1 = np.sum(self.weights * x**2)
        f2 = np.sum(self.weights * (x - 1)**2)
        return np.array([f1, f2])

class LogisticsScheduling:
    """动态物流调度问题"""
    
    def __init__(self, n_vehicles: int = 10, n_orders: int = 50):
        """初始化物流调度问题
        
        Args:
            n_vehicles: 车辆数量
            n_orders: 订单数量
        """
        self.n_vehicles = n_vehicles
        self.n_orders = n_orders
        self.vehicle_capacity = 100  # 车辆容量
        self.reset()
        
    def reset(self) -> None:
        """重置问题状态"""
        # 生成随机订单位置
        self.order_locations = np.random.uniform(0, 100, (self.n_orders, 2))
        # 订单需求量
        self.order_demands = np.random.uniform(10, 30, self.n_orders)
        # 订单时间窗口
        self.time_windows = np.random.uniform(0, 480, (self.n_orders, 2))
        self.time_windows.sort(axis=1)  # 确保开始时间小于结束时间
        
    def add_dynamic_orders(self, n_new_orders: int) -> None:
        """添加新的动态订单
        
        Args:
            n_new_orders: 新订单数量
        """
        # 生成新订单
        new_locations = np.random.uniform(0, 100, (n_new_orders, 2))
        new_demands = np.random.uniform(10, 30, n_new_orders)
        new_time_windows = np.random.uniform(
            self.time_windows[-1, 1],
            self.time_windows[-1, 1] + 240,
            (n_new_orders, 2)
        )
        new_time_windows.sort(axis=1)
        
        # 更新问题状态
        self.order_locations = np.vstack([self.order_locations, new_locations])
        self.order_demands = np.concatenate([self.order_demands, new_demands])
        self.time_windows = np.vstack([self.time_windows, new_time_windows])
        self.n_orders += n_new_orders
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """评估调度方案
        
        Args:
            solution: 调度方案（车辆-订单分配矩阵）
            
        Returns:
            np.ndarray: [总行驶距离, 时间窗口违反度]
        """
        # 解码方案
        assignments = solution.reshape(self.n_vehicles, -1)
        
        total_distance = 0
        time_window_violations = 0
        
        for vehicle_id in range(self.n_vehicles):
            route = np.where(assignments[vehicle_id] > 0.5)[0]
            if len(route) == 0:
                continue
                
            # 计算路径距离
            route_locations = self.order_locations[route]
            distances = np.linalg.norm(route_locations[1:] - route_locations[:-1], axis=1)
            total_distance += np.sum(distances)
            
            # 检查时间窗口约束
            current_time = 0
            for order_id in route:
                service_time = 30  # 服务时间（分钟）
                if current_time < self.time_windows[order_id, 0]:
                    time_window_violations += self.time_windows[order_id, 0] - current_time
                elif current_time > self.time_windows[order_id, 1]:
                    time_window_violations += current_time - self.time_windows[order_id, 1]
                current_time += service_time
        
        return np.array([total_distance, time_window_violations])

def run_dynamic_experiment(
    problem: Callable,
    n_changes: int,
    n_runs: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """运行动态环境实验
    
    Args:
        problem: 动态问题实例
        n_changes: 环境变化次数
        n_runs: 重复运行次数
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 每次环境变化后的Pareto前沿
    """
    results = []
    
    for _ in range(n_runs):
        # 初始化NSGA2算法
        nsga2 = NSGA2(
            pop_size=100,
            n_gen=50,
            n_obj=2,
            n_var=30,
            var_bounds=np.array([[0, 1] for _ in range(30)]),
            crossover_rate=0.9,
            mutation_rate=0.1,
            virus_rate=0.1
        )
        
        change_results = []
        for _ in range(n_changes):
            # 运行算法
            pareto_front, pareto_objectives = nsga2.evolve(problem.evaluate)
            change_results.append((pareto_front, pareto_objectives))
            
            # 更新环境
            if isinstance(problem, HELPProblem):
                problem.update_environment()
            elif isinstance(problem, LogisticsScheduling):
                problem.add_dynamic_orders(5)  # 添加5个新订单
        
        results.append(change_results)
    
    return results