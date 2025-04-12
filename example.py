import numpy as np
from nsga2 import NSGA2

def zdt1(x: np.ndarray) -> np.ndarray:
    """
    ZDT1测试函数
    f1 = x1
    f2 = g(x) * [1 - sqrt(x1/g(x))]
    where g(x) = 1 + 9*sum(x[2:n])/(n-1)
    x_i ∈ [0,1], i=1,2,...,n
    """
    f1 = x[0]
    g = 1.0 + 9.0 * np.mean(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def main():
    # 问题参数设置
    n_var = 30  # 决策变量个数
    pop_size = 300  # 增加种群规模以提供更大的搜索空间
    n_gen = 500  # 迭代代数
    n_obj = 2  # 目标函数个数
    var_bounds = np.array([[0, 1] for _ in range(n_var)])  # 决策变量边界

    # 创建NSGA-II算法实例
    nsga2 = NSGA2(
        pop_size=pop_size,
        n_gen=n_gen,
        n_obj=n_obj,
        n_var=n_var,
        var_bounds=var_bounds,
        crossover_rate=0.98,  # 提高交叉率以增强种群多样性
        mutation_rate=0.1,    # 降低变异率以减少随机扰动
        virus_rate=0.1        # 优化病毒感染率
    )

    # 运行算法
    pareto_front, pareto_objectives = nsga2.evolve(zdt1)

    # 打印结果
    print("\nPareto最优解集:")
    print(pareto_front)
    print("\n对应的目标函数值:")
    print(pareto_objectives)

    # 可视化结果
    from visualization import plot_results
    plot_results(pareto_front, pareto_objectives, n_var)

if __name__ == "__main__":
    main()