import numpy as np
import matplotlib.pyplot as plt
from nsga2 import NSGA2

def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1测试函数"""
    f1 = x[0]
    g = 1.0 + 9.0 * np.mean(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def plot_comparison(nsga2_objectives, vmb_nsga2_objectives):
    """绘制两种算法的对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1a：传统NSGA-II
    ax1.scatter(nsga2_objectives[:, 0], nsga2_objectives[:, 1], 
               c='b', s=30, alpha=0.6, label='Pareto front')
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    ax1.set_title('图1a: 传统NSGA-II')
    ax1.grid(True)
    ax1.legend()

    # 图1b：VMB-NSGA-II
    ax2.scatter(vmb_nsga2_objectives[:, 0], vmb_nsga2_objectives[:, 1], 
               c='r', s=30, alpha=0.6, label='Pareto front')
    ax2.set_xlabel('f1')
    ax2.set_ylabel('f2')
    ax2.set_title('图1b: VMB-NSGA-II')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('output/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 问题参数设置
    n_var = 30
    pop_size = 100
    n_gen = 200
    n_obj = 2
    var_bounds = np.array([[0, 1] for _ in range(n_var)])

    # 传统NSGA-II
    traditional_nsga2 = NSGA2(
        pop_size=pop_size,
        n_gen=n_gen,
        n_obj=n_obj,
        n_var=n_var,
        var_bounds=var_bounds,
        crossover_rate=0.9,
        mutation_rate=0.1,
        virus_rate=0.0  # 不使用病毒机制
    )
    _, nsga2_objectives = traditional_nsga2.evolve(zdt1)

    # VMB-NSGA-II
    vmb_nsga2 = NSGA2(
        pop_size=pop_size,
        n_gen=n_gen,
        n_obj=n_obj,
        n_var=n_var,
        var_bounds=var_bounds,
        crossover_rate=0.9,
        mutation_rate=0.1,
        virus_rate=0.1  # 使用病毒机制
    )
    _, vmb_objectives = vmb_nsga2.evolve(zdt1)

    # 绘制对比图
    plot_comparison(nsga2_objectives, vmb_objectives)

if __name__ == "__main__":
    main()