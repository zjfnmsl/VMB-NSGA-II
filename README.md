# 基于病毒记忆库机制的NSGA-II算法实现

## 项目简介
本项目实现了一个改进的NSGA-II（非支配排序遗传算法II）算法，通过引入病毒记忆库（Virus Memory Bank, VMB）机制来提高算法在动态多目标优化问题中的性能。该算法能够更好地保持种群多样性，加快收敛速度，并提高对环境变化的适应能力。

## 主要功能

### 1. 算法核心功能
- 传统NSGA-II算法实现
- 病毒记忆库机制集成
- 环境变化检测
- 动态种群更新
- Pareto前沿解集生成

### 2. 分析工具
- 理论分析模块
- 实验对比分析
- 统计检验功能
- 可视化结果展示

## 安装说明

### 环境要求
项目依赖包已在requirements.txt中列出，可通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本使用
```python
from nsga2 import NSGA2
import numpy as np

# 设置问题参数
n_var = 30  # 决策变量个数
pop_size = 100  # 种群规模
n_gen = 200  # 迭代代数
n_obj = 2  # 目标函数个数
var_bounds = np.array([[0, 1] for _ in range(n_var)])

# 创建算法实例
nsga2 = NSGA2(
    pop_size=pop_size,
    n_gen=n_gen,
    n_obj=n_obj,
    n_var=n_var,
    var_bounds=var_bounds,
    crossover_rate=0.9,
    mutation_rate=0.1,
    virus_rate=0.1  # 启用病毒机制
)

# 运行算法
pareto_front, objectives = nsga2.evolve(problem_function)
```

### 2. 结果分析
```python
from visualization_manager import VisualizationManager

# 创建可视化管理器
vm = VisualizationManager()

# 可视化结果
vm.visualize_statistical_results(results_dict)
vm.visualize_sensitivity_analysis(sensitivity_results)
vm.visualize_theoretical_analysis(population_history)
```

## 实验结果

### 性能对比
- 与传统NSGA-II相比，VMB-NSGA-II在以下方面表现更优：
  - 收敛速度提升
  - 种群多样性维持
  - 对环境变化的适应能力增强

### 可视化展示
项目在`output`目录下提供了丰富的可视化结果，包括：
- 算法收敛过程
- Pareto前沿对比
- 参数敏感性分析
- 统计检验结果

## 项目结构
```
├── nsga2.py           # 核心算法实现
├── compare.py         # 算法对比实验
├── example.py         # 使用示例
├── experiments.py     # 实验模块
├── theoretical_analysis.py  # 理论分析
├── statistical_tests.py     # 统计检验
├── visualization.py   # 可视化工具
├── output/           # 输出结果目录
└── requirements.txt   # 项目依赖
```

## 参数配置说明

### 核心参数
- `pop_size`: 种群规模
- `n_gen`: 迭代代数
- `n_obj`: 目标函数个数
- `n_var`: 决策变量个数
- `crossover_rate`: 交叉率
- `mutation_rate`: 变异率
- `virus_rate`: 病毒感染率
- `virus_length`: 病毒长度
- `window_size`: 环境变化检测窗口大小
- `variance_threshold`: 环境变化检测阈值