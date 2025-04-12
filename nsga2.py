import numpy as np
from typing import List, Tuple

class NSGA2:
    def __init__(self, pop_size: int, n_gen: int, n_obj: int, n_var: int,
                 var_bounds: np.ndarray, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9, virus_rate: float = 0.1,
                 virus_length: int = None, window_size: int = 5,
                 variance_threshold: float = 0.1):
        """
        初始化NSGA-II算法
        
        参数:
            pop_size: 种群大小
            n_gen: 迭代代数
            n_obj: 目标函数个数
            n_var: 决策变量个数
            var_bounds: 决策变量边界，shape为(n_var, 2)
            mutation_rate: 变异率
            crossover_rate: 交叉率
            virus_rate: 病毒感染率
            virus_length: 病毒长度
            window_size: 环境变化检测窗口大小
            variance_threshold: 环境变化检测阈值
        """
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_obj = n_obj
        self.n_var = n_var
        self.var_bounds = var_bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.virus_rate = virus_rate
        self.virus_length = virus_length if virus_length else self.n_var // 4  # 默认病毒长度为染色体长度的1/4
        self.virus_library = []  # 病毒库
        self.window_size = window_size  # 环境变化检测窗口大小
        self.variance_threshold = variance_threshold  # 环境变化检测阈值
        self.fitness_history = []  # 存储历史适应度值
        self.environment_features = []  # 存储环境特征

    def initialize_population(self) -> np.ndarray:
        """
        初始化种群
        """
        population = np.random.rand(self.pop_size, self.n_var)
        for i in range(self.n_var):
            population[:, i] = population[:, i] * (self.var_bounds[i, 1] - self.var_bounds[i, 0]) + self.var_bounds[i, 0]
        return population

    def fast_non_dominated_sort(self, objectives: np.ndarray) -> List[np.ndarray]:
        """
        快速非支配排序（优化版本）
        
        参数:
            objectives: 目标函数值矩阵，shape为(pop_size, n_obj)
        返回:
            fronts: 非支配层列表
        """
        n_points = objectives.shape[0]
        domination_count = np.zeros(n_points, dtype=np.int32)
        dominated_solutions = [set() for _ in range(n_points)]  # 使用集合提高查找效率
        fronts = [[]]  # 存储不同等级的解集

        # 使用numpy的广播和向量化操作计算支配关系
        for i in range(n_points):
            # 计算当前解与其他解的支配关系
            dominated_mask = np.all(objectives[i] <= objectives, axis=1) & \
                           np.any(objectives[i] < objectives, axis=1)
            dominating_mask = np.all(objectives <= objectives[i], axis=1) & \
                            np.any(objectives < objectives[i], axis=1)
            
            # 更新支配计数和被支配解集
            dominated_indices = np.where(dominated_mask)[0]
            dominating_indices = np.where(dominating_mask)[0]
            
            dominated_solutions[i].update(dominated_indices)
            domination_count[dominated_indices] += 1
            
            # 如果当前解不被任何解支配，加入第一层
            if len(dominating_indices) == 0:
                fronts[0].append(i)

        # 使用队列优化层级分配
        from collections import deque
        current_front = deque(fronts[0])
        front_no = 0
        
        while current_front:
            next_front = []
            while current_front:
                j = current_front.popleft()
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            
            front_no += 1
            if next_front:
                fronts.append(next_front)
                current_front.extend(next_front)

        return fronts

    def crowding_distance(self, objectives: np.ndarray, front: np.ndarray) -> np.ndarray:
        """
        计算拥挤度距离（优化版本）
        
        参数:
            objectives: 目标函数值矩阵
            front: 当前非支配层的索引
        返回:
            distances: 拥挤度距离
        """
        front_size = len(front)
        if front_size <= 2:
            return np.full(front_size, np.inf)

        distances = np.zeros(front_size)
        front_objectives = objectives[front]

        # 使用numpy的向量化操作计算拥挤度距离
        for obj in range(self.n_obj):
            # 对当前目标函数值排序并获取排序索引
            obj_values = front_objectives[:, obj]
            sorted_indices = np.argsort(obj_values)
            sorted_values = obj_values[sorted_indices]
            
            # 计算目标函数的范围
            obj_range = sorted_values[-1] - sorted_values[0]
            if obj_range == 0:
                continue

            # 设置边界点的距离为无穷大
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf

            # 使用向量化操作计算中间点的距离
            if front_size > 2:
                norm_diffs = (sorted_values[2:] - sorted_values[:-2]) / obj_range
                distances[sorted_indices[1:-1]] += norm_diffs

        return distances

    def tournament_selection(self, population: np.ndarray, objectives: np.ndarray,
                           k: int = 2) -> np.ndarray:
        """
        锦标赛选择（优化版本）
        使用批量处理和向量化操作提高选择效率
        """
        # 批量生成候选者索引，确保每次选择的数量不超过种群大小
        pop_len = len(population)
        k = min(k, pop_len)  # 限制k不超过种群大小
        selected = np.zeros((self.pop_size, self.n_var))
        
        # 分批处理选择，避免采样数量超过总体
        candidates = np.zeros((self.pop_size, k), dtype=int)
        for i in range(self.pop_size):
            candidates[i] = np.random.choice(pop_len, k, replace=False)
        
        # 批量处理每组候选者
        for i in range(self.pop_size):
            group_candidates = candidates[i]
            group_objectives = objectives[group_candidates]
            
            # 快速确定非支配关系
            dominated = np.zeros(k, dtype=bool)
            for j in range(k):
                for l in range(k):
                    if j != l:
                        if np.all(group_objectives[l] <= group_objectives[j]) and \
                           np.any(group_objectives[l] < group_objectives[j]):
                            dominated[j] = True
                            break
            
            non_dominated = np.where(~dominated)[0]
            
            if len(non_dominated) == 1:
                selected[i] = population[group_candidates[non_dominated[0]]]
            else:
                # 计算非支配解的拥挤度距离
                crowding_dist = self.crowding_distance(group_objectives, non_dominated)
                winner_idx = non_dominated[np.argmax(crowding_dist)]
                selected[i] = population[group_candidates[winner_idx]]
        
        return selected

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟二进制交叉(SBX)（优化版本）
        使用向量化操作提高交叉效率
        """
        nc = 20  # 交叉分布指数
        child1, child2 = parent1.copy(), parent2.copy()
        
        if np.random.random() < self.crossover_rate:
            # 生成随机掩码，决定哪些位置需要交叉
            cross_mask = np.random.random(self.n_var) <= 0.5
            
            # 计算需要交叉的位置
            diff_mask = np.abs(parent1 - parent2) > 1e-14
            cross_positions = cross_mask & diff_mask
            
            if np.any(cross_positions):
                # 获取交叉位置的边界值
                lb = self.var_bounds[cross_positions, 0]
                ub = self.var_bounds[cross_positions, 1]
                
                # 获取交叉位置的父代值
                p1 = parent1[cross_positions]
                p2 = parent2[cross_positions]
                
                # 确保p1小于p2
                y1 = np.minimum(p1, p2)
                y2 = np.maximum(p1, p2)
                
                # 计算beta值
                rand = np.random.random(len(y1))
                beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                alpha = 2.0 - beta ** -(nc + 1)
                
                # 向量化计算beta
                mask = rand <= 1.0 / alpha
                beta[mask] = (rand[mask] * alpha[mask]) ** (1.0 / (nc + 1))
                beta[~mask] = (1.0 / (2.0 - rand[~mask] * alpha[~mask])) ** (1.0 / (nc + 1))
                
                # 向量化计算子代
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                
                # 边界处理
                child1[cross_positions] = np.clip(c1, lb, ub)
                child2[cross_positions] = np.clip(c2, lb, ub)
        
        return child1, child2

    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        多项式变异（优化版本）
        使用向量化操作提高变异效率
        """
        nm = 20  # 变异分布指数
        mutated = individual.copy()
        
        # 生成变异掩码
        mutation_mask = np.random.random(self.n_var) < self.mutation_rate
        
        if np.any(mutation_mask):
            # 获取需要变异的位置的值和边界
            y = individual[mutation_mask]
            lb = self.var_bounds[mutation_mask, 0]
            ub = self.var_bounds[mutation_mask, 1]
            
            # 计算delta值
            delta1 = (y - lb) / (ub - lb)
            delta2 = (ub - y) / (ub - lb)
            
            # 生成随机值并计算变异幂
            rand = np.random.random(len(y))
            mut_pow = 1.0 / (nm + 1.0)
            
            # 向量化计算变异值
            mask = rand <= 0.5
            deltaq = np.zeros_like(y)
            
            # 处理rand <= 0.5的情况
            xy = 1.0 - delta1[mask]
            val = 2.0 * rand[mask] + (1.0 - 2.0 * rand[mask]) * (xy ** (nm + 1.0))
            deltaq[mask] = val ** mut_pow - 1.0
            # 处理rand > 0.5的情况
            xy = 1.0 - delta2[~mask]
            val = 2.0 * (1.0 - rand[~mask]) + 2.0 * (rand[~mask] - 0.5) * (xy ** (nm + 1.0))
            deltaq[~mask] = 1.0 - val ** mut_pow
            
            # 计算变异后的值
            mutated[mutation_mask] = y + deltaq * (ub - lb)
            
            # 边界处理
            mutated[mutation_mask] = np.clip(mutated[mutation_mask], lb, ub)
                
        return mutated

    def extract_environment_features(self, objectives: np.ndarray) -> np.ndarray:
        """
        提取当前环境的特征
        """
        # 计算目标函数的统计特征
        mean = np.mean(objectives, axis=0)
        std = np.std(objectives, axis=0)
        gradient = np.gradient(objectives.mean(axis=1))
        return np.concatenate([mean, std, [gradient[-1]]])

    def detect_environment_change(self, objectives: np.ndarray) -> bool:
        """
        检测环境是否发生变化
        """
        self.fitness_history.append(objectives.mean())
        if len(self.fitness_history) > self.window_size:
            self.fitness_history.pop(0)
            variance = np.var(self.fitness_history)
            return variance > self.variance_threshold
        return False

    def evaluate_virus_utility(self, virus: np.ndarray, objectives: np.ndarray) -> float:
        """
        评估病毒片段的效用值
        """
        # 计算病毒所在解的拥挤度距离
        fronts = self.fast_non_dominated_sort(objectives)
        crowding_dist = self.crowding_distance(objectives, np.array(fronts[0]))
        # 返回归一化的效用值
        return float(np.mean(crowding_dist))

    def generate_virus(self, population: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        """
        从当前种群中生成病毒
        """
        # 选择最优个体作为病毒来源
        fronts = self.fast_non_dominated_sort(objectives)
        elite_idx = fronts[0][0]
        elite = population[elite_idx]
        
        # 随机选择一段基因序列作为病毒
        start = np.random.randint(0, self.n_var - self.virus_length)
        virus = elite[start:start + self.virus_length]
        
        # 记录环境特征和评估病毒效用
        env_features = self.extract_environment_features(objectives)
        utility = self.evaluate_virus_utility(virus, objectives)
        
        # 将病毒及其相关信息存入病毒库
        self.virus_library.append({
            'virus': virus,
            'env_features': env_features,
            'utility': utility,
            'usage_count': 0
        })
        
        return virus


    def infect_population(self, population: np.ndarray, virus: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        """
        使用病毒感染种群中的部分个体
        """
        infected_pop = population.copy()
        n_infected = int(self.pop_size * self.virus_rate)
        
        # 随机选择要感染的个体
        infected_idx = np.random.choice(self.pop_size, n_infected, replace=False)
        
        for idx in infected_idx:
            # 随机选择感染位置
            start = np.random.randint(0, self.n_var - len(virus))
            # 直接替换基因片段
            infected_pop[idx, start:start + len(virus)] = virus
            
            # 确保变量在边界范围内
            for i in range(start, start + len(virus)):
                infected_pop[idx, i] = np.clip(
                    infected_pop[idx, i],
                    self.var_bounds[i, 0],
                    self.var_bounds[i, 1]
                )
        
        return infected_pop

    def select_best_virus(self, current_env_features: np.ndarray) -> np.ndarray:
        """
        从病毒库中选择最匹配的病毒
        """
        if not self.virus_library:
            return None

        # 计算当前环境与病毒库中环境的相似度
        similarities = []
        for virus_info in self.virus_library:
            env_similarity = np.dot(current_env_features, virus_info['env_features']) / \
                           (np.linalg.norm(current_env_features) * np.linalg.norm(virus_info['env_features']))
            # 综合考虑环境相似度、效用值和使用次数
            score = env_similarity * virus_info['utility'] * (1 + np.log1p(virus_info['usage_count']))
            similarities.append(score)

        best_idx = np.argmax(similarities)
        self.virus_library[best_idx]['usage_count'] += 1
        return self.virus_library[best_idx]['virus']

    def evolve(self, objectives_func) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行NSGA-II算法的主循环
        
        参数:
            objectives_func: 目标函数，接受决策变量返回目标值
        返回:
            最优解集和对应的目标函数值
        """
        # 初始化种群
        population = self.initialize_population()
        objectives = np.array([objectives_func(ind) for ind in population])
        
        for _ in range(self.n_gen):
            # 检测环境变化
            env_changed = self.detect_environment_change(objectives)
            
            # 选择父代
            parents = self.tournament_selection(population, objectives)
            offspring = np.zeros_like(population)
            
            # 产生子代
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[min(i+1, self.pop_size-1)]
                c1, c2 = self.crossover(p1, p2)
                offspring[i] = self.mutation(c1)
                if i+1 < self.pop_size:
                    offspring[i+1] = self.mutation(c2)
            
            # 评价子代
            offspring_objectives = np.array([objectives_func(ind) for ind in offspring])
            
            # 合并父代和子代
            combined_pop = np.vstack((population, offspring))
            combined_obj = np.vstack((objectives, offspring_objectives))
            
            # 非支配排序
            fronts = self.fast_non_dominated_sort(combined_obj)
            
            # 选择下一代种群
            new_pop = []
            new_obj = []
            front_no = 0
            
            while len(new_pop) + len(fronts[front_no]) <= self.pop_size:
                for idx in fronts[front_no]:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
                front_no += 1
                
            if len(new_pop) < self.pop_size:
                crowding_distances = self.crowding_distance(combined_obj, np.array(fronts[front_no]))
                sorted_indices = np.argsort(-crowding_distances)
                for idx in sorted_indices[:self.pop_size - len(new_pop)]:
                    new_pop.append(combined_pop[fronts[front_no][idx]])
                    new_obj.append(combined_obj[fronts[front_no][idx]])
            
            population = np.array(new_pop)
            objectives = np.array(new_obj)
            
            # 病毒感染操作
            if env_changed or np.random.random() < self.virus_rate:
                # 生成新病毒并存入病毒库
                virus = self.generate_virus(population, objectives)
                
                # 如果检测到环境变化，优先使用最匹配的病毒
                if env_changed and len(self.virus_library) > 0:
                    current_env_features = self.extract_environment_features(objectives)
                    best_virus = self.select_best_virus(current_env_features)
                    if best_virus is not None:
                        virus = best_virus
                
                population = self.infect_population(population, virus, objectives)
                objectives = np.array([objectives_func(ind) for ind in population])
        
        # 获取最终的非支配解
        fronts = self.fast_non_dominated_sort(objectives)
        return population[fronts[0]], objectives[fronts[0]]