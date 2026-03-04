"""
强化学习基础 + Q-Learning 实现
==============================
面试常考点：
1. 理解 MDP (马尔可夫决策过程) 的五元组 (S, A, P, R, γ)
2. 贝尔曼方程推导
3. Q-Learning 更新公式：Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
4. ε-greedy 探索策略

环境：简单的 GridWorld（4x4 格子世界）
- 起点: (0,0)  终点: (3,3)
- 障碍格子: (1,1), (2,2)
- 每步奖励: -1，到达终点: +10，撞障碍: -5
"""

import numpy as np
import random
from collections import defaultdict


# ============================================================
# 1. GridWorld 环境（模拟 OpenAI Gym 接口）
# ============================================================

class GridWorld:
    """
    4x4 格子世界环境
    状态: (row, col) 元组
    动作: 0=上, 1=下, 2=左, 3=右
    """

    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(1, 1), (2, 2)}  # 障碍格子
        self.actions = [0, 1, 2, 3]        # 上下左右
        self.action_map = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
        }
        self.state = self.start

    def reset(self):
        """重置环境，返回初始状态"""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done, info)
        这是 Gym 标准接口
        """
        dr, dc = self.action_map[action]
        r, c = self.state
        nr, nc = r + dr, c + dc

        # 边界检查：越界则原地不动
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = (nr, nc)
        else:
            next_state = self.state

        # 计算奖励
        if next_state == self.goal:
            reward = 10.0
            done = True
        elif next_state in self.obstacles:
            reward = -5.0
            done = False
        else:
            reward = -1.0
            done = False

        self.state = next_state
        return next_state, reward, done, {}

    def render(self, q_table=None):
        """可视化当前状态"""
        symbols = {self.goal: 'G', self.start: 'S'}
        for obs in self.obstacles:
            symbols[obs] = 'X'

        print("\n当前格子世界：")
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                pos = (r, c)
                if pos == self.state:
                    row_str += " @ "
                elif pos in symbols:
                    row_str += f" {symbols[pos]} "
                else:
                    row_str += " . "
            print(row_str)
        print()


# ============================================================
# 2. Q-Learning 算法
# ============================================================

class QLearning:
    """
    Q-Learning（离线策略 TD 方法）

    核心公式（贝尔曼最优方程的 TD 近似）：
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]
               ↑             ↑        ↑
             旧值        TD误差     目标值

    特点：off-policy，用贪心策略更新，但用 ε-greedy 探索
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        参数：
        - alpha (α): 学习率，控制更新步长
        - gamma (γ): 折扣因子，控制未来奖励的权重
        - epsilon (ε): ε-greedy 中随机探索的概率
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q 表：Q[state][action] = 期望累计奖励
        # 用 defaultdict 自动初始化为 0
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))

    def choose_action(self, state):
        """
        ε-greedy 策略：
        - 以 ε 概率随机探索（exploration）
        - 以 1-ε 概率选最优动作（exploitation）
        """
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)  # 随机探索
        else:
            return np.argmax(self.q_table[state])   # 贪心利用

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 核心更新步骤
        手撕重点：必须背下这个公式
        """
        # 当前 Q 值
        current_q = self.q_table[state][action]

        # 目标 Q 值（贝尔曼最优方程）
        if done:
            target_q = reward  # 终止状态没有未来奖励
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # TD 误差（Temporal Difference Error）
        td_error = target_q - current_q

        # 更新 Q 值
        self.q_table[state][action] += self.alpha * td_error

        return td_error

    def train(self, num_episodes=500, max_steps=100):
        """训练主循环"""
        rewards_history = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                # 1. 选动作
                action = self.choose_action(state)

                # 2. 执行动作，得到 (s', r, done)
                next_state, reward, done, _ = self.env.step(action)

                # 3. 更新 Q 表
                self.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break

            rewards_history.append(total_reward)

            # 逐渐减小 epsilon（让智能体越来越倾向于利用）
            self.epsilon = max(0.01, self.epsilon * 0.995)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode+1:4d} | "
                      f"平均奖励: {avg_reward:6.2f} | "
                      f"ε: {self.epsilon:.3f}")

        return rewards_history

    def get_policy(self):
        """提取最优策略（每个状态下的最优动作）"""
        action_names = ['↑', '↓', '←', '→']
        policy = {}
        for r in range(self.env.size):
            for c in range(self.env.size):
                state = (r, c)
                best_action = np.argmax(self.q_table[state])
                policy[state] = action_names[best_action]
        return policy


# ============================================================
# 3. SARSA（On-policy TD 方法，和 Q-Learning 对比）
# ============================================================

class SARSA(QLearning):
    """
    SARSA（on-policy TD 方法）

    Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') - Q(s,a)]
                                        ↑
                                  实际执行的 a'（不是 max）

    与 Q-Learning 的区别：
    - Q-Learning: 用 max Q(s',a') → off-policy（乐观估计）
    - SARSA:      用 Q(s', 实际a') → on-policy（保守估计）
    """

    def update(self, state, action, reward, next_state, done, next_action=None):
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            # SARSA 用下一个实际动作的 Q 值（不是 max）
            target_q = reward + self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] += self.alpha * (target_q - current_q)

    def train(self, num_episodes=500, max_steps=100):
        """SARSA 训练：需要提前知道 next_action"""
        rewards_history = []

        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)  # 先选好第一个动作
            total_reward = 0

            for step in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)  # 选下一个动作

                # SARSA 更新：用 (s, a, r, s', a') 五元组
                self.update(state, action, reward, next_state, done, next_action)

                total_reward += reward
                state = next_state
                action = next_action

                if done:
                    break

            rewards_history.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode+1:4d} | 平均奖励: {avg_reward:6.2f}")

        return rewards_history


# ============================================================
# 4. 运行演示
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Q-Learning 训练 GridWorld")
    print("=" * 50)

    env = GridWorld(size=4)
    agent = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.5)
    agent.train(num_episodes=500)

    print("\n学到的最优策略：")
    policy = agent.get_policy()
    for r in range(env.size):
        row = ""
        for c in range(env.size):
            pos = (r, c)
            if pos == env.goal:
                row += " G "
            elif pos in env.obstacles:
                row += " X "
            else:
                row += f" {policy[pos]} "
        print(row)

    print("\n" + "=" * 50)
    print("测试训练好的智能体")
    print("=" * 50)
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    agent.epsilon = 0  # 关闭探索，纯利用

    while not done and steps < 20:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        action_names = ['上', '下', '左', '右']
        print(f"  {state} --{action_names[action]}--> {next_state}, 奖励={reward}")
        total_reward += reward
        state = next_state
        steps += 1

    print(f"\n总奖励: {total_reward}, 用了 {steps} 步")
    print("成功到达终点！" if done else "未能到达终点")
