"""
Policy Gradient - REINFORCE 算法
=================================
面试常考点：
1. 为什么需要 Policy Gradient？（Q-Learning 对连续动作空间无效）
2. 策略梯度定理推导：∇J(θ) = E[∇log π(a|s) · G_t]
3. REINFORCE 的无偏性 vs 高方差
4. Baseline（基线）如何减小方差

PyTorch 实现 CartPole 环境
- 状态: [位置, 速度, 角度, 角速度]（连续）
- 动作: 0=左, 1=右（离散）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. 策略网络（Policy Network）
# ============================================================

class PolicyNetwork(nn.Module):
    """
    参数化策略 π_θ(a|s)
    输入：状态 s
    输出：动作概率分布
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        # 简单的两层 MLP
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        前向传播：输入状态，输出动作概率
        """
        x = F.relu(self.fc1(x))
        # softmax 保证输出是概率分布（和为1）
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

    def get_action(self, state):
        """
        从策略中采样动作（用于 rollout）
        返回动作 和 对应的 log 概率（用于梯度计算）
        """
        state_tensor = torch.FloatTensor(state)
        action_probs = self.forward(state_tensor)

        # 用 Categorical 分布采样
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log π(a|s)

        return action.item(), log_prob


# ============================================================
# 2. REINFORCE 算法
# ============================================================

class REINFORCE:
    """
    蒙特卡洛策略梯度（Williams, 1992）

    算法流程：
    1. 用当前策略 π_θ 跑完一个 episode，得到轨迹 τ = (s0,a0,r0, s1,a1,r1, ...)
    2. 计算每步的折扣累计奖励 G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    3. 计算损失：L = -Σ_t [log π(a_t|s_t) · G_t]
    4. 反向传播更新参数 θ

    关键：负号！因为我们 maximize J(θ)，PyTorch 只做 minimize
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 存储一个 episode 的数据
        self.log_probs = []   # log π(a_t|s_t)
        self.rewards = []     # r_t

    def store(self, log_prob, reward):
        """存储每步的数据"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_returns(self):
        """
        计算折扣累计回报 G_t（从后往前计算）
        G_t = r_t + γ·G_{t+1}

        手撕关键：这个循环方向是从后往前！
        """
        G = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)  # 插到列表头部，保持时序

        returns = torch.FloatTensor(returns)

        # 标准化 returns（减均值除标准差）→ 减小方差，稳定训练
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """
        策略梯度更新

        损失函数：L(θ) = -E[log π_θ(a|s) · G_t]
        ∇L = -E[∇log π_θ(a|s) · G_t]  ← 策略梯度定理
        """
        returns = self.compute_returns()

        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # 每一步的损失：-log π(a|s) · G_t
            policy_loss.append(-log_prob * G)

        # 对所有时步求和
        loss = torch.stack(policy_loss).sum()

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空 episode 数据
        self.log_probs = []
        self.rewards = []

        return loss.item()


# ============================================================
# 3. 带基线的 REINFORCE（Baseline REINFORCE）
# ============================================================

class REINFORCEWithBaseline(REINFORCE):
    """
    用值函数 V(s) 作为基线，减小方差

    更新公式变为：∇J(θ) = E[∇log π(a|s) · (G_t - V(s_t))]
                                                  ↑
                                            优势函数 A_t

    直觉：只有当 G_t 超过平均水平 V(s_t) 时，才加强该动作
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        super().__init__(state_dim, action_dim, lr, gamma)

        # 额外的价值网络（评估 V(s)）
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # 输出标量 V(s)
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.states = []
        self.values = []

    def get_action(self, state):
        action, log_prob = self.policy.get_action(state)

        # 同时计算当前状态的价值估计
        state_tensor = torch.FloatTensor(state)
        value = self.value_net(state_tensor)

        self.states.append(state_tensor)
        self.values.append(value)
        return action, log_prob

    def update(self):
        returns = self.compute_returns()
        values = torch.stack(self.values).squeeze()

        # 优势函数 A_t = G_t - V(s_t)
        advantages = returns - values.detach()  # detach 防止梯度流入 value_net

        # 策略损失（最大化期望优势）
        policy_loss = []
        for log_prob, adv in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * adv)
        policy_loss = torch.stack(policy_loss).sum()

        # 价值网络损失（MSE）
        value_loss = F.mse_loss(values, returns)

        # 更新策略网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 清空
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.values = []

        return policy_loss.item(), value_loss.item()


# ============================================================
# 4. 简单的 CartPole 环境（不依赖 gym）
# ============================================================

class CartPoleSimple:
    """
    简化版 CartPole（面试时如果没有 gym 可以用这个）
    真实训练请用 gym.make('CartPole-v1')
    """

    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.max_steps = 200
        self._step = 0
        self.state = None

    def reset(self):
        # 随机初始化状态（小幅随机噪声）
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self._step = 0
        return self.state.copy()

    def step(self, action):
        """简化的 CartPole 物理模型"""
        x, x_dot, theta, theta_dot = self.state
        force = 10.0 if action == 1 else -10.0

        # 物理常数
        g = 9.8; mc = 1.0; mp = 0.1; l = 0.5
        total_mass = mc + mp

        # 加速度计算
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + mp * l * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (g * sin_theta - cos_theta * temp) / \
                    (l * (4/3 - mp * cos_theta**2 / total_mass))
        x_acc = temp - mp * l * theta_acc * cos_theta / total_mass

        # 欧拉积分
        dt = 0.02
        x += dt * x_dot
        x_dot += dt * x_acc
        theta += dt * theta_dot
        theta_dot += dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self._step += 1

        # 终止条件
        done = (abs(x) > 2.4 or abs(theta) > 0.21 or
                self._step >= self.max_steps)
        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done, {}


# ============================================================
# 5. 训练
# ============================================================

def train_reinforce(num_episodes=300):
    env = CartPoleSimple()
    agent = REINFORCE(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-3,
        gamma=0.99
    )

    print("开始训练 REINFORCE...")
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        # Rollout：跑完整个 episode
        while True:
            action, log_prob = agent.policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store(log_prob, reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # episode 结束后更新
        loss = agent.update()
        rewards_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode+1:4d} | 平均奖励: {avg_reward:6.1f} | Loss: {loss:.3f}")

    return rewards_history


if __name__ == "__main__":
    # 设置随机种子（面试时加上，保证可复现）
    torch.manual_seed(42)
    np.random.seed(42)

    rewards = train_reinforce(num_episodes=300)

    final_avg = np.mean(rewards[-50:])
    print(f"\n最终平均奖励（最后50轮）: {final_avg:.1f}")
    print(f"CartPole 满分是 200，当前: {final_avg:.1f}")
