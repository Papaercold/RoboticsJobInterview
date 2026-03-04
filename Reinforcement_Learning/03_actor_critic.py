"""
Actor-Critic (A2C) 算法
========================
面试常考点：
1. Actor-Critic 和 REINFORCE 的区别（TD vs MC）
2. 优势函数 A(s,a) = Q(s,a) - V(s) 的作用
3. 两个网络的梯度流向（Actor 用优势，Critic 用 TD 误差）
4. PPO 的 clip 技巧（常考）

A2C = Advantage Actor-Critic
Actor:  更新策略 π_θ(a|s)         → 最大化期望优势
Critic: 更新价值函数 V_φ(s)       → 最小化 TD 误差
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. Actor-Critic 网络（共享主干）
# ============================================================

class ActorCriticNetwork(nn.Module):
    """
    Actor 和 Critic 共享前几层（减少参数量，加速训练）

    结构：
    输入 s → 共享层 → Actor头: π(a|s) 概率分布
                    → Critic头: V(s)  标量
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()

        # 共享的特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),           # 注意：RL 中常用 Tanh 而不是 ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor 头：输出动作概率
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic 头：输出状态价值
        self.critic_head = nn.Linear(hidden_dim, 1)

        # 初始化（面试可能问）
        self._init_weights()

    def _init_weights(self):
        """正交初始化（A2C/PPO 常用）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor 头用更小的增益，让初始策略更均匀
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, x):
        """
        返回 (动作概率分布, 状态价值)
        """
        features = self.shared(x)
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic_head(features)
        return action_probs, value

    def get_action_and_value(self, state):
        """采样动作并返回所需信息"""
        state_t = torch.FloatTensor(state)
        action_probs, value = self.forward(state_t)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()   # 熵正则化，鼓励探索

        return action.item(), log_prob, value.squeeze(), entropy


# ============================================================
# 2. A2C 算法
# ============================================================

class A2C:
    """
    Advantage Actor-Critic（同步版本）

    每步更新（在线学习），不需要等整个 episode 结束：
    - Critic 损失: L_V = (r + γV(s') - V(s))²   ← TD 误差
    - Actor 损失: L_π = -log π(a|s) · A(s,a)    ← 优势加权
    - 总损失: L = L_π + c1·L_V - c2·H[π]        ← 熵正则化

    其中优势函数: A(s,a) = r + γV(s') - V(s) （TD 优势）
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):
        """
        参数：
        - value_coef (c1): Critic 损失的权重（通常 0.5）
        - entropy_coef (c2): 熵正则化权重（鼓励探索，通常 0.01）
        """
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr,
                                    eps=1e-5)  # PPO 中常用小 eps

    def update_step(self, state, action, reward, next_state, done, log_prob, value, entropy):
        """
        单步 TD 更新（A2C 在线更新版本）

        面试手撕核心：
        1. 计算 TD 目标
        2. 计算优势函数
        3. 计算三个损失项
        4. 合并并反向传播
        """
        # 计算下一状态的价值（用于 TD 目标）
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state)
            _, next_value = self.net(next_state_t)
            next_value = next_value.squeeze()

        # TD 目标（Bellman 目标）
        td_target = reward + self.gamma * next_value * (1 - done)

        # 优势函数 A = TD目标 - 当前估计值
        advantage = (td_target - value).detach()  # detach 防止梯度通过 advantage 流向 critic

        # === 三个损失项 ===

        # 1. Actor 损失（最大化 advantage 加权的 log 概率）
        actor_loss = -log_prob * advantage

        # 2. Critic 损失（最小化 TD 误差的平方）
        critic_loss = F.mse_loss(value, td_target.detach())

        # 3. 熵正则化（最大化熵 = 鼓励探索）
        entropy_loss = -entropy  # 注意负号（我们最大化熵）

        # 总损失
        total_loss = (actor_loss
                      + self.value_coef * critic_loss
                      + self.entropy_coef * entropy_loss)

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪（防止梯度爆炸，PPO/A2C 标配）
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)

        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), total_loss.item()

    def train(self, env, num_episodes=300):
        """训练主循环"""
        rewards_history = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            losses = []

            while True:
                # 获取动作和价值估计
                action, log_prob, value, entropy = self.net.get_action_and_value(state)

                # 与环境交互
                next_state, reward, done, _ = env.step(action)

                # 单步更新
                losses.append(self.update_step(
                    state, action, reward, next_state,
                    float(done), log_prob, value, entropy
                ))

                total_reward += reward
                state = next_state

                if done:
                    break

            rewards_history.append(total_reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                avg_loss = np.mean([l[2] for l in losses])
                print(f"Episode {episode+1:4d} | 平均奖励: {avg_reward:6.1f} | Loss: {avg_loss:.3f}")

        return rewards_history


# ============================================================
# 3. PPO 核心片段（面试最常考）
# ============================================================

class PPOClip:
    """
    PPO-Clip（Proximal Policy Optimization）
    OpenAI 2017，目前最流行的 RL 算法

    核心思想：限制每次策略更新的幅度，防止策略崩溃

    PPO 损失：
    L_CLIP = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

    其中概率比：r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
              = exp(log π_new - log π_old)

    clip 的作用：
    - A > 0（好动作）：防止概率比过大，限制上界为 1+ε
    - A < 0（坏动作）：防止概率比过小，限制下界为 1-ε
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 n_epochs=10):
        """
        n_epochs: 每批数据重复更新的次数（PPO 的关键超参）
        clip_eps (ε): clip 范围，通常 0.1~0.3
        """
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs

        self.net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # 经验缓冲区（PPO 用 rollout buffer）
        self.buffer = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'values': [], 'dones': []
        }

    def store(self, state, action, log_prob, reward, value, done):
        """存储一步经验"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob.detach())
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value.detach())
        self.buffer['dones'].append(done)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """
        GAE（Generalized Advantage Estimation）
        Schulman 2016，比单步 TD 方差更小

        δ_t = r_t + γV(s_{t+1}) - V(s_t)      ← TD 残差
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        λ=0: 纯 TD（低方差高偏差）
        λ=1: 蒙特卡洛（高方差低偏差）
        λ=0.95: 实践中常用的平衡点
        """
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones']

        advantages = []
        gae = 0

        # 从后往前计算（和 REINFORCE 的 returns 计算一样）
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            # TD 残差
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]

            # GAE 递推
            gae = delta + gamma * lam * (1 - next_done) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.stack(values)  # V + A = Q ≈ returns

        return advantages, returns

    def update(self, last_value):
        """
        PPO 核心更新（重复 n_epochs 次）
        """
        advantages, returns = self.compute_gae(last_value)

        # 标准化优势（减小方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转为 tensor
        states = torch.FloatTensor(np.array(self.buffer['states']))
        actions = torch.LongTensor(self.buffer['actions'])
        old_log_probs = torch.stack(self.buffer['log_probs'])

        # 重复更新 n_epochs 次（PPO 的关键）
        for _ in range(self.n_epochs):
            # 用当前网络重新计算 log prob 和 value
            action_probs, values = self.net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # 概率比 r_t(θ) = π_new / π_old = exp(log_new - log_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # ===== PPO Clip 损失（手撕重点）=====
            # 未裁剪项
            surr1 = ratio * advantages
            # 裁剪项：把 ratio 限制在 [1-ε, 1+ε]
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            # 取 min（保守更新）
            actor_loss = -torch.min(surr1, surr2).mean()
            # ===================================

            # Critic 损失
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())

            # 总损失
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

        # 清空缓冲区
        for key in self.buffer:
            self.buffer[key] = []

        return actor_loss.item(), critic_loss.item()


# ============================================================
# 4. 算法对比总结（面试必背）
# ============================================================

def print_comparison():
    """
    面试常考：各算法的区别和适用场景
    """
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                   强化学习算法对比                               │
    ├──────────────┬──────────────┬──────────────┬────────────────────┤
    │   算法       │   更新方式   │   策略类型   │   特点             │
    ├──────────────┼──────────────┼──────────────┼────────────────────┤
    │ Q-Learning   │  TD(0)       │  Off-policy  │ 离散动作，表格方法 │
    │ REINFORCE    │  MC          │  On-policy   │ 无偏但高方差       │
    │ A2C          │  TD(0)       │  On-policy   │ 低方差，在线更新   │
    │ PPO          │  TD + GAE    │  On-policy   │ 稳定，工程友好     │
    │ SAC          │  TD          │  Off-policy  │ 连续动作，最大熵   │
    │ DDPG/TD3     │  TD          │  Off-policy  │ 确定性连续动作     │
    └──────────────┴──────────────┴──────────────┴────────────────────┘

    机器人常用：PPO（稳定）、SAC（连续动作+样本效率）
    """)


# ============================================================
# 5. 运行 A2C 训练
# ============================================================

# 复用上一节的 CartPole 环境
class CartPoleSimple:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.max_steps = 200
        self._step = 0

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self._step = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = 10.0 if action == 1 else -10.0
        g = 9.8; mc = 1.0; mp = 0.1; l = 0.5
        total_mass = mc + mp
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        temp = (force + mp * l * theta_dot**2 * sin_t) / total_mass
        theta_acc = (g * sin_t - cos_t * temp) / \
                    (l * (4/3 - mp * cos_t**2 / total_mass))
        x_acc = temp - mp * l * theta_acc * cos_t / total_mass
        dt = 0.02
        x += dt * x_dot; x_dot += dt * x_acc
        theta += dt * theta_dot; theta_dot += dt * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self._step += 1
        done = abs(x) > 2.4 or abs(theta) > 0.21 or self._step >= self.max_steps
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done, {}


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print_comparison()

    print("=" * 50)
    print("训练 A2C on CartPole")
    print("=" * 50)

    env = CartPoleSimple()
    agent = A2C(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01
    )
    agent.train(env, num_episodes=300)
