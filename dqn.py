import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ハイパーパラメータ例
GAMMA = 0.99          # 割引率
LR = 1e-4             # 学習率
BUFFER_SIZE = 100000  # リプレイバッファ容量
BATCH_SIZE = 64       # ミニバッチサイズ
TARGET_UPDATE = 1000  # ターゲットネット更新ステップ
EPS_START = 1.0       # ε–greedy 初期値
EPS_END = 0.1         # ε 最低値
EPS_DECAY = 100000    # ε 減衰ステップ数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, n_actions=64):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        """
        x: torch.Tensor shape=(batch,3,8,8)
        returns: Q値 shape=(batch,64)
        """
        x = self.conv(x)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state, next_state は torch.Tensor
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions=64):
        self.n_actions = n_actions
        self.policy_net = QNetwork(n_actions).to(DEVICE)
        self.target_net = QNetwork(n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()
        self.steps_done = 0
        self.epsilon = EPS_START

    def select_action(self, state, legal_actions):
        """
        state: torch.Tensor shape=(3,8,8)
        legal_actions: list of int (0–63)
        returns: action (int)
        """
        # ε–greedy
        self.steps_done += 1
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1.0 * self.steps_done / EPS_DECAY)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_batch = state.unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_batch).cpu().squeeze(0)
            # 合法手以外を大きく負にして選ばれないように
            mask = torch.full((self.n_actions,), -1e9)
            mask[legal_actions] = 0.0
            filtered = q_values + mask
            return int(filtered.argmax().item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # サンプリング
        states, actions, rewards, next_states, dones = self.memory.sample()

        # 現在のQ(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # ターゲット値 r + γ max_a' Q_target(s',a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * next_q * (1 - dones))

        # 損失 & 更新
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ターゲットネットを定期更新
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())