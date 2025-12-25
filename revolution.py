import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import copy

# Настройка
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Устройство: {device} ---")
SEED = 42

def set_seed(env, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.reset(seed=seed) # Gymnasium handles seeding differently now, passed to reset

# ==========================================
# 1. ТВОЯ АРХИТЕКТУРА (REVOLUTION)
# ==========================================

class LatentWorld(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.encoder(state)

class RevolutionAgent:
    def __init__(self, state_dim, latent_dim=16):
        self.latent_dim = latent_dim
        self.model = LatentWorld(state_dim, latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.memory = deque(maxlen=15000)
        self.danger_states = deque(maxlen=200)
        self.peak_states = deque(maxlen=20)
        self.best_reward = 0

    def get_action(self, state, explore=True):
        state_t = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            z_curr = self.model.encoder(state_t)
            
            # Оптимизация: не перекодировать если буферы пусты
            if len(self.peak_states) > 0:
                peak_tensor = torch.FloatTensor(np.array(self.peak_states)).to(device)
                current_peaks = self.model.encoder(peak_tensor)
            else:
                current_peaks = None

            if len(self.danger_states) > 0:
                # Берем случайную выборку опасностей, чтобы не замедлять работу
                sample_size = min(len(self.danger_states), 32)
                danger_sample = random.sample(self.danger_states, sample_size)
                danger_tensor = torch.FloatTensor(np.array(danger_sample)).to(device)
                current_dangers = self.model.encoder(danger_tensor)
            else:
                current_dangers = None

            best_action = 0
            best_score = -float('inf')
            
            is_opposite = explore and (random.random() < 0.05)

            for a in [0, 1]:
                action_t = torch.FloatTensor([a]).to(device)
                z_next = self.model.dynamics(torch.cat([z_curr, action_t]))
                v_score = self.model.value_head(z_next).item()
                
                score = v_score
                
                if current_peaks is not None and len(current_peaks) > 0:
                    idx = random.randint(0, len(current_peaks)-1)
                    target = current_peaks[idx]
                    
                    if is_opposite:
                        dist = torch.dist(z_next, -target).item()
                        score = dist # Тут логика немного странная (dist растет), но оставим как у автора
                    else:
                        dist_to_peak = torch.dist(z_next, target).item()
                        score = v_score - (dist_to_peak * 0.3)

                if current_dangers is not None:
                    # Векторизованный расчет расстояний до всех опасностей
                    dists = torch.norm(current_dangers - z_next, dim=1)
                    if torch.any(dists < 0.3):
                        score -= 40.0

                if score > best_score:
                    best_score = score
                    best_action = a
            
            return best_action

    def train(self, batch_size=64): # Чуть уменьшил батч для скорости сравнения
        if len(self.memory) < batch_size: return
        
        batch = random.sample(self.memory, batch_size)
        s, a, r, next_s, d = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.FloatTensor(np.array(a)).unsqueeze(1).to(device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(device)
        next_s = torch.FloatTensor(np.array(next_s)).to(device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(device)
        
        z_curr = self.model.encoder(s)
        z_next_pred = self.model.dynamics(torch.cat([z_curr, a], dim=1))
        with torch.no_grad():
            z_next_real = self.model.encoder(next_s)
        
        loss_dyn = nn.MSELoss()(z_next_pred, z_next_real)
        
        v_curr = self.model.value_head(z_curr)
        with torch.no_grad():
            v_next = self.model.value_head(z_next_real)
            v_target = r + (0.99 * v_next * (1 - d))
        loss_val = nn.MSELoss()(v_curr, v_target)

        loss_reg = torch.mean(z_curr**2) * 0.01
        
        total_loss = loss_dyn + loss_val + loss_reg
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

# ==========================================
# 2. STANDARD DQN (БАЗОВЫЙ УРОВЕНЬ)
# ==========================================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=15000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_every = 10
        self.step_counter = 0

    def get_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, next_s, d = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.LongTensor(np.array(a)).unsqueeze(1).to(device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(device)
        next_s = torch.FloatTensor(np.array(next_s)).to(device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(device)

        # Q(s, a)
        q_values = self.policy_net(s).gather(1, a)
        
        # Q_target = r + gamma * max(Q(s', a'))
        with torch.no_grad():
            next_q_values = self.target_net(next_s).max(1)[0].unsqueeze(1)
            target_q_values = r + (self.gamma * next_q_values * (1 - d))
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==========================================
# 3. ЦИКЛ СРАВНЕНИЯ
# ==========================================

def run_experiment(agent_type, episodes=150):
    env = gym.make("CartPole-v1")
    set_seed(env, SEED)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type == "Revolution":
        agent = RevolutionAgent(state_dim)
    else:
        agent = DQNAgent(state_dim, action_dim)
    
    rewards_history = []
    
    print(f"Запуск обучения: {agent_type}...")
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED+episode) # Меняем сид каждый эпизод, но одинаково для обоих агентов
        total_reward = 0
        episode_states = []
        
        for t in range(500):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            
            # --- Логика наград для Revolution ---
            if agent_type == "Revolution":
                r_signal = reward
                if terminated and total_reward < agent.best_reward * 0.9:
                    r_signal = -50.0 
                agent.memory.append((state, action, r_signal, next_state, done))
                agent.train(batch_size=64)
            # --- Логика наград для DQN ---
            else:
                agent.memory.append((state, action, reward, next_state, done))
                agent.train()
            
            state = next_state
            total_reward += reward
            if done: break
        
        # Пост-процессинг эпизода
        if agent_type == "Revolution":
            if total_reward >= agent.best_reward and total_reward > 10:
                agent.best_reward = total_reward
                mid_state = episode_states[len(episode_states)//2]
                agent.peak_states.append(mid_state)
            elif total_reward < agent.best_reward * 0.8:
                agent.danger_states.append(episode_states[-1])
        else:
            agent.update_epsilon()

        rewards_history.append(total_reward)
        
        if episode % 20 == 0:
            print(f"  Ep: {episode} | Reward: {total_reward:.1f}")
            
    env.close()
    return rewards_history

# ==========================================
# 4. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================

EPISODES = 150 # Количество эпизодов для теста

# Запускаем DQN
dqn_rewards = run_experiment("DQN", episodes=EPISODES)

# Запускаем Revolution
rev_rewards = run_experiment("Revolution", episodes=EPISODES)

# Сглаживание графика (Moving Average)
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(dqn_rewards, alpha=0.3, color='blue')
plt.plot(moving_average(dqn_rewards), label='Standard DQN (Trend)', color='blue', linewidth=2)

plt.plot(rev_rewards, alpha=0.3, color='red')
plt.plot(moving_average(rev_rewards), label='Revolution Agent (Trend)', color='red', linewidth=2)

plt.title('Сравнение скорости обучения: DQN vs Revolution')
plt.xlabel('Эпизод')
plt.ylabel('Награда')
plt.legend()
plt.grid(True)
plt.show()