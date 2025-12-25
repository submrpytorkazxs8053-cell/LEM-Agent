import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {device} ---")
SEED = 42

def set_seed(env, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==========================================
# 1. REVOLUTION AGENT (YOUR ARCHITECTURE)
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
            
            # Optimization: don't re-encode if buffers are empty
            if len(self.peak_states) > 0:
                peak_tensor = torch.FloatTensor(np.array(self.peak_states)).to(device)
                current_peaks = self.model.encoder(peak_tensor)
            else:
                current_peaks = None

            if len(self.danger_states) > 0:
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
                        score = dist
                    else:
                        dist_to_peak = torch.dist(z_next, target).item()
                        score = v_score - (dist_to_peak * 0.3)

                if current_dangers is not None:
                    dists = torch.norm(current_dangers - z_next, dim=1)
                    if torch.any(dists < 0.3):
                        score -= 40.0

                if score > best_score:
                    best_score = score
                    best_action = a
            
            return best_action, 0 # Return 0 as log_prob (placeholder for interface consistency)

    def train(self, batch_size=64):
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
# 2. PPO AGENT (REPLACING DQN)
# ==========================================

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor: outputs probabilities for actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic: estimates Value(state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = []

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, action_logprob = self.policy_old.act(state)
        return action.item(), action_logprob.item()

    def store_transition(self, transition):
        # transition: (state, action, log_prob, reward, done)
        self.buffer.append(transition)

    def update(self):
        # Unzip buffer
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(device)
        old_logprobs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(device)
        rewards = [t[3] for t in self.buffer]
        is_terminals = [t[4] for t in self.buffer]

        # Monte Carlo Estimate of returns
        rewards_norm = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_norm.insert(0, discounted_reward)
            
        rewards_norm = torch.FloatTensor(rewards_norm).to(device)
        # Normalizing the rewards
        rewards_norm = (rewards_norm - rewards_norm.mean()) / (rewards_norm.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            # Match tensor shapes
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards_norm - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards_norm) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer = []

# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================

def run_experiment(agent_type, episodes=150):
    env = gym.make("CartPole-v1")
    set_seed(env, SEED)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type == "Revolution":
        agent = RevolutionAgent(state_dim)
    else:
        agent = PPOAgent(state_dim, action_dim)
    
    rewards_history = []
    start_time = time.time()
    
    print(f"Starting Training: {agent_type}...")
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED+episode)
        total_reward = 0
        episode_states = []
        
        for t in range(500):
            # Get Action
            if agent_type == "Revolution":
                action, _ = agent.get_action(state) # _ is dummy log_prob
            else:
                action, log_prob = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            
            # --- Revolution Logic ---
            if agent_type == "Revolution":
                r_signal = reward
                if terminated and total_reward < agent.best_reward * 0.9:
                    r_signal = -50.0 
                agent.memory.append((state, action, r_signal, next_state, done))
                agent.train(batch_size=64)
            
            # --- PPO Logic ---
            else:
                # Store data in buffer (PPO updates at end of episode/batch usually)
                agent.store_transition((state, action, log_prob, reward, done))
            
            state = next_state
            total_reward += reward
            if done: break
        
        # Post-processing per episode
        if agent_type == "Revolution":
            if total_reward >= agent.best_reward and total_reward > 10:
                agent.best_reward = total_reward
                mid_state = episode_states[len(episode_states)//2]
                agent.peak_states.append(mid_state)
            elif total_reward < agent.best_reward * 0.8:
                agent.danger_states.append(episode_states[-1])
        else:
            # Update PPO at the end of the episode (On-Policy)
            agent.update()

        rewards_history.append(total_reward)
        
        if episode % 20 == 0:
            print(f"  Ep: {episode} | Reward: {total_reward:.1f}")
            
    total_time = time.time() - start_time
    env.close()
    return rewards_history, total_time

# ==========================================
# 4. EXECUTION AND VISUALIZATION
# ==========================================

EPISODES = 150

# Run PPO
ppo_rewards, ppo_time = run_experiment("PPO", episodes=EPISODES)

# Run Revolution
rev_rewards, rev_time = run_experiment("Revolution", episodes=EPISODES)

# Smoothing function
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Graph 1: Learning Curve (Reward vs Episodes)
ax1.plot(ppo_rewards, alpha=0.3, color='blue')
ax1.plot(moving_average(ppo_rewards), label='PPO (Trend)', color='blue', linewidth=2)
ax1.plot(rev_rewards, alpha=0.3, color='red')
ax1.plot(moving_average(rev_rewards), label='Revolution Agent (Trend)', color='red', linewidth=2)
ax1.set_title('Learning Efficiency: PPO vs Revolution')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True)

# Graph 2: Time Taken (Wall clock)
agents = ['PPO', 'Revolution']
times = [ppo_time, rev_time]
colors = ['blue', 'red']

ax2.bar(agents, times, color=colors, alpha=0.7)
ax2.set_title('Total Training Time (Seconds)')
ax2.set_ylabel('Seconds')
for i, v in enumerate(times):
    ax2.text(i, v + 1, f"{v:.1f}s", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()