import os
import torch
import torch.nn as nn
import torch.optim as optim
from time import sleep

# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase

env = VecEnvBase(headless=False)
# when you need all the extensions of isaac sim use the below line
#  env = VecEnvBase(headless=False, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from sim.crazy_task import CarzyFlyTask

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class RLAttentionNetwork(nn.Module):
    def __init__(self, action_dim, obs_dim, embed_dim, num_heads, hidden_dim, seq_len):
        super(RLAttentionNetwork, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(action_dim + obs_dim, embed_dim)
        self.attention_block = SelfAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_dim * seq_len)

    def forward(self, actions, observations):
        # Concatenate actions and observations along the last dimension
        x = torch.cat((actions, observations), dim=-1)
        # Pass through embedding layer
        x = self.embedding(x)
        # Reshape for attention block (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        # Pass through self-attention block
        x = self.attention_block(x)
        # Reshape back (batch_size, seq_len * embed_dim)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)
        # Pass through linear layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to (batch_size, seq_len, obs_dim)
        next_observations = x.view(x.size(0), self.seq_len, -1)
        return next_observations
    
def update_obs(obs_batch, next_obs):
    obs_batch[:, :-1] = obs_batch[:, 1:]
    obs_batch[:, -1] = next_obs
    return obs_batch

task = CarzyFlyTask(name="Fly")
env.set_task(task, backend="torch")
env._world.reset()
obs, _ = env.reset()

# Example usage    
seq_len = 5
obs_dim = 4
goal_dim = 2
embed_dim = 8
num_heads = 2
hidden_dim = 16
action_dim = 4

# Hyperparams
epochs = 100
learning_rate = 0.001
batch_size = 1
episode_length = 10

# Initialize models
model = RLAttentionNetwork(action_dim, obs_dim, embed_dim, num_heads, hidden_dim, seq_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Init action and observation vars
actions = torch.randn(batch_size, seq_len, action_dim)  # (batch_size, seq_len, action_dim)
observations = torch.zeros(batch_size, seq_len, obs_dim)  # (batch_size, seq_len, obs_dim)
next_observations = torch.zeros(batch_size, seq_len, obs_dim)  # (batch_size, seq_len, obs_dim)
goal = torch.randn(batch_size, goal_dim)  # (batch_size, goal_dim)

# Get atleast 2 observations, put the observations at the end
for i in range(seq_len):
    # Update observations buffer in FIFO manner
    observations[:, :-1, :] = observations[:, 1:, :]
    observations[:, -1, :] = env.step(actions[i])

next_observations[:, :-1, :] = observations[:, 1:, :]

for epoch in range(epochs):
    # load dataset
    for time in episode_length:
        # Forward pass
        predict_next_observations = model(actions, observations)
        
        #get the next obs
        new_obs = env.step(actions[i])
        next_observations[:, :-1, :] = next_observations[:, 1:, :]
        next_observations[:, -1, :] = env.step(actions[i])

        # Compute loss (using observations as target for demonstration)
        loss = criterion(predict_next_observations, next_observations)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update observations buffer in FIFO manner
        observations[:, :-1, :] = observations[:, 1:, :]
        observations[:, -1, :] = new_obs

        #update actions buffer in FIFO manner
        actions[:, :-1, :] = actions[:, 1:, :]
        actions[:, -1, :] = torch.randn(batch_size, 1, action_dim)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')