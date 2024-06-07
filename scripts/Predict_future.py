import torch
import torch.nn as nn

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
    
class ObsGoalAttentionNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, embed_dim, num_heads, hidden_dim, seq_len, action_dim):
        super(ObsGoalAttentionNetwork, self).__init__()
        self.seq_len = seq_len
        self.goal_dim = goal_dim
        self.embedding = nn.Linear(obs_dim + goal_dim, embed_dim)
        self.attention_block = SelfAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim * seq_len)

    def forward(self, observations, goal):
        # Repeat goal vector for each sequence step and concatenate with observations
        goal_repeated = goal.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = torch.cat((observations, goal_repeated), dim=-1)
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
        # Reshape to (batch_size, seq_len, action_dim)
        actions = x.view(x.size(0), self.seq_len, -1)
        return actions

class Discriminator(nn.Module):
    def __init__(self, action_dim, seq_len, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(action_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, actions):
        x = actions.view(actions.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output is a single probability
        return x

class RewardNetwork(nn.Module):
    def __init__(self, action_dim, obs_dim, goal_dim, embed_dim, hidden_dim, seq_len):
        super(RewardNetwork, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(action_dim + obs_dim + goal_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, actions, observations, goal):
        # Repeat goal vector for each sequence step and concatenate with actions and observations
        goal_repeated = goal.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = torch.cat((actions, observations, goal_repeated), dim=-1)
        # Pass through embedding layer
        x = self.embedding(x)
        # Flatten the input for the fully connected layers
        x = x.view(x.size(0), -1)
        # Pass through linear layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reward = torch.sigmoid(self.fc3(x))  # Output a value between 0 and 1
        return reward
    

# Example usage    
seq_len = 5
obs_dim = 4
goal_dim = 2
embed_dim = 8
num_heads = 2
hidden_dim = 16
action_dim = 3

# Initialize models
model = RLAttentionNetwork(action_dim, obs_dim, embed_dim, num_heads, hidden_dim, seq_len)
generator = ObsGoalAttentionNetwork(obs_dim, goal_dim, embed_dim, num_heads, hidden_dim, seq_len, action_dim)
discriminator = Discriminator(action_dim, seq_len, hidden_dim)
reward_network = RewardNetwork(action_dim, obs_dim, goal_dim, embed_dim, hidden_dim, seq_len)

# Dummy input sequences
actions = torch.randn(10, seq_len, action_dim)  # (batch_size, seq_len, action_dim)
observations = torch.randn(10, seq_len, obs_dim)  # (batch_size, seq_len, obs_dim)
goal = torch.randn(10, goal_dim)  # (batch_size, goal_dim)

# Forward pass
next_observations = model(actions, observations)
print(next_observations.shape)  # Expected output: torch.Size([10, seq_len, obs_dim])
# Forward pass through generator
generated_actions = generator(observations, goal)
# Forward pass through discriminator
discriminator_output = discriminator(generated_actions)
print(discriminator_output.shape) # Expected output: torch.Size([10, 1])
# Forward pass through reward network
reward = reward_network(actions, observations, goal)
print(reward.shape)  # Expected output: torch.Size([10, 1])

model = ObsGoalAttentionNetwork(obs_dim, goal_dim, embed_dim, num_heads, hidden_dim, seq_len, action_dim)

# Dummy input sequences
observations = torch.randn(10, seq_len, obs_dim)  # (batch_size, seq_len, obs_dim)
goal = torch.randn(10, goal_dim)  # (batch_size, goal_dim)

# Forward pass
actions = model(observations, goal)
print(actions.shape)  # Expected output: torch.Size([10, seq_len, action_dim])

# Integrate with GAN training loop
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_R = torch.optim.Adam(reward_network.parameters(), lr=0.0002)
criterion = nn.BCELoss()

for epoch in range(epochs):
    # Train reward network
    optimizer_R.zero_grad()
    reward_real = reward_network(real_actions, real_observations, goal)
    reward_fake = reward_network(generated_actions.detach(), observations, goal)
    r_loss = criterion(reward_real, real_labels) + criterion(reward_fake, fake_labels)
    r_loss.backward()
    optimizer_R.step()
    
    # Train generator using reward network's output
    optimizer_G.zero_grad()
    reward_fake = reward_network(generated_actions, observations, goal)
    g_loss = criterion(reward_fake, real_labels)  # We want the reward network to give high rewards for generated actions
    g_loss.backward()
    optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{epochs}]  R Loss: {r_loss.item()}  G Loss: {g_loss.item()}")