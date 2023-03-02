import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN Agent
class DQNAgent():
    def __init__(self, input_size, hidden_size, output_size, lr, gamma, buffer_size, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = QNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

    # Add the current transition to the replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    # Choose an action using the epsilon-greedy policy
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                action = q_values.argmax(dim=1).item()
        else:
            action = random.randrange(self.q_net.output_size)
        return action

    # Train the Q-Network on a batch of transitions from the replay buffer
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert the transition batch into PyTorch tensors
        state_batch_tensor = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch_tensor = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch_tensor = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch_tensor = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch_tensor = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute the Q-Values for the current state-action pairsx
        q_values = self.q_net(state_batch_tensor).gather(1, action_batch_tensor)

        # Compute the Q-Values for the next state using the target network
        target_q_values = self.target_net(next_state_batch_tensor).max(1)[0]

        # Compute the expected Q-Values using the Bellman equation
        expected_q_values = reward_batch_tensor + (self.gamma * target_q_values * (1 - done_batch_tensor))

        # Compute the loss between the predicted and expected Q-Values
        loss = F.smooth_l1_loss(q_values, expected_q_values.detach())

        # Optimize the Q-Net by taking a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_net()

    # Update the target network by copying the weights from the Q-Net
    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # Save the current state of the agent to a file
    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename)

    # Load the last saved state of the agent from a file
    def load(self, filename):
        self.q_net.load_state_dict(torch.load(filename))

