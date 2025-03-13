import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models import DQN_LSTM, TwoLayerNN
from replay_buffer import ReplayBuffer
from learned_optimizer import Learned_Optimizer
from environment import OptimizerEnvironment
from dataset import get_mnist_projected_dataloader
from config import *
import random

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN_LSTM(48, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net = DQN_LSTM(48, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.train_loader = get_mnist_projected_dataloader()

    def select_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, epochs=50):
        for epoch in range(epochs):
            model = TwoLayerNN().to(self.device)
            env = OptimizerEnvironment(model)
            learned_optimizer = Learned_Optimizer(model.parameters(), self.policy_net)
            state = env.get_state()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                
                output = model(data)
                loss = F.cross_entropy(output, target.to(self.device))
                learned_optimizer.zero_grad()
                loss.backward()
                learned_optimizer.step()
                
                next_state = env.get_state()
                reward = env.get_reward(loss)
                done = batch_idx == len(self.train_loader) - 1
                
                self.memory.push(state, 0, reward, next_state, done)
                state = next_state
                
                self.optimize_model()
                
            if epoch % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
