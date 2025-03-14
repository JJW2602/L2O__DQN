import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
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
        self.policy_net = DQN_LSTM(201, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net = DQN_LSTM(201, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.train_loader = get_mnist_projected_dataloader()
        self.action_counts = defaultdict(int)  # 각 optimizer 선택 횟수 기록
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # 모든 텐서를 GPU 또는 CPU로 변환
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, epochs=50):
        print(" Trainer.train() 시작됨!")
        for epoch in range(epochs):
            model = TwoLayerNN().to(self.device)
            params = list(model.parameters())
            if not params:  
                raise ValueError("Model has no trainable parameters!")
            env = OptimizerEnvironment(model)
            learned_optimizer = Learned_Optimizer(params, self.policy_net)
            state = env.get_state()
            total_loss = 0
            correct = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = F.cross_entropy(output, target)
                learned_optimizer.zero_grad()
                loss.backward()
                action = env.select_action(self.policy_net, state)
                self.action_counts[action] += 1  # Action 선택 횟수 기록
                learned_optimizer.step(action)  # 수정된 action 적용
                
                next_state = env.get_state()
                reward = env.get_reward(loss)
                done = batch_idx == len(self.train_loader) - 1                
                
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                
                self.optimize_model()

                # Accuracy 계산
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                total_loss += loss.item()

                 # 로그 출력
                print(f"Epoch {epoch}, Episode {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {correct / total_samples:.4f}, Action Counts: {dict(self.action_counts)}")

            if epoch % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            print(f"Epoch {epoch} Summary: Avg Loss = {total_loss / total_samples:.4f}, Accuracy = {correct / total_samples:.4f}, Action Counts = {dict(self.action_counts)}")
