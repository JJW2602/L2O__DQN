import torch
import random
from config import ACTION_SIZE, EPSILON

class OptimizerEnvironment:
    def __init__(self, model):
        self.model = model
    
    def get_state(self):
        """현재 MLP의 weight를 state로 변환."""
        return torch.tensor(self.model.fc1.weight.data.flatten(), device=self.model.fc1.weight.device)
    
    def select_action(self, policy_net, state):
        """DQN_LSTM을 사용하여 action 선택 (어떤 optimizer를 사용할지)."""
        if random.random() < EPSILON:
            return random.randint(0, ACTION_SIZE - 1)  # Exploration (랜덤 선택)
        with torch.no_grad():
            return torch.argmax(policy_net(state)).item()  # Exploitation (DQN이 선택)
    
    def get_reward(self, loss):
        """손실 값(Loss)에 기반하여 reward를 반환."""
        return -loss.item()