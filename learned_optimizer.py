import torch
import torch.optim as optim
import torch.nn as nn
from config import LR

class Learned_Optimizer(optim.Optimizer):
    def __init__(self, params, model): #params : models parametrs, model : LSTM
        self.model = model
        self.lr = LR
        self.optimizers = [
            optim.Adam(params, lr=self.lr),  # Adam
            optim.SGD(params, lr=self.lr, momentum=0.9),  # SGD
            optim.RMSprop(params, lr=self.lr),  # RMSprop
            optim.Adagrad(params, lr=self.lr),  # Adagrad
            optim.Adam(params, lr=self.lr, betas=(0.85, 0.999))  # Custom Adam variant
        ]
        defaults = dict(lr=self.lr)
        super(Learned_Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = p.data.view(1, -1)  # 현재 weight를 flatten하여 state로 변환
                
                with torch.no_grad():
                    action_values = self.model(state)  # DQN_LSTM이 예측한 Q-values
                    action_idx = torch.argmax(action_values, dim=-1).item()  # 가장 높은 Q-value의 action 선택
                
                # 선택된 optimizer에 따라 step 실행
                self.optimizers[action_idx].zero_grad()
                p.grad = grad  # 현재 gradient 적용
                self.optimizers[action_idx].step()

        return loss