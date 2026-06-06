import random
import torch
import numpy as np

def select_action(state, epsilon, num_actions, dqn_model):
    if np.random.rand() <= epsilon:
        return random.choice(range(num_actions))
    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    with torch.no_grad():
        q_values = dqn_model(state_tensor)
    return np.argmax(q_values.cpu().numpy())