import torch

def state_to_dqn_input(state:int, num_states:int) -> torch.Tensor: # one-hot encoding
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor