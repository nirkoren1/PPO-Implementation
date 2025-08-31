import torch
import torch.nn as nn
import numpy as np
import math

np.random.seed(42)
torch.manual_seed(42)

class FastCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FastCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = torch.nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        h_out = (x.size(2) - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (x.size(3) - self.kernel_size[1]) // self.stride[1] + 1

        x_unf = torch.nn.functional.unfold(x, self.kernel_size, stride=self.stride)
        w_flat = self.weight.view(self.out_channels, -1)
        out_unf = x_unf.transpose(1, 2).matmul(w_flat.t()).transpose(1, 2)
        out_unf = out_unf + self.bias.view(1, -1, 1)
        out = out_unf.view(batch_size, self.out_channels, h_out, w_out)
        
        return out
    
class AtariImageEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(AtariImageEncoder, self).__init__()
        self.activation = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            FastCNN(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            FastCNN(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            FastCNN(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_dim)
        )
        
        
    def forward(self, state):
        state = state.float()
        state = state / 255.0
        return self.cnn(state)
    
    def get_state_dim(self):
        return self.hidden_dim
        
class AtariNoEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(AtariNoEncoder, self).__init__()
        self.state_dim = state_dim
        self.activation = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.fc_simple = nn.Linear(state_dim[0] * state_dim[1] * state_dim[2], hidden_dim)

    def forward(self, state):
        state = state.view(state.size(0), -1)
        x = self.activation(self.fc_simple(state))
        return x
    
    def get_state_dim(self):
        return self.hidden_dim
    
class NoEncoder(nn.Module):
    def __init__(self, state_dim):
        super(NoEncoder, self).__init__()
        self.state_dim = state_dim[0]

    def forward(self, state):
        return state
    
    def get_state_dim(self):
        return self.state_dim