#import torch_dwn
import torchlogix
import torch
import torch.nn as nn

from torchlogix.layers import (
        FixedBinarization,
        GroupSum,
        LogicDense,
)

class RegressionBucketLayer(nn.Module):
    """Input: Popcounts for each dimension, Outputs: rescaled actions"""
    def __init__(self, n, k,
                 init_log_alpha=-0.6931): 
        super().__init__()
        self.n = float(n)
        self.k = float(k)
        
        init = torch.ones(k, dtype=torch.float32) *init_log_alpha
        self.log_alpha = nn.Parameter(init)
        self.beta = nn.Parameter(torch.zeros(k, dtype=torch.float32))
        self.eps = 1e-6
        self.norm_factor = self.n / self.k
    
    def forward(self, x):
        x_norm = x / self.norm_factor    
        # the clamp could be removed, its legacy, 
        # but the runs in the paper where done with it
        # keeping it here for consistency
        x_norm = torch.clamp(x_norm, self.eps, 1 - self.eps)
        y = torch.exp(self.log_alpha) *  (x_norm - 0.5) + self.beta
        return y


class RNNDWN(nn.Module):
    
    def __init__(self, input_dim, 
                 bits,
                 thresholds,
                 hidden_size, 
                 output_dim, 
                 n=6,
                 num_layers=1, 
                 map="learnable", # the first layers connectivity
                 init_log_alpha=-0.6931):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_size = input_dim * 3 #bits
        self.input_dim = input_dim
        self.output_size = output_dim * 25
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bits = bits
        #self.thresholds = thresholds
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        thresholds = torch.tensor([0,1,2], dtype=torch.float32).to(device)
        self.binarization = FixedBinarization(thresholds=thresholds,feature_dim=1).to(device)
        self.preprocess = nn.Flatten().to(device)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(LogicDense(self.input_size + hidden_size, hidden_size, lut_rank=n, device=device, connections=map, parametrization="warp"))

        for _ in range(1, num_layers):
            self.hidden_layers.append(LogicDense(hidden_size + hidden_size, hidden_size, lut_rank=n, device=device, parametrization="warp"))

        #self.hidden_layer = self.hidden_layers[0]
        self.output_layer = nn.Sequential(
            LogicDense(hidden_size , hidden_size // n, parametrization="warp", lut_rank=n, device=device),
            LogicDense(hidden_size // n, self.output_size, parametrization="warp", lut_rank=n, device=device),
            GroupSum(k=output_dim, tau=1.0, device=device)
        )

    def init_hidden(self, batch_size, device=None, dtype=None):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if device is None:
            device = next(self.parameters()).device

        if dtype is None:
            dtype = next(self.parameters()).dtype

        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, hidden_state=None):
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        # elif x.dim() != 2:
        #     raise ValueError("x must have shape (features,) or (batch_size, features)")

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        else:
            if hidden_state.dim() != 3:
                raise ValueError("hidden_state must have shape (num_layers, batch_size, hidden_size)")
            if hidden_state.shape[0] != self.num_layers:
                raise ValueError(f"hidden_state must have {self.num_layers} layers")
            if hidden_state.shape[1] != x.shape[0]:
                raise ValueError("hidden_state batch size must match x batch size")
            if hidden_state.shape[2] != self.hidden_size:
                raise ValueError(f"hidden_state must have hidden size {self.hidden_size}")
            if hidden_state.device != x.device or hidden_state.dtype != x.dtype:
                hidden_state = hidden_state.to(device=x.device, dtype=x.dtype)

        x = self.binarization(x)
        #print(f"After binarization: {x}")
        next_input = self.preprocess(x)
        #print(f"Next input shape: {next_input.shape}")
        next_hidden_states = []

        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            combined = torch.cat((next_input, hidden_state[layer_index]), dim=1)
            layer_hidden_state = hidden_layer(combined)
            next_hidden_states.append(layer_hidden_state)
            next_input = layer_hidden_state

        next_hidden_state = torch.stack(next_hidden_states, dim=0)
        output = self.output_layer(next_hidden_state[-1])

        if squeeze_output:
            return output.squeeze(0), next_hidden_state
        
        return output, next_hidden_state

    
