import torch_dwn
import torch
import torch.nn as nn

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
                 thermometer,
                 hidden_size, 
                 output_dim, 
                 n=6,
                 num_layers=1, 
                 map="learnable", # the first layers connectivity
                 init_log_alpha=-0.6931):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_size = input_dim * bits
        self.output_size = output_dim * 100
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bits = bits
        self.thermometer = thermometer

        self.preprocess = nn.Flatten()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(torch_dwn.LUTLayer(self.input_size + hidden_size, hidden_size, n=n, mapping=map))

        for _ in range(1, num_layers):
            self.hidden_layers.append(torch_dwn.LUTLayer(hidden_size + hidden_size, hidden_size, n=n))

        #self.hidden_layer = self.hidden_layers[0]
        self.output_layer = nn.Sequential(
            torch_dwn.LUTLayer(hidden_size, self.output_size, n=2),
            torch_dwn.GroupSum(k=output_dim, tau=1.0)
        )

        self.register_buffer("hidden_state", torch.zeros(num_layers, 0, hidden_size))

    def reset_hidden_state(self, batch_size=None, device=None, dtype=None):
        if batch_size is None:
            self.hidden_state = self.hidden_state.new_zeros(self.num_layers, 0, self.hidden_size)
            return

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if device is None:
            device = self.hidden_state.device

        if dtype is None:
            dtype = self.hidden_state.dtype

        self.hidden_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

    def _ensure_hidden_state(self, batch_size, x):
        if self.hidden_state.shape[1] != batch_size:
            self.reset_hidden_state(batch_size=batch_size, device=x.device, dtype=x.dtype)
        elif self.hidden_state.device != x.device or self.hidden_state.dtype != x.dtype:
            self.hidden_state = self.hidden_state.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.dim() != 2:
            raise ValueError("x must have shape (features,) or (batch_size, features)")

        self._ensure_hidden_state(x.shape[0], x)

        x = self.thermometer.binarize(x)
        next_input = self.preprocess(x)
        #print(f"Next input shape: {next_input.shape}")
        next_hidden_states = []

        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            combined = torch.cat((next_input, self.hidden_state[layer_index]), dim=1)
            layer_hidden_state = hidden_layer(combined)
            next_hidden_states.append(layer_hidden_state)
            next_input = layer_hidden_state

        self.hidden_state = torch.stack(next_hidden_states, dim=0)
        output = self.output_layer(self.hidden_state[-1])

        if squeeze_output:
            return output.squeeze(0)
        
        return output
    
