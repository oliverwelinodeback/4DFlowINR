import torch.nn as nn
import torch
import numpy as np


class SIREN(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, depth=6, hidden_features=128, first_omega_0=30, hidden_omega_0=30, outermost_linear=True):

        super(SIREN, self).__init__()
        self.net = []
        self.net.append(SineLayer(in_dim, hidden_features, is_first=True, omega_0=first_omega_0))
        
        for i in range(depth):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_dim)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_dim, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        output = self.net(x)
        return output 

class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class FourierFeatureEncoding(nn.Module):
    """
    Applies Fourier feature mapping to input data.
    """
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        self.B = torch.randn((input_dim, mapping_size)) * scale

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.to(x.device)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FFN(nn.Module):
    """
    Fully connected network with Fourier Feature Encoding.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 fourier_mapping_size: int, depth: int, scale: float = 1.0, bias: bool = True):
        super().__init__()

        # Fourier Encoding
        self.fourier_encoder = FourierFeatureEncoding(input_dim, fourier_mapping_size,scale=scale)
        encoded_dim = fourier_mapping_size * 2

        # Build the dynamic network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(encoded_dim, hidden_dim,bias=bias))
        layers.append(nn.GELU())

        # Hidden layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Define the sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x_encoded = self.fourier_encoder(x)
        return self.network(x_encoded)
        

# class GaussianFourierFeatureTransform(torch.nn.Module):
#     """
#     An implementation of Gaussian Fourier feature mapping.

#     "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
#        https://arxiv.org/abs/2006.10739
#        https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

#     Given an input of size [batches, num_input_channels, width, height],
#      returns a tensor of size [batches, mapping_size*2, width, height].
#     """

#     def __init__(self, num_input_channels, mapping_size=256, scale=10):
#         super().__init__()

#         self._num_input_channels = num_input_channels
#         self._mapping_size = mapping_size
#         self._B = torch.randn((num_input_channels, mapping_size)) * scale

#     def forward(self, x):
#         assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

#         batches, channels, width, height = x.shape

#         assert channels == self._num_input_channels,\
#             "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

#         # Make shape compatible for matmul with _B.
#         # From [B, C, W, H] to [(B*W*H), C].
#         x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

#         x = x @ self._B.to(x.device)

#         # From [(B*W*H), C] to [B, W, H, C]
#         x = x.view(batches, width, height, self._mapping_size)
#         # From [B, W, H, C] to [B, C, W, H]
#         x = x.permute(0, 3, 1, 2)

#         x = 2 * pi * x
#         return torch.cat([torch.sin(x), torch.cos(x)], dim=1)