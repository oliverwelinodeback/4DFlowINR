import torch.nn as nn
import torch
import numpy as np

class FF_SIREN(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, depth=6, hidden_features=128, first_omega_0=30, hidden_omega_0=30, 
                 outermost_linear=True, fourier_mapping_size=128, scale = 1.0):

        super(FF_SIREN, self).__init__()

        # Fourier Encoding
        self.fourier_encoder = FourierFeatureEncoding(in_dim, fourier_mapping_size,scale=scale)
        encoded_dim = fourier_mapping_size * 2

        self.net = []
        self.net.append(SineLayer(encoded_dim, hidden_features, is_first=True, omega_0=first_omega_0))
        
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
        x_enc = self.fourier_encoder(x)
        output = self.net(x_enc)
        return output 

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
        # super().__init__()

        super(FFN, self).__init__()

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


###############



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
    
################


class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

        initialization(self.freqs, in_features, float(self.omega_0), float(self.scale_0), is_first=self.is_first)
        initialization(self.scale, in_features, float(self.omega_0), float(self.scale_0), is_first=self.is_first)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

        initialization(self.linear, in_features, float(self.omega_0.item()), float(self.scale_0.item()), is_first=self.is_first)


    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class WIRE(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, depth=6, hidden_features=128, first_omega_0=10.0, hidden_omega_0=10.0, scale=10.0, complex=True):
        
        super(WIRE, self).__init__()
        
        #hidden_features = int(hidden_features/np.sqrt(2))
        
        self.net = []

        if complex:
            dtype = torch.cfloat 

            self.net.append(ComplexGaborLayer(in_dim,
                                        hidden_features, 
                                        omega0=first_omega_0,
                                        is_first=True, 
                                        sigma0=scale,
                                        trainable=False))

            for _ in range(depth):
                self.net.append(ComplexGaborLayer(hidden_features,
                                            hidden_features, 
                                            omega0=hidden_omega_0,
                                            sigma0=scale,
                                            trainable=False))
        
        else:
            dtype = torch.float  

            self.net.append(RealGaborLayer(in_dim,
                                        hidden_features, 
                                        omega0=first_omega_0,
                                        is_first=True, 
                                        sigma0=scale,
                                        trainable=False))

            for _ in range(depth):
                self.net.append(RealGaborLayer(hidden_features,
                                            hidden_features, 
                                            omega0=hidden_omega_0,
                                            sigma0=scale,
                                            trainable=False))

        final_linear = nn.Linear(hidden_features, out_dim, dtype=dtype)        
        with torch.no_grad():
            if hidden_omega_0 != 0:
                b = (6.0 / hidden_features) ** 0.5 / hidden_omega_0
            else:
                b = (6.0 / hidden_features) ** 0.5 / scale

            if torch.is_complex(final_linear.weight):
                final_linear.weight.data.real.uniform_(-b, b)
                final_linear.weight.data.imag.uniform_(-b, b)
            else:
                final_linear.weight.uniform_(-b, b)

            if final_linear.bias is not None:
                if torch.is_complex(final_linear.weight):
                    final_linear.bias.real.uniform_(-b, b)
                    final_linear.bias.imag.uniform_(-b, b)
                else:
                    final_linear.bias.uniform_(-b, b)

        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output.real
    
def initialization(linear: nn.Linear, in_features: int, omega_0: float, sigma_0: float, is_first: bool):
    with torch.no_grad():
        if is_first:
            bound = 1.0 / in_features
        else:
            if omega_0 != 0:
                bound = (6.0 / in_features) ** 0.5 / omega_0
            else:
                bound = (6.0 / in_features) ** 0.5 / sigma_0

        # weights
        if torch.is_complex(linear.weight):
            # Initialize real/imag independently with the same bound
            linear.weight.data.real.uniform_(-bound, bound)
            linear.weight.data.imag.uniform_(-bound, bound)
        else:
            linear.weight.uniform_(-bound, bound)

        # biases (match SIREN style by using same bound)
        if linear.bias is not None:
            if torch.is_complex(linear.bias):
                linear.bias.data.real.uniform_(-bound, bound)
                linear.bias.data.imag.uniform_(-bound, bound)
            else:
                linear.bias.uniform_(-bound, bound)