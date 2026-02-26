"""
Functional (stateless) implementation of WIRE network for meta-learning.

MAML requires computing gradients through parameter updates, which needs
a functional form where parameters are passed explicitly rather than
stored in the module.

This module provides:
- FunctionalWIRE: Wrapper that extracts parameters for functional use
- functional_forward: Stateless forward pass
- Parameter manipulation utilities for MAML/REPTILE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np


def complex_gabor_activation(
    x: torch.Tensor,
    weight_freq: torch.Tensor,
    bias_freq: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    bias_scale: Optional[torch.Tensor],
    omega_0: float,
    sigma_0: float
) -> torch.Tensor:
    """
    Functional Complex Gabor activation (WIRE's core nonlinearity).

    Computes: exp(i * omega_0 * (Wx + b)) * exp(-(sigma_0 * (W'x + b'))^2)
    """
    # Frequency path
    omega = omega_0 * F.linear(x, weight_freq, bias_freq)
    # Scale path
    scale = sigma_0 * F.linear(x, weight_scale, bias_scale)

    # Complex Gabor: exp(i*omega - scale^2)
    return torch.exp(1j * omega - scale.abs().square())


def real_gabor_activation(
    x: torch.Tensor,
    weight_freq: torch.Tensor,
    bias_freq: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    bias_scale: Optional[torch.Tensor],
    omega_0: float,
    sigma_0: float
) -> torch.Tensor:
    """
    Functional Real Gabor activation.

    Computes: cos(omega_0 * (Wx + b)) * exp(-(sigma_0 * (W'x + b'))^2)
    """
    omega = omega_0 * F.linear(x, weight_freq, bias_freq)
    scale = sigma_0 * F.linear(x, weight_scale, bias_scale)

    return torch.cos(omega) * torch.exp(-(scale ** 2))


def functional_forward(
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
    config: dict,
    use_complex: bool = True
) -> torch.Tensor:
    """
    Stateless forward pass through WIRE network.

    Args:
        x: Input coordinates (N, in_dim)
        params: Dictionary of named parameters
        config: Dict with 'depth', 'omega_0', 'sigma_0'
        use_complex: Whether to use complex Gabor (True) or real (False)

    Returns:
        Output predictions (N, out_dim)
    """
    depth = config['depth']
    omega_0 = config['omega_0']
    sigma_0 = config['sigma_0']

    activation_fn = complex_gabor_activation if use_complex else real_gabor_activation

    # First layer (is_first=True uses different initialization but same forward)
    h = activation_fn(
        x,
        params['layer_0.freqs.weight'],
        params.get('layer_0.freqs.bias'),
        params['layer_0.scale.weight'],
        params.get('layer_0.scale.bias'),
        omega_0,
        sigma_0
    )

    # Hidden layers
    for i in range(1, depth + 1):
        h = activation_fn(
            h,
            params[f'layer_{i}.freqs.weight'],
            params.get(f'layer_{i}.freqs.bias'),
            params[f'layer_{i}.scale.weight'],
            params.get(f'layer_{i}.scale.bias'),
            omega_0,
            sigma_0
        )

    # Final linear layer
    output = F.linear(
        h,
        params['final.weight'],
        params.get('final.bias')
    )

    # Return real part if using complex
    if use_complex:
        return output.real
    return output


class FunctionalWIRE(nn.Module):
    """
    WIRE network wrapper that supports functional (stateless) forward passes.

    This class maintains the standard PyTorch module interface but also
    provides methods to:
    - Extract parameters as a flat dictionary
    - Perform forward passes with external parameters
    - Support MAML-style gradient computation

    Args:
        in_dim: Input dimension (typically 4 for t,x,y,z)
        out_dim: Output dimension (typically 3 for u,v,w)
        depth: Number of hidden layers
        hidden_features: Hidden layer width
        omega_0: Frequency parameter for Gabor
        sigma_0: Scale parameter for Gabor
        use_complex: Use complex (True) or real (False) Gabor layers
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 3,
        depth: int = 6,
        hidden_features: int = 128,
        omega_0: float = 30.0,
        sigma_0: float = 30.0,
        use_complex: bool = True
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.hidden_features = hidden_features
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.use_complex = use_complex

        # Store config for functional forward
        self.config = {
            'depth': depth,
            'omega_0': omega_0,
            'sigma_0': sigma_0
        }

        dtype = torch.cfloat if use_complex else torch.float

        # Build layers
        self.layers = nn.ModuleList()

        # First layer - must accept real inputs
        first_layer = nn.ModuleDict({
            'freqs': nn.Linear(in_dim, hidden_features, dtype=torch.float),  # ← Real for first layer
            'scale': nn.Linear(in_dim, hidden_features, dtype=torch.float)
        })
        self._init_gabor_layer(first_layer, in_dim, is_first=True)
        self.layers.append(first_layer)

        # Hidden layers - can be complex
        dtype = torch.cfloat if use_complex else torch.float
        for _ in range(depth):
            layer = nn.ModuleDict({
                'freqs': nn.Linear(hidden_features, hidden_features, dtype=dtype),
                'scale': nn.Linear(hidden_features, hidden_features, dtype=dtype)
            })
            self._init_gabor_layer(layer, hidden_features, is_first=False)
            self.layers.append(layer)

        # Final linear layer
        self.final = nn.Linear(hidden_features, out_dim, dtype=dtype)
        self._init_final_layer()

    def _init_gabor_layer(self, layer: nn.ModuleDict, in_features: int, is_first: bool):
        """Initialize Gabor layer weights following WIRE paper."""
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                if self.omega_0 != 0:
                    bound = (6.0 / in_features) ** 0.5 / self.omega_0
                else:
                    bound = (6.0 / in_features) ** 0.5 / self.sigma_0

            for name in ['freqs', 'scale']:
                linear = layer[name]
                if torch.is_complex(linear.weight):
                    linear.weight.data.real.uniform_(-bound, bound)
                    linear.weight.data.imag.uniform_(-bound, bound)
                else:
                    linear.weight.uniform_(-bound, bound)

                if linear.bias is not None:
                    if torch.is_complex(linear.bias):
                        linear.bias.data.real.uniform_(-bound, bound)
                        linear.bias.data.imag.uniform_(-bound, bound)
                    else:
                        linear.bias.uniform_(-bound, bound)

    def _init_final_layer(self):
        """Initialize final linear layer."""
        with torch.no_grad():
            if self.omega_0 != 0:
                b = (6.0 / self.hidden_features) ** 0.5 / self.omega_0
            else:
                b = (6.0 / self.hidden_features) ** 0.5 / self.sigma_0

            if torch.is_complex(self.final.weight):
                self.final.weight.data.real.uniform_(-b, b)
                self.final.weight.data.imag.uniform_(-b, b)
            else:
                self.final.weight.uniform_(-b, b)

            if self.final.bias is not None:
                if torch.is_complex(self.final.bias):
                    self.final.bias.data.real.uniform_(-b, b)
                    self.final.bias.data.imag.uniform_(-b, b)
                else:
                    self.final.bias.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using module parameters."""
        return self.functional_forward(x, self.get_params())

    def functional_forward(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with external parameters.

        This enables MAML-style gradient computation through parameter updates.

        Args:
            x: Input coordinates (N, in_dim)
            params: Dictionary mapping parameter names to tensors

        Returns:
            Output (N, out_dim)
        """
        activation_fn = complex_gabor_activation if self.use_complex else real_gabor_activation

        # First layer
        h = activation_fn(
            x,
            params['layers.0.freqs.weight'],
            params.get('layers.0.freqs.bias'),
            params['layers.0.scale.weight'],
            params.get('layers.0.scale.bias'),
            self.omega_0,
            self.sigma_0
        )

        # Hidden layers
        for i in range(1, self.depth + 1):
            h = activation_fn(
                h,
                params[f'layers.{i}.freqs.weight'],
                params.get(f'layers.{i}.freqs.bias'),
                params[f'layers.{i}.scale.weight'],
                params.get(f'layers.{i}.scale.bias'),
                self.omega_0,
                self.sigma_0
            )

        # Final layer
        output = F.linear(h, params['final.weight'], params.get('final.bias'))

        if self.use_complex:
            return output.real
        return output

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Extract all parameters as an ordered dictionary."""
        return OrderedDict(
            (name, param) for name, param in self.named_parameters()
        )

    def set_params(self, params: Dict[str, torch.Tensor]):
        """Set parameters from a dictionary (in-place)."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in params:
                    param.copy_(params[name])

    def clone_params(self) -> Dict[str, torch.Tensor]:
        """Create a deep copy of parameters."""
        return OrderedDict(
            (name, param.clone()) for name, param in self.named_parameters()
        )


def sgd_step(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    lr: float
) -> Dict[str, torch.Tensor]:
    """
    Perform a single SGD step on parameters.

    Args:
        params: Current parameters
        grads: Gradients for each parameter
        lr: Learning rate

    Returns:
        Updated parameters (new tensors, preserves computation graph if grads have it)
    """
    return OrderedDict(
        (name, params[name] - lr * grads[name])
        for name in params.keys()
    )


def interpolate_params(
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between two parameter sets.

    Returns: (1 - alpha) * params_a + alpha * params_b
    """
    return OrderedDict(
        (name, (1 - alpha) * params_a[name] + alpha * params_b[name])
        for name in params_a.keys()
    )


def average_params(
    params_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Average multiple parameter sets (for REPTILE)."""
    n = len(params_list)
    avg = OrderedDict()
    for name in params_list[0].keys():
        avg[name] = sum(p[name] for p in params_list) / n
    return avg


def compute_param_norm(params: Dict[str, torch.Tensor]) -> float:
    """Compute L2 norm of all parameters."""
    total = 0.0
    for p in params.values():
        if torch.is_complex(p):
            total += p.real.norm().item() ** 2 + p.imag.norm().item() ** 2
        else:
            total += p.norm().item() ** 2
    return np.sqrt(total)


def compute_param_distance(
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor]
) -> float:
    """Compute L2 distance between two parameter sets."""
    total = 0.0
    for name in params_a.keys():
        diff = params_a[name] - params_b[name]
        if torch.is_complex(diff):
            total += diff.real.norm().item() ** 2 + diff.imag.norm().item() ** 2
        else:
            total += diff.norm().item() ** 2
    return np.sqrt(total)
