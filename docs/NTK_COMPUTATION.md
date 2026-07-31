# Neural Tangent Kernel: Computation and Visualization

## 1. Mathematical Background

The **Neural Tangent Kernel (NTK)** characterizes how a neural network learns. For a network
$f_\theta : \mathbb{R}^d \to \mathbb{R}^m$ with parameters $\theta \in \mathbb{R}^P$, the
empirical NTK is defined as:

$$K(x_1, x_2) = \nabla_\theta f(x_1)^\top \cdot \nabla_\theta f(x_2)$$

where $\nabla_\theta f(x) \in \mathbb{R}^{P \times m}$ is the Jacobian of all network outputs
with respect to all parameters, evaluated at input $x$.

For a dataset $\{x_i\}_{i=1}^n$, this gives an $n \times n$ kernel matrix:

$$K_{ij} = \nabla_\theta f(x_i)^\top \cdot \nabla_\theta f(x_j) = J(x_i) \cdot J(x_j)^\top$$

where each $J(x_i) \in \mathbb{R}^{m \times P}$ is the Jacobian at $x_i$.

### Why the NTK matters

The NTK governs training dynamics under gradient descent. For an MSE loss, the error
$e(t) = f_\theta(X) - y$ evolves as:

$$\dot{e}(t) = -K \cdot e(t)$$

This means:
- **Eigenvalues** of $K$ determine the relative learning rates of its corresponding modes under gradient descent. 
  Modes associated with larger eigenvalues generally converge faster. 
- **Eigenvectors** of $K$ are the spatial patterns the network learns, in order of speed.
- **Condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$ measures optimization difficulty.
  A large condition number means some modes learn much faster than others.

At initialization, the NTK is determined by the network architecture, its initialization, and the 
input coordinates at which it is evaluated; it does not depend on the target values or subsequent 
optimization trajectory. This makes it a pure diagnostic of the network's inductive bias.

### Spectral bias

Networks exhibit **spectral bias**: modes with large eigenvalues (smooth, low-frequency patterns)
are learned first. High-frequency modes have smaller eigenvalues and require more training steps.
The $\omega_0$ parameter in SIREN/WIRE directly controls how much high-frequency content the
network can represent, which is reflected in the NTK spectrum.

---

## 2. Jacobian Computation

Computing $J(x_i) = \nabla_\theta f(x_i)$ for each input point individually and then stacking is
infeasible for large $n$ and $P$. This project uses PyTorch's `torch.func` API for efficient,
batched per-sample Jacobians.

**Core implementation** (`utils/ntk.py`, function `get_ntk_fn`):

```python
from torch.func import functional_call, vmap, jacrev

def get_ntk_fn(model):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def fnet_single(p, x):
        # functional_call evaluates the model with parameter dict p at input x
        return functional_call(model, p, (x.unsqueeze(0),)).squeeze(0)

    def flatten_jacobian(jac_dict):
        # jac_dict maps param_name -> tensor of shape (out_dim, *param_shape)
        # flatten each to (out_dim, -1), then concatenate along the last dim
        jac_tensors = [v.flatten(2) for v in jac_dict.values()]
        return torch.cat(jac_tensors, dim=2)  # (batch, out_dim, total_params)

    def ntk_fn(x1, x2):
        # vmap(jacrev(fnet_single), (None, 0)) applies jacrev over the batch dim of x
        jac1_dict = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac2_dict = vmap(jacrev(fnet_single), (None, 0))(params, x2)

        jac1_flat = flatten_jacobian(jac1_dict)   # (b1, out_dim, P)
        jac2_flat = flatten_jacobian(jac2_dict)   # (b2, out_dim, P)

        # Sum over output dimensions (standard for multi-output networks)
        jac1 = jac1_flat.sum(dim=1)  # (b1, P)
        jac2 = jac2_flat.sum(dim=1)  # (b2, P)

        return jac1 @ jac2.T  # (b1, b2)

    return ntk_fn
```

**Key components:**

- `functional_call(model, params, inputs)`: evaluates the model using an explicit parameter
  dictionary instead of `model.parameters()`. This is necessary for `jacrev` to differentiate
  with respect to the parameters as inputs.

- `jacrev(fnet_single)`: computes the Jacobian of `fnet_single` with respect to its first
  argument (`params`) using reverse-mode autodiff. Result: a dict mapping each parameter name
  to a tensor of shape `(out_dim, *param_shape)`.

- `vmap(..., (None, 0))`: vectorizes the Jacobian computation over the batch dimension of `x`
  (second argument, axis 0), while broadcasting the shared `params` (first argument, `None`).
  This avoids a Python-level loop over data points.

- **Summing over output dims**: For a network with `out_dim=3` (u, v, w), summing collapses
  `(b, 3, P)` to `(b, P)`. This is the standard convention for multi-output NTK analysis,
  treating the kernel as a scalar function of the parameter sensitivity across all outputs.

---

## 3. NTK Matrix Assembly

The full $n \times n$ NTK matrix cannot be computed in one shot for large $n$ (e.g., $n=4096$,
$P \sim 10^5$). The computation is split into a batched double loop over row-blocks and column-blocks.

**Implementation** (`utils/ntk.py`, function `ntk_eigendecomposition`):

```python
ntk_matrix = torch.zeros((n_data, n_data), device=device)

for i in tqdm(range(0, n_data, batch_size), desc="NTK Matrix Rows"):
    x1_batch = data[i : i + batch_size]
    for j in range(0, n_data, batch_size):
        x2_batch = data[j : j + batch_size]
        kernel_block = ntk_fn(x1_batch, x2_batch)  # (b1, b2)
        ntk_matrix[i : i + batch_size, j : j + batch_size] = kernel_block
```

With `n=4096` and `batch_size=128`, this is $32 \times 32 = 1024$ block evaluations. Each block
requires computing Jacobians for two mini-batches and performing a matrix product.

The resulting matrix is symmetric and positive semi-definite (PSD) by construction:
$K_{ij} = \langle J(x_i), J(x_j) \rangle$ is an inner product.

---

## 4. Eigendecomposition

The NTK matrix is moved to CPU and decomposed with `scipy.sparse.linalg.eigsh`:

```python
ntk_np = ntk_matrix.cpu().numpy()

eigvals, eigvecs = scipy.sparse.linalg.eigsh(ntk_np, k=200, which='LM')

# Sort descending (eigsh returns in ascending order)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
```

**Why `eigsh`?**

- `eigsh` is for **symmetric** matrices (the `h` stands for Hermitian). The NTK is symmetric, so
  this is the correct choice.
- `which='LM'` selects the **Largest Magnitude** eigenvalues. For a PSD matrix this equals the
  largest eigenvalues.
- Computing only the top $k=200$ eigenpairs (out of $n=4096$) is much cheaper than full
  decomposition — `eigsh` uses the **ARPACK** iterative method and scales as $O(n \cdot k)$
  rather than $O(n^3)$.

**Interpretation:**

| Quantity | Meaning |
|---|---|
| $\lambda_0 = \lambda_{\max}$ | Fastest-learning mode (smoothest eigenvector) |
| $\lambda_{k-1} = \lambda_{\min}$ | Slowest-learning computed mode |
| $\kappa = \lambda_{\max} / \lambda_{\min}$ | Condition number (higher = harder to optimize) |
| $v_i \in \mathbb{R}^n$ | Spatial pattern for mode $i$ (one value per input point) |

---

## 5. Visualization: 2D Grid Slice

The eigenvectors $v_i \in \mathbb{R}^n$ contain one scalar value per input point. If the input
points are sampled from a **regular 2D grid**, the eigenvector can be directly reshaped into an
image and displayed as a heatmap — no interpolation needed.

**Grid construction** (`tools/run_ntk.py`, `tools/ntk_sanity_check_2d.py`):

```python
RES = 64
xs = np.linspace(0, 1, RES)
ys = np.linspace(0, 1, RES)
grid_x, grid_y = np.meshgrid(xs, ys)

# For 4D networks (in_dim=4): pad x,y grid with fixed t=0.5, z=0.5
coords_np = np.stack([
    np.full(RES * RES, 0.5),   # t (fixed)
    grid_x.ravel(),             # x (varies)
    grid_y.ravel(),             # y (varies)
    np.full(RES * RES, 0.5),   # z (fixed)
], axis=1).astype(np.float32)  # (4096, 4)

# For 2D networks (in_dim=2): native 2D grid
coords_np = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (4096, 2)
```

**Eigenvector heatmap**:

```python
for i in range(5):
    v_i = eigvecs[:, i].reshape(RES, RES)  # direct reshape — exact values, no interpolation
    plt.imshow(v_i, cmap='viridis', origin='lower')
```

Because the coordinates were constructed in row-major (C) order via `meshgrid` + `ravel()`,
the reshape exactly inverts the flattening — each pixel corresponds to exactly one input point.

### 4D slice interpretation

For a 4D network (input: [t, x, y, z]), fixing t=0.5 and z=0.5 produces a 2D cross-section of
the 4D kernel. This slice shows how the network's learning modes look in the x-y plane at a
specific moment in time and depth. The patterns should be comparable to those of a native 2D
network, aside from any distortion introduced by the two constant extra dimensions.

---

## 6. Files in This Project

| File | Purpose |
|---|---|
| `utils/ntk.py` | Core empirical NTK computation and eigendecomposition. |
| `tools/run_ntk.py` | Main script for computing and visualizing NTK eigenfunctions for WIRE and SIREN on a regular 2D slice through the 4D input domain. |
| `tools/ntk_sanity_check_2d.py` | Independent 2D SIREN sanity check of the NTK implementation. |

## 7. References

1. **Jacot, A., Gabriel, F., & Hongler, C. (2018)**. *Neural Tangent Kernel: Convergence and
   Generalization in Neural Networks*. NeurIPS 2018.
   https://arxiv.org/abs/1806.07572

2. **Tancik, M., Srinivasan, P., Mildenhall, B., et al. (2020)**. *Fourier Features Let Networks
   Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020.
   https://arxiv.org/abs/2006.10739
   *(Source of the eigenvector heatmap visualization approach.)*

3. **Saragadam, V., LeJeune, D., Tan, J., et al. (2023)**. *WIRE: Wavelet Implicit neural
   REpresentations*. CVPR 2023.
   https://arxiv.org/abs/2301.05187
   *(WIRE network architecture with Gabor wavelet activations.)*

4. **Sitzmann, V., Martel, J., Bergman, A., et al. (2020)**. *Implicit Neural Representations
   with Periodic Activation Functions*. NeurIPS 2020.
   https://arxiv.org/abs/2006.09661
   *(SIREN network architecture.)*

5. **PyTorch team**. *Neural Tangent Kernels (torch.func tutorial)*.
   https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
   *(Reference implementation for `vmap + jacrev` NTK computation.)*

---
