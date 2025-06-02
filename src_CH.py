"""
Core Cahn-Hilliard simulation implementation using pseudospectral methods.
"""

import torch
from typing import Union, Tuple, Optional
from tqdm import tqdm


@torch.jit.script
def compute_dfdc(c: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Compute derivative of free energy w.r.t. concentration"""
    return 2 * W * (c * (1 - c) ** 2 - (1 - c) * c**2)


@torch.jit.script
def compute_dfdc_hat(dfdc_local: torch.Tensor, dealias: torch.Tensor) -> torch.Tensor:
    """Compute Fourier transform of dfdc with dealiasing"""
    return torch.fft.fftn(dfdc_local, dim=(-1, -2)) * dealias


@torch.jit.script
def compute_chat(
    c_hat: torch.Tensor,
    dfdc_hat: torch.Tensor,
    kappa: torch.Tensor,
    K2: torch.Tensor,
    M: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Update Fourier coefficients of concentration field"""
    return (c_hat - dt.view(-1, 1, 1) * K2 * M.view(-1, 1, 1) * dfdc_hat) / (
        1 + dt.view(-1, 1, 1) * M.view(-1, 1, 1) * kappa * K2**2
    )


@torch.jit.script
def ch_step_single(
    c: torch.Tensor,
    c_hat: torch.Tensor,
    W: torch.Tensor,
    kappa: torch.Tensor,
    K2: torch.Tensor,
    M: torch.Tensor,
    dt: torch.Tensor,
    dealias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single time step of Cahn-Hilliard simulation"""
    dfdc_local = compute_dfdc(c, W)
    dfdc_hat = compute_dfdc_hat(dfdc_local, dealias)
    c_hat = compute_chat(c_hat, dfdc_hat, kappa, K2, M, dt)
    c = torch.fft.ifftn(c_hat, dim=(-1, -2)).real
    return c_hat, c


def cahn_hilliard_simulation(
    c0: torch.Tensor,  # Initial concentration field (n_sim, N_X, N_Y)
    W: Union[float, torch.Tensor],  # Bulk free energy coefficient
    kappa: Union[float, torch.Tensor],  # Gradient energy coefficient
    L: Union[float, torch.Tensor],  # Domain size
    Nsteps: int,  # Number of simulation steps
    dt: Union[float, torch.Tensor] = 0.5,  # Time step size
    M: Union[float, torch.Tensor] = 1.0,  # Mobility
    progress_bar: bool = True,  # Show progress bar
) -> torch.Tensor:
    """Run Cahn-Hilliard simulation with implicit pseudospectral method"""
    device = c0.device
    n_sim, N_X, N_Y = c0.shape

    def prep_parameter(param, name, view_shape=None):
        """Ensure parameters are properly shaped tensors"""
        if view_shape is None:
            view_shape = (n_sim, 1, 1)

        if isinstance(param, (int, float)):
            return torch.full(
                view_shape, float(param), dtype=torch.float32, device=device
            )
        elif isinstance(param, torch.Tensor):
            if param.ndim == 0:
                return param.expand(view_shape[0]).view(view_shape)
            elif param.shape == view_shape:
                return param
            else:
                try:
                    return param.view(view_shape)
                except RuntimeError:
                    raise ValueError(
                        f"{name} shape {param.shape} cannot be reshaped to {view_shape}"
                    )
        else:
            raise ValueError(f"{name} must be float or torch.Tensor")

    # Prepare parameters
    W = prep_parameter(W, "W")
    kappa = prep_parameter(kappa, "kappa")
    L = prep_parameter(L, "L")
    M = prep_parameter(M, "M")
    dt = prep_parameter(dt, "dt", view_shape=(n_sim,))

    # Initialize concentration and compute FFT
    c = c0.clone()
    c_hat = torch.fft.fftn(c, dim=(-1, -2))

    # Compute frequency vectors
    dx = L / N_X
    kx = torch.cat([torch.fft.fftfreq(N_X, d=d_i.item()) * 2 * torch.pi for d_i in dx])
    kx = kx.view(n_sim, N_X)

    K = torch.zeros((n_sim, 2, N_X, N_X), device=device)
    for i in range(n_sim):
        Kx, Ky = torch.meshgrid(kx[i], kx[i], indexing="ij")
        K[i, 0], K[i, 1] = Kx, Ky

    # Compute squared wavenumbers and dealiasing mask
    K2 = torch.sum(K * K, dim=1)
    kcut = kx.max() * 2.0 / 3.0
    dealias = (torch.abs(K[:, 0]) < kcut) * (torch.abs(K[:, 1]) < kcut)

    # Simulation loop
    iterator = range(Nsteps)
    if progress_bar:
        iterator = tqdm(iterator, desc="CH Simulation")

    for _ in iterator:
        c_hat, c = ch_step_single(c, c_hat, W, kappa, K2, M, dt, dealias)

    return c


def generate_initial_conditions(
    grid_size: int,  # Size of simulation grid
    n_sims: int,  # Number of simulations
    c_init: Union[float, torch.Tensor],  # Initial concentration values
    noise_amplitude: float = 0.01,  # Amplitude of random noise
    seed: Optional[int] = None,  # Random seed
    device: Optional[torch.device] = None,  # Computation device
) -> torch.Tensor:
    """Generate initial conditions with noise for CH simulations"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        torch.manual_seed(seed)

    # Handle c_init as tensor or scalar
    if isinstance(c_init, (int, float)):
        c_init = torch.full((n_sims,), float(c_init), device=device)
    elif isinstance(c_init, torch.Tensor):
        if c_init.ndim == 0:
            c_init = c_init.expand(n_sims)
        elif c_init.shape != (n_sims,):
            try:
                c_init = c_init.view(n_sims)
            except RuntimeError:
                raise ValueError(
                    f"c_init shape {c_init.shape} cannot be reshaped to ({n_sims},)"
                )
    else:
        raise ValueError("c_init must be float or torch.Tensor")

    # Generate a single noise pattern for all simulations
    noise_pattern = torch.randn(
        grid_size, grid_size, dtype=torch.float32, device=device
    )

    # Initialize output tensor
    c0 = torch.zeros((n_sims, grid_size, grid_size), dtype=torch.float32, device=device)

    # Create initial conditions
    for i in range(n_sims):
        c0[i] = torch.full(
            (grid_size, grid_size), c_init[i], dtype=torch.float32, device=device
        )

    # Add noise to all simulations at once
    c0 += noise_amplitude * noise_pattern

    return c0
