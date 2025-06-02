import torch
from torch import Tensor
from pykeops.torch import LazyTensor
import torch
import numpy as np
import torch.nn as nn
import ripserplusplus as rpp_py
from typing import Tuple, Dict, Optional, List, Union, Set
from utils_mst import get_mst_edges
from scipy import special
import math


# Device configuration
def get_device(device_str=None):
    if device_str:
        return torch.device(device_str)

    # Check for available accelerated hardware in order of preference
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.backends, "mps")
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def levina_bickel_dim_multi_k(
    X: Tensor,
    k_min: int = 5,
    k_max: int = 10,
    p: float = 1,
    unbiased: bool = True,
) -> Tensor:
    """
    Compute the Levina-Bickel intrinsic dimensionality estimate for a dataset
    for multiple k values efficiently.

    Parameters:
    -----------
    X : torch.Tensor
        The input data tensor of shape (N, D), where N is the number of samples
        and D is the number of features.

    k_min : int, optional (default=5)
        The minimum number of nearest neighbors to use in the estimation.

    k_max : int, optional (default=20)
        The maximum number of nearest neighbors to use in the estimation.

    p : float, optional (default=2)
        The order of Minkowski distance to use.
        p=1 gives Manhattan distance, p=2 gives Euclidean distance, etc.
    unbiased : bool, optional (default=True) Whether to divide by k-2 instead of k-1.

    Returns:
    --------
    mle : torch.Tensor
        A tensor of shape (N, k_max - k_min + 1) containing the local estimates of
        intrinsic dimensionality for each point in the dataset and each k value.
    """
    N, D = X.shape
    device = X.device

    # Input validation
    if k_min < 2 or k_max < k_min or k_max >= N:
        raise ValueError(
            "Invalid k_min or k_max. Must satisfy: 3 <= k_min < k_max < N."
        )

    # Create LazyTensors
    X_i = LazyTensor(X[:, None, :])  # (N, 1, D)
    X_j = LazyTensor(X[None, :, :])  # (1, N, D)

    # Compute Minkowski distances
    if p == 2:
        # Optimization for Euclidean distance
        # WARNING: IF YOU PUT "2.0" INSTEAD OF "2", A SUBTLE KEOPS ERROR WILL FILL NANS
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    else:
        D_ij = ((X_i - X_j).abs() ** p).sum(-1)

    # Find k_max + 1 nearest neighbors (including self)
    distances = D_ij.Kmin(k_max + 1, dim=1)  # (N, k_max + 1)

    # Remove the first column (distance to self) and apply root
    distances = distances[:, 1:] ** (1.0 / float(p))  # (N, k_max)

    # Compute log distances (N, k_max)
    ldists = torch.log(distances)

    # Compute MLE estimates for all k values
    num_k_vals = k_max - k_min + 1
    k_vals = torch.arange(k_min, k_max + 1, device=device)
    val = 2 if unbiased else 1
    mle = (k_vals[None, :] - val) / (
        (k_vals - 1) * ldists[:, -num_k_vals:]
        - torch.cumsum(ldists[:, :-1], dim=1)[:, -num_k_vals:]
    )

    return mle


def estimate_poisson_intensity(
    distances: torch.Tensor, dim: float, k_max: int = 5
) -> float:
    """
    Estimate Poisson process intensity parameter lambda using k-NN distances.

    Args:
        distances: Pairwise distance matrix of shape (n_points, n_points)
        dim: Intrinsic dimensionality of the data
        k_max: Maximum k value for k-NN distances to use

    Returns:
        Estimated intensity parameter lambda
    """
    n_points = distances.shape[0]

    # Compute volume of unit ball in D dimensions in log space: log(ω_D) = log(π^(D/2)) - log(Γ(D/2 + 1))
    # = (D/2)*log(π) - log_gamma(D/2 + 1)
    log_omega_d = (dim / 2) * np.log(np.pi) - special.gammaln(dim / 2 + 1)
    omega_d = np.exp(log_omega_d)

    # Sort distances to get k-NN distances (excluding self)
    sorted_dists, _ = torch.sort(distances, dim=1)

    # Remove self-distances (first column)
    k_nearest_dists = sorted_dists[:, 1 : k_max + 1]

    # Vectorized approach to calculate lambda for all k values at once
    r_k_sums = torch.sum(k_nearest_dists**dim, dim=0)
    lambda_estimates = (
        n_points * torch.arange(1, k_max + 1, device=distances.device)
    ) / (omega_d * r_k_sums)

    return lambda_estimates.mean().item()  # Return the mean of lambda estimates


def beta_zero_from_mst(
    data: torch.Tensor,
    dim: Optional[float] = None,
    k_max: int = 10,
    alpha: float = 3.0,
) -> Tuple[int, float, Dict]:
    """
    Calculate β₀ (number of connected components) using MST and Poisson process model.

    This function estimates the intensity parameter of the underlying Poisson process,
    calculates the expected nearest neighbor distance, and sets a threshold based on this.
    The β₀ value is then determined from the MST edges as one plus the number of MST edges
    longer than twice the threshold.

    Args:
        data: Point cloud tensor of shape (n_points, dim)
        dim: Intrinsic dimensionality. If None, ambient dimension is used.
        k_max: Maximum k value for k-NN distances used in lambda estimation
        alpha: Scaling factor for connectivity threshold

    Returns:
        Tuple of (β₀ value, connectivity threshold, diagnostics dictionary)
    """
    n_points = data.shape[0]

    # Use ambient dimension if intrinsic dim not provided
    if dim is None:
        dim = float(data.shape[1])

    # Compute pairwise distances
    distances = torch.cdist(data, data)

    # Estimate Poisson intensity parameter
    lambda_hat = estimate_poisson_intensity(distances, dim, k_max)

    # Compute volume of unit ball in D dimensions in log space
    log_omega_d = (dim / 2) * np.log(np.pi) - special.gammaln(dim / 2 + 1)
    omega_d = np.exp(log_omega_d)

    # Compute expected 1-NN distance in log space
    # log(r_expected) = log(gamma_term) - (1/dim) * [log(lambda_hat) + log(omega_d)]
    log_gamma_term = special.gammaln(1 + 1 / dim)  # log of gamma(1 + 1/dim)
    gamma_term = np.exp(log_gamma_term)

    # Complete the calculation
    r_expected = gamma_term / (lambda_hat * omega_d) ** (1 / dim)

    # Set connectivity threshold with scaling factor
    connectivity_threshold = alpha * r_expected

    # Compute MST
    mst_pairs = get_mst_edges(distances, one_root=True)
    mst_edge_weights = distances[mst_pairs[:, 0], mst_pairs[:, 1]]

    # B₀ is one plus the number of MST edges longer than 2*threshold
    edge_count = torch.sum(mst_edge_weights > 2 * connectivity_threshold).item()
    beta_zero = int(edge_count + 1)

    # Create diagnostics dictionary
    diagnostics = {
        "lambda_hat": lambda_hat,
        "expected_nn_distance": r_expected,
        "connectivity_threshold": connectivity_threshold,
        "mst_edges": len(mst_edge_weights),
        "long_edges": edge_count,
    }

    return beta_zero, connectivity_threshold, diagnostics


def compute_empirical_lipschitz(
    input: torch.Tensor,
    output: torch.Tensor,
    p: float = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Lipschitz constant using PyKeops reductions to avoid OOM errors.

    :Args:
        input: torch.Tensor
            Input tensor of shape (B, D1).
        output: torch.Tensor
            Output tensor of shape (B, D2).
        p: float
            Order of the norm.
        eps: float
            Small value to avoid division by zero.

    :Returns:
        Maximum Lipschitz constant.
    """
    # check shapes
    if (len(input.shape) != 2) and (len(input.shape) != 1):
        raise ValueError(
            "Inputs must be a 1D or 2D tensors, but got {} and {}".format(
                input.shape, output.shape
            )
        )
    if (len(output.shape) != 2) and (len(output.shape) != 1):
        raise ValueError(
            "Outputs must be a 1D or 2D tensors, but got {} and {}".format(
                input.shape, output.shape
            )
        )
    if input.shape[0] != output.shape[0]:
        raise ValueError(
            "Inputs and Outputs must have the same batch size, but got {} and {}".format(
                input.shape[0], output.shape[0]
            )
        )

    if len(input.shape) == 1:
        input = input.unsqueeze(1)
    if len(output.shape) == 1:
        output = output.unsqueeze(1)

    D1, D2 = input.shape[1], output.shape[1]

    # param_dist = ((input[:, None, :] - input[None, :, :]).abs() ** p).mean(
    #     dim=2
    # )  # (B, B)
    # output_dist = ((output[:, None, :] - output[None, :, :]).abs() ** p).mean(
    #     dim=2
    # )  # (B, B)

    param_dist = torch.cdist(input[None, :, :], input[None, :, :], p=p).squeeze(0)
    output_dist = torch.cdist(output[None, :, :], output[None, :, :], p=p).squeeze(0)

    # Compute Lipschitz ratio and mask diagonal (i=j)
    lip_fwd = output_dist / (param_dist + eps)  # (B, B)
    lip_fwd_mask = param_dist > eps  # (B, B)
    lip_fwd *= lip_fwd_mask  # Apply mask to zero out diagonal entries
    max_fwd_ratios = torch.max(lip_fwd, dim=1).values  # * (D1 / D2)

    lip_inv = param_dist / (output_dist + eps)  # (B, B)
    lip_inv_mask = output_dist > eps  # (B, B)
    lip_inv *= lip_inv_mask  # Apply mask to zero out diagonal entries
    max_inv_ratios = torch.max(lip_inv, dim=1).values  # * (D2 / D1)
    return max_fwd_ratios, max_inv_ratios


class TopoAELoss(nn.Module):
    """Computes the TopoAE topology-preserving loss between input and latent spaces."""

    def __init__(self):
        """
        Initialize TopoAE loss module.

        """
        super().__init__()

    def _get_pairings(self, distances: Tensor) -> Tuple[Tensor, Tensor]:
        """Get persistent homology pairings from distance matrix."""
        return get_mst_edges(distances, one_root=True)

    def _select_distances_from_pairs(
        self, distance_matrix: Tensor, pairs: Tensor
    ) -> Tensor:
        """Select distances corresponding to persistence pairs."""
        return distance_matrix[(pairs[:, 0], pairs[:, 1])]

    def forward(self, input_distances: Tensor, latent_distances: Tensor) -> Tensor:
        """
        Compute TopoAE loss between input and latent space distances.

        Args:
            input_distances: Distance matrix in input space (N x N)
            latent_distances: Distance matrix in latent space (N x N)

        Returns:
            Tuple of (loss value, dictionary of distance components)
        """
        pairs1 = self._get_pairings(input_distances)
        pairs2 = self._get_pairings(latent_distances)

        # Symmetric matching of edges
        sig1 = self._select_distances_from_pairs(input_distances, pairs1)
        sig2 = self._select_distances_from_pairs(latent_distances, pairs2)
        sig1_2 = self._select_distances_from_pairs(latent_distances, pairs1)
        sig2_1 = self._select_distances_from_pairs(input_distances, pairs2)

        distance1_2 = ((sig1 - sig1_2) ** 2).sum()
        distance2_1 = ((sig2 - sig2_1) ** 2).sum()

        distance = distance1_2 + distance2_1

        return distance


def get_indicies(DX, rc, dim, card):
    """
    Extract significant point indices from persistence diagram.
    Fully vectorized implementation for improved performance.

    Args:
        DX: Distance matrix
        rc: Ripser computation results
        dim: Homological dimension
        card: Maximum number of points to return (cardinality)

    Returns:
        List of indices representing the most significant topological features
    """
    dgm = rc["dgms"][dim]
    pairs = rc["pairs"][dim]

    if not pairs:  # Handle empty pairs
        return [0] * (4 * card)

    # Filter valid pairs (correct dimensions)
    valid_pairs_idx = []
    for i, (s1, s2) in enumerate(pairs):
        if len(s1) == dim + 1 and len(s2) > 0:
            valid_pairs_idx.append(i)

    if not valid_pairs_idx:  # Handle no valid pairs
        return [0] * (4 * card)

    valid_pairs_idx = np.array(valid_pairs_idx)
    valid_pairs = [pairs[i] for i in valid_pairs_idx]

    # Calculate persistence values (death - birth) vectorized
    persistence_values = np.array([dgm[i][1] - dgm[i][0] for i in valid_pairs_idx])

    # Vectorized processing of simplices
    s1_arrays = []  # First simplices as arrays
    s2_arrays = []  # Second simplices as arrays

    for s1, s2 in valid_pairs:
        s1_arrays.append(np.array(s1))
        s2_arrays.append(np.array(s2))

    # Collect all representative points
    representative_indices = []

    # Process each pair and extract representative vertices
    for i, (s1, s2) in enumerate(zip(s1_arrays, s2_arrays)):
        # For first simplex: extract submatrix and find vertices with max distance
        dist_submatrix1 = DX[np.ix_(s1, s1)]
        flat_idx1 = np.argmax(dist_submatrix1)
        idx1_i, idx1_j = np.unravel_index(flat_idx1, dist_submatrix1.shape)
        rep_idx1 = [s1[idx1_i], s1[idx1_j]]

        # For second simplex: extract submatrix and find vertices with max distance
        dist_submatrix2 = DX[np.ix_(s2, s2)]
        flat_idx2 = np.argmax(dist_submatrix2)
        idx2_i, idx2_j = np.unravel_index(flat_idx2, dist_submatrix2.shape)
        rep_idx2 = [s2[idx2_i], s2[idx2_j]]

        # Store all four representative vertices
        representative_indices.append(rep_idx1 + rep_idx2)

    # Convert to numpy array for efficient operations
    representative_indices = np.array(representative_indices)

    # Sort by persistence (highest first)
    sorted_indices = np.argsort(-persistence_values)
    sorted_reps = representative_indices[sorted_indices]

    # Flatten the result
    flattened_indices = sorted_reps.flatten()

    # Take top indices and pad if needed
    result_length = 4 * card
    if len(flattened_indices) >= result_length:
        return list(flattened_indices[:result_length].astype(np.int64))
    else:
        # Pad with zeros
        padding = [0] * (result_length - len(flattened_indices))
        return list(np.array(list(flattened_indices) + padding, dtype=np.int64))


def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1

    if engine == "ripser":
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0  # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))
        rc = rpp_py.run("--format distance --dim " + str(dim), DX_)
    # elif engine == "giotto":
    #     rc = ripser_parallel(
    #         DX,
    #         maxdim=dim,
    #         metric="precomputed",
    #         collapse_edges=False,
    #         n_threads=n_threads,
    #     )
    else:
        raise ValueError(f"Unknown engine: {engine}")

    all_indicies = []  # for every dimension
    for d in range(1, dim + 1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies


class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode="minimum", n_threads=8, engine="giotto"):
        super().__init__()

        if dim < 1:
            raise ValueError(
                f"Dimension should be greater than 1. Provided dimension: {dim}"
            )
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine

    def forward(self, Dr1, Dr2, immovable=None):
        # inputs are distance matricies
        d, c = self.dim, self.card

        device = Dr1.device
        # Compute distance matrices
        #         Dr1 = torch.cdist(r1, r1)
        #         Dr2 = torch.cdist(r2, r2)

        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)
        if self.mode == "minimum":
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat(
                    (torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr1), 1)), 0
                )  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat(
                    (torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0
                )  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat(
                    (torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0
                )  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat(
                    (torch.cat((Dzz, Dr2.T), 1), torch.cat((Dr2, Dr2), 1)), 0
                )  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX

        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        all_ids = Rips(
            DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine
        )
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            tmp_idx = np.reshape(ids, [2 * c, 2])
            if self.mode == "minimum":
                dgm = torch.hstack(
                    [
                        torch.reshape(DX[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]),
                        torch.reshape(DX_2[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1]),
                    ]
                )
            else:
                dgm = torch.hstack(
                    [
                        torch.reshape(DX_2[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]),
                        torch.reshape(DX[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1]),
                    ]
                )
            all_dgms.append(dgm)
        return all_dgms


class RTD_differentiable_pairs(nn.Module):
    def __init__(self, dim=1, card=50, mode="minimum", n_threads=8, engine="giotto"):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine

    def forward(self, Dr1, Dr2, immovable=None):
        device = Dr1.device
        n = len(Dr1)
        Dzz = torch.zeros((n, n), device=device)

        if self.mode == "minimum":
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            DX_2 = torch.cat(
                (
                    torch.cat((Dzz, Dr1.T), 1),
                    torch.cat(
                        (
                            Dr1,
                            Dr2 if immovable == 1 else Dr1 if immovable == 2 else Dr12,
                        ),
                        1,
                    ),
                ),
                0,
            )
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            DX_2 = torch.cat(
                (
                    torch.cat((Dzz, Dr1.T), 1),
                    torch.cat(
                        (
                            Dr1,
                            Dr2 if immovable == 2 else Dr2 if immovable == 1 else Dr2,
                        ),
                        1,
                    ),
                ),
                0,
            )

        all_ids = Rips(
            DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine
        )
        all_dgms = []
        all_pairs = []

        for ids in all_ids:
            tmp_idx = np.reshape(ids, [2 * self.card, 2])
            pairs_1 = tmp_idx[::2]
            pairs_2 = tmp_idx[1::2]

            # Convert batch indices back to original point indices
            pairs_1_conv = np.where(pairs_1 >= n, pairs_1 - n, pairs_1)
            pairs_2_conv = np.where(pairs_2 >= n, pairs_2 - n, pairs_2)
            all_pairs.append((pairs_1_conv, pairs_2_conv))

            if self.mode == "minimum":
                dgm = torch.hstack(
                    [
                        torch.reshape(DX[pairs_1[:, 0], pairs_1[:, 1]], [self.card, 1]),
                        torch.reshape(
                            DX_2[pairs_2[:, 0], pairs_2[:, 1]], [self.card, 1]
                        ),
                    ]
                )
            else:
                dgm = torch.hstack(
                    [
                        torch.reshape(
                            DX_2[pairs_1[:, 0], pairs_1[:, 1]], [self.card, 1]
                        ),
                        torch.reshape(DX[pairs_2[:, 0], pairs_2[:, 1]], [self.card, 1]),
                    ]
                )
            all_dgms.append(dgm)

        return all_dgms, all_pairs


class RTDLoss(nn.Module):
    def __init__(
        self,
        dim=1,
        card=50,
        n_threads=25,
        engine="ripser",
        mode="minimum",
        is_sym=True,
        lp=1.0,
        **kwargs,
    ):
        super().__init__()

        self.is_sym = is_sym
        self.mode = mode
        self.p = lp
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)

    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += torch.sum(torch.abs(rtd_xz[d][:, 1] - rtd_xz[d][:, 0]) ** self.p)
            if self.is_sym:
                loss_zx += torch.sum(
                    torch.abs(rtd_zx[d][:, 1] - rtd_zx[d][:, 0]) ** self.p
                )
        loss = (loss_xz + loss_zx) / 2.0
        return loss


class MinMaxRTDLoss(nn.Module):
    def __init__(
        self,
        dim=1,
        card=50,
        n_threads=25,
        engine="ripser",
        is_sym=True,
        lp=1.0,
        **kwargs,
    ):
        super().__init__()

        self.is_sym = is_sym
        self.p = lp
        self.rtd_min = RTD_differentiable(dim, card, "minimum", n_threads, engine)
        self.rtd_max = RTD_differentiable(dim, card, "maximum", n_threads, engine)

    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd_min(x_dist, z_dist, immovable=1) + self.rtd_max(
            x_dist, z_dist, immovable=1
        )
        if self.is_sym:
            rtd_zx = self.rtd_min(z_dist, x_dist, immovable=2) + self.rtd_max(
                z_dist, x_dist, immovable=2
            )
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += torch.sum(torch.abs(rtd_xz[d][:, 1] - rtd_xz[d][:, 0]) ** self.p)
            if self.is_sym:
                loss_zx += torch.sum(
                    torch.abs(rtd_zx[d][:, 1] - rtd_zx[d][:, 0]) ** self.p
                )
        loss = (loss_xz + loss_zx) / 2.0
        return loss


def compute_metrics_consolidated(
    input_points: torch.Tensor,
    output_points: torch.Tensor,
    metrics: Union[List[str], Set[str]] = None,
    tloss: Optional[nn.Module] = None,
    rloss: Optional[nn.Module] = None,
    k_min: int = 5,
    k_max: int = 10,
    k_jac: int = 5,
    p: float = 2.0,
) -> Dict[str, float]:
    """
    Efficiently compute multiple metrics that depend on distance matrices by
    computing the distance matrices only once and reusing them.

    Args:
        input_points: Original point cloud tensor of shape (n_points, dim_in)
        output_points: Transformed point cloud tensor of shape (n_points, dim_out)
        metrics: List or set of metrics to compute. If None, computes all available metrics.
            Supported metrics: 'mle', 'beta0', 'lip', 'topoae', 'rtd', 'ljd', 'mst_error'
        tloss: TopoAE loss function instance (required if 'topoae' in metrics)
        rloss: RTD loss function instance (required if 'rtd' in metrics)
        k_min: Minimum k value for k-NN based metrics (e.g., MLE)
        k_max: Maximum k value for k-NN based metrics (e.g., MLE)
        k_jac: k value for k-NN used in Jacobian linearization error calculation
        p: Distance norm (e.g., p=2 for Euclidean)

    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
    """
    # Set default metrics if None provided
    all_metrics = {"mle", "beta0", "lip", "topoae", "rtd", "ljd", "mst_error"}
    if metrics is None:
        metrics = all_metrics
    elif isinstance(metrics, list):
        metrics = set(metrics)

    # Check for required loss functions
    if "topoae" in metrics and tloss is None:
        raise ValueError(
            "TopoAE loss instance (tloss) must be provided to compute 'topoae' metric"
        )
    if "rtd" in metrics and rloss is None:
        raise ValueError(
            "RTD loss instance (rloss) must be provided to compute 'rtd' metric"
        )

    # Initialize results dictionary
    results = {}

    # Get device
    device = input_points.device
    n_points = input_points.shape[0]

    # Compute distance matrices once for reuse
    dists_input = torch.cdist(input_points, input_points, p=p)
    dists_output = torch.cdist(output_points, output_points, p=p)

    # Get sorted indices for k-NN
    _, input_indices = torch.sort(dists_input, dim=1)
    _, output_indices = torch.sort(dists_output, dim=1)

    # Variables to store MST edges if they're computed
    mst_pairs_input = None
    mst_pairs_output = None
    mst_edge_weights_input = None
    mst_edge_weights_output = None

    # Compute MLE intrinsic dimension if requested
    if "mle" in metrics:
        # Calculate MLE for input data
        if k_max >= n_points:
            k_max = n_points - 1

        # Remove self-distances (first column) and take only up to k_max
        input_knn_dists = dists_input.gather(1, input_indices[:, 1 : k_max + 1])
        output_knn_dists = dists_output.gather(1, output_indices[:, 1 : k_max + 1])

        # Compute log distances
        input_log_dists = torch.log(input_knn_dists)
        output_log_dists = torch.log(output_knn_dists)

        # Compute MLE estimates for all k values from k_min to k_max
        num_k_vals = k_max - k_min + 1
        k_vals = torch.arange(k_min, k_max + 1, device=device)
        val = 2  # For unbiased estimation

        # MLE calculation for input
        input_mle = (k_vals[None, :] - val) / (
            (k_vals - 1) * input_log_dists[:, -num_k_vals:]
            - torch.cumsum(input_log_dists[:, :-1], dim=1)[:, -num_k_vals:]
        )

        # MLE calculation for output
        output_mle = (k_vals[None, :] - val) / (
            (k_vals - 1) * output_log_dists[:, -num_k_vals:]
            - torch.cumsum(output_log_dists[:, :-1], dim=1)[:, -num_k_vals:]
        )

        results["MLE_original"] = input_mle.mean().item()
        results["MLE_transformed"] = output_mle.mean().item()

    # Compute beta0 (connected components) or MST error if requested
    if "beta0" in metrics or "mst_error" in metrics:
        # Compute MST for input if not already computed
        if mst_pairs_input is None:
            mst_pairs_input = get_mst_edges(dists_input, one_root=True)
            mst_edge_weights_input = dists_input[
                mst_pairs_input[:, 0], mst_pairs_input[:, 1]
            ]

        # Compute MST for output if not already computed
        if mst_pairs_output is None:
            mst_pairs_output = get_mst_edges(dists_output, one_root=True)
            mst_edge_weights_output = dists_output[
                mst_pairs_output[:, 0], mst_pairs_output[:, 1]
            ]

    if "beta0" in metrics:
        # Use MLE estimates from above if available, otherwise use ambient dimension
        dim_input = (
            results.get("MLE_original")
            if "mle" in metrics
            else float(input_points.shape[1])
        )
        dim_output = (
            results.get("MLE_transformed")
            if "mle" in metrics
            else float(output_points.shape[1])
        )

        # Input space beta0 calculation
        # Estimate Poisson intensity parameter for input
        lambda_hat_input = estimate_poisson_intensity(dists_input, dim_input, k_max)

        # Compute volume of unit ball in input dimension in log space
        log_omega_d_input = (dim_input / 2) * np.log(np.pi) - special.gammaln(
            dim_input / 2 + 1
        )
        omega_d_input = np.exp(log_omega_d_input)

        # Compute expected 1-NN distance in log space
        log_gamma_term_input = special.gammaln(1 + 1 / dim_input)
        gamma_term_input = np.exp(log_gamma_term_input)
        r_expected_input = gamma_term_input / (lambda_hat_input * omega_d_input) ** (
            1 / dim_input
        )

        # Set connectivity threshold with scaling factor (alpha=3.0)
        connectivity_threshold_input = 3.0 * r_expected_input

        # B₀ is one plus the number of MST edges longer than 2*threshold
        edge_count_input = torch.sum(
            mst_edge_weights_input > 2 * connectivity_threshold_input
        ).item()
        beta_zero_input = int(edge_count_input + 1)

        # Output space beta0 calculation
        # Estimate Poisson intensity parameter for output
        lambda_hat_output = estimate_poisson_intensity(dists_output, dim_output, k_max)

        # Compute volume of unit ball in output dimension in log space
        log_omega_d_output = (dim_output / 2) * np.log(np.pi) - special.gammaln(
            dim_output / 2 + 1
        )
        omega_d_output = np.exp(log_omega_d_output)

        # Compute expected 1-NN distance in log space
        log_gamma_term_output = special.gammaln(1 + 1 / dim_output)
        gamma_term_output = np.exp(log_gamma_term_output)
        r_expected_output = gamma_term_output / (
            lambda_hat_output * omega_d_output
        ) ** (1 / dim_output)

        # Set connectivity threshold with scaling factor (alpha=3.0)
        connectivity_threshold_output = 3.0 * r_expected_output

        # B₀ is one plus the number of MST edges longer than 2*threshold
        edge_count_output = torch.sum(
            mst_edge_weights_output > 2 * connectivity_threshold_output
        ).item()
        beta_zero_output = int(edge_count_output + 1)

        # Store results
        results["B0_original"] = beta_zero_input
        results["B0_transformed"] = beta_zero_output
        results["B0_delta_original"] = connectivity_threshold_input
        results["B0_delta_transformed"] = connectivity_threshold_output
        results["B0_lambda_original"] = lambda_hat_input
        results["B0_lambda_transformed"] = lambda_hat_output

    # Compute MST Error if requested
    if "mst_error" in metrics:
        # Get distances for MST edges in each space
        input_mst_dists = mst_edge_weights_input
        output_mst_dists = mst_edge_weights_output

        # Get corresponding distances in the other space
        output_dists_for_input_mst = dists_output[
            mst_pairs_input[:, 0], mst_pairs_input[:, 1]
        ]
        input_dists_for_output_mst = dists_input[
            mst_pairs_output[:, 0], mst_pairs_output[:, 1]
        ]

        rmse_in_mst_in = input_mst_dists / math.sqrt(dim_input)
        rmse_in_mst_ot = input_dists_for_output_mst / math.sqrt(dim_input)
        rmse_ot_mst_in = output_dists_for_input_mst / math.sqrt(dim_output)
        rmse_ot_mst_ot = output_mst_dists / math.sqrt(dim_output)

        sse_rmse_in = (rmse_in_mst_in - rmse_ot_mst_in) ** 2
        sse_rmse_ot = (rmse_ot_mst_ot - rmse_in_mst_ot) ** 2

        sse_rmse_in = sse_rmse_in / (rmse_in_mst_in**2 + 1e-8)
        sse_rmse_ot = sse_rmse_ot / (rmse_ot_mst_ot**2 + 1e-8)

        mst_error = 0.5 * (sse_rmse_in + sse_rmse_ot)

        results["MST_Error"] = mst_error.mean().item()

    # Compute Lipschitz constants if requested
    if "lip" in metrics:
        # Ensure input and output have proper shapes
        input_dim = input_points.shape[1]
        output_dim = output_points.shape[1]

        # Small value to avoid division by zero
        eps = 1e-8

        # Compute Lipschitz ratio and mask diagonal (i=j)
        lip_fwd = dists_output / (dists_input + eps)  # (B, B)
        lip_fwd_mask = dists_input > eps  # (B, B)
        lip_fwd *= lip_fwd_mask  # Apply mask to zero out diagonal entries
        max_fwd_ratios = torch.max(lip_fwd, dim=1).values * (input_dim / output_dim)

        lip_inv = dists_input / (dists_output + eps)  # (B, B)
        lip_inv_mask = dists_output > eps  # (B, B)
        lip_inv *= lip_inv_mask  # Apply mask to zero out diagonal entries
        max_inv_ratios = torch.max(lip_inv, dim=1).values * (output_dim / input_dim)

        results["Forward_Lipschitz"] = max_fwd_ratios.max().item()
        results["Inverse_Lipschitz"] = max_inv_ratios.max().item()

    # Compute TopoAE loss if requested
    if "topoae" in metrics:
        topoae_loss = tloss(dists_input, dists_output).item()
        results["TopoAE"] = topoae_loss

    # Compute RTD loss if requested
    if "rtd" in metrics:
        rtd_loss = rloss(dists_input, dists_output).item()
        results["RTD"] = rtd_loss

    # Compute Local Jacobian Discrepancy (LJD) if requested
    if "ljd" in metrics:
        # Make sure k_jac is not larger than the number of points
        if k_jac >= n_points:
            k_jac = n_points - 1

        # Get k-NN indices for Jacobian calculation (excluding self)
        domain_indices = input_indices[:, 1 : k_jac + 1]
        codomain_indices = output_indices[:, 1 : k_jac + 1]

        # Calculate Jacobian linearization errors
        domain_errors, codomain_errors = estimate_jacobian_delta_batched(
            input_points,
            output_points,
            domain_indices,
            codomain_indices,
            batch_size=128,
            rcond=1e-5,
            p=p,
            pca_project=(input_points.shape[1] != output_points.shape[1]),
        )

        # Store the average errors as our LJD metrics
        results["LJD_domain"] = domain_errors.mean().item()
        results["LJD_codomain"] = codomain_errors.mean().item()
        results["LJD_combined"] = 0.5 * (
            domain_errors.mean().item() + codomain_errors.mean().item()
        )

    return results


def estimate_jacobian_delta_batched(
    domain_points: Tensor,
    codomain_points: Tensor,
    domain_indices: Tensor,
    codomain_indices: Tensor,
    batch_size: int = 128,
    rcond: float = 1e-5,
    p: float = 2.0,
    pca_project: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Jacobian linearization error using batched operations to prevent OOM errors.

    Parameters:
    -----------
    domain_points : Tensor
        Domain points, shape (n, d1)
    codomain_points : Tensor
        Codomain points, shape (n, d2)
    domain_indices : Tensor
        Indices of k nearest neighbors in domain, shape (n, k)
    codomain_indices : Tensor
        Indices of k nearest neighbors in codomain, shape (n, k)
    batch_size : int
        Size of batches to process at once, default 128
    rcond : float
        Cutoff for small singular values in pseudoinverse calculation
    p : float
        Order of the norm for error calculation
    pca_project : bool
        Whether to project high-dimensional codomain to match domain dimensions

    Returns:
    --------
    domain_errors : Tensor
        Domain linearization errors, shape (n,)
    codomain_errors : Tensor
        Codomain linearization errors, shape (n,)
    """
    n, d1 = domain_points.shape
    d2 = codomain_points.shape[1]
    k = domain_indices.shape[1]
    device = domain_points.device

    # Initialize result tensors
    domain_errors = torch.zeros(n, device=device)
    codomain_errors = torch.zeros(n, device=device)

    # Process in batches to avoid OOM
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batch_slice = slice(i, end_idx)

        # Get batch of neighborhood points using advanced indexing
        batch_domain_indices = domain_indices[batch_slice]
        batch_codomain_indices = codomain_indices[batch_slice]

        # Shape: (batch_size, k, d1) and (batch_size, k, d2)
        domain_nbrs = domain_points[batch_domain_indices]
        domain_nbr_outs = codomain_points[batch_domain_indices]

        codomain_nbrs = codomain_points[batch_codomain_indices]
        codomain_nbr_ins = domain_points[batch_codomain_indices]

        # Center neighborhoods
        # Shape: (batch_size, 1, d1) and (batch_size, 1, d2)
        center_d = domain_points[batch_slice].unsqueeze(1)
        centers_c = codomain_points[batch_slice].unsqueeze(1)

        X_d = domain_nbrs - center_d
        Y_d = domain_nbr_outs - centers_c
        Y_c = codomain_nbrs - centers_c
        X_c = codomain_nbr_ins - center_d

        if pca_project:
            # PCA for domain-based neighborhood codomain points
            U_d, S_d, Vh_d = torch.linalg.svd(Y_d.transpose(1, 2), full_matrices=False)

            # PCA for codomain-based neighborhood codomain points
            U_c, S_c, Vh_c = torch.linalg.svd(Y_c.transpose(1, 2), full_matrices=False)

            # grab the top d1 components for domain and codomain
            Y_d_proj = torch.matmul(Y_d, U_d[:, :, :d1])  # (batch_size, k, d1)
            Y_c_proj = torch.matmul(Y_c, U_c[:, :, :d1])  # (batch_size, k, d1)
        else:
            Y_d_proj = Y_d
            Y_c_proj = Y_c

        # Use the projected data for Jacobian calculations
        # Forward Jacobian in domain neighborhood (maps X_d -> Y_d_proj)
        J_forward_d = torch.linalg.pinv(X_d, rcond=rcond) @ Y_d_proj

        # # Inverse Jacobian in domain neighborhood (maps Y_d_proj -> X_d)
        J_inverse_d = torch.linalg.pinv(Y_d_proj, rcond=rcond) @ X_d

        # Forward Jacobian in codomain neighborhood (maps X_c -> Y_c_proj)
        J_forward_c = torch.linalg.pinv(X_c, rcond=rcond) @ Y_c_proj

        # # Inverse Jacobian in codomain neighborhood (maps Y_c_proj -> X_c)
        J_inverse_c = torch.linalg.pinv(Y_c_proj, rcond=rcond) @ X_c

        # Compute predictions using other neighborhoods' Jacobian estimates
        X_d_pred = torch.matmul(Y_d_proj, J_inverse_c)
        X_c_pred = torch.matmul(Y_c_proj, J_inverse_d)  # (batch_size, k, d1)
        Y_d_pred = torch.matmul(X_d, J_forward_c)
        Y_c_pred = torch.matmul(X_c, J_forward_d)  # (batch_size, k, d2)

        # Compute scales for normalization (batch_size,)
        scale_X_d = torch.norm(X_d, dim=-1, p=p).clamp(min=1e-10).mean(dim=-1)
        scale_X_c = torch.norm(X_c, dim=-1, p=p).clamp(min=1e-10).mean(dim=-1)
        scale_Y_d = torch.norm(Y_d_proj, dim=-1, p=p).clamp(min=1e-10).mean(dim=-1)
        scale_Y_c = torch.norm(Y_c_proj, dim=-1, p=p).clamp(min=1e-10).mean(dim=-1)

        errors_x_d = torch.norm(X_d - X_d_pred, dim=-1, p=p).mean(dim=-1) / scale_X_d
        errors_x_c = torch.norm(X_c - X_c_pred, dim=-1, p=p).mean(dim=-1) / scale_X_c
        errors_y_d = (
            torch.norm(Y_d_proj - Y_d_pred, dim=-1, p=p).mean(dim=-1) / scale_Y_d
        )
        errors_y_c = (
            torch.norm(Y_c_proj - Y_c_pred, dim=-1, p=p).mean(dim=-1) / scale_Y_c
        )

        errors_d = errors_x_d + errors_y_d
        errors_c = errors_x_c + errors_y_c

        # Store results
        domain_errors[batch_slice] = errors_d
        codomain_errors[batch_slice] = errors_c

    return domain_errors, codomain_errors
