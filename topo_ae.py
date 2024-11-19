"""
TopoAE Loss Implementation

This module implements the topology-preserving loss term from the TopoAE paper,
focusing on the core distance matching calculation between input and latent spaces.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union


class UnionFind:
    """
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    """

    def __init__(self, n_vertices):
        """
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        """

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        """
        Finds and returns the parent of u with respect to the hierarchy.
        """

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        """
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        """

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        """
        Generator expression for returning roots, i.e. components that
        are their own parents.
        """

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    """Computes persistent homology pairing using Union-Find data structure."""

    def __call__(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate persistent homology pairs from a distance matrix.

        Args:
            matrix: Distance matrix of shape (n_vertices, n_vertices)

        Returns:
            Tuple of (0-dim pairs, 1-dim pairs)
        """
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        # Get upper triangular indices and corresponding distances
        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind="stable")

        persistence_pairs = []
        for edge_index, _ in zip(edge_indices, edge_weights[edge_indices]):
            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            persistence_pairs.append((min(u, v), max(u, v)))

        return np.array(persistence_pairs), np.array([])


class TopoAELoss(nn.Module):
    """Computes the TopoAE topology-preserving loss between input and latent spaces."""

    def __init__(self, match_edges: str = None):
        """
        Initialize TopoAE loss module.

        Args:
            match_edges: Method for matching edges. Options: None, 'symmetric', 'random'
        """
        super().__init__()
        self.match_edges = match_edges
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Get persistent homology pairings from distance matrix."""
        return self.signature_calculator(distances.detach().cpu().numpy())

    def _select_distances_from_pairs(
        self, distance_matrix: torch.Tensor, pairs: Tuple[np.ndarray, np.ndarray]
    ) -> torch.Tensor:
        """Select distances corresponding to persistence pairs."""
        pairs_0, _ = pairs
        return distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

    @staticmethod
    def _count_matching_pairs(pairs1: np.ndarray, pairs2: np.ndarray) -> float:
        """Count number of matching pairs between two sets of pairs."""
        to_set = lambda arr: set(tuple(elements) for elements in arr)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    def forward(
        self, input_distances: torch.Tensor, latent_distances: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
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

        distance_components = {
            "matched_pairs": self._count_matching_pairs(pairs1[0], pairs2[0])
        }

        if self.match_edges is None:
            # Standard distance calculation
            sig1 = self._select_distances_from_pairs(input_distances, pairs1)
            sig2 = self._select_distances_from_pairs(latent_distances, pairs2)
            distance = ((sig1 - sig2) ** 2).sum(dim=-1)

        elif self.match_edges == "symmetric":
            # Symmetric matching of edges
            sig1 = self._select_distances_from_pairs(input_distances, pairs1)
            sig2 = self._select_distances_from_pairs(latent_distances, pairs2)
            sig1_2 = self._select_distances_from_pairs(latent_distances, pairs1)
            sig2_1 = self._select_distances_from_pairs(input_distances, pairs2)

            distance1_2 = ((sig1 - sig1_2) ** 2).sum(dim=-1)
            distance2_1 = ((sig2 - sig2_1) ** 2).sum(dim=-1)

            distance_components.update(
                {"distance1-2": distance1_2, "distance2-1": distance2_1}
            )
            distance = distance1_2 + distance2_1

        elif self.match_edges == "random":
            # Random matching for ablation studies
            n_instances = len(pairs1[0])
            random_pairs1 = torch.cat(
                [
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None],
                ],
                dim=1,
            )
            random_pairs2 = torch.cat(
                [
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None],
                ],
                dim=1,
            )

            sig1_1 = self._select_distances_from_pairs(
                input_distances, (random_pairs1, None)
            )
            sig1_2 = self._select_distances_from_pairs(
                latent_distances, (random_pairs1, None)
            )
            sig2_2 = self._select_distances_from_pairs(
                latent_distances, (random_pairs2, None)
            )
            sig2_1 = self._select_distances_from_pairs(
                input_distances, (random_pairs2, None)
            )

            distance1_2 = ((sig1_1 - sig1_2) ** 2).sum(dim=-1)
            distance2_1 = ((sig2_1 - sig2_2) ** 2).sum(dim=-1)

            distance_components.update(
                {"distance1-2": distance1_2, "distance2-1": distance2_1}
            )
            distance = distance1_2 + distance2_1

        return distance / input_distances.shape[0], distance_components


# Example usage:
def compute_topoae_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute TopoAE loss between input x and latent representation z.

    Args:
        x: Input data tensor
        z: Latent representation tensor

    Returns:
        TopoAE loss value
    """
    # Compute pairwise distances
    x_distances = torch.cdist(x, x, p=2)
    z_distances = torch.cdist(z, z, p=2)

    # Normalize distances
    x_distances = x_distances / x_distances.max()
    z_distances = z_distances / torch.nn.Parameter(torch.ones(1))

    # Compute loss
    criterion = TopoAELoss(match_edges="symmetric")
    loss, _ = criterion(x_distances, z_distances)

    return loss
