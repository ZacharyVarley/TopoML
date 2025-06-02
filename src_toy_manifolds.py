"""
Generate toy manifolds for testing homeomorphisms.

"""

import torch
from torch import Tensor
import math


def swiss_roll(p: Tensor, t=1.0) -> Tensor:
    u = (p[:, 0] + 1) / 2.0
    x = (1 - t) * u + t * ((1 + u) * torch.cos(3 * torch.pi * t * u))
    z = t * ((1 + u) * torch.sin(3 * torch.pi * t * u))
    return torch.stack((0.5 * x, p[:, 1], 0.5 * z), dim=1)


def waveform(p: Tensor, t=1.0) -> Tensor:
    x, y = p[:, 0], p[:, 1]
    z_new = t * (torch.sin(2 * torch.pi * x) + torch.cos(2 * torch.pi * y)) * 0.5
    return torch.stack((x, y, z_new), dim=1)


import torch
from torch import Tensor
import math


def dog_ear(p: Tensor, t=1.0) -> Tensor:
    """
    Creates a 'dog ear' fold where the top-right flap rotates around a line passing
    through the point (0.5, 0.5) with slope -1. At t=1, the flap will have rotated
    by π radians, creating a non-injective mapping.

    Args:
        p: Tensor of shape (n, 2) containing points in the [-1,1]^2 square
        t: Transformation parameter in [0,1], with t=0 being identity and t=1 full rotation

    Returns:
        Transformed points in R^3
    """
    x, y = p[:, 0], p[:, 1]

    # Only affect points past the line y = -x + 1
    mask = y > -x + 1

    # Create copy of the input coordinates
    x_new = x.clone()
    y_new = y.clone()
    z_new = torch.zeros_like(x)

    if torch.any(mask):
        # Center of rotation is (0.5, 0.5)
        px, py = x[mask], y[mask]
        dx, dy = px - 0.5, py - 0.5

        # Rotate around axis (1, -1, 0) by angle θ = π * t
        theta = math.pi * t
        axis = torch.tensor([1.0, -1.0, 0.0], device=p.device)
        axis = axis / torch.norm(axis)

        # Assemble vectors in 3D for rotation
        points_3d = torch.stack((dx, dy, torch.zeros_like(dx)), dim=1)

        # Rodrigues' rotation formula
        k = axis
        cos_theta = torch.cos(torch.tensor(theta, device=p.device))
        sin_theta = torch.sin(torch.tensor(theta, device=p.device))

        rotated = (
            points_3d * cos_theta
            + torch.cross(k.expand_as(points_3d), points_3d, dim=1) * sin_theta
            + k * torch.sum(points_3d * k, dim=1, keepdim=True) * (1 - cos_theta)
        )

        # Translate back from center
        x_final = rotated[:, 0] + 0.5
        y_final = rotated[:, 1] + 0.5
        z_final = rotated[:, 2]

        x_new[mask] = x_final
        y_new[mask] = y_final
        z_new[mask] = z_final

    return torch.stack((x_new, y_new, z_new), dim=1)


def ribbon(p: Tensor, t=1.0) -> Tensor:
    y_prime = t * (0.75 * torch.pi) * (p[:, 1] + 1.0) + torch.pi / 4
    curl_factor = math.sin(t * torch.pi / 2)
    x = p[:, 0]
    y = t * p[:, 1] * torch.cos(y_prime * curl_factor) + (1 - t) * p[:, 1]
    z = p[:, 1] * torch.sin(y_prime * curl_factor) + 0.5 * t**2
    return torch.stack((x, y, z), dim=1)


def cylinder(p: Tensor, t=1.0) -> Tensor:
    x, y = p[:, 0], p[:, 1]
    theta = x * torch.pi
    points_new = torch.stack((torch.cos(theta), torch.sin(theta), y), dim=1)
    points_old = torch.stack((x, y, torch.zeros_like(x)), dim=1)
    return (1 - t) * points_old + t * points_new


def hole(p: Tensor, t=1.0) -> Tensor:
    x = p[:, 0]
    y = p[:, 1]
    r = torch.sqrt(x**2 + y**2) + 1e-5  # Avoid division by zero
    dist_to_boundary_x = 1 - x.abs()
    dist_to_boundary_y = 1 - y.abs()
    unit_radial_x = x / r
    unit_radial_y = y / r
    k = 0.5  # Adjust for hole size
    dx = t * k * dist_to_boundary_x * unit_radial_x
    dy = t * k * dist_to_boundary_y * unit_radial_y
    x_new = x + dx
    y_new = y + dy
    return torch.stack((x_new, y_new, torch.zeros_like(x)), dim=1)


def split(p: Tensor, t=1.0) -> Tensor:
    x, y = p[:, 0], p[:, 1]
    y_new = torch.where(
        y >= 0, 0.5 * t + y * (1 - 0.5 * t), -0.5 * t + y * (1 - 0.5 * t)
    )
    return torch.stack((x, y_new, torch.zeros_like(x)), dim=1)


def pinch(p: Tensor, t=1.0) -> Tensor:
    x, y = p[:, 0], p[:, 1]
    return torch.stack((x, y * torch.abs(x) ** (2 * t), torch.zeros_like(x)), dim=1)


def collapse(p: Tensor, t=1.0) -> Tensor:
    return torch.stack((p[:, 0], (1 - t) * p[:, 1], torch.zeros_like(p[:, 0])), dim=1)
