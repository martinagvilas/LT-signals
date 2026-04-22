"""Cross-layer trajectory signals used as baselines in Section 4.1 of the paper.

Following Wang et al. (2024), we compute the mean magnitude (LayerMag) and angle
(LayerAng) of layer-to-layer changes for each reasoning segment and then average
over segments to obtain a single per-sample score.
"""

import torch
import torch.nn.functional as F


def compute_l2_norm(vec):
    return torch.norm(vec, p=2, dim=-1)


def compute_magnitude_change(hs, dim):
    """L2 norm of differences between adjacent points along `dim`."""
    diff = torch.diff(hs, dim=dim)
    return compute_l2_norm(diff)


def compute_angle_change(hs, dim):
    """Arc-cosine angle between adjacent points along `dim`."""
    slices1 = [slice(None)] * hs.dim()
    slices2 = [slice(None)] * hs.dim()
    slices1[dim] = slice(0, -1)
    slices2[dim] = slice(1, None)

    h1 = hs[tuple(slices1)]
    h2 = hs[tuple(slices2)]

    cos_sim = torch.sum(h1 * h2, dim=-1) / (compute_l2_norm(h1) * compute_l2_norm(h2) + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    return torch.acos(cos_sim)


def compute_trajectory(hs, traj_dim):
    """Compute magnitude and angle trajectory signals along `traj_dim`.

    Args:
        hs:       Tensor of shape [n_layers, n_segments, hidden_dim]
        traj_dim: 0 for cross-layer signals (LayerMag / LayerAng baselines),
                  1 for cross-segment signals.

    Returns:
        dict with keys '<dim>_mag', '<dim>_ang', and their raw change tensors.
    """
    hs = hs.float()

    mag_changes = compute_magnitude_change(hs, traj_dim)
    ang_changes = compute_angle_change(hs, traj_dim)

    slices_first = [slice(None)] * hs.dim()
    slices_last = [slice(None)] * hs.dim()
    slices_first[traj_dim] = 0
    slices_last[traj_dim] = -1
    first_point = hs[tuple(slices_first)]
    last_point = hs[tuple(slices_last)]

    overall_mag = compute_l2_norm(last_point - first_point)
    overall_cos = torch.sum(first_point * last_point, dim=-1) / (
        compute_l2_norm(first_point) * compute_l2_norm(last_point) + 1e-8
    )
    overall_ang = torch.acos(torch.clamp(overall_cos, -1.0, 1.0))

    # Normalise step-wise changes by overall change
    overall_mag = overall_mag.unsqueeze(traj_dim)
    overall_ang = overall_ang.unsqueeze(traj_dim)
    mag_norm = mag_changes / (overall_mag + 1e-8)
    ang_norm = ang_changes / (overall_ang + 1e-8)

    dim_name = "layerwise" if traj_dim == 0 else "seqwise"
    return {
        f"{dim_name}_mag": mag_norm.mean(dim=traj_dim).cpu().float(),
        f"{dim_name}_ang": ang_norm.mean(dim=traj_dim).cpu().float(),
        f"{dim_name}_mag_changes": mag_changes.cpu().float(),
        f"{dim_name}_ang_changes": ang_changes.cpu().float(),
        f"{dim_name}_mag_overall_change": overall_mag.cpu().float(),
        f"{dim_name}_ang_overall_change": overall_ang.cpu().float(),
    }
