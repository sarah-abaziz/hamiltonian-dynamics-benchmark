import torch
from torch.autograd.functional import jacobian

def trajectory_errors(x_true, x_pred, eps=1e-8):
    """
    Compute absolute and relative trajectory errors for q and p.

    Args:
        x_true: [T, 2n] ground truth states
        x_pred: [T, 2n] predicted states
        eps: small number to avoid divide-by-zero

    Returns:
        dict with abs/rel errors for q and p
    """
    n = x_true.shape[1] // 2
    q_true, p_true = x_true[:, :n], x_true[:, n:]
    q_pred, p_pred = x_pred[:, :n], x_pred[:, n:]

    abs_err_q = torch.norm(q_pred - q_true, dim=1)
    abs_err_p = torch.norm(p_pred - p_true, dim=1)

    rel_err_q = abs_err_q / (torch.norm(q_true, dim=1) + eps)
    rel_err_p = abs_err_p / (torch.norm(p_true, dim=1) + eps)

    return {
        "abs_err_q": abs_err_q,
        "abs_err_p": abs_err_p,
        "rel_err_q": rel_err_q,
        "rel_err_p": rel_err_p,
        "mean_abs_err_q": abs_err_q.mean(),
        "mean_abs_err_p": abs_err_p.mean(),
        "mean_rel_err_q": rel_err_q.mean(),
        "mean_rel_err_p": rel_err_p.mean()
    }


def energy_errors(H_func, x_traj):
    """
    Compute energy conservation errors along a trajectory.

    Args:
        H_func: callable H(x) -> scalar
        x_traj: [T, 2n] trajectory

    Returns:
        dict with absolute and relative energy errors
    """
    H0 = H_func(x_traj[0])
    H_vals = torch.tensor([H_func(x) for x in x_traj])

    abs_err = torch.abs(H_vals - H0)
    rel_err = abs_err / (torch.abs(H0) + 1e-8)

    return {
        "abs_err": abs_err,
        "rel_err": rel_err,
        "max_abs_err": abs_err.max(),
        "max_rel_err": rel_err.max(),
        "mean_abs_err": abs_err.mean(),
        "mean_rel_err": rel_err.mean()
    }
    
def reconstruction_error(x, x_hat, eps=1e-8):
    """
    Compute absolute and relative reconstruction error for an autoencoder.

    Args:
        x: [T, 2n] original states
        x_hat: [T, 2n] reconstructed states
        eps: small number to avoid divide-by-zero

    Returns:
        dict with abs and relative errors
    """
    abs_err = torch.norm(x - x_hat, dim=1)
    rel_err = abs_err / (torch.norm(x, dim=1) + eps)

    return {
        "abs_err": abs_err,
        "rel_err": rel_err,
        "mean_abs_err": abs_err.mean(),
        "mean_rel_err": rel_err.mean(),
        "max_abs_err": abs_err.max(),
        "max_rel_err": rel_err.max()
    }


def ae_symplecticity_error(autoencoder_map, x_traj):
    """
    Check approximate symplecticity of the autoencoder mapping x -> x_hat
    Args:
        autoencoder_map: callable x -> x_hat (encoder + decoder)
        x_traj: [T, 2n] trajectory of original states
    Returns:
        dict with max and mean symplectic deviation across trajectory
    """
    n = x_traj.shape[1] // 2
    device = x_traj.device
    dtype = x_traj.dtype

    # Symplectic form Ω
    Omega = torch.zeros(2*n, 2*n, dtype=dtype, device=device)
    Omega[:n, n:] = torch.eye(n, dtype=dtype, device=device)
    Omega[n:, :n] = -torch.eye(n, dtype=dtype, device=device)

    symp_errors = []

    for x in x_traj:
        x = x.detach().requires_grad_(True)
        x_flat = x.view(-1)  # ensure 1D
        J = jacobian(autoencoder_map, x_flat)  # [2n,2n]
        symp_err = (J.T @ Omega @ J - Omega).abs().max().item()
        symp_errors.append(symp_err)

    symp_errors = torch.tensor(symp_errors)
    return {
        "max_symplectic_error": symp_errors.max().item(),
        "mean_symplectic_error": symp_errors.mean().item()
    }