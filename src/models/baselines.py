import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    """Baseline MLP that directly learns dynamics"""
    def __init__(self, input_dim=2, hidden_dim=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)  # Output is derivative
        )

    def forward(self, x):
        return self.net(x)

class HNN(nn.Module):
    """Hamiltonian Neural Network"""
    def __init__(self, input_dim=2, hidden_dim=64):
        super(HNN, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output is scalar Hamiltonian
        )

    def forward(self, x):
        return self.net(x)

    def get_dynamics(self, x):
        """Compute dynamics using automatic differentiation"""
        # Enable gradient computation for this specific operation
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.forward(x)

            # Compute gradients
            grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

            # Hamiltonian dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            q_dim = self.input_dim // 2
            dq_dt = grad_H[:, q_dim:]   # ∂H/∂p
            dp_dt = -grad_H[:, :q_dim]  # -∂H/∂q

            return torch.cat([dq_dt, dp_dt], dim=1)

class DoubleHeadHNN(nn.Module):
    """DoubleHeadHNN: Separate networks for kinetic (T) and potential (V) energy"""
    def __init__(self, input_dim=2, hidden_dim=64):
        super(DoubleHeadHNN, self).__init__()
        self.q_dim = input_dim // 2
        self.p_dim = input_dim // 2

        # Network for potential energy V(q)
        self.v_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Network for INVERSE inertia matrix M^{-1}(q)
        # Output size: q_dim x q_dim matrix (flattened)
        self.m_inv_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.q_dim * self.q_dim)
        )

    def forward(self, x):
        """Compute Hamiltonian H = T + V"""
        q = x[:, :self.q_dim]
        p = x[:, self.q_dim:]

        # Potential energy
        V = self.v_net(q)

        # Kinetic energy: T = 0.5 * p^T * M^{-1}(q) * p
        M_inv_flat = self.m_inv_net(q)
        M_inv = M_inv_flat.view(-1, self.q_dim, self.q_dim)

        # Make M_inv symmetric
        M_inv = 0.5 * (M_inv + M_inv.transpose(-1, -2))

        # Add small diagonal term for numerical stability (positive definiteness)
        eye = torch.eye(self.q_dim, device=M_inv.device).unsqueeze(0)
        M_inv = M_inv + 1e-3 * eye

        # Compute M^{-1} * p
        p_expanded = p.unsqueeze(-1)
        M_inv_p = torch.bmm(M_inv, p_expanded).squeeze(-1)

        # Kinetic energy
        T = 0.5 * torch.sum(p * M_inv_p, dim=1, keepdim=True)

        return T + V

    def get_dynamics(self, x):
        """Compute dynamics using automatic differentiation"""
        # Enable gradient computation for this specific operation
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.forward(x)

            # Compute gradients
            grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

            # Split gradients for q and p dimensions
            q_dim = self.q_dim
            dH_dq = grad_H[:, :q_dim]
            dH_dp = grad_H[:, q_dim:]

            # Hamiltonian dynamics
            dq_dt = dH_dp
            dp_dt = -dH_dq

            return torch.cat([dq_dt, dp_dt], dim=1)

class CholeskyHNN(nn.Module):
    """HNN with Cholesky decomposition to ensure SPD inertia matrix"""
    def __init__(self, input_dim=2, hidden_dim=64):
        super(CholeskyHNN, self).__init__()
        self.q_dim = input_dim // 2
        self.p_dim = input_dim // 2

        # Network for potential energy V(q)
        self.v_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Network for Cholesky factor L(q)
        # Output size: lower triangular entries of L
        self.l_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.q_dim * (self.q_dim + 1) // 2)
        )

    def _construct_lower_triangular(self, l_params):
        """Construct lower triangular matrix from parameters"""
        batch_size = l_params.shape[0]
        L = torch.zeros(batch_size, self.q_dim, self.q_dim, device=l_params.device)

        # Fill lower triangular part
        idx = 0
        for i in range(self.q_dim):
            for j in range(i + 1):
                if i == j:
                    # Ensure positive diagonal elements
                    L[:, i, j] = torch.exp(l_params[:, idx])
                else:
                    L[:, i, j] = l_params[:, idx]
                idx += 1

        return L

    def forward(self, x):
        """Compute Hamiltonian H = T + V"""
        q = x[:, :self.q_dim]
        p = x[:, self.q_dim:]

        # Potential energy
        V = self.v_net(q)

        # Kinetic energy: T = 0.5 * p^T * M^{-1}(q) * p
        # where M = L * L^T (Cholesky decomposition)
        l_params = self.l_net(q)
        L = self._construct_lower_triangular(l_params)

        # Solve L * y = p for y
        p_expanded = p.unsqueeze(-1)
        y = torch.linalg.solve_triangular(L, p_expanded, upper=False).squeeze(-1)

        # Solve L^T * z = y for z = M^{-1} * p
        z = torch.linalg.solve_triangular(L.transpose(-1, -2), y.unsqueeze(-1), upper=True).squeeze(-1)

        # Kinetic energy
        T = 0.5 * torch.sum(p * z, dim=1, keepdim=True)

        return T + V

    def get_dynamics(self, x):
        """Compute dynamics using automatic differentiation"""
        # Enable gradient computation for this specific operation
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.forward(x)

            # Compute gradients
            grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

            # Split gradients for q and p dimensions
            q_dim = self.q_dim
            dH_dq = grad_H[:, :q_dim]
            dH_dp = grad_H[:, q_dim:]

            # Hamiltonian dynamics
            dq_dt = dH_dp
            dp_dt = -dH_dq

            return torch.cat([dq_dt, dp_dt], dim=1)