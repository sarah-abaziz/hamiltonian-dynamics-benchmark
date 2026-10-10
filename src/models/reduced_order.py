"""ROGeoHNN. Code from Geometric_Hamiltonian_Neural_Network_Reduced_Order.ipynb."""
import enum
import os
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import geoopt
import geoopt as go
import geoopt.tensor as gt
from geoopt import linalg
from geoopt.manifolds import Stiefel, SymmetricPositiveDefinite
from geoopt.manifolds.base import Manifold
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.utils import size2shape

from torchdiffeq import odeint


# ============================================================================
# BIORTHOGONAL MANIFOLD
# ============================================================================
class BiOrthogonal(Manifold):
    """
    BiOrthogonal manifold:
        points are represented as a concatenation of (Phi, Psi)
        Phi: shape (..., n, r)
        Psi: shape (..., n, r)
    with the constraint Psi^T Phi = I_r.
    """

    ndim = 2
    name = 'biorthogonal'


    def __init__(self, n: int, r: int, device="cpu", dtype=torch.float32):
        super().__init__()
        if n < r:
            raise ValueError("Ambient dimension n must be >= latent dimension r.")
        self.n = n
        self.r = r
        self._I_r = torch.eye(r, device=device, dtype=dtype)

    def _split_representation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split concatenated BiOrthogonal representation x into (Phi, Psi).

        Assumes:
            x has shape (..., 2*n, r), where the last two dims are concatenated along -2:
            - Phi = x[..., :n, :]
            - Psi = x[..., n:, :]
        """
        if x.shape[-2] != 2 * self.n or x.shape[-1] != self.r:
            raise ValueError(
                f"Unexpected shape for BiOrthogonal representation: {x.shape}. "
                f"Expected (..., {2*self.n}, {self.r})."
            )
        Phi, Psi = torch.split(x, [self.n, self.n], dim=-2)
        return Phi, Psi

    def _concat_representation(self, Phi: torch.Tensor, Psi: torch.Tensor) -> torch.Tensor:
        """
        Concatenate Phi and Psi along -2 axis to form canonical BiOrthogonal representation.

        Input shapes:
            Phi: (..., n, r)
            Psi: (..., n, r)
        Output shape:
            (..., 2*n, r)
        """
        return torch.cat((Phi, Psi), dim=-2)


    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-6, rtol=1e-6):
        """
        Canonical check: return (ok, msg). We compute Psi^T Phi - I.
        """
        try:
            Phi, Psi = self._split_representation(x)
        except Exception as e:
            return False, f"Invalid representation shape: {e}"

        C = torch.matmul(Psi.transpose(-1, -2), Phi)  # (..., r, r)
        I = torch.eye(self.r, device=C.device, dtype=C.dtype).expand_as(C)
        ok = torch.allclose(C, I, atol=atol, rtol=rtol)
        if not ok:
            max_err = (C - I).abs().max().item()
            return False, f"Psi^T Phi != I (max abs error {max_err:.3e})"
        return True, None

    def random(self, *size, dtype=torch.float32, device=None) -> torch.Tensor:
        """
        Generate a random tensor on the manifold with approximately orthonormal columns.

        This function constructs a random tensor using Xavier (Glorot) normal initialization,
        then normalizes it to enforce an orthogonality-like property through
        matrix inversion and projection. The result is wrapped as a `ManifoldTensor`.

        Args:
            *size: Variable-length shape specification, interpreted by `size2shape`.
                  Typically represents dimensions (N, r) where N >= r.
            dtype (torch.dtype, optional): Data type of the tensor. Defaults to `torch.float32`.
            device (torch.device or str, optional): Device on which to allocate the tensor.
                                                    Defaults to current default device.

        Returns:
            ManifoldTensor: A manifold-aware tensor representing a random point on the manifold.
        """

        # --- Validate and unpack the shape ---
        shape = size2shape(*size)
        self._assert_check_shape(shape, "x")
        rows, cols = shape[1], shape[0]

        # --- Step 1: Random Xavier initialization ---
        # Creates a random matrix with approximately unit variance.
        tens = torch.zeros((rows, cols), device=device, dtype=dtype)
        torch.nn.init.xavier_normal_(tens, gain=2**(-0.5) * 1.4)

        # --- Step 2: Normalize to enforce orthogonality ---
        # tens2 = tensᵀ (tens tensᵀ)⁻¹ ensures columns of tens2 are orthonormal.
        tens2 = tens.transpose(-1, -2) @ torch.linalg.inv(tens @ tens.transpose(-1, -2))

        # --- Step 3: Construct manifold representation ---
        # Combine primal (tens2) and dual (tensᵀ) parts into one manifold tensor.
        p = self._concat_representation(tens2, tens.transpose(-1, -2))

        # --- Step 4: Wrap as ManifoldTensor and return ---
        return ManifoldTensor(p, manifold=self)

    def origin(self, *size, dtype=None, device=None) -> ManifoldTensor:
        """
        Provide a canonical origin: choose Phi as first r canonical basis vectors,
        i.e. shape (n, r) matrix with identity in the first r rows (or embedded identity).
        Then Psi computed accordingly.
        """
        self._assert_check_shape(size2shape(*size), "x")
        batch_shape = size2shape(*size)
        Phi_base = torch.zeros((*batch_shape, self.n, self.r), dtype=dtype, device=device)
        # put identity in top r rows to ensure full column rank
        idx = torch.arange(self.r, device=device)
        Phi_base[..., idx, idx] = 1.0
        Psi_base = self._psi_from_phi(Phi_base)
        p = self._concat_representation(Phi_base, Psi_base)

        return ManifoldTensor(p, manifold=self)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        """
        Project a Euclidean vector u = (X, Y) onto the tangent space at x = (Phi, Psi).
        Tangent constraint: Psi^T X + Y^T Phi = 0  (shape r x r)
        We look for correction matrices M (r x r) and subtract Psi M from X and
        subtract Phi M^T from Y, i.e. X' = X - Psi M, Y' = Y - Phi M^T.
        This leads to Sylvester: (Psi^T Psi) M + M (Phi^T Phi) = Psi^T X + Y^T Phi
        """
        Phi, Psi = self._split_representation(x)
        X, Y = self._split_representation(u)

        # Compute A = Psi^T Psi, B = Phi^T Phi, C = Psi^T X + Y^T Phi
        A = torch.matmul(Psi.transpose(-1, -2), Psi)  # (..., r, r)
        B = torch.matmul(Phi.transpose(-1, -2), Phi)  # (..., r, r)
        C = torch.matmul(Psi.transpose(-1, -2), X) + torch.matmul(Y.transpose(-1, -2), Phi)  # (..., r, r)

        M = self._solve_sylvester_vec(A, B, C)  # (..., r, r) (A, -B, C)

        X_proj = X - torch.matmul(Psi, M)
        Y_proj = Y - torch.matmul(Phi, M.transpose(-1, -2))

        projection = self._concat_representation(X_proj, Y_proj)

        return projection

    def _solve_sylvester_vec(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Solve the Sylvester equation AX + XB = C for symmetric matrices A and B.

        This implementation supports both single matrices and batched matrices.

        Args:
            A (torch.Tensor): Symmetric matrix of shape (..., r, r)
            B (torch.Tensor): Symmetric matrix of shape (..., r, r)
            C (torch.Tensor): Right-hand side matrix of shape (..., r, r)

        Returns:
            X (torch.Tensor): Solution matrix satisfying AX + XB = C,
                              same shape as C (..., r, r)
        """
        # Ensure all tensors are contiguous in memory for safe linear algebra operations
        A, B, C = A.contiguous(), B.contiguous(), C.contiguous()
        r = A.shape[-1]  # dimension of the square matrices

        # Eigendecompose A and B: A = U R U^T, B = V S V^T
        # R, S: eigenvalues (..., r)
        # U, V: eigenvectors (..., r, r)
        R, U = torch.linalg.eigh(A)
        S, V = torch.linalg.eigh(B)

        # Transform C into the eigenbasis of A and B:
        # F = U^T C V
        # This simplifies AX + XB = C to a diagonal system in Y
        F = U.transpose(-1, -2) @ C @ V

        # Compute denominators for element-wise solution:
        # AX + XB = C -> (λ_i + μ_j) Y_ij = F_ij
        denom = R.unsqueeze(-1) + S.unsqueeze(-2)

        # Solve for Y in the eigenbasis
        Y = F / denom

        # Transform back to the original basis: X = U Y V^T
        X = U @ Y @ V.transpose(-1, -2)

        return X

    egrad2rgrad = proju

    def inner(
    self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim: bool = False
) -> torch.Tensor:
        """
        Compute the inner product on the tangent space
        of the BiOrthogonal manifold at point x.

        The BiOrthogonal tangent vectors are represented as concatenated
        (Phi, Psi) matrices. The inner product is computed as the sum of
        Frobenius inner products of the components:

            <u, v> = trace(X_u^T X_v) + trace(Y_u^T Y_v)

        If `v` is None, computes the squared norm of `u`.

        Args:
            x (torch.Tensor): Manifold point (not used in this metric but included
                              for Geoopt compatibility). Shape (..., 2*n, r).
            u (torch.Tensor): Tangent vector at x. Shape (..., 2*n, r).
            v (torch.Tensor, optional): Another tangent vector at x. If None,
                                        uses u for self-inner product.
            keepdim (bool, optional): If True, retains batch dimensions in the output.
                                      If False, reduces to scalar for non-batched inputs
                                      or sums over batch for batched inputs.

        Returns:
            torch.Tensor: Inner product(s). Scalar or tensor matching batch dimensions.
        """
        # ----------------------------
        # Split concatenated tangent vectors into (X, Y)
        # ----------------------------
        X1, Y1 = self._split_representation(u)
        if v is None:
            X2, Y2 = X1, Y1
        else:
            X2, Y2 = self._split_representation(v)

        # ----------------------------
        # Frobenius inner product over last two dims (matrix dims)
        # ----------------------------
        inner_prod = (X1 * X2).sum(dim=(-2, -1)) + (Y1 * Y2).sum(dim=(-2, -1))

        # ----------------------------
        # Reduce batch dims if keepdim=False
        # ----------------------------
        if not keepdim:
            return inner_prod.sum() if inner_prod.ndim == 0 else inner_prod

        return inner_prod

    def _psi_from_phi(self, Phi: torch.Tensor) -> torch.Tensor:
        """
        Given Phi (..., n, r) with full column rank, compute Psi = Phi @ inv(Phi^T Phi)
        so that Psi^T Phi = I_r.
        """
        # compute Gram = Phi^T Phi : shape (..., r, r)
        Gram = torch.matmul(Phi.transpose(-1, -2), Phi)  # (..., r, r)
        # invert Gram (batched)
        Gram_inv = torch.linalg.inv(Gram)
        Psi = torch.matmul(Phi, Gram_inv)  # (..., n, r)
        return Psi

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Retraction on the BiOrthogonal manifold.

        This implements a **first-order retraction** along the tangent vector `u`,
        moving the current point `x` = (Phi, Psi) along the tangent (X, Y) while
        exactly preserving the biorthogonality constraint:

            Psi_new^T Phi_new = I_r

        The retraction formula is:

            Phi_new = (Phi + X) @ inv((Psi + Y)^T @ (Phi + X))
            Psi_new = Psi + Y

        Args:
            x (torch.Tensor): Manifold point, concatenated representation of shape (..., 2*n, r)
                              or class-specific canonical shape.
            u (torch.Tensor): Tangent vector at x, same shape as x.

        Returns:
            torch.Tensor: New manifold point after retraction, same shape as x.
        """
        # ----------------------------
        # Split current point and tangent vector
        # ----------------------------
        Phi, Psi = self._split_representation(x)  # Phi, Psi: (..., n, r)
        X, Y = self._split_representation(u)      # Tangent components

        # ----------------------------
        # Compute retracted Phi using full tangent (X, Y)
        # Ensures Psi_new^T Phi_new = I exactly
        # ----------------------------
        Phi_new = (Phi + X) @ torch.linalg.inv((Psi + Y).transpose(-1, -2) @ (Phi + X))

        # ----------------------------
        # Update Psi along tangent Y
        # ----------------------------
        Psi_new = Psi + Y

        # ----------------------------
        # Concatenate to return manifold point
        # ----------------------------
        return self._concat_representation(Phi_new, Psi_new)

    expmap = retr

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport by projecting v at y (simple projection).
        """
        return self.proju(y, v)

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retraction and parallel transport on the BiOrthogonal manifold.

        This method performs a **first-order retraction** of a point `x` along a tangent vector `u`,
        and then **transports** another tangent vector `v` from the tangent space at `x` to the
        tangent space at the retracted point `y`.

        Steps:
            1. Retract `x` along `u` → y = retr(x, u)
            2. Transport tangent `v` to new tangent space → w = transp(x, y, v)

        Args:
            x (torch.Tensor): Manifold point, concatenated representation (..., 2*n, r)
            u (torch.Tensor): Tangent vector at x for retraction
            v (torch.Tensor): Tangent vector at x to be transported

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                y: New manifold point after retraction
                w: Transported tangent vector at y
        """
        # ----------------------------
        # Step 1: Retract x along u
        # ----------------------------
        y = self.retr(x, u)

        # ----------------------------
        # Step 2: Transport v from x to y
        # ----------------------------
        w = self.transp(x, y, v)

        return y, w

    expmap_transp = retr_transp

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project an ambient representation onto the manifold.
        If given (Phi0, Psi0) we reproject Phi0 -> Phi via QR (to ensure full column rank)
        then compute Psi accordingly. This ensures Psi^T Phi = I exactly.
        """
        # Accept both stacked or concatenated
        # Extract Phi part (we ignore incoming Psi and recompute it)
        Phi, _ = self._split_representation(x)
        # Orthonormalize Phi's columns via QR (reduced) to ensure stability.
        Q, R = torch.linalg.qr(Phi, mode="reduced")
        # If columns are not orthonormal (tall rectangular), Q is good; otherwise use Q
        Phi_proj = Q
        Psi_proj = self._psi_from_phi(Phi_proj)
        return self._concat_representation(Phi_proj, Psi_proj)

    def _check_vector_on_tangent(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check whether u lies in the tangent space T_x M at x.

        Tangent condition for (Φ, Ψ) ∈ BiOrthogonal(n, r):
            Ψᵀ Φ = I_r
            Tangent vectors (X, Y) satisfy: Ψᵀ X + Yᵀ Φ = 0

        Returns:
            (ok, msg) if check fails, or (True, None) if OK.
        """
        Phi, Psi = self._split_representation(x)
        X, Y = self._split_representation(u)

        # Tangency constraint: Ψᵀ X + Yᵀ Φ = 0
        tangent_condition = Psi.transpose(-1, -2) @ X + Y.transpose(-1, -2) @ Phi

        if torch.allclose(
            tangent_condition,
            tangent_condition.new_zeros(tangent_condition.shape),
            atol=atol,
            rtol=rtol,
        ):
            return True, None
        else:
            msg = f"Tangent condition violated: ||ΨᵀX + YᵀΦ|| = {tangent_condition.norm().item():.3e}"
            return False, msg

    def __repr__(self):
        return f"BiOrthogonal(n={self.n}, r={self.r})"


class ProjectionLayerPair(nn.Module):
    """
    ProjectionLayerPair constructs a pair of layers: an encoder mapping from N-dimensional input
    to r-dimensional latent space (r < N), and a corresponding decoder that approximates a
    left inverse of the encoder. The layer pair uses custom nonlinear activation functions
    that are approximately invertible.

    Attributes:
        _dim_N (int): Input dimension of the encoder / output dimension of the decoder.
        _dim_r (int): Output dimension of the encoder / input dimension of the decoder.
        _alpha_act (float): Parameter controlling the shape of the custom activation function.
        _eye_N (Tensor): Identity matrix of size N x N, used for linear algebra operations.
        _eye_r (Tensor): Identity matrix of size r x r.
        _zeros_r (Tensor): Zero vector of size r, for convenience.
        _a, _b, _t1.._t6 (float): Precomputed constants for encoder and decoder activations.
        biorthogonal (BiOrthogonal): BiOrthogonal manifold for weight parametrization.
        weight (ManifoldParameter): Trainable weight parameter constrained to the biorthogonal manifold.
        bias (Parameter): Trainable bias of size N.
        ENC (Tensor): Encoder linear transformation matrix.
        DEC (Tensor): Decoder linear transformation matrix.
        X (Tensor): Internal representation from weight decomposition.
    """

    def __init__(self, input_size, output_size, alpha=np.pi / 8, device=None):
        """
        Initialize the ProjectionLayerPair.

        Args:
            input_size (int): Dimensionality of input vectors (N).
            output_size (int): Dimensionality of latent space (r), with r < N.
            alpha (float, optional): Controls nonlinearity of activation. Defaults to pi/8.
            device (torch.device, optional): Device to place tensors on. Defaults to None.
        """
        super().__init__()

        # Store dimensions and activation parameter
        self._dim_N = input_size
        self._dim_r = output_size
        self._alpha_act = alpha

        if self._alpha_act > np.pi / 4 or self._alpha_act < 0:
            # Clip invalid alpha values to default pi/8
            self._alpha_act = np.pi / 8

        # Identity matrices and zero vector for convenience
        self._eye_N = torch.eye(self._dim_N, dtype=torch.float32).to(device)
        self._eye_r = torch.eye(self._dim_r, dtype=torch.float32).to(device)
        self._zeros_r = torch.zeros(self._dim_r, dtype=torch.float32).to(device)

        # Precompute constants for nonlinear activation functions
        self._a = (1 / np.sin(self._alpha_act)) ** 2 - (1 / np.cos(self._alpha_act)) ** 2
        self._b = (1 / np.sin(self._alpha_act)) ** 2 + (1 / np.cos(self._alpha_act)) ** 2
        self._t1 = self._b / self._a
        self._t2 = np.sqrt(2) / (self._a * np.sin(self._alpha_act))
        self._t3 = 1 / self._a
        self._t4 = 2 / (np.sin(self._alpha_act) * np.cos(self._alpha_act))
        self._t5 = np.sqrt(2) / np.cos(self._alpha_act)
        self._t6 = 2 * self._a

        # Initialize the biorthogonal manifold for the weight
        self.biorthogonal = BiOrthogonal(self._dim_N, self._dim_r, device=device)
        self.weight = gt.ManifoldParameter(
            self.biorthogonal.random(self._dim_N, self._dim_r, device=device),
            manifold=self.biorthogonal,
            requires_grad=True
        )

        # Initialize bias (uninitialized tensor, will be trained)
        self.bias = nn.Parameter(torch.Tensor(self._dim_N).to(device))

    def update_weight_params(self):
        """
        Decompose the manifold weight into encoder and decoder matrices.
        Stores:
            DEC (Tensor): Decoder matrix
            ENC (Tensor): Encoder matrix
            X (Tensor): Intermediate representation
        """

        self.DEC, self.X = self.biorthogonal._split_representation(self.weight)
        self.ENC = self.X.transpose(-1,-2)

    def enc_activation(self, x):
        """
        Encoder nonlinear activation function.

        Args:
            x (Tensor): Input tensor of shape (..., r)

        Returns:
            Tensor: Activated output tensor
        """
        return self._t1 * x + self._t2 - self._t3 * torch.sqrt((self._t4 * x + self._t5) ** 2 + self._t6)

    def dec_activation(self, x):
        """
        Decoder nonlinear activation function, approximately inverse of encoder.

        Args:
            x (Tensor): Input tensor of shape (..., r)

        Returns:
            Tensor: Activated output tensor
        """
        return self._t1 * x - self._t2 + self._t3 * torch.sqrt((self._t4 * x - self._t5) ** 2 + self._t6)

    def d_enc_activation(self, x):
        """
        Derivative of the encoder activation function.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Element-wise derivative
        """
        return self._t1 - self._t3 * self._t4 * (self._t4 * x + self._t5) / torch.sqrt(
            (self._t4 * x + self._t5) ** 2 + self._t6)

    def d_dec_activation(self, x):
        """
        Derivative of the decoder activation function.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Element-wise derivative
        """
        return self._t1 + self._t3 * self._t4 * (self._t4 * x - self._t5) / torch.sqrt(
            (self._t4 * x - self._t5) ** 2 + self._t6)

    def fw_encoder_layer(self, x):
        """
        Forward pass through encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, N)

        Returns:
            Tensor: Encoded output of shape (batch_size, r)
        """
        a = self.ENC @ (x - self.bias).unsqueeze(-1)
        z = self.enc_activation(a.squeeze(-1))
        return z

    def fw_decoder_layer(self, z):
        """
        Forward pass through decoder layer.

        Args:
            z (Tensor): Encoded latent tensor of shape (batch_size, r)

        Returns:
            Tensor: Reconstructed input of shape (batch_size, N)
        """
        x = (self.DEC @ self.dec_activation(z).unsqueeze(-1)).squeeze(-1) + self.bias
        return x


    def fw_encoder_layer_jac(self, x, x_prime):
        """
        Forward pass through encoder layer with Jacobian-vector product.

        Args:
            x (Tensor): Input tensor (batch_size, N)
            x_prime (Tensor): Perturbation tensor (batch_size, N)

        Returns:
            z (Tensor): Encoded output
            jac (Tensor): Jacobian-vector product (ENC @ x_prime scaled by activation derivative)
        """
        a = self.ENC @ (x - self.bias).unsqueeze(-1)
        z = self.enc_activation(a.squeeze(-1))
        jac = self.d_enc_activation(a) * self.ENC @ x_prime
        return z, jac

    def fw_decoder_layer_jac(self, z, x_prime):
        """
        Forward pass through decoder layer with Jacobian-vector product.

        Args:
            z (Tensor): Encoded latent tensor (batch_size, r)
            x_prime (Tensor): Perturbation tensor (batch_size, r)

        Returns:
            x (Tensor): Reconstructed input
            jac (Tensor): Jacobian-vector product (DEC @ x_prime scaled by activation derivative)
        """
        x = (self.DEC @ self.dec_activation(z).unsqueeze(-1)).squeeze(-1) + self.bias
        jac = self.DEC @ (self.d_dec_activation(z.unsqueeze(-1)) * x_prime)
        return x, jac


class ConstrainedAE(nn.Module):
    """
    Constrained Autoencoder on configuration space Q.
    Encoder ρ_Q and decoder φ_Q satisfy projection properties by construction.
    Used to build symplectic autoencoder via cotangent lift.
    """
    def __init__(self, layers_dims = [600, 128, 64, 32, 10], b_init=0.0, alpha= np.pi/10, device="cpu"):
        super().__init__()
        self.device = device

        # Layer dimensions: [n_full, n_1, n_2, ..., n_latent]
        self.layer_dims = layers_dims
        self.b_init = b_init
        self.alpha = alpha

        # Build layer pairs
        self.layers = nn.ModuleList()
        for i in range(1, len(self.layer_dims)):
            self.layers.append(
                ProjectionLayerPair(
                    self.layer_dims[i - 1],
                    self.layer_dims[i],
                    self.alpha,
                    self.device
                )
            )
            torch.nn.init.constant_(self.layers[-1].bias, self.b_init)

        self.update_parameters()

        # Dimensions
        self.n_full = self.layer_dims[0]
        self.n_latent = self.layer_dims[-1]

    def update_parameters(self):
        """Update weight matrices after optimization step"""
        for layer in self.layers:
            layer.update_weight_params()

    def encoder_pos(self, q):
        z = q
        for layer in self.layers:
            z = layer.fw_encoder_layer(z)
        return z

    def decoder_pos(self, z):
        x_hat = z
        for layer in reversed(self.layers):
            x_hat = layer.fw_decoder_layer(x_hat)
        return x_hat


    def encoder_jacobian(self, q):
        """
        Compute encoder Jacobian dρ_Q/dq
        Used for cotangent lift: p̌ = (dφ_Q)^T p
        """
        z = q
        jac = torch.eye(self.n_full, device=self.device).unsqueeze(0).expand(
            q.shape[0], -1, -1
        )

        for layer in self.layers:
            z, jac = layer.fw_encoder_layer_jac(z, jac)

        return jac

    def decoder_jacobian(self, q_check):
        """
        Compute decoder Jacobian dφ_Q/dq̌
        Used for cotangent lift: p = (dρ_Q)^T p̌
        """
        x = q_check
        jac = torch.eye(self.n_latent, device=self.device).unsqueeze(0).expand(
            q_check.shape[0], -1, -1
        )

        for layer in reversed(self.layers):
            x, jac = layer.fw_decoder_layer_jac(x, jac)

        return jac

    def encoder_phase(self, q, p=None, return_jac=False):
        """
        Encode position and momentum via cotangent lift.

        Cotangent lift formula (Eq. 8 in paper):
        ρ(q, p) = [ρ_Q(q), (dφ_Q|_{ρ_Q(q)})^T p]

        Args:
            q: positions (batch_size, n_full)
            p: velocities or momenta (batch_size, n_full)
            return_jac: if True, return Jacobian instead of transforming p

        Returns:
            q_check: reduced positions
            p_check or jac: reduced momenta or Jacobian
        """
        # Encode position
        q_check = self.encoder_pos(q)

        if return_jac:
            # Return decoder Jacobian at encoded point (for cotangent lift)
            jac = self.decoder_jacobian(q_check)
            return jac
        else:
            # Transform momentum: p̌ = (dφ_Q|_{q̌})^T p
            jac = self.decoder_jacobian(q_check)
            p_check = (jac.transpose(-1, -2) @ p.unsqueeze(-1)).squeeze(-1)
            return q_check, p_check

    def decoder_phase(self, q_check, p_check=None, return_jac=False):
        """
        Decode position and momentum via cotangent lift.

        Cotangent lift formula:
        φ(q̌, p̌) = [φ_Q(q̌), (dρ_Q|_{φ_Q(q̌)})^T p̌]

        Args:
            q_check: reduced positions (batch_size, n_latent)
            p_check: reduced momenta (batch_size, n_latent)
            return_jac: if True, return Jacobian instead of transforming dq_check

        Returns:
            q: full positions
            p or jac: full momenta or Jacobian
        """
        # Decode position
        q = self.decoder_pos(q_check)

        if return_jac:
            # Return encoder Jacobian at decoded point (for cotangent lift)
            jac = self.encoder_jacobian(q)
            return jac
        else:
            # Transform momentum: p = (dρ_Q|_q)^T p̌
            jac = self.encoder_jacobian(q)
            p = (jac.transpose(-1, -2) @ p_check.unsqueeze(-1)).squeeze(-1)
            return q, p

    def reduce_forces(self, q_check, tau):
        """
        Reduce external forces to latent space: τ̌ = (dφ_Q)^T τ

        Args:
            q_check: reduced positions
            tau: full-dimensional forces

        Returns:
            tau_check: reduced forces
        """
        jac_dec = self.decoder_jacobian(q_check)
        tau_check = (jac_dec.transpose(-1, -2) @ tau.unsqueeze(-1)).squeeze(-1)
        return tau_check


class BiMapLayer(nn.Module):
    """
    Bilinear mapping for SPD matrices using Geoopt's Stiefel manifold.

    Implements X_out = W @ X_in @ W.T where W has orthonormal rows (W @ W.T = I).
    This preserves the SPD property while changing dimensionality.

    Since Geoopt's Stiefel enforces orthonormal columns, we store W.T on the manifold
    and transpose it during the forward pass.

    Args:
        in_features: Input matrix dimension
        out_features: Output matrix dimension (must be <= in_features)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features <= in_features, \
            f"out_features ({out_features}) must be <= in_features ({in_features})"

        self.in_features = in_features
        self.out_features = out_features

        # Stiefel manifold for orthonormal matrices
        self.manifold = Stiefel()

        # Store W.T (shape: in_features × out_features) on manifold
        # So W = W.T.T will have orthonormal rows
        Wt = torch.randn(in_features, out_features)
        Wt = self.manifold.projx(Wt)  # Project onto manifold
        self.Wt = ManifoldParameter(Wt, manifold=self.manifold)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply bilinear transformation.

        Args:
            X: Input SPD matrix of shape (..., in_features, in_features)

        Returns:
            Output SPD matrix of shape (..., out_features, out_features)
        """
        # Get W with orthonormal rows
        W = self.Wt.transpose(-2, -1)  # (out_features, in_features)

        # Bilinear map: W @ X @ W.T
        # Handle batched operations with broadcasting
        WX = torch.matmul(W.unsqueeze(0), X)  # (..., out_features, in_features)
        out = torch.matmul(WX, W.unsqueeze(0).transpose(-2, -1))  # (..., out_features, out_features)

        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class SPDScaleLayer(nn.Module):
    """
    Learnable scaling layer for SPD matrices.

    Applies X_out = exp(α) * X_in where α is a learnable scalar parameter.
    This allows the network to learn the overall scale of the mass matrix.
    """

    def __init__(self):
        super().__init__()
        # Initialize scale to 0 (corresponding to scale factor of 1)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Scale the SPD matrix."""
        scale = torch.exp(self.log_scale)
        return scale * X

    def extra_repr(self) -> str:
        return f"scale={torch.exp(self.log_scale).item():.4f}"


class ReEigLayer(nn.Module):
    """
    Eigenvalue rectification layer for SPD matrices.

    Ensures all eigenvalues are >= epsilon by computing:
    X_out = U @ diag(max(epsilon, λ_i)) @ U.T

    Args:
        epsilon: Minimum eigenvalue threshold
    """

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Rectify eigenvalues of SPD matrix.

        Args:
            X: Input matrix of shape (..., n, n)

        Returns:
            Rectified SPD matrix with eigenvalues >= epsilon
        """
        # Eigendecomposition
        evals, evecs = torch.linalg.eigh(X)

        # Rectify eigenvalues
        eps_tensor = torch.tensor(
            self.epsilon, device=X.device, dtype=evals.dtype
        )
        rectified_evals = torch.maximum(evals, eps_tensor)

        # Reconstruct matrix
        return evecs @ torch.diag_embed(rectified_evals) @ evecs.transpose(-2, -1)

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}"


class SPDNetwork(nn.Module):
    """
    SPD network with bilinear mappings and eigenvalue rectification.

    Provides more expressive transformations of SPD matrices through:
    1. Multiple bilinear mappings with dimension changes
    2. Eigenvalue rectification for numerical stability
    3. Optional learnable scaling

    Args:
        n_dof: Number of degrees of freedom
        hidden_dims: List of hidden dimensions for BiMap layers
        use_scaling: Whether to use learnable scaling
        epsilon: Minimum eigenvalue for ReEig layers
        device: Device for computations
    """

    def __init__(
        self,
        n_dof: int,
        hidden_dims: Optional[list] = None,
        use_scaling: bool = True,
        epsilon: float = 1e-4,
        device: str = 'cpu'
    ):
        super().__init__()

        self.n_dof = n_dof

        if hidden_dims is None:
            hidden_dims = [min(n_dof * 2, 32), n_dof]

        # Ensure valid dimensions
        dims = [n_dof] + hidden_dims
        for i in range(len(dims) - 1):
            if dims[i+1] > dims[i]:
                raise ValueError(
                    f"Dimension must be non-increasing in BiMap layers: "
                    f"{dims[i]} -> {dims[i+1]}"
                )

        # Build sequence of transformations
        layers = []

        for i in range(len(dims) - 1):
            # BiMap transformation
            layers.append(BiMapLayer(dims[i], dims[i+1]).to(device))
            # Eigenvalue rectification for stability
            layers.append(ReEigLayer(epsilon=epsilon).to(device))

        # Optional scaling layer at the end
        if use_scaling:
            layers.append(SPDScaleLayer().to(device))

        self.layers = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply sequence of SPD transformations.

        Args:
            X: Input SPD matrix

        Returns:
            Transformed SPD matrix
        """
        for layer in self.layers:
            X = layer(X)
        return X



class SPDExpLayer(nn.Module):
    """
    Exponential map layer for SPD manifolds.
    Converts vector input to symmetric matrix, then applies exponential map.

    Args:
        n_dof: Number of degrees of freedom
        parametrized: If True, learn reference point for expmap
        device: Device for computations
    """

    def __init__(self, n_dof: int, parametrized: bool = False, device: str = 'cpu'):
        super().__init__()

        self.n_dof = n_dof
        self.m = (n_dof * (n_dof + 1)) // 2  # Dimension of vectorized symmetric matrix
        self.device = device
        self.parametrized = parametrized

        # Use Geoopt's SPD manifold
        self.manifold = SymmetricPositiveDefinite()

        # Indices for constructing symmetric matrices
        tril_idx = torch.tril_indices(n_dof, n_dof, offset=-1, device=device)
        diag_idx = torch.arange(n_dof, device=device)

        # we register them so they change device when we apply .to(device)
        self.register_buffer('tril_indices', tril_idx)
        self.register_buffer('diag_indices', diag_idx)

        if parametrized:
            # Learnable reference point on manifold
            origin = torch.eye(n_dof, device=device)
            self.reference = ManifoldParameter(
                origin, manifold=self.manifold, requires_grad=True
            )
        else:
            # Fixed identity reference
            self.register_buffer('reference', torch.eye(n_dof, device=device))

    def vec_to_symmetric(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert vector to symmetric matrix."""
        batch_size = vec.shape[0]
        symmat = torch.zeros(batch_size, self.n_dof, self.n_dof,
                           dtype=vec.dtype, device=vec.device)

        # Diagonal elements
        symmat[:, self.diag_indices, self.diag_indices] = vec[:, :self.n_dof]

        # Off-diagonal elements (symmetric)
        symmat[:, self.tril_indices[1], self.tril_indices[0]] = vec[:, self.n_dof:]
        symmat[:, self.tril_indices[0], self.tril_indices[1]] = vec[:, self.n_dof:]

        return symmat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: vector -> symmetric matrix -> SPD matrix via expmap.

        Args:
            x: Input vector of shape (batch, m)

        Returns:
            SPD matrix of shape (batch, n_dof, n_dof)
        """
        # Convert to symmetric matrix (tangent vector)
        tangent = self.vec_to_symmetric(x)

        if self.parametrized:
            # Exponential map from learned reference point
            spd_matrix = self.manifold.expmap(self.reference, tangent)
        else:
            # Direct matrix exponential (expmap at identity)
            spd_matrix = torch.matrix_exp(tangent)

        return spd_matrix


class EuclideanLayer(nn.Module):
    """
    Fully connected layer on Euclidean space with activation.

    Args:
        n_in: Input dimension
        n_out: Output dimension
        activation: Activation function name
        device: Device for computations
    """

    def __init__(self, n_in: int, n_out: int, activation: str = 'softplus', device: str = 'cpu'):
        super().__init__()

        self.manifold = geoopt.manifolds.Euclidean()

        # Manifold parameters
        self.weight = ManifoldParameter(
            torch.empty(n_out, n_in, device=device),
            manifold=self.manifold
        )
        self.bias = ManifoldParameter(
            torch.empty(n_out, device=device),
            manifold=self.manifold
        )

        # Activation function
        if activation.lower() == 'softplus':
            self.activation = nn.Softplus(beta=1.0)
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.activation(F.linear(x, self.weight, self.bias))


# Yoshida 4th-order splitting coefficients
# These provide 4th-order accuracy for symplectic integration
# where w₁ + w₂ + w₃ = 1 and certain error terms cancel to achieve 4th order
YOSHIDA4_COEFS = [
    1.3512071919596576340476878089715,   # w₁: First substep weight
    -1.7024143839193152680953756179429,  # w₂: Second substep weight (negative for symmetry)
    1.3512071919596576340476878089715    # w₃: Third substep weight (same as w₁ for symmetry)
]



class ODELayerFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        context,
        z0: torch.Tensor,           # Initial state [batch, 2*dim]: concatenated [p, q]
        t_range_forward: torch.Tensor,  # Time points for integration
        dynamics: nn.Module,        # Neural network defining the Hamiltonian dynamics
        *theta: torch.Tensor,       # Parameters of the dynamics network
    ) -> torch.Tensor:
        """
        Forward pass: Integrate ODE from t=0 to t=T using Yoshida-4 splitting.

        State representation:
        - z = [p, q]: Primary variables (momentum and position)
        - [x, y]: Shadow variables that track slightly different trajectories

        Algorithm structure per timestep:
        1. Apply 3 Yoshida substeps (with coefficients w₁, w₂, w₃)
        2. Each substep:
           a) Evolve under split Hamiltonian
           b) Project shadow variables to primary variables
           c) Reset shadow variables to match primary variables

        Args:
            context: Autograd context for saving information for backward pass
            z0: Initial condition [batch_size, 2*dim]
            t_range_forward: Time discretization [n_steps+1]
            dynamics: Neural network f_θ that defines Hamiltonian
            *theta: Flattened parameters of dynamics network

        Returns:
            zT: Final state at time T [batch_size, 2*dim]
        """
        # Enable gradient computation (needed for autograd in backward pass)
        torch.set_grad_enabled(True)

        # Split initial state into momentum (p) and position (q)
        # Convention: z = [p, q] where p and q each have dimension 'dim'
        dim = z0.shape[-1] // 2
        p = z0[..., :dim].clone()   # Momentum variables
        q = z0[..., dim:].clone()   # Position variables

        # Initialize shadow variables to match primary variables
        x = p.clone()  # Shadow momentum
        y = q.clone()  # Shadow position

        # Projection coefficients (chosen as in the Yoshida Paper)
        # These control the convex combination during projection steps
        lambd = 1/np.e  # ≈ 0.368 - coefficient for momentum projection
        mu = 1/np.pi    # ≈ 0.318 - coefficient for position projection

        # Time step size
        delta_t = t_range_forward[1] - t_range_forward[0]

        # Store trajectory information for the adjoint computation
        # This enables us to reverse the operations during backpropagation
        # Each entry contains: state before operation, operation type, parameters
        trajectory = []

        # Main integration loop: perform Yoshida splitting for each timestep
        n_steps = len(t_range_forward) - 1
        for i in range(n_steps):
            # Apply the three Yoshida substeps (weighted by YOSHIDA4_COEFS)
            for coef in YOSHIDA4_COEFS:
                # ============================================================
                # Step 1: Yoshida evolution (Hamiltonian flow)
                # ============================================================
                # Save state BEFORE this operation (needed for backward pass)
                trajectory.append({
                    'type': 'yoshida_step',
                    'state_in': (p.detach(), x.detach(), q.detach(), y.detach()),
                    'coef': coef,
                    'dt': delta_t.item()
                })

                # Apply one Yoshida step: evolve under split Hamiltonian
                # This performs: exp(H_A * w*dt/2) ∘ exp(H_B * w*dt) ∘ exp(H_A * w*dt/2)
                p, x, q, y = ODELayerFunc.one_step_yoshida(p, x, q, y, delta_t, coef, dynamics)

                # ============================================================
                # Step 2: Projection
                # ============================================================
                # Save state BEFORE projection
                trajectory.append({
                    'type': 'projection',
                    'state_in': (p.detach(), x.detach(), q.detach(), y.detach()),
                    'lam': mu if i % 2 == 0 else lambd,      # Alternating projection weights
                    'ksi': lambd if i % 2 == 0 else mu
                })

                # Project shadow variables onto primary variables
                # Alternates projection parameters between even/odd steps
                if i % 2 == 0:
                    p, q = ODELayerFunc.project(p, x, q, y, mu, lambd)
                else:
                    p, q = ODELayerFunc.project(p, x, q, y, lambd, mu)

                # ============================================================
                # Step 3: Shadow variable reset
                # ============================================================
                # Save state BEFORE reset
                trajectory.append({
                    'type': 'shadow_reset',
                    'state_in': (p.detach(), None, q.detach(), None)
                })

                # Reset shadow variables to match primary variables
                x, y = p.clone(), q.clone()

        # Concatenate final momentum and position into output state
        zt = torch.cat([p, q], dim=-1)

        # Save all information needed for backward pass
        context.save_for_backward(z0, zt, t_range_forward, *theta)
        context.delta_t = delta_t.item()
        context.dynamics = dynamics
        context.trajectory = trajectory  # Complete forward trajectory for reversal

        return zt

    @staticmethod
    def expHB(p, x, q, y, t, dynamics):
        """
        Evolution under Hamiltonian H_A for time t.

        Mathematical formulation:
        - Input to dynamics: z = [x, q]
        - Dynamics output: [∂H/∂x, ∂H/∂q]
        - Evolution equations:
            dp/dt = -∂H_A/∂q  →  p(t) = p(0) - t * ∂H/∂q
            dy/dt = +∂H_A/∂x  →  y(t) = y(0) + t * ∂H/∂x

        Args:
            p, x, q, y: Current state (primary and shadow variables)
            t: Time step
            dynamics: Neural network defining Hamiltonian

        Returns:
            Updated (p, x, q, y) after evolution under H_A
        """
        # Construct input: [q, x] for this Hamiltonian split
        z = torch.cat([x, q], dim=-1)

        # Compute Hamiltonian derivatives via neural network
        derivatives = dynamics(z, None)
        dim = z.shape[-1] // 2
        Hx = derivatives[..., :dim]  # ∂H/∂x
        Hq = derivatives[..., dim:]  # ∂H/∂q

        # Apply symplectic evolution equations
        p = p - t * Hq  # Momentum evolves with -∂H/∂q
        y = y + t * Hx  # Shadow position evolves with +∂H/∂x

        # Note: x and q are unchanged under H_A evolution
        return p, x, q, y

    @staticmethod
    def expHA(p, x, q, y, t, dynamics):
        """
        Evolution under Hamiltonian H_B for time t.

        Mathematical formulation:
        - Input to dynamics: z = [p, y]
        - Dynamics output: [∂H/∂p, ∂H/∂y]
        - Evolution equations:
            dx/dt = -∂H_B/∂y  →  x(t) = x(0) - t * ∂H/∂y
            dq/dt = +∂H_B/∂p  →  q(t) = q(0) + t * ∂H/∂p

        Args:
            p, x, q, y: Current state (primary and shadow variables)
            t: Time step (can be positive or negative)
            dynamics: Neural network defining Hamiltonian

        Returns:
            Updated (p, x, q, y) after evolution under H_B
        """
        # Construct input: [y, p] for this Hamiltonian split
        z = torch.cat([p, y], dim=-1)

        # Compute Hamiltonian derivatives
        derivatives = dynamics(z, None)
        dim = z.shape[-1] // 2
        Hp = derivatives[..., :dim]  # ∂H/∂p
        Hy = derivatives[..., dim:]  # ∂H/∂y

        # Apply symplectic evolution equations
        x = x - t * Hy  # Shadow momentum evolves with -∂H/∂y
        q = q + t * Hp  # Position evolves with +∂H/∂p

        # Note: p and y are unchanged under H_B evolution
        return p, x, q, y

    @staticmethod
    def project(ptil, xtil, qtil, ytil, lam=0.5, ksi=0.4):
        """
        Projection step: combine shadow and primary variables.


        Mathematical formulation:
        - p_next = λ * p_tilde + (1-λ) * x_tilde  (convex combination)
        - q_next = ξ * q_tilde + (1-ξ) * y_tilde  (convex combination)

        Args:
            ptil, xtil: Momentum and shadow momentum before projection
            qtil, ytil: Position and shadow position before projection
            lam: Weight for momentum projection (0 < λ < 1)
            ksi: Weight for position projection (0 < ξ < 1)

        Returns:
            (p_next, q_next): Projected primary variables
        """
        p_next = lam * ptil + (1 - lam) * xtil
        q_next = ksi * qtil + (1 - ksi) * ytil
        return p_next, q_next

    @staticmethod
    def one_step_yoshida(p, x, q, y, dt, w, dynamics):
        """
        One Yoshida substep with coefficient w.

        Applies symmetric splitting:
        exp(w*dt*H) ≈ exp(w*dt/2 * H_A) ∘ exp(w*dt * H_B) ∘ exp(w*dt/2 * H_A)

        This symmetric composition ensures:
        - Time-reversibility
        - Symplectic structure preservation
        - 2nd-order accuracy for each substep

        Combined with Yoshida coefficients, achieves 4th-order overall accuracy.

        Args:
            p, x, q, y: Current state
            dt: Base time step
            w: Yoshida coefficient (weights this substep)
            dynamics: Neural network defining Hamiltonian

        Returns:
            Updated (p, x, q, y) after this Yoshida substep
        """
        # First half-step under H_A
        p, x, q, y = ODELayerFunc.expHA(p, x, q, y, w * dt/2, dynamics)

        # Full step under H_B
        p, x, q, y = ODELayerFunc.expHB(p, x, q, y, w * dt, dynamics)

        # Second half-step under H_A (symmetric!)
        p, x, q, y = ODELayerFunc.expHA(p, x, q, y, w * dt/2, dynamics)

        return p, x, q, y

    @staticmethod
    def backward(context, adj_end: torch.Tensor):
        """
        Backward pass: Compute gradients via discrete adjoint method.

        MATHEMATICAL FOUNDATION:
        ========================
        The discrete adjoint method reverses the forward computation:
        1. Start with adjoint at final time: λ(T) = ∂L/∂z(T)
        2. Reverse each operation: λ(t-dt) = (∂f/∂z)ᵀ λ(t)
        3. Accumulate parameter gradients: ∂L/∂θ += (∂f/∂θ)ᵀ λ(t)

        IMPLEMENTATION STRATEGY:
        ========================
        We saved the forward trajectory with three types of operations:
        1. yoshida_step: Hamiltonian evolution (nonlinear, use autograd)
        2. projection: Linear combination of variables (analytic derivatives)
        3. shadow_reset: Copy operation (simple adjoint accumulation)

        We reverse through these operations in REVERSE ORDER, computing:
        - Adjoints for state variables (p, x, q, y)
        - Gradients for network parameters θ

        ADJOINT VARIABLE INTERPRETATION:
        ================================
        If L is the loss, then adjoints represent:
        - adj_p = ∂L/∂p: sensitivity of loss to momentum
        - adj_q = ∂L/∂q: sensitivity of loss to position
        - adj_x = ∂L/∂x: sensitivity of loss to shadow momentum
        - adj_y = ∂L/∂y: sensitivity of loss to shadow position

        Args:
            context: Saved context from forward pass
            adj_end: Incoming gradient ∂L/∂z(T) at final time

        Returns:
            Tuple of gradients: (∂L/∂z₀, None, None, ∂L/∂θ₁, ∂L/∂θ₂, ...)
        """
        # Re-enable gradient computation for autograd
        torch.set_grad_enabled(True)

        # Retrieve saved tensors and context from forward pass
        z0, zT, t_range_forward, *theta = context.saved_tensors
        delta_t = context.delta_t
        dynamics = context.dynamics
        trajectory = context.trajectory  # Sequence of operations to reverse

        # Initialize adjoints at final time T
        # Input: adj_end = ∂L/∂z(T) = [∂L/∂p(T), ∂L/∂q(T)]
        dim = zT.shape[-1] // 2
        adj_p = adj_end[..., :dim].clone()   # ∂L/∂p(T)
        adj_q = adj_end[..., dim:].clone()   # ∂L/∂q(T)

        # Shadow variables have zero adjoint at final time
        # (they only exist as intermediate variables during integration)
        adj_x = torch.zeros_like(adj_p)  # ∂L/∂x(T) = 0
        adj_y = torch.zeros_like(adj_q)  # ∂L/∂y(T) = 0

        # Initialize accumulators for parameter gradients
        # These will sum contributions from all timesteps
        adj_theta = [torch.zeros_like(param) for param in theta]

        # ====================================================================
        # MAIN BACKWARD LOOP: Reverse through trajectory
        # ====================================================================
        # Process operations in REVERSE order (last to first)
        for step_info in reversed(trajectory):
            step_type = step_info['type']

            # ================================================================
            # Operation Type 1: Shadow Reset
            # ================================================================
            if step_type == 'shadow_reset':
                """
                Forward operation: x = p.clone(), y = q.clone()

                Adjoint computation:
                - Since x gets its value from p: ∂L/∂p += ∂L/∂x
                - Since y gets its value from q: ∂L/∂q += ∂L/∂y
                - After reset, x and y start fresh: ∂L/∂x = 0, ∂L/∂y = 0
                                """
                adj_p = adj_p + adj_x  # Accumulate x's gradient into p
                adj_q = adj_q + adj_y  # Accumulate y's gradient into q

                # Shadow adjoints reset to zero (they're about to be "created")
                adj_x = torch.zeros_like(adj_x)
                adj_y = torch.zeros_like(adj_y)

            # ================================================================
            # Operation Type 2: Projection
            # ================================================================
            elif step_type == 'projection':
                """
                Forward operation:
                    p_out = λ * p_in + (1-λ) * x_in
                    q_out = ξ * q_in + (1-ξ) * y_in

                Adjoint computation (chain rule for linear operations):
                    ∂L/∂p_in = λ * ∂L/∂p_out
                    ∂L/∂x_in = (1-λ) * ∂L/∂p_out
                    ∂L/∂q_in = ξ * ∂L/∂q_out
                    ∂L/∂y_in = (1-ξ) * ∂L/∂q_out

                This is the derivative of a convex combination.
                """
                # Retrieve saved state and projection parameters
                p_in, x_in, q_in, y_in = step_info['state_in']
                lam = step_info['lam']  # λ
                ksi = step_info['ksi']  # ξ

                # Apply transposed projection (adjoint of linear map)
                # Currently we have adj_p = ∂L/∂p_out, adj_q = ∂L/∂q_out
                adj_p_in = lam * adj_p           # Gradient flows back weighted by λ
                adj_x_in = (1 - lam) * adj_p     # Gradient flows to x weighted by (1-λ)
                adj_q_in = ksi * adj_q           # Gradient flows back weighted by ξ
                adj_y_in = (1 - ksi) * adj_q     # Gradient flows to y weighted by (1-ξ)

                # Update adjoints to reflect the input side of projection
                adj_p = adj_p_in
                adj_x = adj_x_in
                adj_q = adj_q_in
                adj_y = adj_y_in

            # ================================================================
            # Operation Type 3: Yoshida Step
            # ================================================================
            elif step_type == 'yoshida_step':
                """
                Forward operation: Complex nonlinear dynamics
                    (p, x, q, y)_out = one_step_yoshida(p, x, q, y)_in

                Adjoint computation strategy:
                - Too complex for manual derivatives
                - Use PyTorch autograd to compute exact Jacobian-transpose-vector product
                - This is efficient: doesn't form full Jacobian matrix

                Mathematical operation:
                    [∂L/∂(p,x,q,y)_in] = J^T [∂L/∂(p,x,q,y)_out]
                    [∂L/∂θ] += (∂f/∂θ)^T [∂L/∂(p,x,q,y)_out]

                where J = ∂f/∂(p,x,q,y) is the Jacobian of the step.
                """
                # Retrieve the state BEFORE this Yoshida step
                p_in, x_in, q_in, y_in = step_info['state_in']
                coef = step_info['coef']  # Yoshida coefficient w
                dt = step_info['dt']

                # Create fresh tensors with gradient tracking enabled
                # (detach from old graph, create new computational graph)
                p_in = p_in.detach().requires_grad_(True)
                x_in = x_in.detach().requires_grad_(True)
                q_in = q_in.detach().requires_grad_(True)
                y_in = y_in.detach().requires_grad_(True)

                # Re-run the forward operation to build computation graph
                # This creates the graph that autograd will use for derivatives
                p_out, x_out, q_out, y_out = ODELayerFunc.one_step_yoshida(
                    p_in, x_in, q_in, y_in, dt, coef, dynamics
                )

                # Compute gradients using autograd's vector-Jacobian product
                # This efficiently computes: (∂outputs/∂inputs)^T @ grad_outputs
                # Returns: [∂L/∂p_in, ∂L/∂x_in, ∂L/∂q_in, ∂L/∂y_in, ∂L/∂θ₁, ∂L/∂θ₂, ...]
                grads = torch.autograd.grad(
                    outputs=[p_out, x_out, q_out, y_out],  # What we differentiate
                    inputs=[p_in, x_in, q_in, y_in] + list(theta),  # W.r.t. what
                    grad_outputs=[adj_p, adj_x, adj_q, adj_y],  # Incoming gradients (adjoint variables)
                    allow_unused=True,      # Some parameters might not affect output
                    retain_graph=False      # Don't need graph after this (memory efficient)
                )

                # Extract state adjoints (first 4 gradients)
                # These become the adjoints for the PREVIOUS timestep
                adj_p = grads[0] if grads[0] is not None else torch.zeros_like(adj_p)
                adj_x = grads[1] if grads[1] is not None else torch.zeros_like(adj_x)
                adj_q = grads[2] if grads[2] is not None else torch.zeros_like(adj_q)
                adj_y = grads[3] if grads[3] is not None else torch.zeros_like(adj_y)

                # Accumulate parameter gradients (remaining gradients)
                # Sum contributions from all timesteps: ∂L/∂θ = Σ_t (∂f_t/∂θ)^T λ_t
                for k, grad in enumerate(grads[4:]):
                    if grad is not None:
                        adj_theta[k] = adj_theta[k] + grad

        # ====================================================================
        # Finalize and return gradients
        # ====================================================================
        # After reversing through all operations, adj_p and adj_q now contain
        # the gradients w.r.t. the INITIAL state z₀ = [p₀, q₀]
        adj_z0 = torch.cat([adj_p, adj_q], dim=-1)

        # Return tuple matching forward pass signature:
        # (grad_z0, grad_t_range, grad_dynamics, grad_θ₁, grad_θ₂, ...)
        # None for non-tensor inputs (t_range, dynamics module itself)
        # We use * in front of adj_theta to unpack it
        return (adj_z0, None, None, *adj_theta)


class ODELayer(nn.Module):
    """
    Neural ODE layer with Yoshida-4 integrator.

    This is a PyTorch module that can be used like any other layer in a neural network.
    It integrates an ODE defined by a neural network ('dynamics') from t_start to t_end.

    Example usage:
        >>> dynamics = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 4))
        >>> ode_layer = ODELayer(dynamics, t_start=0.0, t_end=1.0, granularity=25)
        >>> z0 = torch.randn(batch_size, 4)  # Initial condition
        >>> zT = ode_layer(z0)  # Integrate to final time
        >>> loss = ((zT - target)**2).sum()
        >>> loss.backward()  # Gradients computed via adjoint method!

    Attributes:
        dynamics: Neural network defining dz/dt = f_θ(z, t)
        t_start: Initial time
        t_end: Final time
        granularity: Number of time steps (more = more accurate but slower)
        t_range: Tensor of time discretization points
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the ODE layer.
        """
        super().__init__()

    def forward(self, input: torch.Tensor, t_range: torch.Tensor, dynamics: nn.Module) -> torch.Tensor:
        """
        Forward pass: integrate the ODE from t_start to t_end.

        Args:
            input: Initial condition z(t_start) with shape [batch_size, state_dim]
                  For Hamiltonian systems: state_dim should be even (half for p, half for q)

        Returns:
            output: Final state z(t_end) with same shape as input
        """
        # Apply the custom autograd function
        # This calls ODELayerFunc.forward and sets up ODELayerFunc.backward for gradients
        return ODELayerFunc.apply(
            input,
            t_range,
            dynamics,
            *dynamics.parameters()  # Pass all trainable parameters
        )


class ROGeoHNN(nn.Module):
    """
    Geometric Hamiltonian Neural Network.
    Models Hamiltonian dynamics with learned kinetic and potential energies.

    Args:
        n_dof: Number of degrees of freedom
        kinetic_width: Hidden layer width for kinetic energy network
        kinetic_depth: Number of hidden layers for kinetic energy
        potential_width: Hidden layer width for potential energy network
        potential_depth: Number of hidden layers for potential energy
        parametrized_mass: If True, use parametrized exponential map
        activation: Activation function name
        device: Device for computations
    """

    def __init__(
        self,
        n_dof: int,
        r_dof: int,
        kinetic_dim: int = 64,
        kinetic_layers: int = 3,
        potential_dim: int = 64,
        potential_layers: int = 3,
        parametrized_mass: bool = False,
        use_spd_network: bool = True,
        spd_hidden_dims: Optional[list] = None,
        spd_network_scaling = True,
        ae_layer_dims: Optional[list] = None,
        activation: str = 'softplus',
        device: str = 'cpu'
    ):
        super().__init__()

        self.n_dof = n_dof
        self.r_dof = r_dof
        self.m = (r_dof * (r_dof + 1)) // 2
        self.device = device

        # should be filtered after
        self.ode_delta_t = -1.0
        self.time_vec = None
        self.delta_time_vec = None
        self.ode_time_vec = None
        self.tau_in_vec = None
        self.last_saved_t = -1.0


        # Build kinetic energy network: q -> M(q)
        kinetic_layers = []

        # Input layer
        kinetic_layers.append(EuclideanLayer(n_dof, kinetic_dim, activation, device))

        # Hidden layers
        for _ in range(kinetic_layers - 1):
            kinetic_layers.append(EuclideanLayer(kinetic_dim, kinetic_dim, activation, device))

        # Output to vectorized symmetric matrix
        kinetic_layers.append(EuclideanLayer(kinetic_dim, self.m, 'linear', device))

        # Exponential map to SPD
        kinetic_layers.append(SPDExpLayer(n_dof, parametrized_mass, device))

        # Optionally, we can apply an SPD network to the mass matrix
        if use_spd_network:
            kinetic_layers.append(
                SPDNetwork(
                    n_dof=n_dof,
                    hidden_dims=spd_hidden_dims,
                    use_scaling=spd_network_scaling,
                    device=device
                )
            )

        self.kinetic_net = nn.Sequential(*kinetic_layers)

        # Build potential energy network: q -> V(q)
        potential_layers = []

        # Input layer
        potential_layers.append(EuclideanLayer(n_dof, potential_dim, activation, device))

        # Hidden layers
        for _ in range(potential_layers - 1):
            potential_layers.append(EuclideanLayer(potential_dim, potential_dim, activation, device))

        # Output scalar potential (positive via softplus)
        potential_layers.append(EuclideanLayer(potential_dim, 1, 'softplus', device))

        self.potential_net = nn.Sequential(*potential_layers)

        # Initialize weights
        self._initialize_weights()

        # Build the autoencoder for high-dimension
        self.autoencoder = ConstrainedAE(ae_layer_dims)
        self.ode_layer = ODELayer()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, EuclideanLayer):
                gain = np.sqrt(2.0) if module.activation != nn.Identity() else 1.0
                nn.init.xavier_normal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def compute_hamiltonian(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Hamiltonian H = T(q,p) + V(q).

        Args:
            q: Positions of shape (batch, n_dof)
            p: Momenta of shape (batch, n_dof)

        Returns:
            H: Total energy
            T: Kinetic energy
            V: Potential energy
            M: Inverse mass matrix (SPD)
        """
        # Compute inverse mass matrix M(q)
        M = self.kinetic_net(q)

        # Kinetic energy: T = 0.5 * p^T @ M @ p
        T = 0.5 * torch.bmm(
            p.unsqueeze(1),
            torch.bmm(M, p.unsqueeze(2))
        ).squeeze(-1).squeeze(-1)

        # Potential energy: V = V(q)
        V = self.potential_net(q).squeeze(-1)

        # Total energy
        H = T + V

        return H


    def set_solver_times(self, t):
        """
        Set up time grids for ODE integration.

        Args:
            t: observation times (batch_size, num_timepoints)
        """
        t = t.to(self.device)

        # Normalized time vector starting from 0
        self.time_vec = (t - t[:, 0].unsqueeze(-1))[0].squeeze().to(self.device)

        # Average timestep
        self.delta_time_vec = float(torch.mean(self.time_vec[1:] - self.time_vec[:-1]))

        # Total time horizon
        time_horizon = self.time_vec[-1].item()

        # Set ODE timestep
        if self.ode_delta_t <= 0:
            self.ode_delta_t = time_horizon / (t.shape[1] - 1)

        # Create fine-grained ODE time grid
        num_steps = int(np.ceil(time_horizon / self.ode_delta_t)) + 2
        self.ode_time_vec = torch.linspace(0, time_horizon + self.ode_delta_t, num_steps).to(self.device)

    def get_x_hat_at_t(self, x_hat_at_ode):
        """Extract predictions at observation times from ODE solution"""
        diff = torch.abs(self.ode_time_vec[:, None] - self.time_vec[None, :])
        mask = torch.any(diff <= 1e-6, dim=1)
        x_hat = x_hat_at_ode[:, mask, :]
        return x_hat

    def forward(self, state: torch.Tensor, actuation: torch.Tensor) -> torch.Tensor:

        # Enable gradient computation
        state = state.requires_grad_(True)

        # Split into position and momentum
        q, p = torch.split(state, self.n_dof, dim=-1)

        # Number of time steps
        n_steps = actuation.shape[1] # (batch, time steps, dimension)

        # Encode the coordinates to latent space using cotangent left
        q_check, p_check = self.autoencoder.encoder_phase(q.view(-1, self.n_dof),
                                                          p.view(-1, self.n_dof))
        q_check = q.view(-1, n_steps, self.r_dof)
        p_check = q.view(-1, n_steps, self.r_dof)
        state_check = torch.cat((q_check, p_check), dim=-1)

        # Predict latent dynamics
        self.tau_in_vec = actuation
        self.last_saved_t = -1.0

        # Integrate using ODE solver
        out = self.ode_layer(state_check[:, 0], self.ode_time_vec, self.ode_fun)
        state_check_pred_at_ode = torch.transpose(out, 0, 1)  # (batch, time, state)
        state_check_pred = self.get_x_hat_at_t(state_check_pred_at_ode)

        # Decoding the latent dynamics
        q_check_pred, p_check_pred = torch.split(state_check_pred,
                                                 self.r_dof,
                                                 dim=-1)

        q_check_hat, p_check_hat =  self.autoencoder.decoder_phase(q_check_pred.view(-1, self.r_dof),
                                                          p_check_pred.view(-1, self.r_dof))

        q_pred_hat = q_pred_hat.view(-1, n_steps, self.n_dof)
        p_pred_hat = p_pred_hat.view(-1, n_steps, self.n_dof)
        state_pred_hat = torch.cat((q_pred_hat, p_pred_hat), dim=-1)

        # Autoencoder reconstruction only (for training)
        q_hat, p_hat = self.autoencoder.decoder_phase(
            q_check.view(-1, self.r_dof),
            p_check.view(-1, self.r_dof)
        )
        q_hat = q_hat.view(-1, n_steps, self.n_dof)
        p_hat = p_hat.view(-1, n_steps, self.n_dof)
        state_hat = torch.cat((q_hat, p_hat), dim=-1)

        return state_check, state_check_pred, state_pred_hat, state_hat


    def ode_fun(self, t, x):
        """ODE function for torchdiffeq"""
        # Determine if we should save outputs at this timestep
        if torch.any(torch.abs(self.time_vec - t) < 1e-6) and t != self.last_saved_t:
            self.last_saved_t = t

        # Get interpolated input
        q, _ = torch.split(x, split_size_or_sections=int(self.n_dof), dim=-1)
        tau = self.input_interpolation(t, q)

        # Compute derivative
        dx = self.fw_prediction(x, tau=tau, save_outputs=save_out)
        return dx

    def fw_prediction(self, x, tau):
        """
        Compute time derivatives according to Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q + τ

        Args:
            x: state [q, p]
            tau: external forces

        Returns:
            dx: time derivatives [dq/dt, dp/dt]
        """
        x = x.requires_grad_(True)
        q, p = torch.split(x, split_size_or_sections=int(self.n_dof), dim=-1)

        # Compute Hamiltonian
        H = self.compute_hamiltonian(q, p)

        # Compute gradients
        dH = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
        dH_dq, dH_dp = dH[0], dH[1]

        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq + tau

        # Combine
        dx = torch.cat((dq_dt, dp_dt), dim=-1)

        return dx

    def input_interpolation(self, t_interp, q):
        """Interpolate external forces at time t_interp"""

        # we look for the time step correspomding to the time of interpolatiopon
        ind_interp = torch.argmin(torch.abs(self.time_vec - t_interp)).item()
        tau_interp = self.tau_in_vec[:, ind_interp, :]
        tau_ctrl = self.autoencoder.reduce_forces(q, tau_interp)
        return tau_ctrl

    def numpy_dynamics(self, t, state):
        """
        Wrapper to compute dynamics for a single state in numpy format.
        Used for midpoint integration.

        Args:
            t: Time (unused for autonomous systems)
            state: State vector as numpy array

        Returns:
            Time derivative as numpy array
        """
        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Compute dynamics (forward pass needs autograd)
        dx = self.forward(state_tensor)

        # Convert back to numpy
        return dx.squeeze(0).detach().numpy()
