"""GeoHNN. Code from Geometric_Hamiltonian_Neural_Networks_GeoHNNs.ipynb."""
import enum
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt as go
import geoopt.tensor as gt
from geoopt import linalg
from geoopt.manifolds.base import Manifold
from torchdiffeq import odeint

__all__ = ["SPD"]


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

class SPDMetricType(enum.Enum):
    AIM = "AIM"


class SPD(Manifold):
    """
    Manifold of SPD matrices with affine-invariant metric.
    This implementation is inspired by https://github.com/geoopt/geoopt.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SPD"
    ndim = 2
    reversible = False

    def __init__(self, default_metric: Union[str, SPDMetricType] = "AIM"):
        super().__init__()
        self.default_metric = SPDMetricType(default_metric)

    def _affine_invariant_metric(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        c = torch.linalg.cholesky(x)
        c_inv = torch.linalg.inv(c)
        e, V = torch.linalg.eigh(c_inv @ y @ c_inv.transpose(-1, -2), 'L')
        eigs = torch.log(e) ** 2
        return torch.sqrt(torch.sum(eigs, dim=-1) + 1e-15)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e, _ = torch.linalg.eigh(x, "U")
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = linalg.sym(x)
        return linalg.sym_funcm(symx, torch.abs)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return linalg.sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x.transpose(-1, -2)

    _dist_metric = {
        SPDMetricType.AIM: _affine_invariant_metric,
    }

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim=False,) -> torch.Tensor:
        return self._dist_metric[self.default_metric](self, x, y)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, keepdim=False) -> torch.Tensor:
        if v is None:
            v = u
        inv_x = linalg.sym_invm(x)
        ret = linalg.trace(inv_x @ u @ inv_x @ v)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = linalg.sym_invm(x)
        return linalg.sym(x + u + 0.5 * u @ inv_x @ u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p_inv_tv = torch.linalg.solve(x, u)
        return x @ torch.matrix_exp(p_inv_tv)

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        c = torch.linalg.cholesky(x, upper=False)
        c_inv = torch.inverse(c)
        mat = c_inv @ u @ c_inv.transpose(-1, -2)
        return c @ self.logm(mat) @ c.transpose(-1, -2)

    def extra_repr(self) -> str:
        return "default_metric={}".format(self.default_metric)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        X_inv = torch.inverse(x)
        A = linalg.sym_inv_sqrtm2(y @ X_inv)[1]
        return A @ v @ A.transpose(-1, -2)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        tens = 0.5 * torch.randn(*size, dtype=dtype, device=device)
        tens = linalg.sym(tens)
        tens = linalg.sym_funcm(tens, torch.exp)
        return tens

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
    ) -> torch.Tensor:
        return torch.diag_embed(torch.ones(*size[:-1], dtype=dtype, device=device))

    def logm(self, X):
        e, V = torch.linalg.eigh(X, 'L')
        U = V @ torch.diag_embed(torch.log(e)) @ V.transpose(-1, -2)
        return U

    def sym(self, x: torch.Tensor):
        return 0.5 * (x.transpose(-1, -2) + x)

    def sym_sqrt_inv(self, x: torch.Tensor):
        e, v = torch.linalg.eigh(x, "L")
        sqrt_e = torch.sqrt(e)
        inv_sqrt_e = torch.reciprocal(sqrt_e)
        return (
            v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
            v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
        )

def matrix_vec_diff(M, x):
    # autodiff square matrix M wrt vector x: M is nxn matrix, x is dim m
    bs = M.shape[0]
    n = M.shape[1]
    m = x.shape[1]

    M_vec = M.view(bs, n * n)
    dMdx = torch.zeros(bs, n * n, m, device=M.device)
    ones_grad = torch.ones_like(M_vec[:, 0], device=M.device)

    for i in range(0, n*n):
        dMdx[:, i] = torch.autograd.grad(M_vec[:, i], x, grad_outputs=ones_grad, retain_graph=True,
                                         create_graph=True, allow_unused=True)[0]

    dMdx = dMdx.view(bs, n, n, m)

    return dMdx


class SPDExpLayer(nn.Module):
    """
    converts an m=n(n+1)/2 dimensional vector input to symmetric matrix, applies exponential map to obtain SPD matrix.
    """
    def __init__(self, n_dof, parametrized_expmap=False, device=None):
        super().__init__()

        # here we initialize a SPD layer (semi positive definite layer), which we will use for finding the inertia matrix
        # it has the following parameters
        # device - run on CPU or GPU
        # n_dof - this represent the number of degrees of freedom
        # m - this represent the dimensionality of symmetric matrices (dimension of the input vector to be turned into a SPD matrix)
        # also the exponential map used to obtain SPD matrix can be parameterized or no (check the paper)

        self.device = device
        self.n_dof = n_dof
        self.m = int((self.n_dof ** 2 + self.n_dof) / 2)


        # init geom stuff
        # here we start by generating our manifold (SPD manifold), we define it as follows:
        # SPD = {S in Mn such that S = S^T and S >= 0}
        # it can be seen as a lie group
        # tangent space is the symmetric matrices
        self.man = SPD()

        # we initialize the matrix that would store the mass inertia (n_dof * n_dof matrix, moved to device)
        self.zero_mat = torch.zeros(self.n_dof, self.n_dof).to(device)

        # Returns the indices of the lower triangular part of a row-by- col matrix in a 2-by-N Tensor,
        # where the first row contains row coordinates of all indices and the second row contains column coordinates.
        # Indices are ordered based on rows and then columns.

        self.tril_indices = torch.tril_indices(self.n_dof, self.n_dof, offset=-1, device=device)

        # Returns the indices of the diagonal (0, 1, 2, .., n_dof-1)
        self.diag_indices = torch.arange(self.n_dof, device=device)

        # we have two different versions of the exponential map (False or True)
        self.parametrized_expmap = parametrized_expmap
        if self.parametrized_expmap:
            # If using the parametrized approach, creates a learnable parameter on the SPD manifold as the reference point for the exponential map.
            self.exp_point = gt.ManifoldParameter(self.man.origin(self.n_dof, self.n_dof, device=device), manifold=self.man, requires_grad=True)
        else:
            # Otherwise, sets the reference point to the identity matrix (standard approach).
            self.exp_point = torch.eye(self.n_dof).to(device)
            # Creates a tensor of zeros that will be used to construct the Jacobian matrix.
            self.der_zeros = torch.zeros(self.n_dof, self.n_dof, int((self.n_dof ** 2 + self.n_dof) / 2), dtype=torch.float32, device=self.device)

            # derivative of rearranging
             # Initializes the Jacobian matrix with zeros.
            self.dUdx = self.der_zeros

            # derivatives across the diag is the identity
            self.dUdx[self.diag_indices, self.diag_indices, self.diag_indices] = 1

            # Sets the derivatives for off-diagonal elements to 1 in the Jacobian (in both lower and upper triangular positions to maintain symmetry).
            self.dUdx[self.tril_indices[1], self.tril_indices[0], torch.arange(self.m - self.n_dof) + self.n_dof] = 1
            self.dUdx[self.tril_indices[0], self.tril_indices[1], torch.arange(self.m - self.n_dof) + self.n_dof] = 1

            # Reshapes the Jacobian to (1, n_dof², m) for batch operations in the forward pass.
            self.dUdx = self.dUdx.reshape(self.n_dof**2, self.m).unsqueeze(0)


    def forward(self, x, dx):
        # forward pass that takes as input a vector x (used to parameterize the sym matrix) and a perturbation dx (small displacement)
        # first, we get our symmetric matrix
        U = self.vec2symmat(x)


        # then, we have two cases
        if self.parametrized_expmap:
            # if parameterized, we apply the exponential map at the parameterized origin to get the corresponding SPD matrix
            out = self.man.expmap(self.exp_point, U)

            # here, we compute the jacobian using numerical differentiation
            md = matrix_vec_diff(out, x)

            # Computes the output derivative by matrix multiplication with the input derivative.
            dout = md @ dx.unsqueeze(1)

        else:
            # otherwise, if the reference point is the identity matrix, we only need to comoute the matrix exponential directlu
            out = torch.matrix_exp(U)

            # now, we need to compute the jacobian
            # we compute the eigenvalues (e) and eigenvectors (v) of U. This is important because matrix exponentiation can be simplified when using the eigendecomposition of the matrix.
            e, v = torch.linalg.eigh(U, 'L')

            # we compute the Kronecker product of the eigenvectors v for batch operations.
            P = self.batch_kron(v, v)

            # then we compute the divided difference, which is used for handling the matrix exponential computation.
            R = self.comp_R(e)

            # then we compute the Jacobian matrix of the matrix exponential.
            D = torch.diag_embed(R.transpose(-1, -2).reshape(R.shape[0], -1))
            der = P @ D @ P.transpose(-1, -2)

            # here we compute the derivatives of U w.r.t its vectorized form
            der = der @ self.dUdx

            # finally , we reshape before final operation
            der = der.view(-1, self.n_dof, self.n_dof, self.m).transpose(1, 2)

            # last, we compute the output derivative w.r.t input
            dout = der @ dx.unsqueeze(1)

        return out, dout

    def batch_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # here we compute the koronecker product between two matrices A, B, the resulting matrix is N^2 * M^2

        b = A.shape[0]
        kron_prod = A.view(b, self.n_dof, 1, self.n_dof, 1) * B.view(b, 1, self.n_dof, 1, self.n_dof)

        return kron_prod.view(b, self.n_dof**2, self.n_dof**2)


    def comp_R(self, b):
        # this function is used to computes the R matrix used in the derivative of the matrix exponential.

        B_diff = b.unsqueeze(2) - b.unsqueeze(1)
        exp_b = torch.exp(b)
        exp_diff = exp_b.unsqueeze(2) - exp_b.unsqueeze(1)
        non_zero_mask = B_diff != 0
        R = torch.zeros_like(B_diff)
        R[non_zero_mask] = exp_diff[non_zero_mask] / B_diff[non_zero_mask]
        R[:, self.diag_indices, self.diag_indices] = exp_b

        return R

    def vec2symmat(self, vec):
        # this function is used to convert vector to a symmetric matrix
        # we start by getting the batch size denoted bs
        bs = vec.shape[0]

        # we have two possibilities
        # case 1 -  vec has a shape (bs, m)
        if len(vec.shape) == 2:
            # in this case we initialize the symmetric matrix using a repeated 0 matrix
            # .repeat() creates a new tensor with physically separate memory for each copy. So, modifying one matrix in the batch does not affect the others.
            symmat = self.zero_mat.repeat(bs, 1, 1)
        elif len(vec.shape) == 3:
            # case 2 - the vec has a shape (bs, m, T)
            # T here can represent different time steps (time series), frames (videos), channels (pictures)
            symmat = self.zero_mat.view(1, self.n_dof, self.n_dof, 1).repeat(bs, 1, 1, vec.shape[2])

        # the first n degrees of freedom represent the diagonal elements
        symmat[:, self.diag_indices, self.diag_indices] = vec[:, :self.n_dof]

        # the rest represent the triangular indices orders in the same manner
        symmat[:, self.tril_indices[1], self.tril_indices[0]] = vec[:, self.n_dof:]
        symmat[:, self.tril_indices[0], self.tril_indices[1]] = vec[:, self.n_dof:]

        return symmat



class EuclideanLayer(nn.Module):
    """
    Standard Euclidean fully connected layer
    n_in represent the number of input features
    n_width represent of output features
    device is either cpu or gpu
    """

    def __init__(self, n_in, n_width, activation="ReLu", device="cpu"):
        super().__init__()

        # we start by initializing our manifold, which is euclidean in this case.
        self.man = go.manifolds.Euclidean(ndim=0)

        # out manifold, is defined through a linear layer comprising Wx + b (parameterized by (W, b))
        self.weight = gt.ManifoldParameter(torch.Tensor(n_width, n_in).to(device), manifold=self.man, requires_grad=True)
        self.bias = gt.ManifoldParameter(torch.Tensor(n_width).to(device), manifold=self.man, requires_grad=True)

        # activation function used to add or no non-linearities
        self.activation = activation

        # it can be either a softplus , this is a soft diffeentialble ReLU
        if self.activation == "SoftPlus":
            self.softplus_beta = int(1.0)
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.dg = self.d_Softplus
        elif self.activation == "Linear":
            self.g = nn.Identity()
            self.dg = self.d_Linear

    def forward(self, q, dq):
        # we compute the output of the layer, and its corresponding derivative
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        dout = (self.dg(a).unsqueeze(-1) * self.weight) @ dq
        return out, dout

    def forward_nodiff(self, q):
        # if we don't need the derivative (no backprop)
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        return out

    def d_Linear(self, x):
        # derivative of a linear activation x --> x
        return torch.full_like(x, 1)

    def d_Softplus(self, x):
        # derivative of a linear activation x --> softplus(x) = log(1 + exp(x))
        # we clamp to avoid saturation
        clamped_x = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self.softplus_beta * clamped_x)
        out = exp_x / (exp_x + 1.0)
        return out


# Geometric HNN
class GeoHNN(nn.Module):
    """
    Geometric HNN, with kinetic energy network on SPD-tangent space.
    """
    def __init__(self, n_dof, **kwargs):
        super().__init__()
        # DoF of system

        # Stores the degrees of freedom of the system (the dimensionality).
        self.n_dof = n_dof

        # Flag indicating whether the model works in a latent (reduced) space.
        self.latent_flag = False

        # this is to be restudied # Retrieves the 'detach' parameter from kwargs with a default value of True. This likely controls whether to detach computational graphs in certain operations.
        self.detach = kwargs.get('detach', True)

       # this is to be restudied #  Gets the number of Jacobian entries from kwargs with a default of 0.
        self.n_jac_entries = kwargs.get('n_jac_entries', 0)

        # euclidean hidden layers: kinetic & potential energy NN dimensions
        ## Kinetic Energy
        # width represents the number of output features
        self.n_width_T_eucl = kwargs.get('n_width_T_eucl', 64)
        # depth represents the number of layers
        self.n_depth_T_eucl = kwargs.get("n_depth_T_eucl", 2)

        ## Potential Energy
        # width represents the number of output features
        self.n_width_V = kwargs.get('n_width_V_eucl', 64)
        # depth represents the number of layers
        self.n_depth_V = kwargs.get("n_depth_V_eucl", 2)

        # here, we specify the activation, by default it is the softplus
        self.activation = kwargs.get("activation", "SoftPlus")

        # Gets the gain parameter for weight initialization with a default of √2.
        self.gain_hidden = kwargs.get("gain_hidden", np.sqrt(2.))

        # if true, exp wrt a weight, if false it's wrt Identity
        self.do_exp_wrt_p = kwargs.get('exp_wrt_p', False)

        # for ode int
        # gets the ODE integration time step with a default of -1 (auto-determined).
        self.ode_delta_t = kwargs.get('ode_timestep', -1.)
        # gets the type of solver to use, bu default we use euler
        self.ode_solver = kwargs.get('ode_solver', 'euler')
        # we set the  control input vector to None.
        self.tau_in_vec = None
        # we set the time vector to None.
        self.time_vec = None

        # for computations (either use cpu or gpu)
        self.device = kwargs.get("device", "cpu")

        # identity matrix, reshaped for batch operations
        self.eye_mat = torch.eye(self.n_dof).reshape(1, self.n_dof, self.n_dof).to(self.device)

        # zero matrix, reshaped for batch operations
        self.zero_mat = torch.zeros(self.n_dof, self.n_dof).reshape(1, self.n_dof, self.n_dof).to(self.device)

        # SPD manifold dimension
        self.m = int((self.n_dof ** 2 + self.n_dof) / 2)

        # for storage, we store quantities throught the training
        self.out_T = torch.zeros(0, )                               # kinetic energy
        self.out_V = torch.zeros(0, )                               # potential energy
        self.out_M = torch.zeros(0, self.n_dof, self.n_dof)         # inertia matrix
        self.out_H = torch.zeros(0, )                               # inverse of M

        # here, we initialize the autoencoder that might be used to reduce order
        self.ae = None

        # function to inialize parameters for each layer
        def init_layer(layer, is_out=False):
            # if it is the output layer, use a gain = 1
            if is_out:
                torch.nn.init.xavier_normal_(layer.weight, 1)
            else:
                # else, the gain depends on the sign and the activation funciton
                if self.gain_hidden <= 0.0:
                    if layer.activation == "SoftPlus":
                        gain = np.sqrt(2. / (1 + np.pi ** 2 / 6))
                else:
                    gain = self.gain_hidden

                torch.nn.init.xavier_normal_(layer.weight, gain)
            # for the bias it is 0
            torch.nn.init.constant_(layer.bias, 0)

        # construct kinetic energy network layers
        self.layers_M = nn.ModuleList()

        # 1st part: normal euclidean network of desired dimensions
        self.layers_M.append(EuclideanLayer(self.n_dof, self.n_width_T_eucl, activation=self.activation, device=self.device))
        init_layer(self.layers_M[-1])
        for _ in range(1, self.n_depth_T_eucl):
            self.layers_M.append(
                EuclideanLayer(self.n_width_T_eucl, self.n_width_T_eucl, activation=self.activation, device=self.device))
            init_layer(self.layers_M[-1])
        self.layers_M.append(EuclideanLayer(self.n_width_T_eucl, self.m, activation="Linear", device=self.device))
        init_layer(self.layers_M[-1], is_out=True)

        # expmap-layer
        self.layers_M.append(SPDExpLayer(self.n_dof, parametrized_expmap=self.do_exp_wrt_p, device=self.device))

        # potential energy network
        self.layers_V = nn.ModuleList()
        self.layers_V.append(EuclideanLayer(self.n_dof, self.n_width_V, activation=self.activation, device=self.device))
        init_layer(self.layers_V[-1])
        for _ in range(1, self.n_depth_V):
            self.layers_V.append(
                EuclideanLayer(self.n_width_V, self.n_width_V, activation=self.activation,
                               device=self.device))
            init_layer(self.layers_V[-1])
        self.layers_V.append(EuclideanLayer(self.n_width_V, 1, activation="SoftPlus", device=self.device))
        init_layer(self.layers_V[-1], is_out=True)


    def compute_hamiltonian(self, q, p):
        """
            Hamiltonian works with position q and momentum p
            H = T(q,p) + V(q)
        """
        # First compute the inverse of the mass matrix M
        y, dy_dq = self.layers_M[0](q, self.eye_mat)
        for i in range(1, self.n_depth_T_eucl+1):
            y, dy_dq = self.layers_M[i](y, dy_dq)
        M, dMdq = self.layers_M[-1](y, dy_dq)

        # Compute kinetic energy T = 1/2 * p^T * M_inv * p (we predict the inverse so M is out inverse mass inertia)
        T = 0.5 * (p.view(-1, 1, self.n_dof) @ M @ p.view(-1, self.n_dof, 1)).view(-1)

        # Compute potential energy V
        V, der_V = self.layers_V[0](q, self.eye_mat)
        for i in range(1, len(self.layers_V)):
            V, der_V = self.layers_V[i](V, der_V)
        V = V.squeeze(-1)

        # Hamiltonian is H = T + V
        H = T + V

        return H, T, V, M

    def forward(self, x):
        """
        HNN forward pass
        x: combined state [q, p]
        do_ode: whether to use ODE solver
        """
        self.reset_outs_and_integrators()

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            q, p = torch.split(x, split_size_or_sections=int(self.n_dof), dim=-1)

            # Compute Hamiltonian
            H, T, V, M = self.compute_hamiltonian(q, p)

            dH = torch.autograd.grad(H.sum(), [q, p], create_graph=True)
            dH_dq, dH_dp = dH[0], dH[1]

            # Hamilton's equations
            dq_dt = dH_dp  # dq/dt = dH/dp
            dp_dt = -dH_dq  # dp/dt = -dH/dq

            # Combine derivatives
            dx = torch.cat((dq_dt, dp_dt), dim=-1)

            return dx

    def reset_outs_and_integrators(self):
        self.out_T = torch.zeros(0, )
        self.out_Vt = torch.zeros(0, )
        self.out_M = torch.zeros(0, self.n_dof, self.n_dof)
        self.out_H = torch.zeros(0, self.n_dof)

        return None