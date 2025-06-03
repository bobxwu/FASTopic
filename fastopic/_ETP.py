import torch
from torch import nn
import torch.nn.functional as F
from ._model_utils import pairwise_euclidean_distance


class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha, init_a_dist=None, init_b_dist=None, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        # Sinkhorn's algorithm in log-space
        M = pairwise_euclidean_distance(x, y)
        device = M.device

        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        # LOG-DOMAIN STABILIZATION: Work in log-space to avoid extreme scaling values
        log_a = torch.log(a + 1e-30)  # Shape: (n, 1)
        log_b = torch.log(b + 1e-30)  # Shape: (m, 1)

        # Initialize log-domain scaling vectors
        log_u = torch.zeros_like(log_a)  # Shape: (n, 1)
        log_v = torch.zeros_like(log_b)  # Shape: (m, 1)

        log_K = -M * self.sinkhorn_alpha  # Shape: (n, m)

        # Sinkhorn iterations in log-domain
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            # LOG-DOMAIN UPDATE: log(v) = log(b) - logsumexp(log(K^T) + log(u), dim=0)
            # This is equivalent to: v = b / (K^T @ u) but numerically stable
            log_Ku = log_K.T + log_u.T  # Shape: (m, n)
            log_v = log_b - torch.logsumexp(log_Ku, dim=1).unsqueeze(1)  # Shape: (m, 1)

            # LOG-DOMAIN UPDATE: log(u) = log(a) - logsumexp(log(K) + log(v^T), dim=1)
            # This is equivalent to: u = a / (K @ v) but numerically stable
            log_Kv = log_K + log_v.T  # Shape: (n, m)
            log_u = log_a - torch.logsumexp(log_Kv, dim=1).unsqueeze(1)  # Shape: (n, 1)

            cpt += 1
            if cpt % 50 == 1:
                # Absorb current scalings
                log_K = log_K + log_u + log_v.T
                # Reset scalings
                log_u = torch.zeros_like(log_a)
                log_v = torch.zeros_like(log_b)
                
                
                err = self.check_convergence(log_K, log_u, log_v, a, b)

        # Convert final results back to linear domain for compatibility
        u = torch.exp(log_u)  # Shape: (n, 1)
        v = torch.exp(log_v)  # Shape: (m, 1)
        K = torch.exp(log_K)  # Shape: (n, m)

        # Compute transport plan: P = diag(u) K diag(v)
        transp = u * (K * v.T)  # Shape: (n, m)

        # Compute transport cost: <P, M>
        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp

    def check_convergence(self, log_K, log_u, log_v, a, b):
        """
        Check convergence by verifying the marginal constraints using absolute error.
        The transport plan is P = diag(u) @ K @ diag(v)
        We need: P @ 1 = a and P^T @ 1 = b
        """
        with torch.no_grad():
            # Row constraint: (u ⊙ (K @ v)) should equal a
            # In log domain: log(u) + log(K @ v) should equal log(a)
            log_Kv = log_K + log_v.T  # Shape: (n, m)
            log_Kv_sum = torch.logsumexp(
                log_Kv, dim=1, keepdim=True
            )  # log(K @ v), Shape: (n, 1)
            log_row_sums = log_u + log_Kv_sum  # log(u ⊙ (K @ v)), Shape: (n, 1)
            row_sums = torch.exp(log_row_sums)  # Shape: (n, 1)

            # Column constraint: (v ⊙ (K^T @ u)) should equal b
            # In log domain: log(v) + log(K^T @ u) should equal log(b)
            log_Ku = log_K.T + log_u.T  # Shape: (m, n)
            log_Ku_sum = torch.logsumexp(
                log_Ku, dim=1, keepdim=True
            )  # log(K^T @ u), Shape: (m, 1)
            log_col_sums = log_v + log_Ku_sum  # log(v ⊙ (K^T @ u)), Shape: (m, 1)
            col_sums = torch.exp(log_col_sums)  # Shape: (m, 1)

            # Compute absolute errors
            row_abs_err = torch.abs(row_sums - a)
            max_row_err = torch.max(row_abs_err)

            col_abs_err = torch.abs(col_sums - b)
            max_col_err = torch.max(col_abs_err)

            return max(max_row_err.item(), max_col_err.item())
