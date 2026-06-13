import torch
import torch.nn as nn

class LIF(nn.Module):

    def __init__(self, in_dim, out_dim, tau=20.0, vth=1.0, vreset=0.0):
        super().__init__()

        # 重み（シナプス）
        self.W = nn.Parameter(
            torch.randn(in_dim, out_dim) * 0.1
        )

        # パラメータ
        self.tau = tau
        self.vth = vth
        self.vreset = vreset

        # 状態
        self.V = None   # membrane potential
        self.S = None   # spike

    def reset_state(self, batch_size=1):
        self.V = torch.zeros(batch_size, self.W.shape[1])
        self.S = torch.zeros_like(self.V)

    def forward(self, X):

        I = X @ self.W

        dV = (-self.V + I) / self.tau#tauは膜電位の減衰の速さを制御するパラメータ
        self.V = self.V + dV

        self.S = (self.V >= self.vth).float()
        self.V = self.V * (1 - self.S) + self.vreset * self.S

        return self.S