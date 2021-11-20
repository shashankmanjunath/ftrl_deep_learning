from torch.optim import Optimizer

import numpy as np
import torch


class MDAOptimizer(Optimizer):
    def __init__(self, params, lr=1, momentum=0.9, weight_decay=1e-4):
        if lr < 0:
            raise ValueError("Learning rate must be greater than 0")
        if momentum < 0:
            raise ValueError("Momentum must be greater than 0")
        if weight_decay < 0:
            raise ValueError("Weight decay must be greater than 0")
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("MDA does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # Initializing s_k value for s_{-1}, x_0 parameter, and step counter
                if len(state) == 0:
                    state["s"] = torch.zeros(grad.shape, device=p.device)
                    state["z"] = p.data
                    state["step"] = 0

                # Extracting learning rate, weight_deacy, and step count
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                k = state["step"]

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Extracting c_{k+1} parameter
                c_k_1 = 1 - group["momentum"]

                # Calculating scaling (beta) parameter
                beta_k = np.sqrt(k + 1)

                # Calculating stepsize (lambda_k) parameter
                lambda_k = lr * np.sqrt(k + 1)

                # Update sum of gradients
                state["s"].add_(lambda_k * grad)

                # Update dual averaging iterate z_{k+1}
                z_k_1 = state["z"] - (state["s"] / beta_k)

                # Update averaged iterate
                p.mul_(1 - c_k_1).add_(c_k_1 * z_k_1)

                # Update step counter
                state["step"] += 1
