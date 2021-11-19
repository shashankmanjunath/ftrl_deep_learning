from torch.optim import Optimizer

import numpy as np
import torch


class MADGRADOptimizer(Optimizer):
    def __init__(self, params, lr=1, momentum=0.9, weight_decay=1e-4, eps=1e-6):
        if lr < 0:
            raise ValueError("Learning rate must be greater than 0")
        if momentum < 0:
            raise ValueError("Momentum must be greater than 0")
        if weight_decay < 0:
            raise ValueError("Weight decay must be greater than 0")
        if eps < 0:
            raise ValueError("Epsilon must be greater than 0")
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay,"eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("MADGRAD does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # Initializing s_k value for s_{-1}, x_0 parameter, and step counter
                if len(state) == 0:
                    state["s"] = torch.zeros(grad.shape, device=p.device)
                    state["v"] = torch.zeros(grad.shape, device=p.device)
                    state["z"] = p.data
                    state["step"] = 0

                # Extracting learning rate, step count, weight decay, and eps value
                lr = group["lr"]
                k = state["step"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Calculating stepsize (lambda_k) parameter
                lambda_k = lr * np.sqrt(k + 1)

                # Update sum of gradients (s_{k+1})
                state["s"].add_(lambda_k * grad)

                # Updaing v_{k+1}
                state["v"].add_(lambda_k * (grad * grad))

                # Update dual averaging iterate z_{k+1}
                iter_val = 1 / (torch.pow(state["v"], 1/3) + eps)
                z_k_1 = state["z"] - (iter_val * state["s"])

                # Extracting c_{k+1} parameter
                c_k_1 = 1 - group["momentum"]

                # Update averaged iterate
                p.mul_(1 - c_k_1).add_(c_k_1 * z_k_1)

                # Update step counter
                state["step"] += 1
