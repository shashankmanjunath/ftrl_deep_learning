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
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("MADGRAD does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # Initializing s_k value for s_{-1}, x_0 parameter, and step counter
                if len(state) == 0:
                    state["s"] = torch.zeros(grad.shape, device=p.device).detach()
                    state["v"] = torch.zeros(grad.shape, device=p.device).detach()
                    state["z"] = torch.clone(p.data).detach()
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
                state["s"].add_(grad, alpha=lambda_k)

                # Updaing v_{k+1}
                state["v"].addcmul_(grad, grad, value=lambda_k)

                # Update dual averaging iterate z_{k+1}
                iter_val = state["v"].pow(1/3).add_(eps)
                z_k_1 = state["z"].addcdiv(state["s"], iter_val, value=-1)

                # Extracting c_{k+1} parameter
                c_k_1 = 1 - group["momentum"]

                # Update averaged iterate
                p.data.mul_(1 - c_k_1).add_(z_k_1, alpha=c_k_1)

                # Update step counter
                state["step"] += 1
