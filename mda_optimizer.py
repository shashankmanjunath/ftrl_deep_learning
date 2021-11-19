from torch.optim import Optimizer

import numpy as np
import torch


class MDAOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4):
        if lr < 0:
            raise ValueError("Learning rate must be greater than 0")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("MDA does not support sparse gradients")

                grad = p.grad.data
                state = self.state[p]

                # Initializing s_k value for s_{-1}
                if len(state) == 0:
                    state["s"] = torch.zeros(p.grad.shape).to(p.device)
                    state["z"] = p.data
                    state["step"] = 0

                # Extracting learning rate and step count
                lr = group["lr"]
                k = state["step"]

                # Calculating scaling (beta) parameter
                beta_k = np.sqrt(k + 1)

                # Calculating stepsize (lambda_k) parameter
                lambda_k = lr * np.sqrt(k + 1)

                # Calculating c_{k+1} parameter
                c_k_1 = 0.1

                # Update sum of gradients
                state["s"].add_(lambda_k * grad)

                # Update dual averaging iterate z_{k+1}
                # TODO: Make this more efficient
                z_k_1 = state["z"] - state["s"] / beta_k

                # Update averaged iterate
                p.data.mul_(1 - c_k_1).add_(c_k_1 * z_k_1)

                # Update step counter
                state["step"] += 1
        pass
