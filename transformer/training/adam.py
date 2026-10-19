"""Dense Adam matching the pinned Fairseq epsilon/bias-correction convention."""
import math
import torch


class ReferenceAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(.9, .98), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ReferenceAdam requires dense gradients')
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state['step'] += 1
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                correction = math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                p.addcdiv_(m, v.sqrt().add_(group['eps']), value=-group['lr'] * correction)
        return loss
