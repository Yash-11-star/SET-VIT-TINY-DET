import torch
import torch.nn as nn

class APIAuxPath(nn.Module):
    """
    Builds an auxiliary forward on (features + epsilon_adv),
    where epsilon_adv is computed from gradients of the cls loss.
    """
    def __init__(self, head, rho=0.5, lambda_cls=1.0, lambda_reg=0.0):
        super().__init__()
        self.head = head        # share weights with main head or a lightweight copy
        self.rho = rho
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg

    @torch.no_grad()
    def _l2_normalize(self, grad_list):
        normed = []
        for g in grad_list:
            g2 = g.clone()
            denom = g2.flatten(1).norm(dim=1).view(-1,1,1,1) + 1e-8
            normed.append(g2 / denom)
        return normed

    def forward(self, feats, criterion, targets):
        """
        feats: list of multi-scale features (requires_grad=True)
        criterion: detection loss fn returning dict with 'loss_cls' (and 'loss_reg' optional)
        targets: list of dicts (COCO-style)
        """
        # 1) forward to get gradients wrt features
        for f in feats:
            f.requires_grad_(True)

        outputs = self.head(feats)
        loss_dict = criterion(outputs, targets)
        loss = self.lambda_cls * loss_dict.get('loss_cls', 0.0) + \
               self.lambda_reg * loss_dict.get('loss_reg', 0.0)

        grads = torch.autograd.grad(loss, feats, retain_graph=False, create_graph=False, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(f) for g, f in zip(grads, feats)]

        with torch.no_grad():
            grads = self._l2_normalize(grads)
            eps = [self.rho * g for g in grads]
            feats_adv = [f + e for f, e in zip(feats, eps)]

        # 2) auxiliary forward on adversarially-perturbed features
        aux_outputs = self.head(feats_adv)
        aux_loss_dict = criterion(aux_outputs, targets)
        return aux_loss_dict  # to be weighted as aux branch
