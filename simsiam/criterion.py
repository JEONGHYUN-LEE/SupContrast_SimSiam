import torch
from torch import nn


class SupSimSiamLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, p, z, targets):
        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        mask_anchor_out = (1 - torch.eye(dot_product.shape[0])).to(self.device)
        mask_combined = mask_similar_class * mask_anchor_out

        dot_product_selected = dot_product * mask_combined
        return dot_product_selected[dot_product_selected.nonzero(as_tuple=True)].mean()
    
    