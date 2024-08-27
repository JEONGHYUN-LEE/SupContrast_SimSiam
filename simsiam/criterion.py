import torch
from torch import nn


class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2



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
    
    