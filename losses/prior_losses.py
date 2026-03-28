import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B, D, H, W] (long)
    shape = (labels.size(0), num_classes) + tuple(labels.shape[1:])
    y = torch.zeros(shape, device=labels.device, dtype=torch.float32)
    return y.scatter_(1, labels.unsqueeze(1), 1.0)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # logits: [B, C, D, H, W], targets: [B, D, H, W] (long)
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    tgt = one_hot(targets, num_classes)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * tgt, dim=dims)
    union = torch.sum(probs + tgt, dim=dims)
    dice = (2. * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    return loss


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    # Multiclass focal loss with softmax
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)  # [B,C,D,H,W]
    probs = log_probs.exp()
    tgt = one_hot(targets, num_classes)       # [B,C,D,H,W]
    pt = (probs * tgt).clamp_min(1e-6)        # avoid log(0)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return loss.sum(dim=1).mean()


def total_variation_3d(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,D,H,W]
    tv_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
    tv_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
    return (tv_d + tv_h + tv_w) / 3.0


def physics_prior_loss(coupling: torch.Tensor) -> torch.Tensor:
    # coupling: [M, M], row-stochastic (softmax). Encourage symmetry and mild diagonal dominance.
    # Symmetry penalty
    sym = torch.mean(torch.abs(coupling - coupling.t()))
    # Diagonal dominance (encourage diagonal not too small)
    diag = torch.diag(coupling)
    diag_penalty = F.relu(0.3 - diag).mean()  # encourage diag >= 0.3 but softly
    return sym + diag_penalty


def topology_prior_loss(betti_maps: torch.Tensor) -> torch.Tensor:
    # betti_maps: [B, 3, D, H, W] in [0,1]. Encourage smoothness (low TV) and non-collapse (avoid all-zero).
    tv = total_variation_3d(betti_maps)
    non_trivial = F.relu(0.05 - betti_maps.mean())  # discourage trivial zeros
    return tv + non_trivial


def dynamics_prior_loss(rho: torch.Tensor, D: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    # rho, D: [B,1,1,1,1] in (0,1). Weak L2 to mid-range to avoid saturation.
    loss = (rho - target) ** 2 + (D - target) ** 2
    return loss.mean()


class HMPCompositeLoss(nn.Module):
    """
    复合损失：监督（Dice+Focal）+ 少量先验约束（稳定、弱权重）
    使用方式：
        criterion = HMPCompositeLoss(num_classes=4)
        total_loss, loss_dict = criterion(outputs, targets, extras.get('priors', {}), aux_outputs=extras.get('aux'))
    """
    def __init__(
        self,
        num_classes: int,
        lambda_focal: float = 1.0,
        lambda_phys: float = 0.1,
        lambda_topo: float = 0.1,
        lambda_dyn: float = 0.1,
        deep_supervision_weights=(1.0, 0.4, 0.3)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_focal = lambda_focal
        self.lambda_phys = lambda_phys
        self.lambda_topo = lambda_topo
        self.lambda_dyn = lambda_dyn
        self.ds_weights = deep_supervision_weights

    def forward(self, logits, targets, priors: dict = None, aux_outputs=None):
        priors = priors or {}

        # Supervised losses
        loss_dice = dice_loss(logits, targets)
        loss_focal = focal_loss(logits, targets)
        loss_sup = loss_dice + self.lambda_focal * loss_focal

        # Deep supervision (if provided): expect list of 3 aux logits upsampled to target size externally or by head
        loss_ds = 0.0
        if aux_outputs is not None:
            for w, aux in zip(self.ds_weights, aux_outputs):
                loss_ds = loss_ds + w * (dice_loss(aux, targets) + self.lambda_focal * focal_loss(aux, targets))

        # Prior losses (all optional)
        loss_phys = torch.tensor(0.0, device=logits.device)
        if "pse" in priors and "coupling" in priors["pse"]:
            loss_phys = physics_prior_loss(priors["pse"]["coupling"])

        loss_topo = torch.tensor(0.0, device=logits.device)
        if "tsa" in priors and "betti" in priors["tsa"]:
            loss_topo = topology_prior_loss(priors["tsa"]["betti"])

        loss_dyn = torch.tensor(0.0, device=logits.device)
        if "tdm" in priors and "rho" in priors["tdm"] and "D" in priors["tdm"]:
            loss_dyn = dynamics_prior_loss(priors["tdm"]["rho"], priors["tdm"]["D"])

        total = loss_sup + loss_ds + self.lambda_phys * loss_phys + self.lambda_topo * loss_topo + self.lambda_dyn * loss_dyn

        return total, {
            "dice": loss_dice.detach(),
            "focal": loss_focal.detach(),
            "deep_supervision": torch.as_tensor(loss_ds).detach(),
            "phys_prior": loss_phys.detach(),
            "topo_prior": loss_topo.detach(),
            "dyn_prior": loss_dyn.detach(),
            "total": total.detach()
        }


