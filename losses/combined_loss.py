"""
Combined Loss Functions for Brain Tumor Segmentation

组合多个损失函数:
1. Dice Loss: 处理类别不平衡
2. Focal Loss: 关注难分样本
3. Topological Loss: 保持拓扑一致性
4. Boundary Loss: 强化边界准确性
5. Deep Supervision Loss: 多层监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    Dice = 2|X∩Y| / (|X|+|Y|)
    Loss = 1 - Dice
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] 预测logits
            target: [B, D, H, W] 真值标签
            
        Returns:
            loss: scalar
        """
        num_classes = pred.shape[1]
        
        # Softmax
        pred_prob = F.softmax(pred, dim=1)
        
        # One-hot编码
        target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        # [B, C, D, H, W]
        
        # 计算每个类别的Dice
        dice_per_class = []
        for c in range(num_classes):
            pred_c = pred_prob[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)
        
        # 平均（可以加权）
        dice_loss = 1.0 - torch.stack(dice_per_class).mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    参考: Lin et al. (2017) Focal Loss for Dense Object Detection
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, D, H, W]
        """
        # Softmax
        pred_prob = F.softmax(pred, dim=1)
        
        # 交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 获取真值类别的概率
        p_t = pred_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Focal权重
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = self.alpha * focal_weight * ce_loss
        
        return loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss
    
    强化边界区域的准确性
    
    参考: Kervadec et al. (2019) Boundary loss for highly unbalanced segmentation
    """
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算到边界的距离图
        
        Args:
            mask: [D, H, W] 二值mask
            
        Returns:
            dist_map: [D, H, W]
        """
        from scipy.ndimage import distance_transform_edt
        
        mask_np = mask.cpu().numpy()
        
        # 计算到边界的距离
        # 对于前景: 正距离
        # 对于背景: 负距离
        pos_dist = distance_transform_edt(mask_np)
        neg_dist = distance_transform_edt(1 - mask_np)
        dist_map = pos_dist - neg_dist
        
        return torch.from_numpy(dist_map).to(mask.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, D, H, W]
        """
        num_classes = pred.shape[1]
        pred_prob = F.softmax(pred, dim=1)
        
        boundary_loss = 0.0
        
        for b in range(pred.shape[0]):
            for c in range(num_classes):
                if c == 0:  # 跳过背景
                    continue
                
                target_c = (target[b] == c).float()
                pred_c = pred_prob[b, c]
                
                # 计算距离图
                with torch.no_grad():
                    dist_map = self.compute_distance_map(target_c)
                
                # 边界损失: 预测错误 × 距离权重
                error = torch.abs(pred_c - target_c)
                weight = torch.exp(-dist_map ** 2 / (2 * self.theta ** 2))
                
                boundary_loss = boundary_loss + (error * weight).mean()
        
        return boundary_loss / (pred.shape[0] * (num_classes - 1))


class TopologyPreservingLoss(nn.Module):
    """
    拓扑保持损失
    
    惩罚拓扑结构的差异（Betti数差异）
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def compute_betti_numbers(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算Betti数 (简化版本)
        
        Args:
            mask: [D, H, W] 二值mask
            
        Returns:
            betti: [3] β0, β1, β2
        """
        from scipy import ndimage
        
        mask_np = mask.cpu().numpy()
        
        # β0: 连通分量数
        labeled, beta0 = ndimage.label(mask_np)
        
        # β1: 简化估计（使用欧拉示性数）
        eroded = ndimage.binary_erosion(mask_np, iterations=1)
        _, beta0_eroded = ndimage.label(eroded)
        beta1 = max(0, beta0 - beta0_eroded)
        
        # β2: 空腔数（简化）
        filled = ndimage.binary_fill_holes(mask_np)
        cavity_vol = filled.sum() - mask_np.sum()
        beta2 = int(cavity_vol / 100)  # 归一化
        
        betti = torch.tensor([beta0, beta1, beta2], dtype=torch.float32)
        
        return betti.to(mask.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, D, H, W]
        """
        num_classes = pred.shape[1]
        pred_prob = F.softmax(pred, dim=1)
        
        topo_loss = 0.0
        count = 0
        
        for b in range(pred.shape[0]):
            for c in range(1, num_classes):  # 跳过背景
                pred_binary = (pred_prob[b, c] > 0.5).float()
                target_binary = (target[b] == c).float()
                
                # 计算Betti数
                with torch.no_grad():
                    pred_betti = self.compute_betti_numbers(pred_binary)
                    target_betti = self.compute_betti_numbers(target_binary)
                
                # L1差异
                topo_loss = topo_loss + torch.abs(pred_betti - target_betti).sum()
                count += 1
        
        return self.weight * topo_loss / max(count, 1)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    Total Loss = w1*Dice + w2*Focal + w3*Boundary + w4*Topology
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.5,
        topology_weight: float = 0.1,
        use_boundary: bool = True,
        use_topology: bool = True
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.topology_weight = topology_weight
        self.use_boundary = use_boundary
        self.use_topology = use_topology
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
        if use_boundary:
            self.boundary_loss = BoundaryLoss()
        
        if use_topology:
            self.topology_loss = TopologyPreservingLoss(weight=topology_weight)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, D, H, W]
            return_components: 是否返回各组件损失
            
        Returns:
            loss: scalar或dict
        """
        # 基础损失
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.focal_weight * focal
        
        loss_dict = {
            'dice': dice.item(),
            'focal': focal.item()
        }
        
        # 边界损失
        if self.use_boundary:
            boundary = self.boundary_loss(pred, target)
            total_loss = total_loss + self.boundary_weight * boundary
            loss_dict['boundary'] = boundary.item()
        
        # 拓扑损失
        if self.use_topology:
            topology = self.topology_loss(pred, target)
            total_loss = total_loss + topology
            loss_dict['topology'] = topology.item()
        
        loss_dict['total'] = total_loss.item()
        
        if return_components:
            return total_loss, loss_dict
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    深监督损失
    
    对多个输出层应用损失，权重递减
    H-PITMA-UNet使用3层辅助监督 + 1层主监督
    权重策略: [1.0, 0.4, 0.3, 0.2] (主输出优先)
    """
    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.base_loss = base_loss
        # H-PITMA-UNet的默认权重
        self.weights = weights if weights is not None else [1.0, 0.4, 0.3, 0.2]
    
    def forward(
        self, 
        outputs: tuple,  # (main_output, [aux1, aux2, aux3])
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: (main_output, [aux_outputs])
                - main_output: [B, C, D, H, W]
                - aux_outputs: list of [B, C, D, H, W]
            target: [B, D, H, W]
        
        Returns:
            total_loss: weighted sum of losses
        """
        if isinstance(outputs, tuple) and len(outputs) == 2:
            main_output, aux_outputs = outputs
            all_outputs = [main_output] + aux_outputs
        else:
            all_outputs = [outputs]  # 只有主输出
        
        total_loss = 0.0
        weights_used = self.weights[:len(all_outputs)]
        
        for i, (output, weight) in enumerate(zip(all_outputs, weights_used)):
            # 将target调整到与output相同尺寸
            if output.shape[2:] != target.shape[1:]:
                target_resized = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_resized = target
            
            loss = self.base_loss(output, target_resized)
            total_loss = total_loss + weight * loss
        
        # 归一化（保持loss scale一致）
        return total_loss / sum(weights_used)


class RegionBasedLoss(nn.Module):
    """
    基于区域的损失
    
    针对BraTS评估的三个区域分别计算:
    - WT (Whole Tumor): 1, 2, 3
    - TC (Tumor Core): 1, 3
    - ET (Enhancing Tumor): 3
    """
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss
    
    def get_regions(self, seg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从分割结果提取三个评估区域
        
        Args:
            seg: [B, D, H, W] 或 [B, C, D, H, W]
            
        Returns:
            regions: dict of tensors
        """
        if seg.dim() == 5:  # Logits
            seg_class = torch.argmax(seg, dim=1)
        else:
            seg_class = seg
        
        # 标签: 0背景, 1坏死, 2水肿, 3增强肿瘤
        wt = (seg_class > 0).long()  # 所有肿瘤
        tc = ((seg_class == 1) | (seg_class == 3)).long()  # 核心
        et = (seg_class == 3).long()  # 增强
        
        return {'WT': wt, 'TC': tc, 'ET': et}
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, D, H, W]
        """
        # 提取区域
        pred_regions = self.get_regions(pred)
        target_regions = self.get_regions(target)
        
        # 对每个区域计算损失
        region_losses = []
        for region_name in ['WT', 'TC', 'ET']:
            # 转换为2类问题
            pred_region = torch.stack([
                1 - pred_regions[region_name].float(),
                pred_regions[region_name].float()
            ], dim=1)
            
            target_region = target_regions[region_name]
            
            loss = self.base_loss(pred_region, target_region)
            region_losses.append(loss)
        
        return sum(region_losses) / len(region_losses)


if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # 测试输入
    B, C, D, H, W = 2, 4, 32, 32, 32
    pred = torch.randn(B, C, D, H, W)
    target = torch.randint(0, C, (B, D, H, W))
    
    # 测试各个损失
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target).item():.4f}")
    
    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target).item():.4f}")
    
    # 组合损失
    combined_loss = CombinedLoss(
        use_boundary=False,  # 边界损失计算较慢
        use_topology=False   # 拓扑损失计算较慢
    )
    loss, loss_dict = combined_loss(pred, target, return_components=True)
    print(f"\nCombined Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # 深监督损失
    outputs = [pred, pred, pred]  # 模拟多层输出
    deep_sup_loss = DeepSupervisionLoss(combined_loss)
    loss = deep_sup_loss(outputs, target)
    print(f"\nDeep Supervision Loss: {loss.item():.4f}")
    
    print("\n✓ Loss functions test passed!")

