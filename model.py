"""Model + loss + region-Dice factory functions."""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from constants import NUM_CLASSES, NUM_MODALITIES, REGION_LABELS


def build_model(encoder_name="resnet34", encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=NUM_MODALITIES,
        classes=NUM_CLASSES,
        activation=None,
    )


class CombinedDiceCELoss(nn.Module):
    """0.5 * multiclass Dice + 0.5 * CrossEntropy, both computed on raw logits."""

    def __init__(self):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, mask):
        return 0.5 * self.dice_loss(logits, mask) + 0.5 * self.ce_loss(logits, mask)


def build_loss():
    return CombinedDiceCELoss()


def region_mask(labels, region):
    """labels: integer tensor/array of class ids. Returns a boolean mask that is
    True wherever `labels` belongs to the given BraTS region (WT/TC/ET)."""
    region_ids = REGION_LABELS[region]
    out = labels == region_ids[0]
    for r in region_ids[1:]:
        out = out | (labels == r)
    return out


def dice_score(pred_mask, gt_mask, eps=1e-8):
    """Binary Dice over two boolean tensors of identical shape."""
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    intersection = (pred_mask * gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom.item() == 0:
        return torch.tensor(1.0)  # both empty -> perfect agreement
    return (2.0 * intersection + eps) / (denom + eps)


def region_dice_scores(logits, mask):
    """Given raw logits (N,C,H,W) and integer ground-truth mask (N,H,W),
    return a dict {"WT": dice, "TC": dice, "ET": dice} averaged over the batch."""
    preds = torch.argmax(logits, dim=1)
    scores = {}
    for region in REGION_LABELS:
        pred_region = region_mask(preds, region)
        gt_region = region_mask(mask, region)
        scores[region] = dice_score(pred_region, gt_region).item()
    return scores
