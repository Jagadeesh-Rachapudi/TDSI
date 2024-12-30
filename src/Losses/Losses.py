"""
Utility module to handle adversarial losses without requiring to mess up the main training loop.
"""

import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F

ADVERSARIAL_LOSSES = ["mse", "hinge", "hinge2"]

AdvLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


class AdversarialLoss(nn.Module):
    """
    Adversary training wrapper for computing adversarial losses in both generator and detector.

    Args:
        adversary (nn.Module): The adversary module will estimate logits for fake and real samples.
        optimizer (torch.optim.Optimizer): Optimizer for adversary module.
        loss (AdvLossType): Loss function for generator.
        loss_real (AdvLossType): Loss function for adversary on real samples.
        loss_fake (AdvLossType): Loss function for adversary on fake samples.
        loss_feat (FeatLossType): Feature matching loss for generator training (optional).
        normalize (bool): Whether to normalize the loss based on sub-discriminator count.
    """

    def __init__(
        self,
        adversary: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: AdvLossType,
        loss_real: AdvLossType,
        loss_fake: AdvLossType,
        loss_feat: tp.Optional[FeatLossType] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.adversary = adversary
        self.optimizer = optimizer
        self.loss = loss
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    def get_adversary_pred(self, x: torch.Tensor) -> tp.Tuple[tp.List[torch.Tensor], tp.List[tp.List[torch.Tensor]]]:
        """Run adversary model and validate expected output format."""
        logits, fmaps = self.adversary(x)
        assert isinstance(logits, list) and all(isinstance(t, torch.Tensor) for t in logits)
        assert isinstance(fmaps, list)
        for fmap in fmaps:
            assert isinstance(fmap, list) and all(isinstance(f, torch.Tensor) for f in fmap)
        return logits, fmaps

    def train_adv(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Train the adversary with fake and real samples.

        Args:
            fake (torch.Tensor): Fake audio samples.
            real (torch.Tensor): Real audio samples.

        Returns:
            torch.Tensor: Adversarial loss.
        """
        loss = torch.tensor(0.0, device=fake.device)
        all_logits_fake, _ = self.get_adversary_pred(fake.detach())
        all_logits_real, _ = self.get_adversary_pred(real.detach())

        n_sub_adversaries = len(all_logits_fake)
        for logit_fake, logit_real in zip(all_logits_fake, all_logits_real):
            loss += self.loss_fake(logit_fake) + self.loss_real(logit_real)

        if self.normalize:
            loss /= n_sub_adversaries

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor, predicted_message: torch.Tensor, actual_message: torch.Tensor
    ) -> tp.Dict[str, torch.Tensor]:
        """
        Calculate losses for generator and detector.

        Args:
            fake (torch.Tensor): Generated watermarked audio.
            real (torch.Tensor): Original audio.
            predicted_message (torch.Tensor): Message decoded by the detector.
            actual_message (torch.Tensor): Original ground-truth message.

        Returns:
            dict: Dictionary containing generator and detector losses.
        """
        adv_loss = torch.tensor(0.0, device=fake.device)
        feat_loss = torch.tensor(0.0, device=fake.device)

        all_logits_fake, all_fmap_fake = self.get_adversary_pred(fake)
        all_logits_real, all_fmap_real = self.get_adversary_pred(real)

        n_sub_adversaries = len(all_logits_fake)
        for logit_fake in all_logits_fake:
            adv_loss += self.loss(logit_fake)

        if self.loss_feat:
            for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                feat_loss += self.loss_feat(fmap_fake, fmap_real)

        if self.normalize:
            adv_loss /= n_sub_adversaries
            feat_loss /= n_sub_adversaries

        # Compute decoding accuracy loss for detector
        message_loss = F.mse_loss(predicted_message, actual_message)

        return {
            "generator_loss": adv_loss + feat_loss,
            "detector_loss": message_loss,
        }


# Utility functions for adversarial loss

def mse_real_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(1.0, device=x.device).expand_as(x))


def mse_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(0.0, device=x.device).expand_as(x))


def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0.0, device=x.device).expand_as(x)))


def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(-x - 1, torch.tensor(0.0, device=x.device).expand_as(x)))


def mse_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(1.0, device=x.device).expand_as(x))


def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    return -x.mean()


def hinge2_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0.0, device=x.device).expand_as(x)))


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training."""
    def __init__(self, loss: nn.Module = nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            loss += self.loss(feat_fake, feat_real)
        if self.normalize:
            loss /= len(fmap_fake)
        return loss
