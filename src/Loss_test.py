# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from Losses import (
    MelSpectrogramL1Loss,
    MultiScaleMelSpectrogramLoss,
    MRSTFTLoss,
    SISNR,
    STFTLoss,
)
from Losses.loudnessloss import TFLoudnessRatio
from Losses.wmloss import WMMbLoss
from utils.utility_functions import get_white_noise


def test_mel_l1_loss():
    print("Testing MelSpectrogramL1Loss...")
    N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    mel_l1 = MelSpectrogramL1Loss(sample_rate=22_050)
    loss = mel_l1(t1, t2)
    loss_same = mel_l1(t1, t1)

    print(f"MelSpectrogramL1Loss: Loss={loss.item():.4f}, Loss Same={loss_same.item():.4f}")
    assert isinstance(loss, torch.Tensor)
    assert loss_same.item() == 0.0


def test_msspec_loss():
    print("Testing MultiScaleMelSpectrogramLoss...")
    N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    msspec = MultiScaleMelSpectrogramLoss(sample_rate=22_050)
    loss = msspec(t1, t2)
    loss_same = msspec(t1, t1)

    print(f"MultiScaleMelSpectrogramLoss: Loss={loss.item():.4f}, Loss Same={loss_same.item():.4f}")
    assert isinstance(loss, torch.Tensor)
    assert loss_same.item() == 0.0


def test_mrstft_loss():
    print("Testing MRSTFTLoss...")
    N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    mrstft = MRSTFTLoss()
    loss = mrstft(t1, t2)

    print(f"MRSTFTLoss: Loss={loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)


def test_sisnr_loss():
    print("Testing SISNR...")
    N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    sisnr = SISNR()
    loss = sisnr(t1, t2)

    print(f"SISNR: Loss={loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)


def test_stft_loss():
    print("Testing STFTLoss...")
    N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    stft = STFTLoss()
    loss = stft(t1, t2)

    print(f"STFTLoss: Loss={loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)


def test_wm_loss():
    print("Testing WMMbLoss...")
    N, nbits, T = 2, 16, random.randrange(1000, 20_000)
    positive = torch.randn(N, 2 + nbits, T)
    mask = torch.ones(N, 1, T)
    message = torch.randint(0, 2, (N, nbits)).float()

    wm_loss = WMMbLoss(temperature=0.3, loss_type="mse")
    loss = wm_loss(positive, None, mask, message)

    print(f"WMMbLoss: Loss={loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)


def test_loudness_loss():
    print("Testing TFLoudnessRatio...")
    sr = 16_000
    duration = 1.0
    wav = get_white_noise(1, int(sr * duration)).unsqueeze(0)
    tfloss = TFLoudnessRatio(sample_rate=sr, n_bands=1)

    loss = tfloss(wav, wav)
    print(f"TFLoudnessRatio: Loss={loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    print("Running all tests...")
    test_mel_l1_loss()
    test_msspec_loss()
    test_mrstft_loss()
    test_sisnr_loss()
    test_stft_loss()
    test_wm_loss()
    test_loudness_loss()
    print("All tests completed successfully.")
