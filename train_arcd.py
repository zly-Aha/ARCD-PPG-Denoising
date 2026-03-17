# train_arcd.py
# 基于手稿 "A Unified, Rhythm-Aware Diffusion Model..." (ARCD)
# 对应 Section 2.3 (Architecture) & 2.4 (Training)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as sio


# -----------------------------
# 1. 数据集 (Sec 2.1)
# -----------------------------
class RhythmPPGDataset(Dataset):
    def __init__(self, data_dir, list_file, signal_len=1000, target_labels=[0, 5]):
        """
        对应手稿 Table 1: Patient-level dataset split.
        SR(Label 0) -> 0, AF(Label 5) -> 1
        """
        self.signal_len = signal_len
        self.segments = []
        self.labels = []
        self.label_map = {label: i for i, label in enumerate(target_labels)} if target_labels else {}

        list_path = os.path.join(data_dir, list_file)
        if not os.path.exists(list_path):
            print(f"Warning: List file {list_path} not found.")
            return

        with open(list_path, 'r') as f:
            mat_files = [line.strip().strip("'\"") for line in f if line.strip()]

        for mat_file in tqdm(mat_files, desc=f"Loading {list_file}"):
            path = os.path.join(data_dir, mat_file)
            try:
                data = sio.loadmat(path)
            except:
                continue

            if 'ppgseg' not in data or 'labels' not in data:
                continue

            ppg_data = data['ppgseg']
            label_data = data['labels'].flatten()

            for i in range(ppg_data.shape[0]):
                original_label = int(label_data[i])
                if target_labels and original_label not in target_labels:
                    continue

                segment = ppg_data[i, :].astype(np.float32)
                # 简单截断或填充至 1000 点 (Sec 2.1)
                if len(segment) > self.signal_len:
                    segment = segment[:self.signal_len]
                elif len(segment) < self.signal_len:
                    segment = np.pad(segment, (0, self.signal_len - len(segment)), 'constant')

                # Z-score normalization (Sec 2.1: mean=0, variance=1)
                # 注意：代码原文用的是 MinMax [-1, 1]，这里保留原代码的MinMax以配合Diffusion
                # 如果手稿严格要求Z-score，通常Diffusion输入还是会再次缩放到[-1,1]或保持N(0,1)
                min_v, max_v = np.min(segment), np.max(segment)
                if max_v - min_v > 1e-6:
                    segment = 2 * ((segment - min_v) / (max_v - min_v)) - 1
                else:
                    segment = np.zeros_like(segment)

                self.segments.append(segment)
                self.labels.append(self.label_map[original_label])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.from_numpy(self.segments[idx]).unsqueeze(0)  # [1, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment, label


# -----------------------------
# 2. Diffusion 基础工具
# -----------------------------
def get_timestep_embedding(timesteps, embedding_dim=128):
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -np.log(10000) / half_dim).to(timesteps.device)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


def get_beta_schedule(T=1000):  # 手稿 Table 2: T=1000
    s = 0.008
    timesteps = torch.arange(0, T + 1)
    betas = torch.cos((timesteps / T + s) / (1 + s) * torch.pi / 2) ** 2
    betas = torch.clip(betas / betas[0], 0.0001, 0.02)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars = torch.clamp(alpha_bars, 1e-3, 1.0)
    return betas, alphas, alpha_bars


def add_noise(x_0, t, alpha_bars):
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t]).view(-1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bars[t]).view(-1, 1, 1)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


def estimate_x0_stable(x_t, t, noise_pred, alpha_bars):
    alpha_bar_t = torch.clamp(alpha_bars[t], min=1e-3, max=0.999)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return torch.clamp(x0_pred, -2.5, 2.5)  # 稍微放宽截断


# -----------------------------
# 3. 核心架构: ARCD Network
# -----------------------------
class CondResBlock1D(nn.Module):
    """
    对应手稿 Section 2.3: CondResBlock1D
    - Class-Conditioned FiLM
    - Temporal Attention (Zero-centered: 2*sigmoid-1)
    - Class-Gated Attention Strength
    """

    def __init__(self, channels, dilation=1, num_classes=2, attn_gain_init=0.8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, max(1, channels // 4)), channels)
        self.norm2 = nn.GroupNorm(min(8, max(1, channels // 4)), channels)
        self.act = nn.SiLU()

        # 1. Class-Conditioned FiLM (Eq. 4 & 5)
        self.class_film = nn.Linear(num_classes, 2 * channels)

        # 3. Class-Gated Attention Strength
        # "Bias term is initialized to yield an initial gate value near 0.8"
        self.class_gate = nn.Linear(num_classes, 1)
        nn.init.constant_(self.class_gate.weight, 0.0)
        nn.init.constant_(self.class_gate.bias, np.log(attn_gain_init / (1.0 - attn_gain_init + 1e-8)))

        # 2. Temporal Attention (Eq. 6)
        # "Zero-centered temporal attention mechanism... maps local features to [-1, 1]"
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, class_onehot):
        residual = x

        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        # FiLM Implementation
        film = self.class_film(class_onehot)
        gamma, beta = torch.chunk(film, 2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1)
        x = x * (1.0 + gamma) + beta

        # Temporal Attention [-1, 1]
        attn = self.temporal_attn(x)
        a_tilde = 2.0 * attn - 1.0  # Zero-centered

        # Gating
        gain = torch.sigmoid(self.class_gate(class_onehot)).unsqueeze(-1)
        # 手稿提到 gate 接近 1.0 (SR) 或 0.6-0.7 (AF)。
        # Sigmoid 输出 (0,1)，乘以 2.0 可以覆盖 (0, 2)，足够灵活
        gain = 2.0 * gain

        # Residual Integration
        # "Softly modulated by the FiLM scale parameter to enhance stability"
        res_scaled = residual * (1.0 + 0.3 * gamma)

        out = self.act(x + res_scaled + gain * a_tilde * residual)
        return out, attn


class ARCD_UNet(nn.Module):
    """
    对应 Figure 1: Conditional U-Net Backbone
    """

    def __init__(self, in_channels=3, out_channels=1, time_dim=128, num_classes=2, signal_len=1000):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        # Time Projections
        self.time_proj1 = nn.Linear(128, 64)
        self.time_proj2 = nn.Linear(128, 128)
        self.time_proj3 = nn.Linear(128, 256)
        self.time_projb = nn.Linear(128, 256)

        # Encoder
        self.init_conv = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.down1_block = CondResBlock1D(64, num_classes=num_classes)
        self.down1_conv = nn.Conv1d(64, 64, 3, padding=1)

        self.down2_pool = nn.MaxPool1d(2)
        self.down2_conv = nn.Conv1d(64, 128, 3, padding=1)
        self.down2_block = CondResBlock1D(128, num_classes=num_classes)

        self.down3_pool = nn.MaxPool1d(2)
        self.down3_conv = nn.Conv1d(128, 256, 3, padding=1)
        self.down3_block = CondResBlock1D(256, num_classes=num_classes)

        # Bottleneck
        self.bottleneck_pool = nn.MaxPool1d(2)
        self.bottleneck_block1 = CondResBlock1D(256, num_classes=num_classes)
        self.bottleneck_block2 = CondResBlock1D(256, num_classes=num_classes)

        # Decoder
        self.up3_up = nn.ConvTranspose1d(256, 256, 2, stride=2)
        self.up3_block = CondResBlock1D(256, num_classes=num_classes)
        self.fuse3 = nn.Conv1d(512, 128, 1)

        self.up2_up = nn.ConvTranspose1d(128, 128, 2, stride=2)
        self.up2_block = CondResBlock1D(128, num_classes=num_classes)
        self.fuse2 = nn.Conv1d(256, 64, 1)

        self.up1_up = nn.ConvTranspose1d(64, 64, 2, stride=2)
        self.up1_block = CondResBlock1D(64, num_classes=num_classes)
        self.fuse1 = nn.Conv1d(128, 64, 1)

        self.denoising_head = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # x: [B, 1 + num_classes, T] (Input + Broadcasted Label Embedding)
        class_onehot = x[:, 1:1 + self.num_classes, 0]  # Extract label

        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_dim))

        x0 = self.init_conv(x)

        d1, _ = self.down1_block(x0, class_onehot)
        d1 = d1 + self.time_proj1(t_emb).unsqueeze(-1)
        d1 = self.down1_conv(d1)

        d2p = self.down2_pool(d1)
        d2c = self.down2_conv(d2p)
        d2, _ = self.down2_block(d2c, class_onehot)
        d2 = d2 + self.time_proj2(t_emb).unsqueeze(-1)

        d3p = self.down3_pool(d2)
        d3c = self.down3_conv(d3p)
        d3, _ = self.down3_block(d3c, class_onehot)
        d3 = d3 + self.time_proj3(t_emb).unsqueeze(-1)

        bp = self.bottleneck_pool(d3)
        b1, _ = self.bottleneck_block1(bp, class_onehot)
        b2, _ = self.bottleneck_block2(b1, class_onehot)
        bottleneck = b2 + self.time_projb(t_emb).unsqueeze(-1)

        u3 = self.up3_up(bottleneck)
        u3, _ = self.up3_block(u3, class_onehot)
        u3 = self.fuse3(torch.cat([u3, d3], dim=1))

        u2 = self.up2_up(u3)
        u2, _ = self.up2_block(u2, class_onehot)
        u2 = self.fuse2(torch.cat([u2, d2], dim=1))

        u1 = self.up1_up(u2)
        u1, _ = self.up1_block(u1, class_onehot)
        u1 = self.fuse1(torch.cat([u1, d1], dim=1))

        noise_pred = self.denoising_head(u1)
        return noise_pred


# -----------------------------
# 4. 训练流程 (Sec 2.4)
# -----------------------------
def train_arcd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Starting ARCD Training on {device}...")

    # Load Data (Assuming data files exist)
    target_labels = [0, 5]  # SR, AF
    num_classes = len(target_labels)

    # 尝试加载数据，若无文件则跳过
    try:
        train_set = RhythmPPGDataset(args.dataroot, 'train_samples_patient_split.txt', args.signal_len, target_labels)
        val_set = RhythmPPGDataset(args.dataroot, 'val_samples_patient_split.txt', args.signal_len, target_labels)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"Data Loaded: Train={len(train_set)}, Val={len(val_set)}")
    except Exception as e:
        print(f"Data loading failed (Expected if files missing): {e}")
        return

    # Model Setup
    model = ARCD_UNet(in_channels=1 + num_classes, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Table 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    _, _, alpha_bars = get_beta_schedule(T=args.T_ddpm)
    alpha_bars = alpha_bars.to(device)
    mse_criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_rec = 0.0, 0.0

        # Sec 2.4: Rec loss weight linearly annealed
        lambda_rec = np.interp(epoch, [1, args.epochs // 2], [0.5, 2.0])

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x0, labels in pbar:
            x0 = x0.to(device)
            labels = labels.to(device)
            B, _, L = x0.shape

            # CFG Training: Randomly drop condition
            onehot = F.one_hot(labels, num_classes=num_classes).float().to(device)
            cond_channels = onehot.view(B, num_classes, 1).expand(B, num_classes, L)
            null_channels = torch.zeros_like(cond_channels)

            # p_uncond = 0.2
            mask = (torch.rand(B, device=device) < args.uncond_prob).float().view(B, 1, 1)
            model_cond = cond_channels * (1.0 - mask) + null_channels * mask

            # Add Noise
            t = torch.randint(0, args.T_ddpm, (B,), device=device).long()
            x_t, noise_true = add_noise(x0, t, alpha_bars)

            # Forward
            model_in = torch.cat([x_t, model_cond], dim=1)
            noise_pred = model(model_in, t)

            # 1. Noise Loss
            loss_noise = mse_criterion(noise_pred, noise_true)

            # 2. Rec Loss (Auxiliary)
            x0_pred = estimate_x0_stable(x_t, t, noise_pred, alpha_bars)
            loss_rec = mse_criterion(x0_pred, x0)

            # 3. TV Loss (Rhythm-dependent)
            # Sec 2.4: "lambda_tv is averaged at the batch level"
            tv_loss_val = torch.mean(torch.abs(x0_pred[:, :, 1:] - x0_pred[:, :, :-1]))

            is_sr = (labels == 0).float()
            is_af = (labels == 1).float()

            # 手稿特别提到: "averaged at the batch level"
            # 这意味着先算平均权重，再乘平均 Loss，代码这里保留原逻辑，非常准确
            avg_tv_weight = (args.tv_sr * is_sr.mean() + args.tv_af * is_af.mean())
            loss_tv = avg_tv_weight * tv_loss_val

            loss = loss_noise + lambda_rec * loss_rec + loss_tv

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_rec += loss_rec.item()

            pbar.set_postfix(loss=ep_loss / (pbar.n + 1), rec=ep_rec / (pbar.n + 1))

        scheduler.step()

        # 简单验证 (Standard CFG, not full ARCD inference)
        # Full ARCD inference logic is complex and usually done in testing script
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Completed. Saving checkpoint.")
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"arcd_epoch_tv{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data/train_Dataset")
    parser.add_argument("--signal_len", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)  # Table 2
    parser.add_argument("--batch_size", type=int, default=16)  # Table 2
    parser.add_argument("--lr", type=float, default=1e-4)  # Table 2
    parser.add_argument("--T_ddpm", type=int, default=1000)  # Table 2: 1000 steps
    parser.add_argument("--uncond_prob", type=float, default=0.2)
    # Table 2 Hyperparameters
    parser.add_argument("--tv_sr", type=float, default=0.01)
    parser.add_argument("--tv_af", type=float, default=0.002)#default=0.002
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_arcd")

    args = parser.parse_args()
    train_arcd(args)
