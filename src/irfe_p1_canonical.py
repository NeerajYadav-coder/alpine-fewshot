# irfe_p1_canonical.py
# CANONICAL FROZEN REFERENCE MODEL: IRFE Exp-P1 (22,248 Trainable Parameters)
#
# Rationale:
# Exp-P1 achieves the full capacity benefit from Gabor-augmented input (7 channels)
# while maintaining parameter parsimony (22,248 parameters, strictly < 50k budget).
#
# Locked Canonical Benchmarks (8-seed, excluding seed 21):
# - CIFAR-FS 5-shot:          57.65% ± 0.85% (vs ProtoNet 54.46% ± 0.80%, +3.19%)
# - MiniImageNet Native 5-shot: 53.51% ± 0.57% (vs ProtoNet 48.86% ± 0.52%, +4.65%)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_gabor_kernel_bank(ksize=5, sigma=1.5, lambd=3.0, gamma=0.5, psi=0.0):
    orientations = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    half_k = ksize // 2
    y, x = np.mgrid[-half_k:half_k+1, -half_k:half_k+1]
    filters = []
    for theta in orientations:
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta)**2) / (sigma**2)) * np.cos(2 * np.pi * x_theta / lambd + psi)
        gb = gb - gb.mean()
        gb = gb / (np.linalg.norm(gb) + 1e-6)
        filters.append(gb)
    filters = np.stack(filters, axis=0)
    return torch.tensor(filters, dtype=torch.float32).unsqueeze(1)

class GaborPreprocess(nn.Module):
    def __init__(self, ksize=5):
        super().__init__()
        gabor_kernels = create_gabor_kernel_bank(ksize=ksize)
        self.register_buffer("gabor_weight", gabor_kernels)
        self.padding = ksize // 2

    def forward(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gabor_out = F.conv2d(gray, self.gabor_weight, padding=self.padding)
        return torch.cat([x, gabor_out], dim=1) # (B, 7, H, W)

# CIFAR-FS Exp-P1 (32x32)
class PatchEncoderCIFARP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv1 = nn.Conv2d(7, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(24 * 4 * 4, out_dim)
        self.ln    = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.ln(self.fc(x.view(x.size(0), -1)))

class SubPatchEncoderCIFARP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, 12, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(12 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class BoundaryEncoderCIFARP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, 12, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(12 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class IRFEExpP1CIFAR(nn.Module):
    def __init__(self, num_classes=5, embed_dim=16, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        self.encoder          = PatchEncoderCIFARP1(out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderCIFARP1(out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderCIFARP1(out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, embed_dim * 2)
        )

    def _split_patches(self, x):
        return (x[:,:, 0:16, 0:16], x[:,:, 0:16, 16:32],
                x[:,:, 8:24, 8:24], x[:,:, 16:32, 0:16], x[:,:, 16:32, 16:32])

    def _split_boundaries(self, x):
        return (x[:,:, 8:16, 8:16], x[:,:, 8:16, 16:24],
                x[:,:, 16:24, 8:16], x[:,:, 16:24, 16:24])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+8, c:c+8] for r in [0, 8] for c in [0, 8]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract(self, x):
        B = x.size(0)
        patches   = [self._encode_patch(p) for p in self._split_patches(x)]
        edges     = [self._rel(patches[i], patches[j]) for i in range(5) for j in range(i+1, 5)]
        boundaries= [self.boundary_encoder(b) for b in self._split_boundaries(x)]
        
        tokens  = torch.stack(patches + edges + boundaries, dim=1) # 19 tokens
        attn, _ = self.mha(tokens, tokens, tokens)
        refined = self.ln_attn(tokens + attn)
        ws      = self.ws_query.expand(B, -1, -1)
        scores  = torch.bmm(ws, refined.transpose(1, 2)) / math.sqrt(self.embed_dim)
        W       = torch.bmm(F.softmax(scores, dim=-1), refined).squeeze(1)
        
        patch_pool = refined[:, :5, :].mean(dim=1)
        g       = torch.cat([patch_pool, W], dim=1)
        gate    = 2.0 * torch.sigmoid(self.gate_net(g)) - 1.0
        return g + g * gate

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

# MiniImageNet Native Exp-P1 (84x84)
class PatchEncoderNativeP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv1 = nn.Conv2d(7, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.adap  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(24 * 4 * 4, out_dim)
        self.ln    = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adap(x)
        return self.ln(self.fc(x.view(x.size(0), -1)))

class SubPatchEncoderNativeP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, 12, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(12 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class BoundaryEncoderNativeP1(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, 12, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(12 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class IRFEExpP1Native(nn.Module):
    def __init__(self, num_classes=5, embed_dim=16, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        self.encoder          = PatchEncoderNativeP1(out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderNativeP1(out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderNativeP1(out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, embed_dim * 2)
        )

    def _split_patches(self, x):
        return (x[:,:, 0:48,  0:48], x[:,:, 0:48, 36:84],
                x[:,:,18:66, 18:66], x[:,:,36:84,  0:48], x[:,:,36:84, 36:84])

    def _split_boundaries(self, x):
        return (x[:,:, 18:48, 18:48], x[:,:, 18:48, 36:66],
                x[:,:, 36:66, 18:48], x[:,:, 36:66, 36:66])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+24, c:c+24] for r in [0, 24] for c in [0, 24]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract(self, x):
        B = x.size(0)
        patches   = [self._encode_patch(p) for p in self._split_patches(x)]
        edges     = [self._rel(patches[i], patches[j]) for i in range(5) for j in range(i+1, 5)]
        boundaries= [self.boundary_encoder(b) for b in self._split_boundaries(x)]
        
        tokens  = torch.stack(patches + edges + boundaries, dim=1) # 19 tokens
        attn, _ = self.mha(tokens, tokens, tokens)
        refined = self.ln_attn(tokens + attn)
        ws      = self.ws_query.expand(B, -1, -1)
        scores  = torch.bmm(ws, refined.transpose(1, 2)) / math.sqrt(self.embed_dim)
        W       = torch.bmm(F.softmax(scores, dim=-1), refined).squeeze(1)
        
        patch_pool = refined[:, :5, :].mean(dim=1)
        g       = torch.cat([patch_pool, W], dim=1)
        gate    = 2.0 * torch.sigmoid(self.gate_net(g)) - 1.0
        return g + g * gate

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)
