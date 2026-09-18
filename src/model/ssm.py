"""State-space heads copied from the active notebook pipelines.

The implementations intentionally keep the notebook tensor operations and layer
names so existing ``state_dict`` files can be loaded without key conversion.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state-space layer."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a.unsqueeze(0).expand(d_model, -1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        # Present in the notebook/checkpoints although the forward pass does not use it.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, steps, width = x.shape
        x_ssm, _gate = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :steps].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        a = -torch.exp(self.A_log)
        b = self.B_proj(x_conv)
        c = self.C_proj(x_conv)

        state = torch.zeros(batch_size, width, self.d_state, device=x.device)
        outputs = []
        for step in range(steps):
            discrete_a = torch.exp(a[None, :, :] * dt[:, step, :, None])
            discrete_b = dt[:, step, :, None] * b[:, step, None, :]
            state = state * discrete_a + x[:, step, :, None] * discrete_b
            outputs.append((state * c[:, step, None, :]).sum(-1))
        return torch.stack(outputs, dim=1) + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized)
        x = residual + attended
        return x + self.ffn(self.norm2(x))


class ProtoSSM(nn.Module):
    """Model_22 ProtoSSM v4/v5 architecture from the notebook."""

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 192,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.2,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)]
        )
        self.ssm_bwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)]
        )
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)]
        )
        self.ssm_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_ssm_layers)]
        )
        self.ssm_drop = nn.Dropout(dropout)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(
                d_model, n_heads=cross_attn_heads, dropout=dropout
            )
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
        self.n_families = 0
        self.family_head: nn.Linear | None = None

    def init_prototypes_from_data(self, embeddings: Tensor, labels: Tensor) -> None:
        with torch.no_grad():
            hidden = self.input_proj(embeddings)
            for class_index in range(self.n_classes):
                mask = labels[:, class_index] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[class_index] = F.normalize(
                        hidden[mask].mean(0), dim=0
                    )

    def init_family_head(self, n_families: int, class_to_family: list[int]) -> None:
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer(
            "class_to_family", torch.tensor(class_to_family, dtype=torch.long)
        )

    def forward(
        self,
        emb: Tensor,
        perch_logits: Tensor | None = None,
        site_ids: Tensor | None = None,
        hours: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        _, steps, _ = emb.shape
        hidden = self.input_proj(emb) + self.pos_enc[:, :steps, :]
        if site_ids is not None and hours is not None:
            metadata = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            hidden = hidden + metadata[:, None, :]
        for forward, backward, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = hidden
            hidden_forward = forward(hidden)
            hidden_backward = backward(hidden.flip(1)).flip(1)
            hidden = merge(torch.cat([hidden_forward, hidden_backward], dim=-1))
            hidden = norm(self.ssm_drop(hidden) + residual)
        if self.use_cross_attn:
            hidden = self.cross_attn(hidden)

        similarity = (
            torch.matmul(F.normalize(hidden, dim=-1), F.normalize(self.prototypes, dim=-1).T)
            * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * similarity + (1 - alpha) * perch_logits
        else:
            species_logits = similarity
        family_logits = (
            self.family_head(hidden.mean(dim=1)) if self.family_head is not None else None
        )
        return species_logits, family_logits, hidden

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightProtoSSM(nn.Module):
    """Model_51's two-layer ProtoSSM, including per-layer cross attention."""

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model, cross_attn_heads, dropout=dropout, batch_first=True
                    )
                    for _ in range(2)
                ]
            )
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, embeddings: Tensor, labels: Tensor) -> None:
        with torch.no_grad():
            hidden = self.input_proj(embeddings)
            for class_index in range(self.n_classes):
                mask = labels[:, class_index] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[class_index] = F.normalize(
                        hidden[mask].mean(0), dim=0
                    )

    def forward(
        self,
        emb: Tensor,
        perch_logits: Tensor | None = None,
        site_ids: Tensor | None = None,
        hours: Tensor | None = None,
    ) -> Tensor:
        _, steps, _ = emb.shape
        hidden = self.input_proj(emb) + self.pos_enc[:, :steps, :]
        if site_ids is not None and hours is not None:
            metadata = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            hidden = hidden + metadata[:, None, :]
        for index, (forward, backward, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            residual = hidden
            hidden = self.drop(
                merge(torch.cat([forward(hidden), backward(hidden.flip(1)).flip(1)], dim=-1))
            )
            hidden = norm(hidden + residual)
            if self.use_cross_attn:
                attended, _ = self.cross_attn[index](hidden, hidden, hidden)
                hidden = self.cross_norm[index](hidden + attended)
        similarity = (
            torch.matmul(F.normalize(hidden, dim=-1), F.normalize(self.prototypes, dim=-1).T)
            * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )
        if perch_logits is None:
            return similarity
        alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
        return alpha * similarity + (1 - alpha) * perch_logits


class ResidualSSM(nn.Module):
    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: Tensor,
        first_pass_scores: Tensor,
        site_ids: Tensor | None = None,
        hours: Tensor | None = None,
    ) -> Tensor:
        _, steps, _ = emb.shape
        hidden = self.input_proj(torch.cat([emb, first_pass_scores], dim=-1))
        if site_ids is not None and hours is not None:
            metadata = self.meta_proj(
                torch.cat(
                    [
                        self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                        self.hour_emb(hours.clamp(0, 23)),
                    ],
                    dim=-1,
                )
            )
            hidden = hidden + metadata.unsqueeze(1)
        hidden = hidden + self.pos_enc[:, :steps, :]
        residual = hidden
        hidden = self.ssm_merge(
            torch.cat([self.ssm_fwd(hidden), self.ssm_bwd(hidden.flip(1)).flip(1)], dim=-1)
        )
        hidden = self.ssm_norm(self.ssm_drop(hidden) + residual)
        return self.output_head(hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
