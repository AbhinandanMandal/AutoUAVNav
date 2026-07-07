
import torch
import torch.nn as nn

class GoalConditionedAttentionDuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-modality encoders. A-IDDQN expects fixed modality slots:
        # base(5), lidar(12), camera(8), imu(4), optional active mask(3).
        self.base_encoder = nn.Sequential(nn.Linear(5, embed_dim), nn.ReLU())
        self.lidar_encoder = nn.Sequential(nn.Linear(12, embed_dim), nn.ReLU())
        self.camera_encoder = nn.Sequential(nn.Linear(8, embed_dim), nn.ReLU())
        self.imu_encoder = nn.Sequential(nn.Linear(4, embed_dim), nn.ReLU())

        # Learnable token-type embeddings for the 3 sensor (key/value) tokens
        self.token_type_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # Cross-attention: query = goal/base state, key/value = sensor tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,  # dropout removed: it was active during "greedy" eval
            batch_first=True,
        )
        # Zero-init the output projection so the attention block starts as
        # an identity/no-op (ReZero/Fixup-style stabilization).
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

        # Transformer-style feed-forward sub-layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Fuse the (query) base embedding with the attended sensor context
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # Shared trunk + dueling heads
        self.shared = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU())
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size))

        # Populated on every forward pass: shape (batch, 1, 3) attention
        # over [lidar, camera, imu] tokens, used for visualization and the
        # entropy regularizer.
        self.last_attention = None

    def forward(self, x):
        if x.shape[1] < 29:
            raise ValueError(
                f"A-IDDQN needs fixed modality slots with at least 29 features, got {x.shape[1]}")

        base = x[:, 0:5]
        lidar = x[:, 5:17]
        camera = x[:, 17:25]
        imu = x[:, 25:29]
        sensor_mask = x[:, 29:32] if x.shape[1] >= 32 else torch.ones(
            (x.shape[0], 3), dtype=x.dtype, device=x.device)
        sensor_mask = sensor_mask.clamp(0.0, 1.0)

        # (B, E) -- goal/navigation state
        base_feat = self.base_encoder(base)
        lidar_feat = self.lidar_encoder(lidar)    # (B, E)
        camera_feat = self.camera_encoder(camera)  # (B, E)
        imu_feat = self.imu_encoder(imu)          # (B, E)

        sensor_tokens = torch.stack(
            [lidar_feat, camera_feat, imu_feat], dim=1)  # (B, 3, E)
        sensor_tokens = sensor_tokens + self.token_type_embed.unsqueeze(0)
        sensor_tokens = self.norm_kv(sensor_tokens)

        query = self.norm_q(base_feat).unsqueeze(1)  # (B, 1, E)
        key_padding_mask = sensor_mask <= 0.5
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            # Defensive fallback; normal SENSOR_MODES always enable at least one sensor.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        attn_out, attn_weights = self.attention(
            query, sensor_tokens, sensor_tokens, key_padding_mask=key_padding_mask, need_weights=True
        )  # attn_out: (B, 1, E), attn_weights: (B, 1, 3)
        self.last_attention = attn_weights.detach().cpu()

        x1 = self.norm1(query + attn_out)
        ff = self.ffn(x1)
        x2 = self.norm2(x1 + ff)
        attended = x2.squeeze(1)  # (B, E) -- goal-conditioned sensor context

        fused = self.fusion(torch.cat([base_feat, attended], dim=1))

        z = self.shared(fused)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# # Kept as an alias so any external references to the old class name still work.
# SensorAttentionDuelingQNetwork = GoalConditionedAttentionDuelingQNetwork
