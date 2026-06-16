import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Helper: extract patches =========
def extract_patches(img, patch_size):
    """
    img: [B, C, H, W]
    returns: [B, N, patch_dim] where patch_dim = C * patch_size * patch_size
    """
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, H//P, W//P, P, P]
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    # [B, C, N, P, P]
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, N, C, P, P]
    patches = patches.reshape(B, -1, C * patch_size * patch_size)  # [B, N, patch_dim]
    return patches


# ========= Multi-Head Self-Attention block =========
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One linear for Q, K, V together: [D -> 3D]
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, D]
        """
        B, N, D = x.shape

        qkv = self.qkv(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, D]

        # reshape for multi-head: [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        def reshape_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        # q, k, v: [B, H, N, Hd]

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, Hd]

        # merge heads back: [B, N, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, D)

        out = self.out_proj(attn_output)  # [B, N, D]
        return out


# ========= Transformer Encoder Block (ViT style) =========
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        # Self-attention + residual
        x = x + self.attn(self.norm1(x))
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ========= ViT Model =========
class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        patch_dim = in_channels * patch_size * patch_size  # e.g. 3*16*16=768

        # Patch embedding: [patch_dim -> d_model]
        self.patch_embedding = nn.Linear(patch_dim, d_model)

        # CLS token (global token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional embedding: one for each patch + CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # (Optional) classification head or projection head – not needed if you just
        # want the CLS embedding like CLIP. You can add this later if you want.
        # self.head = nn.Linear(d_model, num_classes)

        # Init parameters (simple)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, img):
        """
        img: [B, 3, H, W], here H=W=image_size
        returns: CLS embedding [B, d_model]
        """
        B, C, H, W = img.shape
        assert H == self.image_size and W == self.image_size, "Resize image first"

        # 1) Extract patches: [B, N, patch_dim]
        x = extract_patches(img, self.patch_size)

        # 2) Patch embedding: [B, N, d_model]
        x = self.patch_embedding(x)

        # 3) Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, d_model]

        # 4) Add positional embedding
        x = x + self.pos_embed                        # [B, N+1, d_model]
        x = self.pos_drop(x)

        # 5) Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)                                # [B, N+1, d_model]

        # 6) Final norm
        x = self.norm(x)                              # [B, N+1, d_model]

        # 7) Take CLS token as global embedding
        cls_embedding = x[:, 0]                       # [B, d_model]
        return cls_embedding  # like CLIP's pre-projection image embedding

if __name__ == "__main__":
    image_size = 224
    model = ViT(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        d_model=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    )

    # Dummy input
    batch_size = 1
    img = torch.randn(batch_size, 3, image_size, image_size)

    cls_embedding = model(img)
    print("CLS embedding shape:", cls_embedding.shape)  # [1, 768]
