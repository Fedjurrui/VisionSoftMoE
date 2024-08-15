import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
###########
# Based on simple_vit.py at https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
###########
def normalize(x, axis=-1, eps=1e-6):
    m = torch.rsqrt(torch.square(x).sum(axis=axis, keepdim=True) + eps)
    return x * m


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

######################################################
# SoftMoE implementation
# Based on https://github.com/google-research/vmoe/
# and https://github.com/lucidrains/soft-moe-pytorch/tree/main
######################################################

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

    def forward(self, x):
        return self.net(x)


class Experts(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, dim * 2) for _ in range(num_experts)])

    def forward(self, x):
        outs = []
        x = rearrange(x, 'b e n d -> e b n d')
        for expert, expert_input in zip(self.experts, x):
            out = expert(expert_input)
            outs.append(out)
        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()
        outs = rearrange(outs, 'e b n d -> b e n d')
        return outs


class SoftMoE(nn.Module):
    def __init__(self, dim, num_experts, num_slots, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_slots = num_slots
        self.dim = dim
        self.num_experts = num_experts
        self.experts = Experts(dim, num_experts)
        self.phi = normalize(nn.Parameter(torch.randn(self.num_experts, self.num_slots, self.dim)), axis=0)

    def forward(self, x):
        """
        Einstein notation using the paper notation:
        b: batch size
        m: number of tokens
        d: dimension of the tokens
        n: number of experts
        p: number of slots
        phi: Matrix of (d, n*p)
        x: Input matrix: (m,d)
        D: Dispatch matrix: (m,n,p)
        C: Combined matrix: (m,n*p)
        Xs: Input slots for the experts. From the paper they are a weighted average of all the input tokens
        Ys: Output of the experts. From the paper the application of the corresponding expert to the input slots
        Y: Output. From the paper the weighted average of all the output slots, given by the combined weights
        """
        device = x.device
        logits = torch.einsum('b m d, n p d ->b m n p', x, self.phi.to(device))
        D = logits.softmax(dim=1)
        C = rearrange(logits, 'b m n p -> b m (n p)')
        C = C.softmax(dim=-1)
        Xs = torch.einsum('b m d, b m n p -> b n p d', x, D)
        Ys = self.experts(Xs)
        Ys = rearrange(Ys, ' b n p d -> b (n p) d')
        Y = torch.einsum('b p d, b m p -> b m d', Ys, C)
        return Y


class SoftMoETransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, num_experts, num_slots, num_tokens):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                # FeedForward(dim, mlp_dim)
                SoftMoE(dim, num_experts, num_slots, num_tokens)
            ]))

    def forward(self, x):
        for attn, softmoe in self.layers:
            x = attn(x) + x
            x = softmoe(x)
        return self.norm(x)


class ViTSoftMoE(nn.Module):
    def __init__(self, *, image_size, patch_size,
                 num_classes,
                 dim, depth, heads,
                 num_experts, num_slots, num_tokens,
                 channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )
        self.transformer = SoftMoETransformer(dim, depth, heads, dim_head, num_experts, num_slots, num_tokens)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)