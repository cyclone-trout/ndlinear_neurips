# This code is modified based on the implementation from https://github.com/lucidrains/vit-pytorch.

import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from ndlinear import NdLinear


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class DistillMixin:
    def forward(self, img, distill_token=None):
        distilling = distill_token is not None
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b=b)
            x = torch.cat((x, distill_tokens), dim=1)
        x = self._attend(x)
        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        out = self.mlp_head(x)
        if distilling:
            return out, distill_tokens
        return out


class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class DistillWrapper(nn.Module):
    def __init__(self, *, teacher, student, temperature=1., alpha=0.5, hard=False, mlp_layernorm=False):
        super().__init__()
        self.teacher = teacher
        self.student = student
        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard
        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))
        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature=None, alpha=None, **kwargs):
        alpha = alpha if alpha is not None else self.alpha
        T = temperature if temperature is not None else self.temperature
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        student_logits, distill_tokens = self.student(img, distill_token=self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)
        loss = F.cross_entropy(student_logits, labels)
        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1).detach(),
                reduction='batchmean'
            )
            distill_loss *= T ** 2
        else:
            teacher_labels = teacher_logits.argmax(dim=-1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)
        return loss * (1 - alpha) + distill_loss * alpha


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = NdLinear((dim, 1), (dim, 1))
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = NdLinear((dim, 1), (dim, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        if x.shape[1] < 198:
            padding_size = 198 - x.shape[1]
            padding = torch.zeros(x.size(0), padding_size, x.size(2), device=x.device)
            x = torch.cat((x, padding), dim=1)
        x = self.layer_norm(x)
        x_dim0, x_dim1, x_dim2 = x.shape
        x = x.reshape(x_dim0 * x_dim1, x_dim2, 1)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = x.reshape(x_dim0, x_dim1, x_dim2)
        x = self.dropout2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            if x.shape[1] < 198:
                padding_size = 198 - x.shape[1]
                padding = torch.zeros(x.size(0), padding_size, x.size(2), device=x.device)
                x = torch.cat((x, padding), dim=1)
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
