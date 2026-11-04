from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict, namedtuple
from typing import Any, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from utils.paths import ensure_parent_dir, runs_path

log = logging.getLogger(__name__)


def _stack_episode_tensors(episodes: Sequence[Any], key: str) -> torch.Tensor:
    values = [getattr(episode, key) for episode in episodes]
    if not values:
        raise RuntimeError(f"No demo episodes available to compute `{key}` normalization statistics.")
    return torch.cat(values, dim=0)


def compute_normalizer_stats(episodes: Sequence[Any]) -> dict[str, torch.Tensor]:
    observations = _stack_episode_tensors(episodes, "observations").float()
    actions = _stack_episode_tensors(episodes, "actions").float()
    return {
        "obs_mean": observations.mean(dim=0),
        "obs_std": observations.std(dim=0, unbiased=False).clamp_min(1e-6),
        "obs_min": observations.amin(dim=0),
        "obs_max": observations.amax(dim=0),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0, unbiased=False).clamp_min(1e-6),
        "action_min": actions.amin(dim=0),
        "action_max": actions.amax(dim=0),
    }


def make_identity_normalizer_stats(obs_dim: int, action_dim: int) -> dict[str, torch.Tensor]:
    return {
        "obs_mean": torch.zeros(obs_dim, dtype=torch.float32),
        "obs_std": torch.ones(obs_dim, dtype=torch.float32),
        "obs_min": -torch.ones(obs_dim, dtype=torch.float32),
        "obs_max": torch.ones(obs_dim, dtype=torch.float32),
        "action_mean": torch.zeros(action_dim, dtype=torch.float32),
        "action_std": torch.ones(action_dim, dtype=torch.float32),
        "action_min": -torch.ones(action_dim, dtype=torch.float32),
        "action_max": torch.ones(action_dim, dtype=torch.float32),
    }


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32) -> torch.Tensor:
    """Cosine schedule from https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_timesteps(batch_size: int, i: int, device) -> torch.Tensor:
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


_ACTIVATIONS = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        activation_type: str = "Tanh",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
        use_layernorm_final: bool = False,
        dropout: float = 0,
    ):
        super().__init__()
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim, o_dim = dim_list[idx], dim_list[idx + 1]
            layers = [("linear_1", nn.Linear(i_dim, o_dim))]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and idx < num_layer - 1:
                layers.append(("dropout_1", nn.Dropout(dropout)))
            act = _ACTIVATIONS[activation_type] if idx != num_layer - 1 else _ACTIVATIONS[out_activation_type]
            layers.append(("act_1", act))
            self.moduleList.append(nn.Sequential(OrderedDict(layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.moduleList:
            x = m(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(self, hidden_dim: int, activation_type: str = "Mish", use_layernorm: bool = False):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = _ACTIVATIONS[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input


class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim_list,
        activation_type: str = "Mish",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
        use_layernorm_final: bool = False,
    ):
        super().__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            TwoLayerPreActivationResNetLinear(hidden_dim, activation_type, use_layernorm)
            for _ in range(1, num_hidden_layers, 2)
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(_ACTIVATIONS[out_activation_type])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        transition_dim: int,
        horizon_steps: int,
        cond_dim: int,
        time_dim: int = 16,
        mlp_dims=(256, 256),
        cond_mlp_dims=None,
        activation_type: str = "Mish",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
        residual_style: bool = False,
    ):
        super().__init__()
        output_dim = transition_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        model = ResidualMLP if residual_style else MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + list(cond_mlp_dims),
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + transition_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + transition_dim * horizon_steps + cond_dim
        self.mlp_mean = model(
            [input_dim] + list(mlp_dims) + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        """x: (B, Ta, Da); time: (B,); cond["state"]: (B, To, Do)."""
        B, Ta, Da = x.shape
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)


class CriticObs(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        mlp_dims=(256, 256),
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        residual_style: bool = False,
    ):
        super().__init__()
        dims = [cond_dim] + list(mlp_dims) + [1]
        model = ResidualMLP if residual_style else MLP
        self.Q1 = model(
            dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]) -> torch.Tensor:
        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            state = cond
        return self.Q1(state)


class EtaFixed(nn.Module):
    """Learnable-but-input-independent DDIM eta (unused under DDPM defaults)."""

    def __init__(self, base_eta: float = 0.5, min_eta: float = 0.1, max_eta: float = 1.0, **kwargs):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta
        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, cond: dict) -> torch.Tensor:
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        eta_normalized = torch.tanh(self.eta_logit)
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(sample_data.device)

def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError("Sinusoidal embedding dimension must be at least 2.")
        emb_scale = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb_scale)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(_resolve_group_count(out_channels, n_groups), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = bool(use_film_scale_modulation)
        self.out_channels = int(out_channels)

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if self.use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionUNet(nn.Module):
    def __init__(self, *, action_dim: int, global_cond_dim: int, args):
        super().__init__()
        self.action_dim = int(action_dim)
        self.args = args

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(args.diffusion_step_embed_dim),
            nn.Linear(args.diffusion_step_embed_dim, args.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(args.diffusion_step_embed_dim * 4, args.diffusion_step_embed_dim),
        )

        cond_dim = args.diffusion_step_embed_dim + int(global_cond_dim)
        in_out = [(self.action_dim, args.down_dims[0])] + list(zip(args.down_dims[:-1], args.down_dims[1:]))
        common_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": args.kernel_size,
            "n_groups": args.n_groups,
            "use_film_scale_modulation": args.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(args.down_dims[-1], args.down_dims[-1], **common_kwargs),
                DiffusionConditionalResidualBlock1d(args.down_dims[-1], args.down_dims[-1], **common_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(args.down_dims[0], args.down_dims[0], kernel_size=args.kernel_size, n_groups=args.n_groups),
            nn.Conv1d(args.down_dims[0], self.action_dim, 1),
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        global_cond = cond["state"].reshape(B, -1)
        x = x.transpose(1, 2)
        diffusion_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([diffusion_emb, global_cond], dim=-1)

        encoder_skip_features: list[torch.Tensor] = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):
    """DDPM (optionally DDIM) gaussian diffusion with a noise-prediction network."""

    def __init__(
        self,
        network,
        horizon_steps: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda:0",
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 10,
        final_action_clip_value=None,
        eps_clip_value=None,
        denoising_steps: int = 100,
        predict_epsilon: bool = True,
        use_ddim: bool = False,
        ddim_discretize: str = "uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value
        self.eps_clip_value = eps_clip_value

        self.network = network.to(device)

        # ---- DDPM coefficients ----
        self.betas = cosine_beta_schedule(self.denoising_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.ddpm_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        self.ddpm_mu_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        # ---- DDIM coefficients ----
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            assert ddim_discretize == "uniform", "Unknown discretization method for DDIM."
            step_ratio = self.denoising_steps // ddim_steps
            self.ddim_t = torch.arange(0, ddim_steps, device=device) * step_ratio
            self.ddim_alphas = self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [torch.tensor([1.0]).to(torch.float32).to(device), self.alphas_cumprod[self.ddim_t[:-1]]]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5
            ddim_eta = 0
            self.ddim_sigmas = ddim_eta * (
                (1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            ) ** 0.5
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(self.ddim_sqrt_one_minus_alphas, [0])
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ----------
    def p_mean_var(self, x, t, cond, index=None):
        noise = self.network(x, t, cond=cond)
        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)
        if self.use_ddim:
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * noise
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            logvar = torch.log(sigma**2)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = self.ddim_t if self.use_ddim else list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(x=x, t=t_b, cond=cond, index=index_b)
            std = torch.exp(0.5 * logvar)
            if self.use_ddim:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.zeros_like(std) if t == 0 else torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=1e-3)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
        return Sample(x, None)

    # ---------- Supervised (BC) training ----------
    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.denoising_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def p_losses(self, x_start, cond: dict, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        return F.mse_loss(x_recon, x_start, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class VPGDiffusion(DiffusionModel):
    """Diffusion policy as a denoising MDP with per-step gaussian logprobs."""

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps: int,
        ft_denoising_steps_d: int = 0,
        ft_denoising_steps_t: int = 0,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        eta=None,
        learn_eta: bool = False,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d
        self.ft_denoising_steps_t = ft_denoising_steps_t
        self.ft_denoising_steps_cnt = 0
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std

        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False

        # Frozen base actor + trainable fine-tuned copy
        self.actor = self.network
        self.actor_ft = copy.deepcopy(self.actor)
        for param in self.actor.parameters():
            param.requires_grad = False
        log.info("Cloned diffusion actor for fine-tuning; froze base actor.")

        self.critic = critic.to(self.device)

    def step(self):
        """Anneal sampling std / fine-tuning denoising steps (no-op by default)."""
        if not isinstance(self.min_sampling_denoising_std, float):
            self.min_sampling_denoising_std.step()
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(0, self.ft_denoising_steps - self.ft_denoising_steps_d)
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False

    def get_min_sampling_denoising_std(self):
        if isinstance(self.min_sampling_denoising_std, float):
            return self.min_sampling_denoising_std
        return self.min_sampling_denoising_std()

    # override
    def p_mean_var(self, x, t, cond, index=None, use_base_policy=False, deterministic=False):
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(index >= (self.ddim_steps - self.ft_denoising_steps))[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        actor = self.actor if use_base_policy else self.actor_ft
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        if self.use_ddim:
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)
            sigma = (etas * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            logvar = torch.log(sigma**2)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)
        return mu, logvar, etas

    # override
    @torch.no_grad()
    def forward(self, cond, deterministic=False, return_chain=True, use_base_policy=False):
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = self.ddim_t if self.use_ddim else list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x, t=t_b, cond=cond, index=index_b,
                use_base_policy=use_base_policy, deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)
            if self.use_ddim:
                std = torch.zeros_like(std) if deterministic else torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (self.ddim_steps - self.ft_denoising_steps - 1):
                    chain.append(x)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    # ---------- RL training ----------
    def get_logprobs(self, cond, chains, get_ent: bool = False, use_base_policy: bool = False):
        """Logprobs of each denoising transition. Returns (B*K_ft, Ta, Da)."""
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1, end=-1, step=-1, device=self.device
            )
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps, end=self.ddim_steps, device=self.device
            )
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None

        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)

        next_mean, logvar, eta = self.p_mean_var(
            chains_prev, t_all, cond=cond, index=indices, use_base_policy=use_base_policy
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob


class Normalizer(nn.Module):
    def __init__(self, stats: dict[str, torch.Tensor], obs_mode: str = "min_max", action_mode: str = "min_max"):
        super().__init__()
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        for key, value in stats.items():
            self.register_buffer(key, value.float().clone())

    @staticmethod
    def _normalize(x, mode, mean, std, lo, hi):
        if mode == "mean_std":
            return (x - mean) / std.clamp_min(1e-6)
        scale = (hi - lo).clamp_min(1e-6)
        return ((x - lo) / scale) * 2.0 - 1.0

    @staticmethod
    def _unnormalize(x, mode, mean, std, lo, hi):
        if mode == "mean_std":
            return x * std.clamp_min(1e-6) + mean
        scale = (hi - lo).clamp_min(1e-6)
        return ((x + 1.0) * 0.5) * scale + lo

    def normalize_obs(self, x):
        return self._normalize(x, self.obs_mode, self.obs_mean, self.obs_std, self.obs_min, self.obs_max)

    def normalize_action(self, x):
        return self._normalize(x, self.action_mode, self.action_mean, self.action_std, self.action_min, self.action_max)

    def unnormalize_action(self, x):
        return self._unnormalize(x, self.action_mode, self.action_mean, self.action_std, self.action_min, self.action_max)


def build_backbone(args, *, obs_dim: int, action_dim: int) -> nn.Module:
    backbone = getattr(args, "backbone", "mlp")
    cond_dim = obs_dim * args.n_obs_steps
    if backbone == "mlp":
        return DiffusionMLP(
            transition_dim=action_dim,
            horizon_steps=args.horizon,
            cond_dim=cond_dim,
            time_dim=args.time_dim,
            mlp_dims=tuple(args.mlp_dims),
            cond_mlp_dims=tuple(args.cond_mlp_dims) if args.cond_mlp_dims else None,
            residual_style=args.residual_style,
            use_layernorm=args.use_layernorm,
        )
    if backbone == "unet":
        return DiffusionUNet(
            action_dim=action_dim, global_cond_dim=cond_dim, args=args
        )
    raise ValueError(f"Unknown backbone `{backbone}`; expected 'mlp' or 'unet'.")


def build_critic(args, *, obs_dim: int) -> CriticObs:
    cond_dim = obs_dim * args.n_obs_steps
    return CriticObs(
        cond_dim=cond_dim,
        mlp_dims=tuple(args.critic_mlp_dims),
        residual_style=args.residual_style,
        use_layernorm=args.use_layernorm,
    )


class ObsHistory:
    def __init__(self, num_envs: int, n_obs_steps: int, obs_dim: int, device):
        self.n_obs_steps = n_obs_steps
        self.device = device
        # (num_envs, To, obs_dim)
        self._buf = torch.zeros((num_envs, n_obs_steps, obs_dim), device=device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, obs: torch.Tensor, env_mask: torch.Tensor | None = None):
        """Fill history with ``obs`` (pad-by-repeat). ``env_mask`` resets a subset."""
        obs = obs.to(self.device)
        repeated = obs.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
        if env_mask is None:
            self._buf.copy_(repeated)
            self._initialized.fill_(True)
        else:
            self._buf[env_mask] = repeated[env_mask]
            self._initialized[env_mask] = True

    def append(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        obs = obs.to(self.device)
        N = self._buf.shape[0]
        if mask is None:
            mask = torch.ones(N, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)
        # Initialize uninitialized envs (within mask) by filling every slot.
        uninit = (~self._initialized) & mask
        if uninit.any():
            self._buf[uninit] = obs[uninit].unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            self._initialized[uninit] = True
        # Shift-left + append newest for already-initialized masked envs.
        upd = mask & self._initialized & (~uninit)
        if upd.any():
            rolled = torch.roll(self._buf[upd], shifts=-1, dims=1)
            rolled[:, -1] = obs[upd]
            self._buf[upd] = rolled

    @property
    def cond(self) -> dict:
        return {"state": self._buf.clone()}


class DiffusionAgent(nn.Module):
    def __init__(self, args, *, obs_dim: int, action_dim: int, normalizer_stats: dict, run_name: str, device):
        from utils.logger import Logger, SummaryWriter

        super().__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.backbone = getattr(args, "backbone", "mlp")

        self.normalizer = Normalizer(
            normalizer_stats, obs_mode=args.obs_normalization_mode, action_mode=args.action_normalization_mode
        ).to(device)
        actor = build_backbone(args, obs_dim=obs_dim, action_dim=action_dim)
        self.model = DiffusionModel(
            network=actor,
            horizon_steps=args.horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(device),
            denoising_steps=args.denoising_steps,
            predict_epsilon=args.predict_epsilon,
            final_action_clip_value=args.final_action_clip_value,
        ).to(device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # EMA of model weights (standard diffusion-policy practice): sampling and saved
        # checkpoints use the averaged weights, which removes late-training checkpoint churn
        self.ema_model = None
        self.ema_step = 0
        if getattr(args, "use_ema", False):
            self.ema_model = copy.deepcopy(self.model).requires_grad_(False)

        writer = SummaryWriter(runs_path(run_name))
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

    def _cond_from_batch(self, batch: dict) -> dict:
        # batch["observation.state"]: (B, To, Do)
        return {"state": self.normalizer.normalize_obs(batch["observation.state"].to(self.device))}

    @torch.no_grad()
    def sample_action_chunk(self, cond_raw: dict, *, deterministic: bool = True) -> torch.Tensor:
        cond = {"state": self.normalizer.normalize_obs(cond_raw["state"].to(self.device))}
        model = self.ema_model if self.ema_model is not None else self.model
        sample = model(cond, deterministic=deterministic)
        start = self.args.n_obs_steps - 1
        action_norm = sample.trajectories[:, start : start + self.args.n_action_steps]
        return self.normalizer.unnormalize_action(action_norm)

    @torch.no_grad()
    def evaluate_loss(self, loader, device=None) -> float:
        """Mean denoising (validation) loss over ``loader``. A forward-pass loss, not a rollout."""
        if loader is None:
            return float("nan")
        self.eval()
        total, n = 0.0, 0
        for batch in loader:
            cond = self._cond_from_batch(batch)
            action = self.normalizer.normalize_action(batch["action"].to(self.device))
            total += float(self.model.loss(action, cond).item())
            n += 1
        return total / n if n else float("nan")

    def update(self, batch: dict) -> dict:
        self.train()
        cond = self._cond_from_batch(batch)
        action = self.normalizer.normalize_action(batch["action"].to(self.device))  # (B, horizon, Da)
        loss = self.model.loss(action, cond)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        if self.ema_model is not None:
            self._ema_update()
        return {"loss": float(loss.item()), "grad_norm": float(grad_norm)}

    @torch.no_grad()
    def _ema_update(self):
        self.ema_step += 1
        inv_gamma = float(getattr(self.args, "ema_inv_gamma", 1.0))
        power = float(getattr(self.args, "ema_power", 0.75))
        decay_max = float(getattr(self.args, "ema_decay_max", 0.9999))
        decay = 1.0 - (1.0 + self.ema_step / inv_gamma) ** (-power)
        decay = min(decay_max, max(0.0, decay))
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - decay)
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.copy_(b)

    def save(self, path: str, *, epoch: int, global_step: int):
        path = ensure_parent_dir(path)
        payload = {
            "model": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "normalizer_stats": {k: v.detach().cpu() for k, v in self.normalizer.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "ema_model": None if self.ema_model is None else self.ema_model.state_dict(),
            "ema_step": int(self.ema_step),
            "backbone": self.backbone,
            # sampling with a different K than training silently produces garbage
            # actions (no error raised) — consumers must check this on load
            "denoising_steps": int(getattr(self.args, "denoising_steps", 0)) or None,
            "epoch": int(epoch),
            "global_step": int(global_step),
        }
        torch.save(payload, path)
        print(f"BC diffusion checkpoint saved to {path}")

    def load(self, path: str, *, load_optimizer: bool = False) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_k = ckpt.get("denoising_steps")
        if ckpt_k is not None and int(ckpt_k) != int(self.args.denoising_steps):
            raise ValueError(
                f"denoising_steps mismatch: checkpoint was trained with K={ckpt_k} but this "
                f"agent samples with K={self.args.denoising_steps}. Sampling with a different "
                f"K silently produces garbage actions — pass --denoising_steps {ckpt_k}."
            )
        self.model.load_state_dict(ckpt["model"])
        if self.ema_model is not None:
            if ckpt.get("ema_model") is not None:
                self.ema_model.load_state_dict(ckpt["ema_model"])
            else:
                # old checkpoint without EMA: fall back to the raw weights so eval still works
                self.ema_model.load_state_dict(ckpt["model"])
            self.ema_step = int(ckpt.get("ema_step", 0))
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return {"epoch": int(ckpt.get("epoch", 0)), "global_step": int(ckpt.get("global_step", 0))}
