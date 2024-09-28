"""
Adapted from
https://github.com/fbxiang/NeuTex/blob/master/models/texture/texture_mlp.py
"""
import torch
import torch.nn as nn
import numpy as np
from .network_utils import init_seq


class FourierFeatEnc(nn.Module):
    """
    Inspired by
    https://github.com/facebookresearch/pytorch3d/blob/fc4dd80208bbcf6f834e7e1594db0e106961fd60/pytorch3d/renderer/implicit/harmonic_embedding.py#L10
    """
    def __init__(self, k, include_input=True, use_logspace=True, max_freq=None):
        super(FourierFeatEnc, self).__init__()
        if use_logspace:
            freq_bands = 2 ** torch.arange(0, k) * torch.pi
        else:
            assert max_freq is not None
            freq_bands = 2 ** torch.linspace(0, max_freq, steps=k+1)[:-1] * torch.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None] * self.freq_bands).view(*x.size()[:-1], -1)
        if self.include_input:
            return torch.cat((embed.cos(), embed.sin(), x), dim=-1)
        return torch.cat((embed.cos(), embed.sin()), dim=-1)


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def torch_logit(x):
    return torch.log(x / (1 - x + 1e-5))


class TextureMlpMix(nn.Module):
    def __init__(self, primitive_count, out_channels, num_freqs, uv_dim, num_layers, width, use_logspace, max_freq):
        super().__init__()
        self.textures = nn.ModuleList(
            [
                TextureMlp(uv_dim, out_channels, num_freqs, num_layers, width, use_logspace, max_freq)
                for _ in range(primitive_count)
            ]
        )

    def forward(self, uvs, weights):
        values = []
        for uv, texture in zip(torch.unbind(uvs, dim=-2), self.textures):
            values.append(texture(uv))
        # Use the weights for combining the predicted texture values of the different primitives
        value = (torch.stack(values, dim=-2) * weights[..., None]).sum(-2)
        return value


class TextureMlp(nn.Module):
    def __init__(self, uv_dim, out_channels, num_freqs, num_layers, width, use_logspace, max_freq):
        super().__init__()
        self.uv_dim = uv_dim
        self.num_freqs = max(num_freqs, 0)

        self.pos_enc = FourierFeatEnc(num_freqs, use_logspace=use_logspace, max_freq=max_freq)

        self.channels = out_channels

        block1 = []
        block1.append(nn.Linear(uv_dim + 2 * uv_dim * self.num_freqs, width))
        block1.append(nn.LeakyReLU(0.2))
        for _ in range(num_layers):
            block1.append(nn.Linear(width, width))
            block1.append(nn.LeakyReLU(0.2))
        block1.append(nn.Linear(width, self.channels))
        self.block1 = nn.Sequential(*block1)
        init_seq(self.block1)

        self.cubemap_ = None
        self.specular_cubemap_ = None

    def forward(self, uv):
        """
        Args:
            uvs: :math:`(N,Rays,Samples,U)` /////// dim(N x U)
            input_encoding: :math:`(N,E)`
        """
        assert torch.allclose(torch.linalg.norm(uv, axis=-1), torch.tensor(1.0))

        if self.cubemap_ is None:
            value = self.block1(
                self.pos_enc(uv)
                # torch.cat([uv, positional_encoding(uv, self.num_freqs)], dim=-1)
            )
            # Squash color between 0 and 1
            return torch.sigmoid(value)
        else:
            raise NotImplementedError("Not supported")

# Note: Cube code does not matter for our experiments
"""
            value = self.block1(
                torch.cat([uv, positional_encoding(uv, self.num_freqs)], dim=-1)
            )
            cubemap_color = sample_cubemap(self.cubemap_, uv)
            if self.specular_cubemap_ is None:
                return torch.cat(
                    [cubemap_color[..., :3] ** 2.2, value[..., 3:]], dim=-1
                )
            else:
                s_color = sample_cubemap(self.specular_cubemap_, uv)
                value[..., 3][s_color[..., 0] > 0] = torch_logit(
                    s_color[..., 1][s_color[..., 0] > 0]
                )
                value[..., 4][s_color[..., 0] > 0] = torch_logit(
                    s_color[..., 2][s_color[..., 0] > 0]
                )

                return torch.cat(
                    [cubemap_color[..., :3] ** 2.2, value[..., 3:]], dim=-1,
                )

    def _export_cube(self, resolution):
        device = next(self.block1.parameters()).device

        grid = torch.tensor(generate_grid(2, resolution)).float().to(device)
        textures = []
        for index in range(6):
            xyz = convert_cube_uv_to_xyz(index, grid)
            textures.append(self.forward(xyz))

        return torch.stack(textures, dim=0)

    def import_cubemap(self, filename, s2=None):
        assert self.uv_dim == 3
        device = next(self.block1.parameters()).device
        cube = load_cube_from_single_texture(filename)
        self.cubemap_ = torch.tensor(cube).float().to(device)

        if s2 is not None:
            device = next(self.block1.parameters()).device
            cube = load_cube_from_single_texture(s2)
            self.specular_cubemap_ = torch.tensor(cube).float().to(device)

    def export_textures(self, resolution=512):
        # only support sphere for now
        if self.uv_dim == 3:
            with torch.no_grad():
                return self._export_cube(resolution)
        else:
            with torch.no_grad():
                return self._export_square(resolution)
"""


"""
class TextureViewMlpMix(nn.Module):
    def __init__(
        self, count, out_channels, num_freqs, view_freqs, uv_dim, layers, width, clamp,
    ):
        super().__init__()
        self.textures = nn.ModuleList(
            [
                TextureViewMlp(
                    uv_dim,
                    out_channels,
                    num_freqs,
                    view_freqs,
                    layers=layers,
                    width=width,
                    clamp=clamp,
                )
                for _ in range(count)
            ]
        )

    def forward(self, encoding, uvs, view_dir, weights):
        values = []
        for uv, texture in zip(torch.unbind(uvs, dim=-2), self.textures):
            values.append(texture(uv, view_dir))
        value = (torch.stack(values, dim=-2) * weights[..., None]).sum(-2)
        return value


class TextureViewMlp(nn.Module):
    def __init__(
        self, uv_dim, out_channels, num_freqs, view_freqs, layers, width, clamp
    ):
        super().__init__()
        self.uv_dim = uv_dim
        self.num_freqs = max(num_freqs, 0)
        self.view_freqs = max(view_freqs, 0)

        self.channels = out_channels

        block1 = []
        block1.append(nn.Linear(uv_dim + 2 * uv_dim * self.num_freqs, width))
        block1.append(nn.LeakyReLU(0.2))
        for i in range(layers[0]):
            block1.append(nn.Linear(width, width))
            block1.append(nn.LeakyReLU(0.2))
        self.block1 = nn.Sequential(*block1)
        self.color1 = nn.Linear(width, self.channels)
        init_seq(self.block1)

        block2 = []
        block2.append(nn.Linear(width + 3 + 2 * 3 * self.view_freqs, width))
        block2.append(nn.LeakyReLU(0.2))
        for i in range(layers[1]):
            block2.append(nn.Linear(width, width))
            block2.append(nn.LeakyReLU(0.2))
        block2.append(nn.Linear(width, self.channels))
        self.block2 = nn.Sequential(*block2)
        # init_seq(self.block2)

        self.cubemap_ = None

        self.clamp_texture = clamp

    def forward(self, uv, view_dir):
        if self.cubemap_ is None:
            out = self.block1(
                torch.cat([uv, positional_encoding(uv, self.num_freqs)], dim=-1)
            )
            if self.clamp_texture:
                color1 = torch.sigmoid(self.color1(out))
            else:
                color1 = F.softplus(self.color1(out))
            return color1
            # view_dir = view_dir.expand(out.shape[:-1] + (view_dir.shape[-1],))
            # vp = positional_encoding(view_dir, self.view_freqs)
            # out = torch.cat([out, view_dir, vp], dim=-1)
            # if self.clamp_texture:
            #    color2 = torch.sigmoid(self.block2(out))
            # else:
            #     color2 = self.block2(out)
            # return (color1 + color2).clamp(min=0)
        else:
            raise NotImplementedError("Cube Map Code not ported!")

            out = self.block1(
                torch.cat([uv, positional_encoding(uv, self.num_freqs)], dim=-1)
            )
            if self.clamp_texture:
                color1 = torch.sigmoid(self.color1(out))
            else:
                color1 = F.softplus(self.color1(out))

            view_dir = view_dir.expand(out.shape[:-1] + (view_dir.shape[-1],))
            vp = positional_encoding(view_dir, self.view_freqs)
            out = torch.cat([out, view_dir, vp], dim=-1)
            if self.clamp_texture:
                color2 = torch.sigmoid(self.block2(out))
            else:
                color2 = self.block2(out)

            cubemap_color = sample_cubemap(self.cubemap_, uv)
            original_color = color1 + color2
            if self.cubemap_mode_ == 0:
                original_color = (original_color * 3).clamp(min=0, max=1)  # * 0.4 + 0.3
                return cubemap_color * original_color.mean(dim=-1, keepdim=True)
            elif self.cubemap_mode_ == 1:
                original_color = (original_color).clamp(min=0, max=1)  # * 0.4 + 0.3
                original_color[cubemap_color[..., 0] < 0.99] *= cubemap_color[
                    cubemap_color[..., 0] < 0.99
                ][..., :3]
                return original_color
            elif self.cubemap_mode_ == 2:
                original_color = (original_color).clamp(min=0, max=1)
                original_color[cubemap_color[..., 0] < 0.99] *= (
                    1 / cubemap_color[cubemap_color[..., 0] < 0.99][..., :3]
                )
                return original_color
            elif self.cubemap_mode_ == 3:
                original_color = (original_color).clamp(min=0, max=1)

                mask = cubemap_color[..., :3].sum(-1) > 0.01
                original_color[mask] = (
                    2
                    * original_color[mask].mean(-1)[..., None]
                    * cubemap_color[..., :3][mask]
                )

                original_color += cubemap_color[..., :3]
                return original_color

    def _export_cube(self, resolution, viewdir):
        device = next(self.block1.parameters()).device

        grid = torch.tensor(generate_grid(2, resolution)).float().to(device)
        textures = []
        for index in range(6):
            xyz = convert_cube_uv_to_xyz(index, grid)

            if viewdir is not None:
                view = torch.tensor(viewdir).float().to(device).expand_as(xyz)
                textures.append(self.forward(xyz, view))
            else:
                out = self.block1(
                    torch.cat([xyz, positional_encoding(xyz, self.num_freqs)], dim=-1)
                )
                textures.append(torch.sigmoid(self.color1(out)))

        return torch.stack(textures, dim=0)

    def _export_sphere(self, resolution, viewdir):
        with torch.no_grad():
            device = next(self.block1.parameters()).device

            grid = np.stack(
                np.meshgrid(
                    np.arange(2 * resolution), np.arange(resolution), indexing="xy"
                ),
                axis=-1,
            )
            grid = grid / np.array([2 * resolution, resolution]) * np.array(
                [2 * np.pi, np.pi]
            ) + np.array([np.pi, 0])
            x = grid[..., 0]
            y = grid[..., 1]
            xyz = np.stack(
                [-np.sin(x) * np.sin(y), -np.cos(y), -np.cos(x) * np.sin(y)], -1
            )
            xyz = torch.tensor(xyz).float().to(device)

            if viewdir is not None:
                view = torch.tensor(viewdir).float().to(device).expand_as(xyz)
                texture = self.forward(xyz, view)
            else:
                out = self.block1(
                    torch.cat([xyz, positional_encoding(xyz, self.num_freqs)], dim=-1)
                )
                texture = torch.sigmoid(self.color1(out))
            return texture.flip(0)

    def _export_square(self, resolution, viewdir):
        device = next(self.block1.parameters()).device

        grid = torch.tensor(generate_grid(2, resolution)).float().to(device)

        if viewdir is not None:
            view = (
                torch.tensor(viewdir).float().to(device).expand(grid.shape[:-1] + (3,))
            )
            texture = self.forward(grid, view)
        else:
            out = self.block1(
                torch.cat([grid, positional_encoding(grid, self.num_freqs)], dim=-1)
            )
            texture = torch.sigmoid(self.color1(out))

        return texture

    def export_textures(self, resolution=512, viewdir=[0, 0, 1]):
        # only support sphere for now
        if self.uv_dim == 3:
            with torch.no_grad():
                return self._export_cube(resolution, viewdir)
        else:
            with torch.no_grad():
                return self._export_square(resolution, viewdir)

    def import_cubemap(self, filename, mode=0):
        assert self.uv_dim == 3
        device = next(self.block1.parameters()).device
        if isinstance(filename, str):
            w, h = np.array(Image.open(filename)).shape[:2]
            if w == h:
                cube = load_cubemap([filename] * 6)
            else:
                cube = load_cube_from_single_texture(filename)
        else:
            cube = load_cubemap(filename)
        self.cubemap_ = torch.tensor(cube).float().to(device)
        self.cubemap_mode_ = mode
"""
