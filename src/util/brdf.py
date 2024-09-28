import sys
sys.path.append("src/")

import torch
from util.coordinate_system import ShadingCoordinateSystem


"""
Code for Disney BRDF adapted from https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
"""


def disney_schlick(u, **kwargs):
    '''(1-u)^5'''
    m = torch.clip(1 - u, 0.0, 1.0)
    m2 = m * m
    return m2 * m2 * m


def gtr1(h, alpha, **kwargs):
    # We are in the shading coordinate system, i.e. dot(normal, m) = m[2]
    result = torch.ones_like(alpha) / torch.pi
    ind_alpha_smaller1 = alpha < 1

    a2 = alpha[ind_alpha_smaller1] ** 2
    t = 1 + (a2 - 1) * h[ind_alpha_smaller1, 2]
    result[ind_alpha_smaller1] = (a2 - 1) / (torch.pi * torch.log(a2) * t)
    return result


def gtr2(h, alpha, **kwargs):
    # We are in the shading coordinate system, i.e. dot(normal, m) = m[2]
    a2 = alpha ** 2
    t = 1 + (a2 - 1) * h[..., 2] ** 2
    return a2 / (torch.pi * t ** 2 + 1e-12)


def smithG_GGX_disney(dir, alpha, **kwargs):
    # We are in the shading coordinate system, i.e. dot(normal, dir) = dir[2]
    a = alpha ** 2
    b = dir[..., 2] ** 2
    return 1 / (dir[..., 2] + torch.sqrt(a + b - a * b) + 1e-16)


def map_disney_parameters(mlp_out):
    return {
        'baseColor': mlp_out[:, :3],
        'metallic': mlp_out[:, 3],
        'subsurface': mlp_out[:, 4],
        'specular': mlp_out[:, 5],
        'roughness': mlp_out[:, 6],
        'specularTint': mlp_out[:, 7],
        'sheen': mlp_out[:, 8],
        'sheenTint': mlp_out[:, 9],
        'clearcoat': mlp_out[:, 10],
        'clearcoatGloss': mlp_out[:, 11],
    }


def evaluate_disney_brdf(
    disney_parameters,
    view_dirs,
    light_dirs,
):
    """ For now expects viewing dirs and light dirs to be given in the shading coordinate system
    Disney parameters: N x 12 
    """
    NdotL = light_dirs[..., 2]
    NdotV = view_dirs[..., 2]
    H = torch.nn.functional.normalize(view_dirs + light_dirs, dim=-1)
    LdotH = torch.sum(light_dirs * H, dim=-1)

    def mix(a, b, t):
        if len(a.shape) > 1:
            t = t.unsqueeze(1)
        return a * (1 - t) + b * t

    # Original has gamma trafo from base color. We directly work in linear space and therefore do not need it
    # Cdlin = mon2lin(vec3(self.baseColor))
    Cdlin = disney_parameters['baseColor']
    Cdlum = .3 * Cdlin[..., 0] + .6 * Cdlin[..., 1] + .1 * Cdlin[..., 2]  # luminance approx.

    Ctint = torch.ones_like(Cdlin)
    Cdlum_pos = Cdlum > 0
    Ctint[Cdlum_pos, :] = Cdlin[Cdlum_pos, :] / Cdlum[Cdlum_pos, None]        # normalize lum. to isolate hue+sat

    Cspec0 = mix(
        disney_parameters['specular'].unsqueeze(1) * .08 * mix(torch.ones_like(Ctint), Ctint, disney_parameters['specularTint']),
        Cdlin,
        disney_parameters['metallic']
    )
    Csheen = mix(torch.ones_like(Ctint), Ctint, disney_parameters['sheenTint'])

    # Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    # and mix in diffuse retro-reflection based on roughness
    FL = disney_schlick(NdotL)
    FV = disney_schlick(NdotV)
    Fd90 = 0.5 + 2 * LdotH * LdotH * disney_parameters['roughness']
    Fd = mix(torch.ones_like(Fd90), Fd90, FL) * mix(torch.ones_like(Fd90), Fd90, FV)

    # Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    # 1.25 scale is used to (roughly) preserve albedo
    # Fss90 used to "flatten" retroreflection based on roughness
    Fss90 = LdotH * LdotH * disney_parameters['roughness']
    Fss = mix(torch.ones_like(Fss90), Fss90, FL) * mix(torch.ones_like(Fss90), Fss90, FV)
    ss = 1.25 * (Fss * (1 / (NdotL + NdotV + 1e-16) - .5) + .5)

    # specular. We deactivated the anisotropic part (anisotropic = 0) and adjusted accrodingly
    alpha = torch.clip(disney_parameters['roughness'] ** 2, .001)
    Ds = gtr2(H, alpha)
    FH = disney_schlick(LdotH)
    Fs = mix(Cspec0, torch.ones_like(Cspec0), FH)
    Gs = smithG_GGX_disney(light_dirs, alpha)
    Gs *= smithG_GGX_disney(view_dirs, alpha)

    # sheen
    Fsheen = (FH * disney_parameters['sheen']).unsqueeze(1) * Csheen

    # clearcoat (ior = 1.5 -> F0 = 0.04)
    Dr = gtr1(
        H,
        mix(
            .1 * torch.ones_like(disney_parameters['clearcoatGloss']),
            .001 * torch.ones_like(disney_parameters['clearcoatGloss']),
            disney_parameters['clearcoatGloss']
        )
    )
    Fr = mix(.04*torch.ones_like(FH), 1.0*torch.ones_like(FH), FH)
    Gr = smithG_GGX_disney(light_dirs, .25*torch.ones_like(light_dirs[:, 0])) *\
        smithG_GGX_disney(view_dirs, .25*torch.ones_like(view_dirs[:, 0]))

    brdf_vals = ((1 / torch.pi) * mix(Fd, ss, disney_parameters['subsurface']).unsqueeze(1) * Cdlin + Fsheen) *\
        (1 - disney_parameters['metallic'].unsqueeze(1)) \
        + Gs.unsqueeze(1) * Fs * Ds.unsqueeze(1) \
        + .25 * (disney_parameters['clearcoat'] * Gr * Fr * Dr).unsqueeze(1)

    return brdf_vals


def render_disney_brdf(
    model_out,
    view_dirs_shading_cosy,
    light_dirs_shading_cosy,
    is_in_shade
):
    """
    Rendering the Disney BRDF for a single directional light
    model_out: N x 12 tensor, output of the MLPs
    """

    disney_parameters = map_disney_parameters(model_out)
    # Compute BRDF values
    brdf_vals = evaluate_disney_brdf(disney_parameters, view_dirs_shading_cosy, light_dirs_shading_cosy)

    # Get cosine of angle
    cos_theta_light = ShadingCoordinateSystem.cos_theta(light_dirs_shading_cosy)

    # Clip negative cos values away (they are shaddows!)
    cos_theta_light = torch.clip(cos_theta_light, min=0.0)

    # Rendering equation (lights = 1 due to normalization, point light source, i.e. no integral)
    rgb = brdf_vals * (cos_theta_light * (1 - is_in_shade.float())).unsqueeze(1)

    return rgb
