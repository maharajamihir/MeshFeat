######################################################################################
#   File has been adapted from https://github.com/tum-vision/intrinsic-neural-fields #
######################################################################################
import numpy as np
import torch
import torch.nn as nn
import sys
from torchinfo import summary
import os
import imageio
import yaml
import sys
import igl
import trimesh
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

sys.path.append("src/models/")

# Make sure loading .exr works for imageio
try:
    imageio.plugins.freeimage.download()
except FileExistsError:
    # Ignore
    pass

def load_preprocessed_data(preproc_data_path):
    data = {}
    
    vertex_idxs_of_hit_faces = np.load(os.path.join(preproc_data_path, "vids_of_hit_faces.npy"))
    data["vertex_idxs_of_hit_faces"] = torch.from_numpy(vertex_idxs_of_hit_faces).to(dtype=torch.int64)

    barycentric_coords = np.load(os.path.join(preproc_data_path, "barycentric_coords.npy"))
    data["barycentric_coords"] = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)

    expected_rgbs = np.load(os.path.join(preproc_data_path, "expected_rgbs.npy"))
    data["expected_rgbs"] = torch.from_numpy(expected_rgbs).to(dtype=torch.float32)
    
    unit_ray_dirs_path = os.path.join(preproc_data_path, "unit_ray_dirs.npy")
    face_idxs_path = os.path.join(preproc_data_path, "face_idxs.npy")
    if os.path.exists(unit_ray_dirs_path) and os.path.exists(face_idxs_path):
        unit_ray_dirs = np.load(unit_ray_dirs_path)
        data["unit_ray_dirs"] = torch.from_numpy(unit_ray_dirs).to(dtype=torch.float32)

        face_idxs = np.load(face_idxs_path)
        data["face_idxs"] = torch.from_numpy(face_idxs).to(dtype=torch.int64)
    
    return data

def tensor_mem_size_in_bytes(x):
    return sys.getsizeof(x.untyped_storage())


def load_trained_model(weights_path, device):
    model=torch.load(weights_path)
    return model

def load_mesh(path):
    # Note: We load using libigl because trimesh does some unwanted preprocessing and vertex
    # reordering (even if process=False and maintain_order=True is set). Hence, we load it
    # using libigl and then convert the loaded mesh into a Trimesh object.
    v, f = igl.read_triangle_mesh(path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)
    # assert mesh.is_watertight and (v == mesh.vertices).all() and (f == mesh.faces).all()
    assert np.array_equal(v, mesh.vertices) and np.array_equal(f, mesh.faces)
    # mesh.export("meshes/loaded.ply")
    return mesh


def load_cameras(view_path):
    cameras = np.load(os.path.join(view_path, "depth", "cameras.npz"))
    camCv2world = torch.from_numpy(cameras["world_mat_0"]).to(dtype=torch.float32)
    K = torch.from_numpy(cameras["camera_mat_0"]).to(dtype=torch.float32)
    return camCv2world, K


def model_summary(model, data):
    data_batch = next(iter(data["train"]))
    summary(model, input_data=[data_batch])


def load_obj_mask_as_tensor(view_path):
    if view_path.endswith(".npy"):
        return np.load(view_path)

    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    if os.path.exists(depth_path):
        depth_map = imageio.imread(depth_path)[..., 0]

        mask_value = 1.e+10
        obj_mask = depth_map != mask_value
    else:
        mask_path = os.path.join(view_path, "depth", "mask.png")
        assert os.path.exists(mask_path), "Must have depth or mask"
        mask = imageio.imread(mask_path)
        obj_mask = mask != 0  # 0 is invalid

    obj_mask = torch.from_numpy(obj_mask)
    return obj_mask


def load_depth_as_numpy(view_path):
    depth_path = os.path.join(view_path, "depth", "depth_0000.exr")
    assert os.path.exists(depth_path)
    depth_map = imageio.imread(depth_path)[..., 0]

    return depth_map


def batchify_dict_data(data_dict, input_total_size, batch_size):
    idxs = np.arange(0, input_total_size)
    batch_idxs = np.split(idxs, np.arange(batch_size, input_total_size, batch_size), axis=0)
    batches = []
    for cur_idxs in batch_idxs:
        data = {}
        for key in data_dict.keys():
            data[key] = data_dict[key][cur_idxs]
        batches.append(data)
    return batches

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_loss_fn(loss_type):
    # L1 loss
    if loss_type == "L1":
        return nn.L1Loss()
    elif loss_type == "L2":
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")


def get_combine_res_fn(combine_res_type):
    if combine_res_type == "cat":
        return torch.cat
    elif combine_res_type == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Combine res type {combine_res_type} not implemented")

##########################################################################################
# The following is taken from:
# https://github.com/tum-vision/tandem/blob/master/cva_mvsnet/utils.py
##########################################################################################


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, **kwargs)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def to_device(x, *, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, str):
        return x
    else:
        raise NotImplementedError(f"Invalid type for to_device: {type(x)}")

##########################################################################################
    
class LossWithGammaCorrection(torch.nn.Module):
    def __init__(self, loss_type='L1'):
        super().__init__()

        if loss_type == 'L1':
            self.criterion = torch.nn.L1Loss()
        elif loss_type == 'MSE':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss_type. Must be 'L1' or 'MSE'")

    def forward(self, x, y):
        x = linear2sRGB(x)
        y = linear2sRGB(y)

        return self.criterion(x, y)


def linear2sRGB(linear, eps=None):
    """
    Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB.
    From https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/image.py#L48
    """
    if eps is None:
        eps = torch.tensor(torch.finfo(torch.float32).eps)

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(eps, linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def compute_psnr(x, y):
    # x and y must be numpy arrays in the range [0.0 1.0] with shape [H, W, C]
    return psnr(x, y, data_range=1.0)


def compute_ssim(x, y):
    # x and y must be numpy arrays in the range [0.0 1.0] with shape [H, W, C]
    ssim_val = ssim(
        x,
        y,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        data_range=1.0,
        channel_axis=2
        )
    return ssim_val


def dssim(pred_image, real_image, data_range=1.0):
    """
    Structural Dissimilarity based on Structural Similarity Index Metric (SSIM)
    """
    assert pred_image.shape == real_image.shape and pred_image.shape[2] == 3
    return (1 - ssim(pred_image, real_image, channel_axis=-1, data_range=data_range)) / 2


class Metrics:
    def __init__(self):
        self.lpips_eval = lpips.LPIPS(net='alex', version='0.1')        

    def compute_metrics(self, renderings):
        # ######### Compute psnr and ssim
        psnrs_linear = []
        psnrs_srgb = []
        ssims = []
        dssims = []

        # Loop over images
        for im_linear, im_gt_linear, im_srgb, im_gt_srgb in zip(
            renderings['images_linear'],
            renderings['images_gt_linear'],
            renderings['images_srgb'],
            renderings['images_gt_srgb']
        ):
            psnrs_linear.append(compute_psnr(im_linear.numpy(), im_gt_linear.numpy()))
            psnrs_srgb.append(compute_psnr(im_srgb.numpy(), im_gt_srgb.numpy()))
            ssims.append(compute_ssim(im_srgb.numpy(), im_gt_srgb.numpy()))
            dssims.append(dssim(im_srgb.numpy(), im_gt_srgb.numpy()))
            

        psnrs_linear = np.stack(psnrs_linear)
        psnrs_srgb = np.stack(psnrs_srgb)
        ssims = np.stack(ssims)
        dssims = np.stack(dssims)

        # ################ Compute lpips
        with torch.no_grad():
            lpips = self.lpips_eval(
                2 * torch.stack(renderings['images_srgb']).permute(0, 3, 1, 2) - 1.0,
                2 * torch.stack(renderings['images_gt_srgb']).permute(0, 3, 1, 2) - 1.0
            ).squeeze()

        result = {
            'psnrs_linear': psnrs_linear,
            'psnrs_srgb': psnrs_srgb,
            'ssims': ssims,
            'dssims': dssims,
            'lpips': lpips
        }

        return result



def time_method(model, dummy_input, repetitions=300):
    """
    Model and dummy_input must be on the target device
    dummy_input must be sucht that model(**dummy_input) can be evaluated
    Taken from here: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    """

    with torch.no_grad():
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            tmp_out = model(**dummy_input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(**dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_eval_time = np.sum(timings) / repetitions

    return mean_eval_time