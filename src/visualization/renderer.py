import os
import torch
import numpy as np

from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh
import trimesh
from PIL import Image

from util.mesh import get_ray_mesh_intersector, ray_tracing_xyz
from other_methods.mesh import ray_tracing, load_first_k_eigenfunctions
from util.utils import to_device, load_trained_model, batchify_dict_data, load_mesh
from models.model import MeshFeatModel

def make_renderer_with_trained_model(config, device="cuda"):
    # Load mesh
    mesh = load_mesh(config["data"]["mesh_path"])
    if config["method"] == "intrinsic":
        efuncs = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                             config["model"].get("k"),
                                             rescale_strategy=config["data"].get("rescale_strategy",
                                                                                 "standard"),
                                             embed_strategy=config["data"].get("embed_strategy"),
                                             eigenvalues_path=config["data"].get("eigenvalues_path"))
    else: 
        efuncs = None
    # Load trained model
    weights_path = os.path.join(config["training"]["out_dir"], config["experiment"]["split"], "model.pt")
    model = load_trained_model(weights_path, device)

    return Renderer(model=model, mesh=mesh, method=config["method"], efuncs=efuncs,
                    device=device, H=config["data"]["img_height"], W=config["data"]["img_width"])

def make_reference_renderer(config, mesh_path, device="cpu"):
    # Load mesh
    mesh = load_mesh(mesh_path)
    return Renderer(model=None, mesh=mesh, method=config["method"], efuncs=None,
                    device=device, H=config["data"]["img_height"], W=config["data"]["img_width"])

def make_renderer_during_training(model, config, device="cuda"):
    # Load mesh
    mesh = load_mesh(config["data"]["mesh_path"])

    if config["method"] == "intrinsic":
        efuncs = load_first_k_eigenfunctions(config["data"]["eigenfunctions_path"],
                                             config["model"].get("k"),
                                             rescale_strategy=config["data"].get("rescale_strategy",
                                                                                 "standard"),
                                             embed_strategy=config["data"].get("embed_strategy"),
                                             eigenvalues_path=config["data"].get("eigenvalues_path"))
    else: 
        efuncs = None
    
    return Renderer(model, mesh, method=config["method"], efuncs=efuncs,
                    device=device, H=config["data"]["img_height"], W=config["data"]["img_width"])


class Renderer:
    def __init__(self, model, mesh, method="MeshFeat", efuncs=None, ref_mesh=None, background="white", device="cpu", *, H, W):
        self.method = method
        self.shading = True
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = None
        self.mesh = mesh
        if method == "intrinsic":
            self.features = efuncs
        else:
            self.features = mesh.vertices
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)
        self.H = H
        self.W = W
        self.background = background
        self.device = device

    def set_height(self, height):
        self.H = height

    def set_width(self, width):
        self.W = width
    
    def apply_mesh_transform(self, transform):
        self.mesh.apply_transform(transform)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)

    @torch.no_grad()
    def render(self, camCv2world, K, obj_mask_1d=None, eval_render=False, distortion_coeffs=None, distortion_type=None, file_path=None, tex_path=None, light=None, ref_mesh=None):
        assert obj_mask_1d is None or obj_mask_1d.size()[0] == self.H*self.W

        if self.model is not None:
            # self.model.encoding_layer.active_resolutions = [1,1,1,1]
            self.model.eval()
        
        if self.method == "intrinsic":

            eigenfunction_vector_values, hit_ray_idxs, unit_ray_dirs, face_idxs = ray_tracing(self.ray_mesh_intersector,
                                                                                    self.mesh,
                                                                                    self.features,
                                                                                    camCv2world,
                                                                                    K,
                                                                                    obj_mask_1d=obj_mask_1d,
                                                                                    H=self.H,
                                                                                    W=self.W,
                                                                                    batched=True,
                                                                                    distortion_coeffs=distortion_coeffs,
                                                                                    distortion_type=distortion_type)
            assert eigenfunction_vector_values.dtype == torch.float32
            data = {
                "eigenfunctions": eigenfunction_vector_values,
                "unit_ray_dirs": unit_ray_dirs,
                "hit_face_idxs": face_idxs,
            }
            num_rays = eigenfunction_vector_values.shape[0]
        else:
            hit_points_bary, hit_ray_idxs, unit_ray_dirs, face_idxs, hit_points_xyz = ray_tracing_xyz(self.ray_mesh_intersector,
                                                                                    self.mesh,
                                                                                    self.features,
                                                                                    # ref_mesh.vertices,
                                                                                    camCv2world,
                                                                                    K,
                                                                                    obj_mask_1d=obj_mask_1d,
                                                                                    H=self.H,
                                                                                    W=self.W,
                                                                                    batched=True,
                                                                                    distortion_coeffs=distortion_coeffs,
                                                                                    distortion_type=distortion_type)
            data = {
                "barys": hit_points_bary,
                "unit_ray_dirs": unit_ray_dirs,
                "hit_face_idxs": face_idxs,
                "xyz": hit_points_xyz
            }
            num_rays = hit_points_bary.shape[0]
            assert num_rays > 0

        # Inference in batches to support rendering large views
        total_pred_rgbs = []
        batch_size = 2**15
        faces_all = torch.from_numpy(self.mesh.faces).to(self.device)
        for batch in batchify_dict_data(data, num_rays, batch_size):
            batch = to_device(batch, device=self.device)
            faces = faces_all[batch["hit_face_idxs"]]
            if self.method == "MeshFeat":
                pred_rgbs = self.model(batch["barys"], faces)
            else:
                pred_rgbs = self.model(batch)
                
            if light is not None:
                hit_face_idxs = batch["hit_face_idxs"].cpu().detach().numpy()
                face_normals = self.mesh.face_normals[hit_face_idxs]

                # Calculate the shading factor
                fact = np.dot(face_normals, light)
                fact = np.clip(fact, 0.00001, 0.99999)
                fact = torch.from_numpy(fact.astype(np.float32)).to(self.device)

                # Apply shading by element-wise multiplication
                pred_rgbs_shaded = torch.mul(pred_rgbs, fact.unsqueeze(1))

                total_pred_rgbs.append(pred_rgbs_shaded)
            else:
                total_pred_rgbs.append(pred_rgbs)
        pred_rgbs = torch.concat(total_pred_rgbs, dim=0).to("cpu")

        # We now need to bring the predicted RGB colors into the correct ordering again
        # since the ray-mesh intersection does not preserve ordering
        assert obj_mask_1d is None or obj_mask_1d.dtype == torch.bool
        N = self.H * self.W if obj_mask_1d is None else obj_mask_1d.sum()
        if self.background == "white":
            img = torch.ones((N, 3), device="cpu", dtype=torch.float32)
        else:
            assert self.background == "black"
            img = torch.zeros((N, 3), device="cpu", dtype=torch.float32)
        img[hit_ray_idxs] = pred_rgbs
        
        # If we only kept the object, then img does not have the correct resolution yet.
        # Therefore, we must upscale it one more time taking the object mask into account.
        if obj_mask_1d is not None:
            M = self.H * self.W
            if self.background == "white":
                img_unmasked = torch.ones((M, 3), device="cpu", dtype=torch.float32)
            else:
                assert self.background == "black"
                img_unmasked = torch.zeros((M, 3), device="cpu", dtype=torch.float32)
            img_unmasked[obj_mask_1d] = img
            img = img_unmasked

        if eval_render:
            return img.reshape(self.H, self.W, 3), hit_ray_idxs
        return img.reshape(self.H, self.W, 3).numpy()
