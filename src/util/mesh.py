######################################################################################
#   File has been adapted from https://github.com/tum-vision/intrinsic-neural-fields #
######################################################################################
import glob
import logging
import numpy as np
import torch
import igl
import trimesh
import os
import gc
import scipy as sp
import fast_simplification
import robust_laplacian
import warnings
warnings.filterwarnings('ignore')

from util.cameras import undistort_pixels_meshroom_radial_k3, DistortionTypes
from util.utils import tensor_mem_size_in_bytes, load_mesh


log = logging.getLogger(__name__)

def upsample_mesh(mesh, max_vertices=100000):
    v, f = mesh.vertices, mesh.faces
    while len(v) < max_vertices:
        v, f = igl.upsample(v, f)
    upsampled = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_order=True)
    return upsampled


def simplify_mesh(mesh, resolution):
    # Note: We load using libigl because trimesh does some unwanted preprocessing and vertex
    # reordering (even if process=False and maintain_order=True is set). Hence, we load it
    # using libigl and then convert the loaded mesh into a Trimesh object.
    v, f = mesh.vertices, mesh.faces
    # print(f"Resolution: {resolution}, targetting {len(v)*resolution} vertices.")
    # print(f"Loaded mesh with {len(v)} vertices and {len(f)} faces.")
    # simplify mesh
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.int32)
    u, g, collapses = fast_simplification.simplify(v,f, 1.-resolution, return_collapses=True)
    u, g, i = fast_simplification.replay_simplification(v, f, collapses)
    mesh = trimesh.Trimesh(vertices=u, faces=g, process=False, maintain_order=True)
    # print(f"Simplified mesh with {len(u)} vertices and {len(g)} faces.")
    # assert mesh.is_watertight and (v == mesh.vertices).all() and (f == mesh.faces).all()
    assert np.array_equal(u, mesh.vertices) and np.array_equal(g, mesh.faces)
    return mesh, i

def compute_cotan_laplacian(v, f, return_torch=True):
    """
    Compute the uniform Laplacian of a mesh. 
    v is the list of vertices of the mesh.
    """
    ## Construct and slice up Laplacian
    l = igl.cotmatrix(v, f)

    if return_torch:
        l = torch.sparse_csc_tensor(
            ccol_indices=l.indptr,
            row_indices=l.indices,
            values=l.data,
            dtype=torch.float32
        )
        
    return l

def compute_robust_laplacian(v, f, return_torch=True):
    l, _ = robust_laplacian.mesh_laplacian(np.array(v), np.array(f))
    norm = sp.sparse.linalg.norm(l, ord=2)
    l = l / norm # norm laplacian
    if return_torch:
        l = torch.sparse_csc_tensor(
            ccol_indices=l.indptr,
            row_indices=l.indices,
            values=l.data,
            dtype=torch.float32
        )
    return l

def compute_uniform_laplacian(f, return_torch=True):
    """
    Compute the uniform Laplacian of a mesh. 
    f is the list of faces of the mesh.
    """
    a = igl.adjacency_matrix(f)
    a_sum = np.asarray(np.sum(a, axis=1)).squeeze()
    l = sp.sparse.diags(a_sum, 0, format='csc') - a

    if return_torch:
        l = torch.sparse_csc_tensor(
            ccol_indices=l.indptr,
            row_indices=l.indices,
            values=l.data,
            dtype=torch.float32
        )
        
    return l


def get_ray_mesh_intersector(mesh):
    try:
        import pyembree
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except ImportError:
        print("Warning: Could not find pyembree, the ray-mesh intersection will be significantly slower.")
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def create_ray_origins_and_directions(camCv2world, K, mask_1d, *, H, W, distortion_coeffs=None, distortion_type=None):
    # Let L be the number of pixels where the object is seen in the view
    L = mask_1d.sum()

    try:
        # This does not work for older PyTorch versions.
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    except TypeError:
        # Workaround for older versions. Simulate indexing='xy'
        coord2d_x, coord2d_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        coord2d_x, coord2d_y = coord2d_x.T, coord2d_y.T

    coord2d = torch.cat([coord2d_x[..., None], coord2d_y[..., None]], dim=-1).reshape(-1, 2)  # N*M x 2
    selected_coord2d = coord2d[mask_1d]  # L x 2
    
    # If the views are distorted, remove the distortion from the 2D pixel coordinates
    if distortion_type is not None:
        assert distortion_coeffs is not None
        if distortion_type == DistortionTypes.MESHROOM_RADIAL_K3:
            selected_coord2d = undistort_pixels_meshroom_radial_k3(selected_coord2d.numpy(), K.numpy(), distortion_coeffs)
            selected_coord2d = torch.from_numpy(selected_coord2d).to(torch.float32)
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

    # Get the 3D world coordinates of the ray origins as well as the 3D unit direction vector

    # Origin of the rays of the current view (it is already in 3D world coordinates)
    ray_origins = camCv2world[:, 3].unsqueeze(0).expand(L, -1)  # L x 3

    # Transform 2d coordinates into homogeneous coordinates.
    selected_coord2d = torch.cat((selected_coord2d, torch.ones((L, 1))), dim=-1)  # L x 3
    # Calculate the ray direction: R (K^-1_{3x3} [u v 1]^T)
    ray_dirs = camCv2world[:3, :3].matmul(K[:3, :3].inverse().matmul(selected_coord2d.T)).T  # L x 3
    unit_ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    assert unit_ray_dirs.dtype == torch.float32

    return ray_origins, unit_ray_dirs


def ray_mesh_intersect(ray_mesh_intersector, mesh, ray_origins, ray_directions, return_depth=False, camCv2world=None):
    # Compute the intersection points between the mesh and the rays

    # Note: It might happen that M <= N where M is the number of returned hits
    intersect_locs, hit_ray_idxs, face_idxs = \
        ray_mesh_intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

    # Next, we need to determine the barycentric coordinates of the hit points.

    vertex_idxs_of_hit_faces = torch.from_numpy(mesh.faces[face_idxs]).reshape(-1)  # M*3
    hit_triangles = mesh.vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3)  # M x 3 x 3

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1, 3)  # M x 3

    barycentric_coords = trimesh.triangles.points_to_barycentric(hit_triangles, intersect_locs, method='cramer')  # M x 3

    if return_depth:
        assert camCv2world is not None
        camCv2world = camCv2world.cpu().numpy()
        camCv2world = np.concatenate([camCv2world, np.array([[0., 0, 0, 1]], dtype=camCv2world.dtype)], 0)

        vertices_world = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], -1)  # V, 4

        camWorld2Cv = np.linalg.inv(camCv2world)
        vertices_cam = np.dot(vertices_world, camWorld2Cv.T)
        z_vals = vertices_cam[:, 2][vertex_idxs_of_hit_faces]
        assert np.all(z_vals > 0)

        assert z_vals.shape == barycentric_coords.shape
        assert np.allclose(np.sum(barycentric_coords, -1), 1)

        hit_depth = np.sum(z_vals * barycentric_coords, -1)
        hit_depth = torch.from_numpy(hit_depth)

    barycentric_coords = torch.from_numpy(barycentric_coords).to(dtype=torch.float32)  # M x 3

    hit_ray_idxs = torch.from_numpy(hit_ray_idxs)
    face_idxs = torch.from_numpy(face_idxs).to(dtype=torch.int64)

    if return_depth:
        return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs, hit_depth
    return vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs


def ray_mesh_intersect_batched(ray_mesh_intersector, mesh, ray_origins, ray_directions):
    batch_size = 1 << 18
    num_rays = ray_origins.shape[0]
    idxs = np.arange(0, num_rays)
    batch_idxs = np.split(idxs, np.arange(batch_size, num_rays, batch_size), axis=0)

    total_vertex_idxs_of_hit_faces = []
    total_barycentric_coords = []
    total_hit_ray_idxs = []
    total_face_idxs = []

    total_hits = 0
    hit_ray_idx_offset = 0
    for cur_idxs in batch_idxs:
        cur_ray_origins = ray_origins[cur_idxs]
        cur_ray_dirs = ray_directions[cur_idxs]

        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   cur_ray_origins,
                                                                                                   cur_ray_dirs)

        # Correct the hit_ray_idxs
        hit_ray_idxs += hit_ray_idx_offset

        num_hits = vertex_idxs_of_hit_faces.shape[0]

        # Append results to output
        if num_hits > 0:
            total_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces)
            total_barycentric_coords.append(barycentric_coords)
            total_hit_ray_idxs.append(hit_ray_idxs)
            total_face_idxs.append(face_idxs)

        hit_ray_idx_offset += cur_idxs.shape[0]
        total_hits += num_hits

    # Concatenate results
    out_vertex_idxs_of_hit_faces = torch.zeros((total_hits, 3), dtype=torch.int64)
    out_barycentric_coords = torch.zeros((total_hits, 3), dtype=torch.float32)
    out_hit_ray_idxs = torch.zeros(total_hits, dtype=torch.int64)
    out_face_idxs = torch.zeros(total_hits, dtype=torch.int64)

    offset = 0
    for i in range(len(total_vertex_idxs_of_hit_faces)):
        hits_of_batch = total_vertex_idxs_of_hit_faces[i].shape[0]
        low = offset
        high = low + hits_of_batch

        out_vertex_idxs_of_hit_faces[low:high] = total_vertex_idxs_of_hit_faces[i]
        out_barycentric_coords[low:high] = total_barycentric_coords[i]
        out_hit_ray_idxs[low:high] = total_hit_ray_idxs[i]
        out_face_idxs[low:high] = total_face_idxs[i]

        offset = high

    return out_vertex_idxs_of_hit_faces, out_barycentric_coords, out_hit_ray_idxs, out_face_idxs


def ray_tracing_xyz(ray_mesh_intersector,
                    mesh,
                    vertices,
                    camCv2world,
                    K,
                    obj_mask_1d=None,
                    *,
                    H,
                    W,
                    batched=True,
                    distortion_coeffs=None, 
                    distortion_type=None):
    if obj_mask_1d is None:
        mask = torch.tensor([True]).expand(H * W)
    else:
        mask = obj_mask_1d
    ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                   K, 
                                                                   mask, 
                                                                   H=H, 
                                                                   W=W, 
                                                                   distortion_coeffs=distortion_coeffs,
                                                                   distortion_type=distortion_type)
    if batched:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect_batched(
            ray_mesh_intersector,
            mesh,
            ray_origins,
            unit_ray_dirs)
    else:
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = ray_mesh_intersect(ray_mesh_intersector,
                                                                                                   mesh,
                                                                                                   ray_origins,
                                                                                                   unit_ray_dirs)

    # Calculate the xyz hit points using the barycentric coordinates    
    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1)  # M*3
    face_vertices = torch.tensor(vertices[vertex_idxs_of_hit_faces].reshape(-1, 3, 3), dtype=torch.float32)  # M x 3 x 3
    hit_points_xyz = torch.einsum('bij,bi->bj', face_vertices, barycentric_coords)  # M x 3
 
    return barycentric_coords, hit_ray_idxs, unit_ray_dirs[hit_ray_idxs], face_idxs, hit_points_xyz


class MeshViewPreProcessor:
    def __init__(self, path_to_mesh, out_directory):
        self.out_dir = out_directory
        self.mesh = load_mesh(path_to_mesh)
        self.ray_mesh_intersector = get_ray_mesh_intersector(self.mesh)

        self.cache_vertex_idxs_of_hit_faces = []
        self.cache_barycentric_coords = []
        self.cache_expected_rgbs = []
        self.cache_unit_ray_dirs = []
        self.cache_face_idxs = []

    def _ray_mesh_intersect(self, ray_origins, ray_directions, return_depth=False, camCv2world=None):

        return ray_mesh_intersect(self.ray_mesh_intersector, 
                                  self.mesh, 
                                  ray_origins, 
                                  ray_directions, 
                                  return_depth=return_depth, 
                                  camCv2world=camCv2world)

    def cache_single_view(self, camCv2world, K, mask, img, distortion_coeffs=None, distortion_type=None):
        H, W = mask.shape

        mask = mask.reshape(-1)  # H*W
        img = img.reshape(H * W, -1)  # H*W x 3

        # Let L be the number of pixels where the object is seen in the view

        # Get the expected RGB value of the intersection points with the mesh
        expected_rgbs = img[mask]  # L x 3

        # Get the ray origins and unit directions
        ray_origins, unit_ray_dirs = create_ray_origins_and_directions(camCv2world, 
                                                                       K, 
                                                                       mask, 
                                                                       H=H, 
                                                                       W=W, 
                                                                       distortion_coeffs=distortion_coeffs, 
                                                                       distortion_type=distortion_type)

        # Then, we can compute the ray-mesh-intersections
        vertex_idxs_of_hit_faces, barycentric_coords, hit_ray_idxs, face_idxs = self._ray_mesh_intersect(ray_origins, unit_ray_dirs)

        

        # Choose the correct GTs and viewing directions for the hits.
        num_hits = hit_ray_idxs.size()[0]
        expected_rgbs = expected_rgbs[hit_ray_idxs]
        unit_ray_dirs = unit_ray_dirs[hit_ray_idxs]

        # Some clean up to free memory
        del ray_origins, hit_ray_idxs, mask, img
        gc.collect()  # Force garbage collection

        # Cast the indices down to int32 to save memory. Usually indices have to be int64, however, we assume that
        # the indices from 0 to 2^31-1 are sufficient. Therefore, we can savely cast down

        assert torch.all(face_idxs <= (2<<31)-1)
        face_idxs = face_idxs.to(torch.int32)
        assert torch.all(vertex_idxs_of_hit_faces <= (2<<31)-1)
        vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.to(torch.int32)
        barycentric_coords = barycentric_coords.to(torch.float32)
        expected_rgbs = expected_rgbs.to(torch.float32)
        unit_ray_dirs = unit_ray_dirs.to(torch.float32)

        # And finally, we store the results in the cache
        for idx in range(num_hits):
            self.cache_face_idxs.append(face_idxs[idx])
            self.cache_vertex_idxs_of_hit_faces.append(vertex_idxs_of_hit_faces[idx])
            self.cache_barycentric_coords.append(barycentric_coords[idx])
            self.cache_expected_rgbs.append(expected_rgbs[idx])
            self.cache_unit_ray_dirs.append(unit_ray_dirs[idx])


    def write_to_disk(self):
        print("Starting to write to disk...")

        # Write the cached eigenfuncs and cached expected RGBs to disk
        os.makedirs(self.out_dir, exist_ok=True)

        # Stack the results, write to disk, and then free up memory

        self.cache_face_idxs = torch.stack(self.cache_face_idxs)
        print(
            f"Face Idxs: dim={self.cache_face_idxs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_face_idxs)}B, dtype={self.cache_face_idxs.dtype}")
        np.save(os.path.join(self.out_dir, "face_idxs.npy"), self.cache_face_idxs, allow_pickle=False)
        del self.cache_face_idxs
        gc.collect()  # Force garbage collection

        self.cache_vertex_idxs_of_hit_faces = torch.stack(self.cache_vertex_idxs_of_hit_faces)
        print(
            f"Vertex Idxs of Hit Faces: dim={self.cache_vertex_idxs_of_hit_faces.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_vertex_idxs_of_hit_faces)}B, dtype={self.cache_vertex_idxs_of_hit_faces.dtype}")
        np.save(os.path.join(self.out_dir, "vids_of_hit_faces.npy"), self.cache_vertex_idxs_of_hit_faces,
                allow_pickle=False)
        del self.cache_vertex_idxs_of_hit_faces
        gc.collect()  # Force garbage collection

        self.cache_barycentric_coords = torch.stack(self.cache_barycentric_coords)
        print(
            f"Barycentric Coords: dim={self.cache_barycentric_coords.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_barycentric_coords)}B, dtype={self.cache_barycentric_coords.dtype}")
        np.save(os.path.join(self.out_dir, "barycentric_coords.npy"), self.cache_barycentric_coords, allow_pickle=False)
        del self.cache_barycentric_coords
        gc.collect()  # Force garbage collection

        self.cache_expected_rgbs = torch.stack(self.cache_expected_rgbs)
        print(
            f"Expected RGBs: dim={self.cache_expected_rgbs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_expected_rgbs)}B, dtype={self.cache_expected_rgbs.dtype}")
        np.save(os.path.join(self.out_dir, "expected_rgbs.npy"), self.cache_expected_rgbs, allow_pickle=False)
        del self.cache_expected_rgbs
        gc.collect()  # Force garbage collection

        self.cache_unit_ray_dirs = torch.stack(self.cache_unit_ray_dirs)
        print(
            f"Unit Ray Dirs: dim={self.cache_unit_ray_dirs.size()}, mem_size={tensor_mem_size_in_bytes(self.cache_unit_ray_dirs)}B, dtype={self.cache_unit_ray_dirs.dtype}")
        np.save(os.path.join(self.out_dir, "unit_ray_dirs.npy"), self.cache_unit_ray_dirs, allow_pickle=False)
        del self.cache_unit_ray_dirs
        gc.collect()  # Force garbage collection


######################################################################################
# Functions for the BRDF dataset
#   For the Diligent dataset, we observed occasional intersection misses, therefore we used the
#   following "safe" version (which checks for misses) of the ray-mesh intersection.
#   Functions adapted from https://github.com/tum-vision/intrinsic-neural-fields
######################################################################################
def cast_camera_rays_on_mesh_brdf(mesh, camera_intrinsics, H, W, trafo_cam2world, object_mask=None):
    if object_mask is not None:
        object_mask = object_mask.flatten()

    # Create camera rays
    ray_origins, unit_ray_directions = create_ray_origins_and_directions(
        trafo_cam2world,
        camera_intrinsics,
        object_mask,
        H=H,
        W=W
    )

    # Compute ray mesh intersection
    intersection_pts, intersection_pts_normals, hit_ray_idxs, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces =\
        ray_mesh_intersect_safe_brdf(
            mesh,
            ray_origins,
            unit_ray_directions
        )

    return {
        'intersection_pts': intersection_pts,
        'intersection_pts_normals': intersection_pts_normals,
        'hit_ray_idxs': hit_ray_idxs,
        'barycentric_coords': barycentric_coords,
        'vertex_idxs_of_hit_faces': vertex_idxs_of_hit_faces,
        'idx_hit_faces': idx_hit_faces,
        'ray_origins': ray_origins,
        'unit_ray_directions': unit_ray_directions
    }


def ray_mesh_intersect_safe_brdf(mesh, ray_origins, unit_ray_directions):
    """
    Safe function to compute the ray mesh intersection points as well as the normals.

    Uses the fast pyembree-enhanced version of the trimesh RayMeshIntersector to compute most of the rays. Since this might result
    in intersections on the backside of the mesh, the rays are tested and all rays that result in backfacing intersections are
    re-computed with the (slower) safe variant.
    """
    # Compute ray mesh intersection with pyembree support
    intersection_pts, intersection_pts_normals, hit_ray_idxs, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces = ray_mesh_intersect_brdf(
        mesh, ray_origins, unit_ray_directions
    )

    # Find rays for which the intersection is on the backside of the mesh
    # Those are the ones, for which the scalar product of ray direction and normal is positive (ray direction towards mesh!)
    is_backfacing = torch.sum(unit_ray_directions[hit_ray_idxs] * intersection_pts_normals, dim=-1) > 0
    if torch.sum(is_backfacing) > 0:
        ray_idxs_backfacing = hit_ray_idxs[is_backfacing]

        # For the backfacing rays, recompute the ray mesh intersection with the slow but save variant of the ray mesh intersection
        intersection_pts_recomp, intersection_pts_normals_recomp, hit_ray_idxs_recomp, barycentric_coords_recomp, vertex_idxs_of_hit_faces_recomp, idx_hit_faces_recomp = ray_mesh_intersect_brdf(
            mesh, ray_origins[ray_idxs_backfacing], unit_ray_directions[ray_idxs_backfacing], slow_but_save=True
        )

        # Update the new values
        inds_update = is_backfacing.nonzero().flatten()[hit_ray_idxs_recomp]
        vertex_idxs_of_hit_faces[inds_update] = vertex_idxs_of_hit_faces_recomp
        barycentric_coords[inds_update] = barycentric_coords_recomp
        intersection_pts[inds_update] = intersection_pts_recomp
        intersection_pts_normals[inds_update] = intersection_pts_normals_recomp
        idx_hit_faces[inds_update] = idx_hit_faces_recomp

    return intersection_pts, intersection_pts_normals, hit_ray_idxs, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces


def ray_mesh_intersect_brdf(mesh, ray_origins, unit_ray_directions, slow_but_save=False, return_normals=True):
    """
    Function to compute the ray mesh intersection points as well as the normals.

    By default, the pyembree-enhanced version of the trimesh RayMeshIntersector is used, which is a lot faster. However,
    this version might miss intersections (which results usually in intersection points on the backside of the mesh).

    If slow_but_save=True is used, the standard trimesh RayMeshIntersector is used. This one is much more robust to misses,
    however, it is much slower, than the pyembree variant.

    [Remark] The original code did have a functionality to return depth. However, this was not yet tested
                and therefore removed for now (since depth is probably not needed)
    """
    # Get ray mesh intersector
    if slow_but_save:
        ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    else:
        ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # Note: It might happen that M <= N where M is the number of returned hits
    intersection_pts, hit_ray_idxs, idx_hit_faces = \
        ray_mesh_intersector.intersects_location(ray_origins, unit_ray_directions, multiple_hits=False)

    # Compute Barycentric coordinates for intersection points
    vertex_idxs_of_hit_faces, barycentric_coords = barycentric_coords_from_intersections_brdf(
        mesh, intersection_pts, idx_hit_faces
    )

    # Create torch tensors
    hit_ray_idxs = torch.from_numpy(hit_ray_idxs)
    idx_hit_faces = torch.from_numpy(idx_hit_faces).to(dtype=torch.int64)
    intersection_pts = torch.from_numpy(intersection_pts).float()
    barycentric_coords = barycentric_coords.float()

    if return_normals:
        # Compute Normals at intersection points
        intersection_pts_normals = mesh.vertex_normals[vertex_idxs_of_hit_faces]
        intersection_pts_normals = torch.from_numpy(intersection_pts_normals).float()
        intersection_pts_normals = torch.bmm(barycentric_coords.unsqueeze(1), intersection_pts_normals).squeeze(1)
        intersection_pts_normals = torch.nn.functional.normalize(intersection_pts_normals)

        # Sort results with ascending ray indices
        hit_ray_idxs, inds_reordering = torch.sort(hit_ray_idxs)
        vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces[inds_reordering]
        barycentric_coords = barycentric_coords[inds_reordering]
        intersection_pts = intersection_pts[inds_reordering]
        intersection_pts_normals = intersection_pts_normals[inds_reordering]

    else:
        intersection_pts_normals = None

    return intersection_pts, intersection_pts_normals, hit_ray_idxs, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces


def cast_shadow_rays_brdf(mesh, intersection_pts, intersection_pts_normals, light_directions, shadow_bias):
    # Apply shadow bias
    intersection_points_offset = intersection_pts + shadow_bias * intersection_pts_normals

    # Perform ray mesh intersection. We only need the indices of the rays that intersect since we are doing no more tests to check if
    # the shadow ray intersection is reasonable. We do not need the safe version here since also an intersection at the backface means,
    # that the shadow ray hit something
    _, _, idx_shadow_rays, _, _, _ = ray_mesh_intersect_brdf(
        mesh,
        intersection_points_offset,
        light_directions,
        return_normals=False
    )

    # Construct and return mask for the rays that are in the shade
    is_in_shade = torch.zeros(intersection_pts.shape[0], dtype=torch.bool)
    is_in_shade[idx_shadow_rays] = True

    return is_in_shade


def barycentric_coords_from_intersections_brdf(mesh, intersection_pts, idx_hit_faces):
    # Get vertices of hit faces
    vertex_idxs_of_hit_faces = mesh.faces[idx_hit_faces]
    vertices_hit_faces = mesh.vertices[vertex_idxs_of_hit_faces.flatten()].reshape((-1, 3, 3))

    barycentric_coords = trimesh.triangles.points_to_barycentric(vertices_hit_faces, intersection_pts,
                                                                 method='cramer')  # M x 3

    vertex_idxs_of_hit_faces = torch.from_numpy(vertex_idxs_of_hit_faces)
    barycentric_coords = torch.from_numpy(barycentric_coords)

    return vertex_idxs_of_hit_faces, barycentric_coords


def compute_eigenfunctions_LBO_brdf(mesh, n_eigenfunctions, laplacian_type="cotan", skip_first_efunc=True, return_evalues=False):
    if laplacian_type == "cotan":
        # Compute cotan and mass matrix
        L = -igl.cotmatrix(mesh.vertices, mesh.faces)  # -L to make eigenvalues positive
        M = igl.massmatrix(mesh.vertices, mesh.faces, igl.MASSMATRIX_TYPE_VORONOI)
    elif laplacian_type == "robust":
        # Use Robust Laplacian from: Sharp and Crane "A Laplacian for Nonmanifold Triangle Meshes"
        L, M = robust_laplacian.mesh_laplacian(np.array(mesh.vertices), np.array(mesh.faces))
    elif laplacian_type == "pc_vert_robust":
        # Use vertices of mesh as point in a point cloud
        # Use Robust Laplacian from: Sharp and Crane "A Laplacian for Nonmanifold Triangle Meshes"
        L, M = robust_laplacian.point_cloud_laplacian(np.array(mesh.vertices))
    else:
        raise RuntimeError(f"Laplacian type {laplacian_type} not implemented.")

    # k + 1 because we will remove the first eigenfunction since it is always a constant
    # but we still want k eigenfunctions.
    try:
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, n_eigenfunctions + 1, M, sigma=0, which="LM")
    except RuntimeError as e:
        if len(e.args) == 1 and e.args[0] == "Factor is exactly singular":
            print(
                "Stiffness matrix L is singular because L is most likely badly conditioned. Retrying with improved condition...")
            # https://stackoverflow.com/questions/18754324/improving-a-badly-conditioned-matrix
            c = 1e-10
            L = L + c * sp.sparse.eye(L.shape[0])
            # Retry
            eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, n_eigenfunctions + 1, M, sigma=0, which="LM")

    # This must hold true otherwise we would divide by 0 later on!
    assert np.all(np.max(eigenfunctions, axis=0) != np.min(eigenfunctions, axis=0))

    # We remove the first eigenfunction since it is constant. This implies that for
    # any linear layer Wx+b (assuming W and b are scalars for now), we would get W+b
    # which is always a bias.
    if skip_first_efunc:
        eigenfunctions = eigenfunctions[:, 1:]
        eigenvalues = eigenvalues[1:]
    else:
        # Remove the +1 again
        eigenfunctions = eigenfunctions[:, :-1]
        eigenvalues = eigenvalues[:-1]

    if return_evalues:
        return eigenfunctions, eigenvalues

    return eigenfunctions


def get_eigenfunctions_mesh_brdf(mesh, max_nr_efun, path_precomputed, indices=None):
    # Search for existing results_LBO files and extract k from the name
    existing_files = glob.glob(os.path.join(path_precomputed, "results_LBO_*.pt"))
    existing_nr_efun = torch.tensor([int(name.split('_')[-2]) for name in existing_files])

    # Check if suitable file exist and load if possible
    if torch.sum(max_nr_efun <= existing_nr_efun) > 0:
        # Load
        file_idx = torch.nonzero(max_nr_efun <= existing_nr_efun)[0].item()
        file = existing_files[file_idx]
        log.info(f"Found existing LBO results file. Using {file} for the eigenfunctions")
        data = torch.load(file)
        efuns = data[:, :max_nr_efun]

    else:
        # Compute and save the file
        log.info(f"No suitable LBO results file found or using precomputed disabled. Computing first {max_nr_efun} eigenfunctions. This may take a considerable amount of time (hours)...")
        efuns = torch.from_numpy(compute_eigenfunctions_LBO_brdf(mesh, max_nr_efun, laplacian_type='robust')).float()

        # Construct file name
        file_name = f"results_LBO_{max_nr_efun}_efuns.pt"
        path = os.path.join(path_precomputed, file_name)

        # Delete existing file
        for file in existing_files:
            os.remove(file)

        # Save the new file
        torch.save(efuns, path)

        log.info("Computation of eigenfunctions done!")

    # Select eigenfunctions depending on indices
    if indices:
        efuns = efuns[:, indices]

    return efuns


def compute_eigenfunctions_at_intersections_brdf(E, vertex_idxs_of_hit_faces, barycentric_coords):
    B = vertex_idxs_of_hit_faces.size()[0]

    vertex_idxs_of_hit_faces = vertex_idxs_of_hit_faces.reshape(-1)  # B*3

    # Get for each vertex of the hit face their corresponding eigenfunction vector values
    eigenfuncs_triangle = E[vertex_idxs_of_hit_faces]  # B*3 x k
    eigenfuncs_triangle = eigenfuncs_triangle.reshape(B, 3, -1)  # B x 3 x k

    # Using the barycentric coordinates, we compute the eigenfunction vector values of the point
    eigenfunc_vec_vals = torch.bmm(barycentric_coords.unsqueeze(1), eigenfuncs_triangle)  # B x 1 x k
    return eigenfunc_vec_vals.squeeze(1), eigenfuncs_triangle  # B x k