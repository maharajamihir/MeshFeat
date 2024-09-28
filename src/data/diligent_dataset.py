import os
from tqdm import tqdm
import logging

import torch
import numpy as np
import cv2
import scipy.io as sio

import sys
sys.path.append("src/")

from util.utils import load_mesh
from util.mesh import cast_camera_rays_on_mesh_brdf, cast_shadow_rays_brdf, compute_eigenfunctions_at_intersections_brdf, get_eigenfunctions_mesh_brdf, cast_camera_rays_on_mesh_brdf
from util.coordinate_system import ShadingCoordinateSystem


DILIGENT_BIT_DEPTH = 16
DILIGENT_N_VIEWS = 20
DILIGENT_N_LIGHTS = 96
DILIGENT_SHADOW_BIAS = 1e-4  # https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/ligth-and-shadows.html
NAME_FOLDER_PRECOMPUTED = '_precomputed_quantities'

log = logging.getLogger(__name__)


class Dataset_Diligent_MV():
    @staticmethod
    def create_train_test_split(
        n_views_train,
        n_lights_per_view_train,
        n_lights_per_view_test,
        return_only_index_dicts=False,
        LBO_indices=None,
        offset_lights_test=2,       # Use offset to avoid that light directions of test are too similar to train
        **kwargs
    ):
        """ Function to create a train test split for the diligent mv dataset.
        Returns the train and test dataset
        """
        # Distribute train views and train lights per view as evenly as possible
        spacing_views = DILIGENT_N_VIEWS / n_views_train
        spacing_lights_train = DILIGENT_N_LIGHTS / n_lights_per_view_train
        spacing_lights_test = (DILIGENT_N_LIGHTS - offset_lights_test) / n_lights_per_view_test
        train_view_inds = [round(i * spacing_views) + 1 for i in range(n_views_train)]
        train_light_inds = [round(i * spacing_lights_train) + 1 for i in
                            range(n_lights_per_view_train)]
        test_light_inds = [round(i * spacing_lights_test) + 1 + offset_lights_test for i in
                           range(n_lights_per_view_test)]
        log.info(f"Using train light Indices: {train_light_inds}")
        log.info(f"Using test light indices: {test_light_inds}")

        # Create test inds via set differences
        all_views = [i + 1 for i in range(DILIGENT_N_VIEWS)]
        test_view_inds = list(set(all_views) - set(train_view_inds))

        # Construct view light selection dicts
        train_view_light_selection = {f"view_{view_idx:02d}": train_light_inds for view_idx in
                                      train_view_inds}
        test_view_light_selection = {f"view_{view_idx:02d}": test_light_inds for view_idx in
                                     test_view_inds}

        # Check that train and test views and train and light indices do not overlap
        entries_train_set = [(v, l) for v in train_view_light_selection.keys() for l in
                             train_view_light_selection[v]]
        entries_test_set = [(v, l) for v in test_view_light_selection.keys() for l in
                            test_view_light_selection[v]]
        assert len(set(entries_train_set).intersection(set(entries_test_set))) == 0, "[create_train_test_split()] - Train and test set are overlapping"

        log.info(
            f"[create_train_test_split()] - Using {len(entries_train_set)} view-light combinations for training and {len(entries_test_set)} for testing.")

        if return_only_index_dicts:
            return train_view_light_selection, test_view_light_selection
        else:
            # Construct datasets
            log.info("[create_train_test_split()] - Loading train dataset")
            train_dataset = Dataset_Diligent_MV(
                view_light_selection=train_view_light_selection,
                add_mesh_to_dataset=True,
                LBO_indices=LBO_indices,
                **kwargs
            )
            log.info("[create_train_test_split()] - Train dataset loading done")
            log.info("[create_train_test_split()] - Loading test dataset")
            test_dataset = Dataset_Diligent_MV(
                view_light_selection=test_view_light_selection,
                LBO_indices=LBO_indices,
                save_efuns_per_vertex=True,
                **kwargs
            )
            log.info("[create_train_test_split()] - Test dataset loading done")

            return train_dataset, test_dataset

    def __init__(
            self,
            path_diligent_mv,
            name_object,
            view_light_selection,
            add_mesh_to_dataset=False,
            LBO_indices=None,
            save_efuns_per_vertex=False,
            **kwargs
    ):
        """
        - Views/Lights passed as dict that contains the views as keys ('view_01')
          and a list of the light indices ([1, 3, 4, 5,...]) as elements
        - Light direction is from intersection towards light
        - Viewing direction is from intersection towards camera
        - Viewing directions and light directions are given in the shading coordinate system
                (https://www.pbr-book.org/3ed-2018/Reflection_Models#x0-GeometricSetting)
                Normals are [0, 0, 1] in this coordinate system
        """

        # Initial setup
        self.initial_setup(path_diligent_mv, name_object, view_light_selection)
        image_index_counter = 0
        index_counter_duplicates = 0

        # Load mesh
        mesh = self.load_mesh(add_mesh_to_dataset)

        # Get eigenfunctions
        self.load_efuns = LBO_indices is not None
        if self.load_efuns:
            efuns = self.get_eigenfunctions(mesh, add_mesh_to_dataset, LBO_indices)

        # Load camera calibration
        self.load_camera_calibration(view_light_selection)

        # Save available views
        self.n_available_views = len(view_light_selection.keys())
        self.viewIdx2_view = {}
        self.viewLightIdx2_ViewLight = {}
        view_light_idx = 0

        # Loop over views
        log.info(
            "Looping over the views. If no precomputed results are found, this can take a while.")
        for view_idx, view in enumerate(tqdm(view_light_selection.keys())):
            self.image2index_dict[view] = {}

            # Load light directions and intensities
            self.load_lights(view, view_light_selection)

            # Load the initial object mask
            self.load_initial_object_mask(view)

            # Compute ray mesh intersection and update mask (rays from the loaded object mask might not actually hit!)
            intersection_pts, intersection_pts_normals, viewing_directions, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces = \
                self.ray_mesh_intersection(view, mesh)

            # Get trafo to shading coordinate system and transform viewing directions
            world2shading_intersection_pts = ShadingCoordinateSystem.get_trafo_to_shading_cosy_fun_handle(
                intersection_pts_normals)
            view_dir_shading_cosy = world2shading_intersection_pts(viewing_directions)

            # Compute values of eigenfunctions at intersection points
            if self.load_efuns:
                efuns_vals, efuns_vals_triangle = compute_eigenfunctions_at_intersections_brdf(efuns, vertex_idxs_of_hit_faces,
                                                                    barycentric_coords)
                self.ray_mesh_intersections_efuns.append(efuns_vals.float())

                if save_efuns_per_vertex:
                    self.ray_mesh_intersections_efuns_triangle.append(efuns_vals_triangle.float())

            # Add quantities that are not light dependent to data fields
            # Scale intersection points into unit cube. Otherwise TF will have issues
            self.ray_mesh_intersections.append(
                ((intersection_pts - self.mesh_center) / self.mesh_scale).float()
            )
            self.ray_mesh_intersections_barycentric_coords.append(barycentric_coords.float())
            self.ray_mesh_intersections_idx_vertices_hit_faces.append(vertex_idxs_of_hit_faces)
            self.normals_world_cosy.append(intersection_pts_normals.float())
            self.viewing_directions.append(view_dir_shading_cosy.float())

            # Add mapping from light idx to view
            self.viewIdx2_view[view_idx] = view

            # Loop over lights for the current view
            cur_lights = self.lights[view]
            for light_idx, light_intensity, light_direction in zip(
                    cur_lights['light_indices'],
                    cur_lights['intensities'],
                    cur_lights['light_direction']
            ):
                self.image2index_dict[view][light_idx] = {}

                # Cast shadow rays for the intersection points
                is_in_shade = self.cast_shadow_rays(
                    intersection_pts,
                    intersection_pts_normals,
                    light_direction,
                    mesh,
                    view,
                    light_idx,
                    shadow_bias=DILIGENT_SHADOW_BIAS
                )

                # Load RGB values
                rgb_values = self.load_image_and_normalize_by_light(view, light_idx, light_intensity)

                # Transform light directions in the shading coordinate system
                light_dirs_shading_cosy = world2shading_intersection_pts(
                    light_direction.unsqueeze(0).repeat(intersection_pts_normals.shape[0], 1)
                )

                # Append values that are light dependent
                self.rgb_values.append(rgb_values.float())
                self.light_directions.append(light_dirs_shading_cosy.float())
                self.is_in_shade.append(is_in_shade)
                self.image2index_dict[view][light_idx]['start_idx'] = image_index_counter
                self.image2index_dict[view][light_idx]['n_elements'] = rgb_values.shape[0]
                image_index_counter += rgb_values.shape[0]

                self.viewLightIdx2_ViewLight[view_light_idx] = [view, light_idx]
                view_light_idx += 1

                indices_duplicates = torch.arange(rgb_values.shape[0]) + index_counter_duplicates
                self.indices_duplicates.append(indices_duplicates)

            index_counter_duplicates += rgb_values.shape[0]

        # Concatenate all the appended values into single tensors
        self.finalize_data_fields()

    def initial_setup(self, path_diligent_mv, name_object, view_light_selection):
        # Construct path to data and create folder for precomputed results
        self.path_data = os.path.join(path_diligent_mv, 'mvpmsData', name_object)
        assert os.path.isdir(self.path_data), f"The folder {self.path_data} does not exist."
        self.path_mesh = os.path.join(path_diligent_mv, 'simplifiedMeshes', f"{name_object}_simplified.ply")
        self.path_precomputed = os.path.join(path_diligent_mv, NAME_FOLDER_PRECOMPUTED, name_object)
        os.makedirs(self.path_precomputed, exist_ok=True)

        self.view_light_selection = view_light_selection
        self.available_view_light_pairs = [[v, l] for v in view_light_selection.keys() for l in
                                           view_light_selection[v]]
        self.n_available_view_light_pairs = len(self.available_view_light_pairs)

        self.precomputation_msg_intersection_done = False
        self.precomputation_msg_normals_done = False
        self.precomputation_msg_shadow_done = False

        # Init dicts to store additional data (not returned directly as an item of the dataset)
        self.lights = {}
        self.object_masks = {}
        self.image_dimensions = {}
        self.image2index_dict = {}

        self.rgb_values = []
        self.viewing_directions = []
        self.light_directions = []
        self.ray_mesh_intersections = []
        self.ray_mesh_intersections_barycentric_coords = []
        self.ray_mesh_intersections_idx_vertices_hit_faces = []
        self.ray_mesh_intersections_efuns = []
        self.ray_mesh_intersections_efuns_triangle = []
        self.normals_world_cosy = []
        self.is_in_shade = []
        self.indices_duplicates = []

    def finalize_data_fields(self):
        """ Concatenate Tensors. We store everything as float
        """
        self.rgb_values = torch.cat(self.rgb_values)
        self.viewing_directions = torch.cat(self.viewing_directions)
        self.light_directions = torch.cat(self.light_directions)
        self.normals_world_cosy = torch.cat(self.normals_world_cosy)
        self.ray_mesh_intersections = torch.cat(self.ray_mesh_intersections)
        self.ray_mesh_intersections_barycentric_coords = torch.cat(self.ray_mesh_intersections_barycentric_coords)
        self.ray_mesh_intersections_idx_vertices_hit_faces = torch.cat(self.ray_mesh_intersections_idx_vertices_hit_faces)

        if self.load_efuns:
            self.ray_mesh_intersections_efuns = torch.cat(self.ray_mesh_intersections_efuns)

        self.is_in_shade = torch.cat(self.is_in_shade)

        self.indices_duplicates = torch.cat(self.indices_duplicates)

        if len(self.ray_mesh_intersections_efuns_triangle) > 0:
            self.ray_mesh_intersections_efuns_triangle = torch.cat(self.ray_mesh_intersections_efuns_triangle)
        else:
            self.ray_mesh_intersections_efuns_triangle = None

    def load_mesh(self, save_mesh):
        mesh = load_mesh(self.path_mesh)
        self.mesh = mesh if save_mesh else None

        # Since the meshes are not in unit cube and centered, compute scale and center
        # to later scale the intersection points into the unit cube. Otherwise TF will have problems
        # Use safety margin of 1.05 for scale
        mesh_max = np.max(mesh.vertices, axis=0)
        mesh_min = np.min(mesh.vertices, axis=0)
        self.mesh_center = (mesh_max + mesh_min) / 2
        self.mesh_scale = 1.05 * np.max(mesh_max - mesh_min) / 2

        return mesh

    def get_eigenfunctions(
            self,
            mesh,
            add_to_dataset,
            LBO_indices,
    ):
        max_nr_efun = max(LBO_indices) + 1

        efuns = get_eigenfunctions_mesh_brdf(
            mesh,
            max_nr_efun,
            self.path_precomputed,
            indices=LBO_indices
        )
        self.efuns_mesh = efuns if add_to_dataset else None
        self.n_efuns_used = len(LBO_indices)
        self.efuns_indicies = torch.tensor(LBO_indices)

        return efuns

    @staticmethod
    def load_camera_calibration_from_file(path_data, list_views):
        trafos_cam2world = {}
        trafos_world2cam = {}

        # Load calibration. The trafos stored here are world2cam
        path_calib = os.path.join(path_data, 'Calib_Results.mat')
        calib_data = sio.loadmat(path_calib)

        # Extract intrinsics matrix
        camera_intrinsics = torch.from_numpy(calib_data['KK']).float()

        # Extract the relevant transformations
        for view in list_views:
            view_idx = int(view.split('_')[-1])

            # Load current transformation and invert to get cam2world
            R_world2cam = torch.from_numpy(calib_data[f"Rc_{view_idx}"]).float()
            T_world2cam = torch.from_numpy(calib_data[f"Tc_{view_idx}"]).float()
                                                      
            R_cam2world = torch.from_numpy(calib_data[f"Rc_{view_idx}"]).transpose(0, 1).float()
            T_cam2world = - torch.matmul(R_cam2world, torch.from_numpy(calib_data[f"Tc_{view_idx}"]).float())
            trafos_cam2world[view] = torch.cat([R_cam2world, T_cam2world], dim=1)
            trafos_world2cam[view] = torch.cat([R_world2cam, T_world2cam], dim=1)

        return {
            'trafos_cam2world': trafos_cam2world,
            'trafos_world2cam': trafos_world2cam,
            'camera_intrinsics': camera_intrinsics
        }

    def load_camera_calibration(self, views_lights_selection):
        cam_calibration = self.load_camera_calibration_from_file(
            self.path_data,
            views_lights_selection.keys()
        )

        self.camera_intrinsics = cam_calibration['camera_intrinsics']
        self.trafos_cam2world = cam_calibration['trafos_cam2world']

    @staticmethod
    def load_lights_from_file(path_data, view):
        # Load intensities for this view
        path_light_intensities = os.path.join(path_data, view, 'light_intensities.txt')
        light_intensities = torch.from_numpy(np.loadtxt(path_light_intensities)).float()

        # Load directions towards light for this view
        path_light_directions = os.path.join(path_data, view, 'light_directions.txt')
        dirs_to_light = torch.from_numpy(np.loadtxt(path_light_directions)).float()

        return light_intensities, dirs_to_light

    def load_lights(self, view, views_lights_selection):
        light_intensities, dirs_to_light = self.load_lights_from_file(self.path_data, view)

        # Transform light directions to world coordinates
        dirs_to_light = self.transform_directions_cam2world(self.trafos_cam2world[view],
                                                            dirs_to_light)

        # (-1 for the indexing, since the image numbering starts at 1!)
        self.lights[view] = {
            'light_indices': [i for i in views_lights_selection[view]],
            'intensities': light_intensities[[i - 1 for i in views_lights_selection[view]]],
            'light_direction': dirs_to_light[[i - 1 for i in views_lights_selection[view]]]
        }

    @staticmethod
    def load_object_mask(path_data, view):
        path_mask = os.path.join(path_data, view, "mask.png")
        object_mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED) != 0
        return object_mask

    def load_initial_object_mask(self, view):
        obj_mask = self.load_object_mask(self.path_data, view)

        H, W = obj_mask.shape
        obj_mask = obj_mask.flatten()

        self.object_masks[view] = obj_mask
        self.image_dimensions[view] = [H, W]

    def ray_mesh_intersection(self, view, mesh):
        # Path handling for precomputation
        path_precomputed = os.path.join(self.path_precomputed, 'ray_mesh_intersections')
        path_file = os.path.join(path_precomputed, f"{view}_ray_mesh_intersections.pt")

        # Check if precomputed file can be found
        if os.path.exists(path_file):
            if not self.precomputation_msg_intersection_done:
                log.info(
                    f"Found existing files for ray mesh intersection. Using all available files in {path_precomputed}")
                self.precomputation_msg_intersection_done = True

            output_raycast = torch.load(path_file)

        else:
            H, W = self.image_dimensions[view]
            output_raycast = cast_camera_rays_on_mesh_brdf(
                mesh,
                self.camera_intrinsics,
                H,
                W,
                self.trafos_cam2world[view],
                self.object_masks[view]
            )

            # Save the results as precomputed
            os.makedirs(path_precomputed, exist_ok=True)
            torch.save(
                output_raycast,
                path_file
            )

        # Read values
        vertex_idxs_of_hit_faces = output_raycast['vertex_idxs_of_hit_faces']
        barycentric_coords = output_raycast['barycentric_coords']
        hit_ray_idxs = output_raycast['hit_ray_idxs']
        intersection_pts = output_raycast['intersection_pts']
        unit_ray_directions = output_raycast['unit_ray_directions']
        intersection_pts_normals = output_raycast['intersection_pts_normals']
        idx_hit_faces = output_raycast['idx_hit_faces']

        # Store viewing directions as negative ray directions of hitting rays
        viewing_directions = -unit_ray_directions[hit_ray_idxs]

        # Update object mask (some rays from the mask might not have hit)
        self._update_object_mask(view, hit_ray_idxs)

        return intersection_pts, intersection_pts_normals, viewing_directions, barycentric_coords, vertex_idxs_of_hit_faces, idx_hit_faces

    def cast_shadow_rays(self, intersection_points, intersection_pts_normals, light_direction, mesh,
                         view, light_idx, shadow_bias=0.0):
        """ Function to cast shadow rays from the intersection_points towards the light_direction and check for intersections with the mesh

        To avoid self intersections, we follow the shadow bias strategy
        (see e.g. here https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/ligth-and-shadows.html)
        """
        # Path handling for precomputation
        path_precomputed = os.path.join(self.path_precomputed, 'shadow_maps')
        path_precomputed_view = os.path.join(path_precomputed, view)
        path_file = os.path.join(path_precomputed_view, f"shaddow_map_{light_idx}_{shadow_bias}.pt")

        # Check if precomputed file can be found
        if os.path.exists(path_file):
            if not self.precomputation_msg_shadow_done:
                log.info(
                    f"Found existing files for shadow maps. Using all available files in {path_precomputed}")
                self.precomputation_msg_shadow_done = True

            is_in_shade = torch.load(path_file)

        else:
            is_in_shade = cast_shadow_rays_brdf(
                mesh,
                intersection_points,
                intersection_pts_normals,
                light_direction.unsqueeze(0).repeat(intersection_points.shape[0], 1),
                shadow_bias
            )

            # Save the results as precomputed
            os.makedirs(path_precomputed_view, exist_ok=True)
            torch.save(is_in_shade, path_file)

        return is_in_shade

    def _update_object_mask(self, view, indices_still_mask):
        """ Function to update the mask (e.g. after ray mesh intersection)

        indices_still_mask holds the indices to the entries in the flattend mask defined by self.masks[view] that are
        still valid after the update.
        """
        object_mask = self.object_masks[view]

        # We want to update all positive mask values, therefore the update mask has this many entries
        update_mask = np.zeros(np.sum(object_mask))
        update_mask[indices_still_mask] = 1
        object_mask[object_mask] = update_mask

    @staticmethod
    def transform_directions_cam2world(trafo_cam2world, directions, normalize=True):
        """ Function to transform directions from the camera frame of the diligent dataset into the world frame
        The multiplication by [1, -1, -1] corresponds to the change between x-right, y-down, z-towards the scene
        and x-right, y-up, z-away from scen
        """
        directions = directions * torch.tensor([1., -1., -1.])
        directions = (trafo_cam2world[:3, :3] @ directions.transpose(0, 1)).transpose(0, 1)

        if normalize:
            directions = torch.nn.functional.normalize(directions)
        return directions

    def load_image_and_normalize_by_light(self, view, light_idx, light_intensity):
        """
        Since we assume a directional light, the intensity of the light is the same for all pixels
        Normalize by light, so we do not have to take care of light intensity during training
        """
        path_im = os.path.join(self.path_data, view, f"{light_idx:03d}.png")

        # Load image, convert from bgr to rgb, scale to [0, 1] and extract masked pixels
        im = cv2.imread(path_im, cv2.IMREAD_UNCHANGED)[..., ::-1]
        im = torch.tensor(im / (2 ** DILIGENT_BIT_DEPTH - 1)).reshape(-1, 3)
        im = im[self.object_masks[view]]

        # Normalize by light intensity
        im /= light_intensity

        return im

    def get_light_intensity(self, view, light_idx):
        # Get index of lists in lights[view][index]
        idx = self.lights[view]['light_indices'].index(light_idx)
        return self.lights[view]['intensities'][idx]

    def get_data_full_image(self, view, light_idx):
        # Get data for image
        start_idx = self.image2index_dict[view][light_idx]['start_idx']
        top_idx = start_idx + self.image2index_dict[view][light_idx]['n_elements']
        item = self.__getitem__(range(start_idx, top_idx))

        # Read mask
        H, W = self.image_dimensions[view]
        object_mask = self.object_masks[view]

        # Re-assemble image
        image = torch.zeros([H * W, 3])
        image[object_mask] = item['rgb_values']

        result = {
            'item': item,
            'image': image.view(H, W, 3),
            'object_mask': object_mask.reshape(H, W)
        }

        return result

    def __len__(self):
        return self.rgb_values.shape[0]

    def __getitem__(self, idx: int):
        indices_duplicate = self.indices_duplicates[idx]

        item = {
            'rgb_values': self.rgb_values[idx],
            'ray_mesh_intersections': self.ray_mesh_intersections[indices_duplicate],
            'ray_mesh_intersections_barycentric_coords': self.ray_mesh_intersections_barycentric_coords[indices_duplicate],
            'ray_mesh_intersections_idx_vertices_hit_faces': self.ray_mesh_intersections_idx_vertices_hit_faces[indices_duplicate],
            'viewing_directions': self.viewing_directions[indices_duplicate],
            'light_directions': self.light_directions[idx],
            'ray_mesh_intersections_efuns': self.ray_mesh_intersections_efuns[indices_duplicate] if self.load_efuns else None,
            'ray_mesh_intersections_efuns_triangle': self.ray_mesh_intersections_efuns_triangle[indices_duplicate] if self.ray_mesh_intersections_efuns_triangle is not None else None,
            'is_in_shade': self.is_in_shade[idx]
        }

        return item
