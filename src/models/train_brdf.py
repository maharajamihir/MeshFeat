import os
import sys
sys.path.append("src/")

import argparse
import logging
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import save_image

from util.utils import load_config
from data.diligent_dataset import Dataset_Diligent_MV
from data.dataset import MeshFeatDataLoader
from models.model import MeshFeatModel
from other_methods.other_model import TextureField
from util.utils import LossWithGammaCorrection, Metrics, linear2sRGB, time_method
from util.brdf import render_disney_brdf


VALID_OBJECT_NAMES = ['bear', 'buddha', 'cow', 'pot2', 'reading']
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results_brdf'
))
PATH_DILIGENT_MV = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', 'data', 'DiLiGenT-MV'
))

log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model on brdf data")
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument(
        "--object",
        type=str,
        default="cow",
        choices=VALID_OBJECT_NAMES,
        help=f"Object to train on. Can be one of {VALID_OBJECT_NAMES}",
    )
    args = parser.parse_args()
    return args

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


class Trainer():
    def __init__(self, cfg, path_output):
        self.cfg = cfg
        self.path_output = path_output
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        train_data, test_data = Dataset_Diligent_MV.create_train_test_split(
            **cfg['data'],
            path_diligent_mv=PATH_DILIGENT_MV,
            LBO_indices=cfg['model']['k'] if cfg['method'] == 'INF' else None
        )
        self.train_data = train_data
        self.test_data = test_data

        self.train_dataloader = MeshFeatDataLoader(
            self.train_data,
            batch_size=2**self.cfg['dataloader']['batch_size_exp'],
            shuffle=True
        )

        # Create model
        output_dim = 12

        if cfg['method'] == 'MeshFeat':
            self.reg_weight =  float(cfg['training']['regularization_weight'])

            self.model = MeshFeatModel(
                **self.cfg['model'],
                mesh=self.train_data.mesh,
                output_dim=output_dim
            )
            self.model.to(self.device)
            self.model_size = sum(p.numel() for p in self.model.parameters())

            # Create an Adam optimizer and LR scheduler
            self.optimizer = optim.Adam(
                [
                    {
                        "params": self.model.layers.parameters(),
                        "lr": self.cfg['training']['learning_rate_mlp'],
                        "weight_decay": float(self.cfg['optimizer']['weight_decay_mlp'])
                    },
                    {
                        "params": self.model.encoding_layer.parameters(),
                        "lr": self.cfg['training']['learning_rate_features'],
                        "weight_decay": float(self.cfg['optimizer']['weight_decay_features'])
                    },
                ],
                eps=1e-15
            )
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg['optimizer']['lr_sched_gamma'])

        elif cfg['method'] == 'INF':
            self.model = TextureField(
                num_layers=cfg['model']['num_layers'],
                in_dim=self.train_data.n_efuns_used,
                hidden_dim=cfg['model']['mlp_hidden_dim'],
                skip_layer_idx=cfg['model']['skip_layer_idx'],
                return_rgb=True,
                out_dim=output_dim,
                batchnorm=cfg['model']['batchnorm'],
                no_output_nonlin=True
            )
            self.model.to(self.device)

            # Save model size
            self.model_size = sum(p.numel() for p in self.model.parameters())
            self.model_size += self.train_data.n_efuns_used * self.train_data.mesh.vertices.shape[0]

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg['training']['learning_rate']
            )
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg['optimizer']['lr_sched_gamma'])

        elif cfg['method'] == 'TF_RFF':
            self.model = TextureField(
                num_layers=cfg['model']['num_layers'],
                in_dim=cfg['model']['k'],
                hidden_dim=cfg['model']['mlp_hidden_dim'],
                skip_layer_idx=cfg['model']['skip_layer_idx'],
                return_rgb=True,
                out_dim=output_dim,
                batchnorm=cfg['model']['batchnorm'],
                input_feature_embed=cfg['model']['feature_strategy'],
                embed_dim=cfg['model']['k'],
                embed_include_input=cfg['model']['embed_include_input'],
                embed_std=cfg['model']['embed_std'],
                no_output_nonlin=True
            )
            self.model.to(self.device)

            # Save model size
            self.model_size = sum(p.numel() for p in self.model.parameters())
            self.model_size += self.model.embedding.B.numel()

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg['training']['learning_rate']
            )
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg['optimizer']['lr_sched_gamma'])
                
        else:
            raise ValueError(f"Method {cfg['method']} not implemented")

        # Create loss function
        self.loss_fun_data = LossWithGammaCorrection(cfg['training']['loss_type'])

        # Use subset of train images for rendering during training
        indices_rendering_train_set = torch.linspace(0, self.train_data.n_available_view_light_pairs-1, cfg['data']['n_train_views_for_metrics']).long()
        self.view_lights_rendering_train = [train_data.available_view_light_pairs[ind] for ind in indices_rendering_train_set]

        self.metrics = Metrics()

        log.info("Done with initialization")

    def render_images_dataset(self, dataset='test', view_lights=None):
        assert dataset in ['test', 'train'], "Dataset must be 'test' or 'train'"
        data = self.test_data if dataset == 'test' else self.train_data

        self.model.eval()
        with torch.no_grad():
            if view_lights is None:
                view_lights = data.available_view_light_pairs

            images_linear = []
            images_gt_linear = []
            images_srgb = []
            images_gt_srgb = []
            masks = []

            for v, l in view_lights:
                data_full_image = data.get_data_full_image(v, l)
                item = data_full_image['item']
                image_gt = data_full_image['image']
                mask = data_full_image['object_mask']

                # Evaluate the model
                if self.cfg['method'] == 'MeshFeat':
                    model_out = self.model(
                        item['ray_mesh_intersections_barycentric_coords'].to(self.device),
                        item['ray_mesh_intersections_idx_vertices_hit_faces'].to(self.device)
                    )
                elif self.cfg['method'] == 'INF':
                    model_in = {
                        'xyz': None,
                        'eigenfunctions': item['ray_mesh_intersections_efuns'].to(self.device),
                    }
                    model_out = self.model(model_in)
                elif self.cfg['method'] == 'TF_RFF':
                    model_in = {
                        'xyz': item['ray_mesh_intersections'].to(self.device),
                        'eigenfunctions': None,
                    }
                    model_out = self.model(model_in)
                else:
                    raise ValueError(f"Method {self.cfg['method']} not implemented")
                
                # Render the rays
                rgb_rendering = render_disney_brdf(
                    model_out,
                    view_dirs_shading_cosy=item['viewing_directions'].to(self.device),
                    light_dirs_shading_cosy=item['light_directions'].to(self.device),
                    is_in_shade=item['is_in_shade'].to(self.device),
                )

                # Load light and re-scale loaded images
                light_intensity = data.get_light_intensity(v, l).float()
                image_gt *= light_intensity

                # Assemble the rendered image
                rendering = torch.zeros_like(image_gt)
                rendering[mask] = light_intensity * rgb_rendering.cpu().float()

                images_linear.append(rendering)
                images_gt_linear.append(image_gt)
                images_srgb.append(linear2sRGB(torch.clip(rendering, 0.0, 1.0)))
                images_gt_srgb.append(linear2sRGB(image_gt))
                masks.append(mask)

        self.model.train()

        return {
            'images_linear': images_linear,
            'images_gt_linear': images_gt_linear,
            'images_srgb': images_srgb,
            'images_gt_srgb': images_gt_srgb,
            'masks': masks,
        }

    def train_step(self, batch, add_reg_loss=False):
        rgb_gt = batch['rgb_values'].to(self.device)
        viewing_dirs = batch['viewing_directions'].to(self.device)
        light_dirs = batch['light_directions'].to(self.device)
        is_in_shade = batch['is_in_shade'].to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Evaluate the model
        if self.cfg['method'] == 'MeshFeat':
            intersection_pts_barycentric_coords = batch['ray_mesh_intersections_barycentric_coords'].to(self.device)
            itersection_pts_idx_vertices_hit_faces = batch['ray_mesh_intersections_idx_vertices_hit_faces'].to(self.device)
            model_out = self.model(intersection_pts_barycentric_coords, itersection_pts_idx_vertices_hit_faces)

        elif self.cfg['method'] == 'INF':
            model_in = {
                'xyz': None,
                'eigenfunctions': batch['ray_mesh_intersections_efuns'].to(self.device),
            }
            model_out = self.model(model_in)

        elif self.cfg['method'] == 'TF_RFF':
            model_in = {
                'xyz': batch['ray_mesh_intersections'].to(self.device),
                'eigenfunctions': None,
            }
            model_out = self.model(model_in)

        else:
            raise ValueError(f"Method {self.cfg['method']} not implemented")
        
        # Render the rays
        rgb_rendering = render_disney_brdf(
            model_out,
            viewing_dirs,
            light_dirs,
            is_in_shade
        )

        # Clip the results
        rgb = torch.clip(rgb_rendering, 0.0, 1.0)

        # Compute loss
        # For points in shade, the rendering is always zero. In the data not (due to ambient light)
        mask_loss = torch.logical_not(is_in_shade)
        loss_data = self.loss_fun_data(rgb[mask_loss], rgb_gt[mask_loss])
        loss = loss_data

        loss_reg = None
        if self.cfg['method'] == 'MeshFeat':
            if self.reg_weight > 0:
                loss_reg = self.model.encoding_layer.get_regularization(self.reg_weight)
                loss = loss_data + loss_reg

        # Optimization step
        loss.backward()
        self.optimizer.step()

        # Accumulate losses
        self.loss_epoch_full += loss
        self.loss_epoch_data += loss_data
        if loss_reg is not None:
            self.loss_epoch_reg += loss_reg
        else:
            self.loss_epoch_reg = None 
        self.n_batches += 1
    
    def zero_losses_epoch(self):
        self.loss_epoch_full = torch.zeros([]).to(self.device)
        self.loss_epoch_data = torch.zeros([]).to(self.device)
        self.loss_epoch_reg = torch.zeros([]).to(self.device)

        self.n_batches = 0

    def train(self):
        add_reg_loss = True

        for epoch in range(self.cfg['training']['num_epochs']):
            self.zero_losses_epoch()

            # Train epoch
            for batch in self.train_dataloader:
                self.train_step(batch, add_reg_loss=add_reg_loss)

            # Update lr scheduler
            self.lr_scheduler.step()

            # Log loss
            log.info(f"Epoch [{epoch+1}/{self.cfg['training']['num_epochs']}], Training Loss: {self.loss_epoch_full}")

            # Render images
            if (self.cfg['logging']['rendering_interval'] > 0 and epoch % self.cfg['logging']['rendering_interval'] == 0) or epoch == self.cfg['training']['num_epochs']-1:
                log.info("Doing rendering")
                renderings_train = self.render_images_dataset('train', view_lights=self.view_lights_rendering_train)
                renderings_test = self.render_images_dataset('test')

                self.save_images(renderings_test, suffix=epoch)

                log.info("Computing metrics")
                metrics_train = self.metrics.compute_metrics(renderings_train)
                metrics_test = self.metrics.compute_metrics(renderings_test)

                # Print metrics
                log.info(f"Epoch [{epoch+1}/{self.cfg['training']['num_epochs']}], Train PSNR: {np.mean(metrics_train['psnrs_srgb'])}, Test PSNR: {np.mean(metrics_test['psnrs_srgb'])}")

                # Store final numbers and render images
                if epoch == self.cfg['training']['num_epochs']-1:
                    self.results_final = {
                        'psnr (final)':  np.mean(metrics_test['psnrs_srgb']).item(),
                        'ssim (final)':  np.mean(metrics_test['ssims']).item(),
                        'dssim (final)': np.mean(metrics_test['dssims']).item(),
                        'lpips (final)': torch.mean(metrics_test['lpips']).item(),
                    }

        # Save model
        log.info("Saving Model")
        torch.save(self.model.state_dict(), os.path.join(self.path_output, 'trained_model.pth'))

        self.final_epoch = epoch

    def save_images(
        self,
        renderings,
        sample_renderings_n_pixel_edge_top=5,
        sample_renderings_n_pixel_edge_left=5,
        sample_renderings_n_pixel_edge_right=5,
        sample_renderings_n_pixel_edge_bottom=5,
        suffix="",
        im_idx=0
    ):
        rendering = renderings['images_srgb'][im_idx]
        rendering_gt = renderings['images_gt_srgb'][im_idx]
        rendering_mask = renderings['masks'][im_idx]
        rendering[np.logical_not(rendering_mask)] = 1.0
        rendering_gt[np.logical_not(rendering_mask)] = 1.0

        # Crop images to remove large boundaries
        y_inds_nonzero, x_inds_nonzero = np.nonzero(rendering_mask)
        x_min = np.maximum(x_inds_nonzero.min() - sample_renderings_n_pixel_edge_left, 0)
        x_max = np.minimum(x_inds_nonzero.max() + sample_renderings_n_pixel_edge_right, rendering_mask.shape[1])
        y_min = np.maximum(y_inds_nonzero.min() - sample_renderings_n_pixel_edge_top, 0)
        y_max = np.minimum(y_inds_nonzero.max() + sample_renderings_n_pixel_edge_bottom, rendering_mask.shape[0])
        rendering = rendering[y_min:y_max, x_min:x_max, :]
        rendering_gt = rendering_gt[y_min:y_max, x_min:x_max, :]

        # Create paths and save images
        path_ims = os.path.join(self.path_output, 'renderings')
        os.makedirs(path_ims, exist_ok=True)

        save_image(
            rendering.permute(2, 0, 1),
            os.path.join(path_ims, f"rendering_{suffix}.png")
        )

        save_image(
            rendering_gt.permute(2, 0, 1),
            os.path.join(path_ims, f"rendering_{suffix}_gt.png")
        )

    def time_method(self):
        # Get input batch:
        batch = self.test_data[range(2 ** self.cfg['timing']['batch_size_exp'])]

        if self.cfg['method'] == 'MeshFeat':
            dummy_input = {
                'bary': batch['ray_mesh_intersections_barycentric_coords'].to(self.device),
                'triangle': batch['ray_mesh_intersections_idx_vertices_hit_faces'].to(self.device)
            }
            call_fun = self.model
            
        elif self.cfg['method'] == 'INF':
            model_in = {
                'xyz': None,
                'eigenfunctions': batch['ray_mesh_intersections_efuns'].to(self.device),
            }
            dummy_input = {
                'efuns_triangle': batch['ray_mesh_intersections_efuns_triangle'].to(self.device),
                'barycentric_coordinates': batch['ray_mesh_intersections_barycentric_coords'].to(self.device)
            }
            call_fun = self.model.eval_inf_with_interpolation
            # Make sure that calling the model with the changed function does yield the same result
            assert torch.max(torch.abs(call_fun(**dummy_input) - self.model(model_in))) < 1e-4, f"INF timing function evaluated differently than the original function! Max error: {torch.max(torch.abs(call_fun(**dummy_input) - self.model(model_in)))}"

        elif self.cfg['method'] == 'TF_RFF':
            model_in = {
                'xyz': batch['ray_mesh_intersections'].to(self.device),
                'eigenfunctions': None,
            }
            dummy_input = {
                'batch': model_in
            }
            call_fun = self.model

        self.model.eval()
        self.timing_inference = time_method(call_fun, dummy_input, repetitions=self.cfg['timing']['repetitions'])
        log.info(f"Timing: {self.timing_inference}")

    def write_results(self):
        # Store results
        with open(os.path.join(self.path_output, 'results.yaml'), 'w') as f:
            log.info("storing results!")
            model_specs = {
                'number_params': self.model_size,
                'eval_time': self.timing_inference.item(),
                'batch_size_timing': 2 ** self.cfg['timing']['batch_size_exp'],
            }
            log.info(model_specs)
            log.info(self.results_final)
            yaml.dump(model_specs, f, sort_keys=False)
            yaml.dump(self.results_final, f, sort_keys=False)


if __name__ == '__main__':
    args = parse_args()

    # load config file
    config = load_config(args.config_file)
    config['data']['name_object'] = args.object + 'PNG'

    # output directory
    path_output = os.path.join(OUTPUT_DIR, config['method'], args.object)
    os.makedirs(path_output, exist_ok=True)

    # set up logging
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        os.path.join(path_output, 'training_brdf.log'),
        mode='w'
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stdout_handler)
    log.info(f"Starting training: {config['method']} - {args.object}")

    # set seeds
    set_seeds(config['seed'])

    # train model
    trainer = Trainer(config, path_output)

    # Train the model
    trainer.train()

    trainer.time_method()
    trainer.write_results()

    log.info("Training done")