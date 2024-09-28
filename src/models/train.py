import numpy as np
import torch 
import argparse
import os
import time
import random
import trimesh
import igl
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")

from models.model import MeshFeatModel
from util.utils import load_config, get_loss_fn, load_mesh, compute_psnr
from visualization.render_view import render_during_training
from other_methods.other_model import make_model
from other_methods.ray_dataloader import create_ray_dataloader
from data.dataset import MeshFeatDataset, MeshFeatDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--method", type=str, default="MeshFeat", help="Method to use for training")
    args = parser.parse_args()
    return args



class Trainer():

    def __init__(self, preproc_data_path, mesh, config, method="MeshFeat"):
        print(config["model"])
        print(config["training"])
        # if you don't want to use wandb, simply run `wandb disabled` in your terminal
        wandb.init(project="MeshFeat", config=config)
        wandb.log({"Experiment name": config["experiment"]})
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = config["method"] 
        self.mesh = mesh
        # load training data
        preproc_data_path_train = os.path.join(preproc_data_path, "train")
        barys_path_train = os.path.join(preproc_data_path_train,"barycentric_coords.npy")
        target_rgbs_path_train = os.path.join(preproc_data_path_train,"expected_rgbs.npy")
        face_idxs_path_train = os.path.join(preproc_data_path_train,"face_idxs.npy")
        self.barys_train = torch.from_numpy(np.load(barys_path_train)).to(self.device)
        self.target_rgbs_train = torch.from_numpy(np.load(target_rgbs_path_train)).to(self.device)
        face_idx_train_np = np.load(face_idxs_path_train)
        triangles_train_np = self.mesh.faces[face_idx_train_np]
        self.face_idxs_train = torch.from_numpy(face_idx_train_np).to(self.device)
        self.triangles = torch.tensor(triangles_train_np).to(self.device)
        triangles_coords = torch.tensor(self.mesh.vertices[triangles_train_np], dtype=torch.float32).to(self.device)  # B x 3 x 3
        self.points_xyz_train = torch.matmul(self.barys_train.unsqueeze(1), triangles_coords).squeeze(1)  # B x 3

        self.calculate_validation_loss = config["training"]["calculate_validation_loss"]
        if self.calculate_validation_loss:
            # load validation data
            preproc_data_path_val = os.path.join(preproc_data_path, "val")
            barys_path_val = os.path.join(preproc_data_path_val,"barycentric_coords.npy")
            target_rgbs_path_val = os.path.join(preproc_data_path_val,"expected_rgbs.npy")
            face_idxs_path_val = os.path.join(preproc_data_path_val,"face_idxs.npy")
            self.barys_val = torch.from_numpy(np.load(barys_path_val)).to(self.device)
            self.target_rgbs_val = torch.from_numpy(np.load(target_rgbs_path_val)).to(self.device)
            face_idx_val_np = np.load(face_idxs_path_val)
            triangles_val_np = self.mesh.faces[face_idx_val_np]
            self.face_idxs_val = torch.from_numpy(face_idx_val_np).to(self.device)
            self.triangles_val = torch.tensor(triangles_val_np).to(self.device)
            triangles_coords_val = torch.tensor(self.mesh.vertices[triangles_val_np], dtype=torch.float32).to(self.device)  # B x 3 x 3
            self.points_xyz_val = torch.matmul(self.barys_val.unsqueeze(1), triangles_coords_val).squeeze(1)  # B x 3

        # experiment
        self.split = config["experiment"]["split"]
        self.experiment_name = config["experiment"]["name"]

        # training config
        self.out_dir = os.path.join(config["training"]["out_dir"], self.split)
        self.loss_type = config["training"]["loss_type"]
        self.loss_fn = get_loss_fn(self.loss_type)

    def loss_func_with_reg(self, pred, target):
        loss = self.loss_fn(pred, target)
        if float(config["model"]["regularization_lat_feat"]) > 0.:
            regularization_term = self.model.encoding_layer.get_regularization(float(config["model"]["regularization_lat_feat"]))
            total_loss = torch.add(loss, regularization_term)
            wandb.log({"L1_loss": loss, "reg_term": regularization_term})
            return total_loss
        return loss

    def _training_step(self, batch):  
        # Zero the gradients
        self.optimizer.zero_grad(set_to_none=True)
        # Forward pass       
        if self.method == "MeshFeat":
            input_tensors = batch['x']
            target_rgbs = batch['y']
            triangles = batch['triangle']
            output_rgb = self.model(input_tensors, triangles)

        else:
            target_rgbs = batch["expected_rgbs"].to(self.device)
            output_rgb = self.model(batch)

        self.loss = self.criterion(output_rgb, target_rgbs)
        self.loss.backward()
        self.optimizer.step()
        acc_loss = self.loss.item()
        psnr = compute_psnr(output_rgb.cpu().detach().numpy(), target_rgbs.cpu().detach().numpy())
        return acc_loss, psnr

    def _validation_step(self, batch):
        if self.method == "MeshFeat":
            input_tensors = batch['x']
            target_rgb_val = batch['y']
            tris = batch['triangle']
            output_rgb_val = self.model(input_tensors, tris)
        else:
            target_rgb_val = batch["expected_rgbs"].to(self.device)
            output_rgb_val = self.model(batch)
        loss_val = self.criterion(output_rgb_val, target_rgb_val).item()
        val_psnr = compute_psnr(output_rgb_val.cpu().detach().numpy(), target_rgb_val.cpu().detach().numpy())
        return loss_val, val_psnr
        
    def train(self):
        print("Training started")
        print("Number of training samples: ", len(self.barys_train))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        config = self.config

        batch_size = self.config["training"]["batch_size"]
        num_epochs = self.config["training"]["num_epochs"]
        render_every = self.config["training"].get("render_every", 0)
        print_every = self.config["training"].get("print_every", 1)
        
        # define model
        if self.method == "MeshFeat":
            self.model = MeshFeatModel(
                num_layers=self.config["model"]["num_layers"],
                hidden_dim=self.config["model"]["mlp_hidden_dim"], 
                output_dim=self.config["model"].get("output_dim", 3), 
                len_feature_vec=self.config["model"]["latent_corner_features_dim"], 
                mesh=self.mesh,
                resolutions=self.config["model"]["resolutions"],
                reg_type=self.config["training"]["reg_type"],
                use_zero_init=self.config["model"]["use_zero_init"],
                neural=self.config["model"].get("neural", True)
                )
            points_xyz = None
            # Define the loss
            self.criterion = self.loss_func_with_reg 

        else:
            print("Loading model") 
            self.model = make_model(self.config["model"], self.mesh)
            points_xyz = self.points_xyz_train
            self.criterion = self.loss_fn

        # move everything to device
        self.model = self.model.to(self.device)
        print(f"Model {self.method} has {sum(p.numel() for p in self.model.parameters())} parameters")
        


        self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=float(self.config['training']['learning_rate_mlp']), 
                )
        if self.method == "MeshFeat":
            # Create a Dataset instance
            dataset = MeshFeatDataset(x=self.barys_train.detach(), y=self.target_rgbs_train, triangles=self.triangles, points_xyz=points_xyz)
            dataset_val = MeshFeatDataset(x=self.barys_val.detach(), y=self.target_rgbs_val, triangles=self.triangles_val)
            shuffle = self.config["training"]["shuffle_dataloading"]

            # Create an Adam optimizer and LR scheduler
            self.optimizer = optim.Adam(
                [
                
                    {
                        "params": self.model.layers.parameters(),
                        "lr": float(self.config['training']['learning_rate_mlp']),
                        "weight_decay": float(self.config['training']['weight_decay_mlp'])
                    },
                    {
                        "params": self.model.encoding_layer.parameters(),
                        "lr": float(self.config['training']['learning_rate_lat_feat']),
                        "weight_decay": float(self.config['training']['weight_decay_lat_feat'])
                    },
                ],
                eps=1e-15
            )

            # Create a DataLoader
            dataloader = MeshFeatDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            dataloader_val = MeshFeatDataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle)
        else: 
            dataloader = create_ray_dataloader(
                os.path.join(self.config["data"]["preproc_data_path"],"train"),
                self.config["data"]["eigenfunctions_path"],
                self.config["model"].get("k"),
                self.config["model"].get("feature_strategy", "efuncs"),
                self.mesh,
                self.config["data"].get("rescale_strategy", "standard"),
                self.config["data"].get("embed_strategy"),
                self.config["data"].get("eigenvalues_path"),
                batch_size,
                shuffle=True,
                drop_last=self.config["data"].get("train_drop_last", True),
                device=self.device)
            dataloader_val = create_ray_dataloader(
                os.path.join(self.config["data"]["preproc_data_path"],"val"),
                self.config["data"]["eigenfunctions_path"],
                self.config["model"].get("k"),
                self.config["model"].get("feature_strategy", "efuncs"),
                self.mesh,
                self.config["data"].get("rescale_strategy", "standard"),
                self.config["data"].get("embed_strategy"),
                self.config["data"].get("eigenvalues_path"),
                batch_size,
                shuffle=True,
                drop_last=self.config["data"].get("train_drop_last", True),
                device=self.device)
            
        print(f"Number of batches: {len(dataloader)}")
        # Training loop
        losses = []
        val_losses = []
        best_val_psnr = 0.
        best_epoch = num_epochs
        self.val_psnrs = []
        print(f"Dataset Length {self.barys_train.shape[0]}")
        print(f"Number of Epochs: {num_epochs}")
        for epoch in range(num_epochs):
            acc_loss = 0.0
            acc_psnr = 0.0
            start_time = time.time()
            for batch in dataloader:
                acc_loss_step, psnr_step = self._training_step(batch)
                acc_loss += acc_loss_step
                acc_psnr += psnr_step
            
            epoch_time = time.time() - start_time
            if self.calculate_validation_loss:
                loss_val_acc = 0.0
                val_psnr_acc = 0.0
                for batch in dataloader_val:
                    loss_val, val_psnr = self._validation_step(batch) 
                    loss_val_acc+=loss_val
                    val_psnr_acc+=val_psnr

                
                acc_loss_val = loss_val_acc/len(dataloader_val)
                val_psnr = val_psnr_acc/len(dataloader_val)
                self.val_psnrs.append(val_psnr)

            acc_loss = acc_loss / len(dataloader)
            acc_psnr = acc_psnr / len(dataloader) # train psnr

            # Print the loss based on printing frequency
            if print_every > 0 and epoch % print_every == 0:
                lr = [round(param_group['lr'], 5) for param_group in self.optimizer.param_groups]
                if self.calculate_validation_loss:
                    print(f'Epoch [{epoch+1}/{num_epochs}], \tTraining Loss: {round(acc_loss, 6)}, \tTrain PSNR: {round((acc_psnr), 5)}, \tValidation Loss: {round(acc_loss_val, 5)}, \tValidation PSNR: {round((val_psnr), 5)}, \tElapsed Time: {round(epoch_time, 4)}, Learning rate: {lr}')
                    wandb.log({"loss_train": acc_loss, "loss_val": acc_loss_val, "train_psnr": acc_psnr, "val_psnr": val_psnr})
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}], \tTraining Loss: {round(acc_loss,6)}, \tElapsed Time: {round(epoch_time, 4)}, \tLearning rate: {lr}')

            # Render image based on rendering frequency
            if render_every > 0 and epoch % render_every == 0:
                print(f"Rendering Image for Epoch [{epoch+1}/{num_epochs}]")
                render_during_training(config=self.config, model=self.model, epoch=epoch+1, device=self.device)
        
            # Save the model if best val loss:
            if self.calculate_validation_loss and val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch+1
                print(f"Saving Model with Validation PSNR: {best_val_psnr}")
                torch.save(self.model, os.path.join(self.out_dir, "model.pt"))
            # store losses
            losses.append(acc_loss)
            if self.calculate_validation_loss:
                val_losses.append(acc_loss_val)

        # Plot the loss per epoch in case you're not using wandb
        self.plot_loss(losses, best_epoch, val_losses)

        print("Training complete")
        print("Saving model")
        model_save_path = os.path.join(self.out_dir, "model.pt")
        if not self.calculate_validation_loss:
            torch.save(self.model, model_save_path)
        return self.model

    def plot_loss(self, losses, best_epoch, val_losses=None):
        save_dir = os.path.join("results_texture", self.split, "figures")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        x = np.arange(1,  len(losses)+1)
        plt.title(f"Training Loss {self.experiment_name}")
        plt.xlabel("Epoch")
        plt.ylabel(f"{self.loss_type} Loss")
        plt.plot(x, losses, label ='training loss')
        if self.calculate_validation_loss:
            plt.plot(x, val_losses, '-.', label ='validation loss')
            plt.plot(best_epoch, val_losses[best_epoch-1], 'o', label ='best validation loss')
            with open(os.path.join("results_texture", self.split, "log.txt"), "w") as f:
                f.write(f"Best epoch: {best_epoch}")
            np.save(os.path.join("results_texture", self.split, "val_psnr.npy"), self.val_psnrs)
        plt.xscale('log')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(save_dir, "loss_per_epoch.png"))

if __name__ == "__main__":
    args = parse_args()

    # load config file with paths and hyperparams etc
    config = load_config(args.config_file)
    preproc_data_path = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    seed = config["seed"]

    # set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

    # load the preprocessed data
    mesh = load_mesh(mesh_path)
    trainer = Trainer(preproc_data_path=preproc_data_path,
                      mesh=mesh, 
                      config=config,
                      method=args.method)
    trainer.train()
