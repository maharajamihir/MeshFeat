######################################################################################
#   File has been adapted from https://github.com/tum-vision/intrinsic-neural-fields #
######################################################################################


import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import imageio.v2 as imageio
import sys
sys.path.append("src/")

from util.mesh import MeshViewPreProcessor
from util.utils import load_obj_mask_as_tensor, load_cameras, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--split", type=str, help="Dataset split")
    args = parser.parse_args()
    return args


def preprocess_views(mesh_view_pre_proc, mesh_views_list_train, dataset_path):
    for mesh_view in tqdm(mesh_views_list_train):
        mesh_view_path = os.path.join(dataset_path, mesh_view)
        camCv2world, K = load_cameras(mesh_view_path)

        # Load depth map for building a mask
        obj_mask = load_obj_mask_as_tensor(mesh_view_path)

        # Load image
        img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))

        img[~obj_mask] = [255, 255, 255]
        imageio.imwrite(os.path.join(mesh_view_path, "image", "001.png"), img)
        img = torch.from_numpy(img).to(dtype=torch.float32)
        img /= 255.


        # Preprocess and cache the current view
        mesh_view_pre_proc.cache_single_view(camCv2world, K, obj_mask, img)

    mesh_view_pre_proc.write_to_disk()

def preprocess_dataset(split, dataset_path, path_to_mesh, out_dir, mesh_views_list_train):
    split_out_dir = os.path.join(out_dir, split)

    if os.path.exists(split_out_dir):
        raise RuntimeError(f"Error: You are trying to overwrite the following directory: {split_out_dir}")
    os.makedirs(split_out_dir, exist_ok=True)

    mesh_view_pre_proc = MeshViewPreProcessor(path_to_mesh, split_out_dir)

    preprocess_views(mesh_view_pre_proc, mesh_views_list_train, dataset_path)

def main():
    args = parse_args()
    config = load_config(args.config_path)
    dataset_path = config["data"]["raw_data_path"]
    out_dir = config["data"]["preproc_data_path"]
    mesh_path = config["data"]["mesh_path"]
    data_split = load_config(config["data"]["data_split"])
    mesh_views_list_train = data_split[f"mesh_views_list_{args.split}"]

    print(f"Preprocessing dataset: {args.split}")
    preprocess_dataset(split=args.split, 
                       dataset_path=dataset_path, 
                       path_to_mesh=mesh_path, 
                       out_dir=out_dir, 
                       mesh_views_list_train=mesh_views_list_train
                       )


if __name__ == "__main__":
    main()