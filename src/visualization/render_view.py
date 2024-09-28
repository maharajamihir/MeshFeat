######################################################################################
#   File has been adapted from https://github.com/tum-vision/intrinsic-neural-fields #
######################################################################################
import argparse
import cv2
import torch
import sys
from tqdm import tqdm
sys.path.append("src/")

from util.cameras import load_extr_and_intr_camera
from util.utils import load_config
from visualization.renderer import make_renderer_with_trained_model, make_renderer_during_training
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    return args

def _render_view(renderer, camCv2world, cam_intrinsic):
    renderer.set_height(cam_intrinsic["height"])
    renderer.set_width(cam_intrinsic["width"])
    return renderer.render(camCv2world, cam_intrinsic["K"])

def execute_rendering(renderer, height, width, render_imgs,data_path, render_dir, epoch=None):
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    print(f"Rendering {len(render_imgs)} images")
    for img in tqdm(render_imgs):
        camera_path = os.path.join(data_path, img, "depth", "cameras.npz")
        camCv2world, K = load_extr_and_intr_camera(camera_path)
        cam_intrinsic = {
            "K": K,
            "height": height,
            "width": width,
        }

        view = _render_view(renderer, camCv2world, cam_intrinsic)
        view = view * 255
        img_name = f"{img}_{epoch}" if epoch is not None else img
        out_path = os.path.join(render_dir, (img_name + ".png")) 
        cv2.imwrite(out_path, view[..., ::-1])

def render_during_training(config, model, epoch, device):
    renderer = make_renderer_during_training(model, config) 
    height = config["data"]["img_height"]
    width = config["data"]["img_width"]

    render_imgs = config["data"]["train_render_img_names"]
    data_path = config["data"]["raw_data_path"]
    render_dir = os.path.join(config["data"]["render_img_directory"], config["experiment"]["split"], "train")
    os.makedirs(render_dir, exist_ok=True)
    execute_rendering(renderer, height, width, render_imgs,data_path, render_dir, epoch)



def main(config=None):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if config == None:
        args = parse_args()

        config = load_config(args.config_path)

    renderer = make_renderer_with_trained_model(config, device=device) 

    height = config["data"]["img_height"]
    width = config["data"]["img_width"]

    data_split = load_config(config["data"]["data_split"])
    render_imgs = data_split["mesh_views_list_test"]
    data_path = config["data"]["raw_data_path"]
    render_dir = os.path.join(config["data"]["render_img_directory"], config["experiment"]["split"], "eval")
    execute_rendering(renderer, height, width, render_imgs,data_path, render_dir)
   

if __name__ == "__main__":
    main()
