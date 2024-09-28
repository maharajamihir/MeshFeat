import numpy as np
import pandas as pd
import torch
import gc
import imageio.v2 as imageio
import os
import argparse
import json
from tqdm import tqdm
from lpips import LPIPS
import sys
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
sys.path.append("src/")
from util.utils import load_config, load_obj_mask_as_tensor, load_trained_model, batchify_dict_data, load_mesh, time_method
from other_methods.mesh import load_first_k_eigenfunctions, get_k_eigenfunc_vec_vals
import warnings
warnings.filterwarnings("ignore")


TEST_RUNS = 50

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--test_speed", action="store_true")
    args = parser.parse_args()
    return args

def dssim(pred_image, real_image):
    """
    Structural Dissimilarity based on Structural Similarity Index Metric (SSIM)
    """
    assert pred_image.shape == real_image.shape and pred_image.shape[2] == 3
    return (1 - structural_similarity(pred_image, real_image, channel_axis=-1)) / 2

def test_speed(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load mesh
    mesh = load_mesh(config["data"]["mesh_path"])
    method = config["method"]
    if method == "intrinsic":
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
    model.eval()
    model.to(device)
    # load training data
    preproc_data_path = config["data"]["preproc_data_path"]
    preproc_data_path_train = os.path.join(preproc_data_path, "val")
    barys_path_train = os.path.join(preproc_data_path_train,"barycentric_coords.npy")
    unit_ray_dirs_path_train = os.path.join(preproc_data_path_train,"unit_ray_dirs.npy")
    face_idxs_path_train = os.path.join(preproc_data_path_train,"face_idxs.npy")
    v_idxs_face_path_train = os.path.join(preproc_data_path_train,"vids_of_hit_faces.npy")
    
    face_idx_train_np = np.load(face_idxs_path_train)
    triangles_train_np = mesh.faces[face_idx_train_np]
    # convert to tensors
    barys_train = torch.from_numpy(np.load(barys_path_train)).to(device)
    face_idxs_train = torch.from_numpy(face_idx_train_np).to(device)
    triangles_coords = torch.tensor(mesh.vertices[triangles_train_np], dtype=torch.float32).to(device)  # B x 3 x 3
    points_xyz_train = torch.matmul(barys_train.unsqueeze(1), triangles_coords).squeeze(1)  # B x 3
    unit_ray_dirs_train = torch.from_numpy(np.load(unit_ray_dirs_path_train)).to(device)
    v_idxs_face_train = torch.from_numpy(np.load(v_idxs_face_path_train)).to(device)

    batch_size = 2**15

    if method == "intrinsic":
        efuncs_local = get_k_eigenfunc_vec_vals(efuncs,
                                        v_idxs_face_train.to("cpu"),
                                        barys_train.to("cpu"))
                                        
        vertex_idxs_of_hit_faces = v_idxs_face_train.to("cpu").reshape(-1)  # B*3
        eigenfuncs_triangle = efuncs[vertex_idxs_of_hit_faces]  # B*3 x k
        eigenfuncs_triangle = eigenfuncs_triangle.reshape(v_idxs_face_train.size()[0], 3, -1)  # B x 3 x k

        data = {
            "eigenfunctions" : efuncs_local[:batch_size],
            "unit_ray_dirs": unit_ray_dirs_train[:batch_size],
            "hit_face_idxs": face_idxs_train[:batch_size],
            "eigenfunctions_triangle": eigenfuncs_triangle[:batch_size],
            "barycentric_coordinates": barys_train[:batch_size]
        }
        del barys_train
        del points_xyz_train
        del triangles_coords
        gc.collect()
    else: 
        data = {
            "barys": barys_train[:batch_size],
            "xyz": points_xyz_train[:batch_size], 
            "unit_ray_dirs": unit_ray_dirs_train[:batch_size], 
            "hit_face_idxs": face_idxs_train[:batch_size] 
        }


    #num_rays = unit_ray_dirs_train.shape[0]
    #assert num_rays > 0
    faces_all = torch.from_numpy(mesh.faces).to(device)
    print("Calculating time")
    # breakpoint()
    # batches = batchify_dict_data(data, num_rays, batch_size)
    # batch = batches[0]
    model.eval()        
    faces = faces_all[data["hit_face_idxs"]]
    
    if method == "MeshFeat":
         time_taken = time_method(model, {"bary": data["barys"], "triangle": faces})

    # TODO: Check that this runs. Not yet checked
    elif method == "intrinsic":
        time_taken = time_method(
            model.eval_inf_with_interpolation,
            {
                'efuns_triangle': data['eigenfunctions_triangle'].to(device),
                'barycentric_coordinates': data['barycentric_coordinates'].to(device),
            }
        )
    else:
         time_taken = time_method(model, {"batch": data})
    report = f"""
    Method: {method}
    Batch Size: {batch_size}
    Number of iterations: {TEST_RUNS}  (simulating rendering)
    Time taken: {time_taken} seconds
    """
    print(report)
    save_pth = os.path.join(config["data"]["render_img_directory"], config["experiment"]["split"])
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    with open(os.path.join(save_pth, "speed.txt"), "w") as f:
        f.write(report)
    return time_taken

    

def main(config=None, speed_test=False):
    report = {}
    if config is None:
        args = parse_args()
        config = load_config(args.config_path)
        if args.test_speed:
            speed = test_speed(config)
            report["speed"] = speed
    elif speed_test:
        speed = test_speed(config)
        report["speed"] = speed


    data_split = load_config(config["data"]["data_split"])
    images = data_split["mesh_views_list_test"]
    gt_dir = config["data"]["raw_data_path"]
    pred_dir = os.path.join(config["data"]["render_img_directory"], config["experiment"]["split"], "eval")
    # Load the images
    results = {
        "image": [],
        "psnr": [],
        "mse": [],
        "ssim": [],
        "dssim": [],
        "lpips": []
        }
    
    lpips_loss_fn = LPIPS(net='alex', version='0.1')
    print("Running experiments")
    for img in tqdm(images):
        gt_img_path = os.path.join(gt_dir, img, "image", "001.png")
        pred_img_path = os.path.join(pred_dir, img + ".png")
        mesh_view_path = os.path.join(gt_dir, img)
        obj_mask = load_obj_mask_as_tensor(mesh_view_path)
        try:
            gt_img = imageio.imread(gt_img_path)
        except:
            gt_img = imageio.imread(os.path.join(mesh_view_path, "image", "000.png"))
            gt_img[~obj_mask] = [255, 255, 255]
            imageio.imwrite(os.path.join(mesh_view_path, "image", "001.png"), gt_img)
        pred_img = imageio.imread(pred_img_path)

        gt_img_masked = gt_img[obj_mask]
        pred_img_masked = pred_img[obj_mask]
        
        # Calculate the PSNR between the two images
        psnr = peak_signal_noise_ratio(image_true=gt_img_masked, image_test=pred_img_masked, data_range=255.)
        mse = mean_squared_error(image0=gt_img_masked, image1=pred_img_masked)

        ssim = structural_similarity(im1=gt_img, im2=pred_img, channel_axis=-1)
        dssim_loss = dssim(pred_image=pred_img, real_image=gt_img) * 100.
        # LPIPS
        gt_img_normalized = torch.from_numpy((gt_img / 127.5 - 1.0).astype('float32').transpose(2,0,1)).unsqueeze(0)
        pred_img_normalized = torch.from_numpy((pred_img / 127.5 - 1.0).astype('float32').transpose(2,0,1)).unsqueeze(0)
        lpips_score = lpips_loss_fn(gt_img_normalized, pred_img_normalized).item() * 100.

        results["image"].append(img)
        results["psnr"].append(psnr)
        results["mse"].append(mse)
        results["ssim"].append(ssim)
        results["dssim"].append(dssim_loss)
        results["lpips"].append(lpips_score)
    
    results["image"].insert(0, "AVERAGE")
    results["psnr"].insert(0, np.mean(results["psnr"]))
    results["mse"].insert(0, np.mean(results["mse"]))
    results["ssim"].insert(0, np.mean(results["ssim"]))
    results["dssim"].insert(0, np.mean(results["dssim"]))
    results["lpips"].insert(0, np.mean(results["lpips"]))

    model_pth = os.path.join(config["training"]["out_dir"], config["experiment"]["split"], "model.pt")
    model = torch.load(model_pth)
    num_params = sum(p.numel() for p in model.parameters())
    if config["method"] == "intrinsic":
        mesh = load_mesh(config["data"]["mesh_path"])
        num_efuncs = len(config["model"]["k"])
        print(f"INF has {num_efuncs} efuncs")
        num_params += len(mesh.vertices) * num_efuncs
    elif config["method"] == "tf_rff":
        num_params += 3*1023 # TODO change if other k

    save_pth = os.path.join(config["data"]["render_img_directory"], config["experiment"]["split"])
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_pth, "eval_metrics.csv"))
    print(df.head(1))
    with open(os.path.join(save_pth, 'experiment.md'), 'w') as f:
        f.write(f'# Experiment: {config["experiment"]["name"]}\n')
        f.write('## Description\n')
        f.write(f'{config["experiment"]["description"]}\n')
        f.write(f"## Number of parameters of our model: {num_params}\n")
        f.write('## Metrics\n')
        f.write(df.to_markdown())

    avg_psnr = results["psnr"][0]
    avg_dssim = results["dssim"][0]
    avg_lpips = results["lpips"][0]

    report["psnr"] = avg_psnr
    report["dssim"] = avg_dssim
    report["lpips"] = avg_lpips
    report["num_params"] = num_params
    with open(os.path.join(save_pth,"report.json"), 'w') as json_file:
        json.dump(report, json_file, indent=2)

    return avg_psnr, avg_dssim, avg_lpips

if __name__ == '__main__':
    main()
