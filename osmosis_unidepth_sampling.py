"""
Osmosis Sampling with UniDepth Depth Guidance

This script integrates UniDepth depth maps into the Osmosis diffusion sampling process.
Based on original osmosis_sampling.py with UniDepth depth replacement at each timestep.

Usage:
    python osmosis_unidepth_sampling.py -c ./configs/osmosis_unidepth_config.yaml -d 0
"""

import sys
import numpy as np
from functools import partial
import os
from os.path import join as pjoin
from argparse import ArgumentParser
from PIL import Image
import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvtf
from torchvision.utils import make_grid
import torch.nn.functional as F

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from osmosis_utils import logger

import osmosis_utils.utils as utilso
import osmosis_utils.data as datao


# =============================================================================
# UniDepth Loading Functions
# =============================================================================

def load_unidepth_depth(depth_dir, image_name, device, image_size=256):
    """
    Load UniDepth depth map and preprocess it for diffusion sampling.
    """
    base_name = os.path.splitext(image_name)[0]
    depth_path = os.path.join(depth_dir, f"{base_name}.npy")
    
    if not os.path.exists(depth_path):
        print(f"[UniDepth] Warning: Depth file not found: {depth_path}")
        return None
    
    depth = np.load(depth_path)
    
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[2] == 1:
            depth = depth[:, :, 0]
    
    depth_tensor = torch.from_numpy(depth).float()
    
    if depth_tensor.ndim == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    elif depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.unsqueeze(0)
    
    _, _, h, w = depth_tensor.shape
    if h < w:
        new_h = image_size
        new_w = int(w * image_size / h)
    else:
        new_w = image_size
        new_h = int(h * image_size / w)
    
    depth_tensor = F.interpolate(depth_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    _, _, h, w = depth_tensor.shape
    top = (h - image_size) // 2
    left = (w - image_size) // 2
    depth_tensor = depth_tensor[:, :, top:top+image_size, left:left+image_size]
    
    d_min = depth_tensor.min()
    d_max = depth_tensor.max()
    if d_max - d_min > 1e-8:
        depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
        depth_tensor = 2 * depth_tensor - 1
    else:
        depth_tensor = torch.zeros_like(depth_tensor)
    
    print(f"[UniDepth] Loaded: {depth_path}, Shape: {depth_tensor.shape}, Range: [{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")
    
    return depth_tensor.to(device)


# =============================================================================
# Main Sampling Function
# =============================================================================

def main() -> None:
    # Read the config file
    args = utilso.arguments_from_file(CONFIG_FILE)
    args.image_size = args.unet_model['image_size']
    args.unet_model['model_path'] = os.path.abspath(args.unet_model['model_path'])
    
    # Device setting
    torch.cuda.set_device(DEVICE)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device set to: {device}")
    
    # ==========================================================================
    # UniDepth Configuration
    # ==========================================================================
    unidepth_config = getattr(args, 'unidepth_guidance', {})
    use_unidepth = unidepth_config.get('enabled', False)
    unidepth_depth_dir = unidepth_config.get('depth_dir', './data/underwater/depth2')
    
    if use_unidepth:
        print(f"[UniDepth] Guidance ENABLED")
        print(f"[UniDepth] Depth directory: {unidepth_depth_dir}")
    else:
        print(f"[UniDepth] Guidance DISABLED")
    # ==========================================================================
    
    # Prepare dataloader
    data_config = args.data
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=[256, 256]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if data_config['ground_truth']:
        gt_flag = True
        dataset = datao.ImagesFolder_GT(
            root_dir=data_config['root'],
            gt_rgb_dir=data_config['gt_rgb'],
            gt_depth_dir=data_config['gt_depth'],
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)
    else:
        gt_flag = False
        dataset = datao.ImagesFolder(data_config['root'], transform)
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)
    
    # Load UNet model
    model = create_model(**args.unet_model)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.unet_model['model_path']}")
    
    # Extract configurations
    measure_config = args.measurement
    cond_config = args.conditioning
    diffusion_config = args.diffusion
    sample_pattern_config = args.sample_pattern
    aux_loss_config = args.aux_loss
    
    # Output directory
    measurement_name = measure_config['operator']['name']
    out_path = os.path.abspath(pjoin(args.save_dir, measurement_name, args.data['name']))
    out_path = utilso.update_save_dir_date(out_path)
    
    # Save config
    utilso.yaml_to_txt(CONFIG_FILE, pjoin(out_path, "configurations.txt"))
    
    # Create output directories
    if args.save_singles:
        save_singles_path = pjoin(out_path, "single_images")
        os.makedirs(save_singles_path)
        
        save_input_path = pjoin(save_singles_path, "input")
        os.makedirs(save_input_path)
        save_rgb_path = pjoin(save_singles_path, "rgb")
        os.makedirs(save_rgb_path)
        save_depth_pmm_color_path = pjoin(save_singles_path, "depth_color")
        os.makedirs(save_depth_pmm_color_path)
        save_depth_mm_path = pjoin(save_singles_path, "depth_raw")
        os.makedirs(save_depth_mm_path)
        
        if use_unidepth:
            save_unidepth_path = pjoin(save_singles_path, "unidepth_input")
            os.makedirs(save_unidepth_path)
    else:
        save_singles_path = None
    
    if args.save_grids:
        save_grids_path = pjoin(out_path, "grid_results")
        os.makedirs(save_grids_path)
    else:
        save_grids_path = None
    
    # Logger
    logger.configure(dir=out_path)
    logger.log(f"pretrained model file: {args.unet_model['model_path']}")
    
    if use_unidepth:
        logger.log(f"[UniDepth] Guidance ENABLED, depth_dir: {unidepth_depth_dir}")
    
    if not args.rgb_guidance:
        log_txt_tmp = utilso.log_text(args=args)
        logger.log(log_txt_tmp)
    
    # ==========================================================================
    # Main Inference Loop
    # ==========================================================================
    for i, (ref_img, ref_img_name) in enumerate(loader):
        
        if gt_flag:
            gt_rgb_img = ref_img[1].squeeze()
            gt_rgb_img_01 = 0.5 * (gt_rgb_img + 1)
            gt_depth_img = ref_img[2].squeeze()
            gt_depth_img_01 = 0.5 * (gt_depth_img + 1)
            gt_depth_img_01 = utilso.depth_tensor_to_color_image(gt_depth_img_01)
            ref_img = ref_img[0]
        
        start_run_time_ii = datetime.datetime.now()
        
        ref_img_01 = 0.5 * (ref_img.detach().cpu()[0] + 1)
        ref_img_name = ref_img_name[0]
        orig_file_name = os.path.splitext(ref_img_name)[0]
        
        # Stop early if requested
        if i == args.data['stop_after']:
            break
        
        # ======================================================================
        # Load UniDepth depth for this image
        # ======================================================================
        if use_unidepth:
            unidepth_depth = load_unidepth_depth(
                depth_dir=unidepth_depth_dir,
                image_name=ref_img_name,
                device=device,
                image_size=args.image_size
            )
        else:
            unidepth_depth = None
        # ======================================================================
        
        # Initialize operator and conditioning for each image
        measure_config['operator']['batch_size'] = args.data['batch_size']
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        
        cond_method = get_conditioning_method(
            cond_config['method'], operator, noiser,
            **cond_config['params'],
            **sample_pattern_config,
            **aux_loss_config
        )
        measurement_cond_fn = cond_method.conditioning
        
        sampler = create_sampler(**diffusion_config)
        
        # ======================================================================
        # Create sample function with UniDepth depth
        # ======================================================================
        sample_fn = partial(
            sampler.p_sample_loop,
            model=model,
            measurement_cond_fn=measurement_cond_fn,
            pretrain_model=args.unet_model['pretrain_model'],
            rgb_guidance=args.rgb_guidance,
            sample_pattern=args.sample_pattern,
            record=args.record_process,
            save_root=out_path,
            image_idx=i,
            record_every=args.record_every,
            original_file_name=orig_file_name,
            save_grids_path=save_grids_path,
            unidepth_depth=unidepth_depth  # ★ Pass UniDepth depth here!
        )
        # ======================================================================
        
        logger.log(f"\nInference image {i}: {ref_img_name}\n")
        ref_img = ref_img.to(device)
        
        y_n = noiser(ref_img)
        
        if args.degamma_input:
            y_n_tmp = 0.5 * (y_n + 1)
            y_n = 2 * torch.pow(y_n_tmp, 2.2) - 1
        
        x_start_shape = list(ref_img.shape)
        x_start_shape[1] = 4 if (args.unet_model["pretrain_model"] == 'osmosis') else x_start_shape[1]
        
        if args.sample_pattern['pattern'] == "original":
            global_N = 1
        elif args.sample_pattern['pattern'] == "pcgs":
            global_N = args.sample_pattern['global_N']
        else:
            raise ValueError(f"Unrecognized sample pattern: {args.sample_pattern['pattern']}")
        
        for global_ii in range(global_N):
            
            logger.log(f"global iteration: {global_ii}\n")
            torch.manual_seed(args.manual_seed)
            
            x_start = torch.randn(x_start_shape, device=device).requires_grad_()
            
            if args.unet_model["pretrain_model"] == 'osmosis' and not args.rgb_guidance:
                
                sample, variable_dict, loss, out_xstart = sample_fn(
                    x_start=x_start,
                    measurement=y_n,
                    global_iteration=global_ii
                )
                
                sample_rgb = out_xstart[0, 0:-1, :, :]
                sample_depth_tmp = out_xstart[0, -1, :, :].unsqueeze(0)
                sample_depth_tmp_rep = sample_depth_tmp.repeat(3, 1, 1)
                
                sample_rgb_01 = 0.5 * (sample_rgb + 1)
                sample_rgb_01_clip = torch.clamp(sample_rgb_01, min=0, max=1)
                
                sample_depth_mm = utilso.min_max_norm_range(sample_depth_tmp[0].unsqueeze(0))
                sample_depth_vis_pmm = utilso.min_max_norm_range_percentile(
                    sample_depth_tmp, vmin=0, vmax=1,
                    percent_low=0.03, percent_high=0.99, is_uint8=False
                )
                sample_depth_vis_pmm_color = utilso.depth_tensor_to_color_image(sample_depth_vis_pmm)
                
                sample_depth_calc = utilso.convert_depth(
                    sample_depth_tmp_rep,
                    depth_type=args.measurement['operator']['depth_type'],
                    value=args.measurement['operator']['value']
                )
                
                phi_inf = variable_dict['phi_inf'].cpu().squeeze(0)
                phi_inf_image = phi_inf * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                
                # ==================== Underwater Model ====================
                if 'underwater_physical_revised' in args.measurement['operator']['name']:
                    
                    phi_a = variable_dict['phi_a'].cpu().squeeze(0)
                    phi_a_image = phi_a * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    phi_b = variable_dict['phi_b'].cpu().squeeze(0)
                    phi_b_image = phi_b * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    
                    backscatter_image = phi_inf_image * (1 - torch.exp(-phi_b_image * sample_depth_calc))
                    attenuation_image = torch.exp(-phi_a_image * sample_depth_calc)
                    forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image
                    
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round([torch.linalg.norm(
                        degraded_image - ref_img.detach().cpu()).numpy()], decimals=3)
                    
                    attenuation_flip_image = torch.exp(phi_a_image * sample_depth_calc)
                    sample_rgb_recon = attenuation_flip_image * (ref_img_01 - backscatter_image)
                    
                    print_phi_a = [np.round(x, decimals=3) for x in phi_a.cpu().squeeze().tolist()]
                    print_phi_b = [np.round(x, decimals=3) for x in phi_b.cpu().squeeze().tolist()]
                    print_phi_inf = [np.round(x, decimals=3) for x in phi_inf.cpu().squeeze().tolist()]
                    
                    log_value_txt = f"\nInitialized values: " \
                                    f"\nphi_a: [{measure_config['operator']['phi_a']}], lr: {measure_config['operator']['phi_a_eta']}" \
                                    f"\nphi_b: [{measure_config['operator']['phi_b']}], lr: {measure_config['operator']['phi_b_eta']}" \
                                    f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\nphi_a: {print_phi_a}" \
                                    f"\nphi_b: {print_phi_b}" \
                                    f"\nphi_inf: {print_phi_inf}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=3)}"
                    
                    if use_unidepth:
                        log_value_txt += f"\n[UniDepth] Depth guidance was ENABLED"
                    
                    logger.log(log_value_txt)
                
                # ==================== Haze Model ====================
                elif ('haze' in args.measurement['operator']['name']) or \
                     ('underwater_physical' in args.measurement['operator']['name']):
                    
                    phi_ab = variable_dict['phi_ab'].cpu().squeeze(0)
                    phi_ab_image = phi_ab * torch.ones_like(sample_rgb, device=torch.device('cpu'))
                    backscatter_image = phi_inf_image * (1 - torch.exp(-phi_ab_image * sample_depth_calc))
                    attenuation_image = torch.exp(-phi_ab_image * sample_depth_calc)
                    forward_predicted_image = sample_rgb_01 * attenuation_image + backscatter_image
                    
                    attenuation_flip_image = torch.exp(phi_ab_image * sample_depth_calc)
                    sample_rgb_recon = attenuation_flip_image * (ref_img_01 - backscatter_image)
                    
                    degraded_image = 2 * forward_predicted_image - 1
                    norm_loss_final = np.round(
                        [torch.linalg.norm(degraded_image.cpu() - ref_img.detach().cpu()).numpy()],
                        decimals=3)
                    
                    print_phi_ab = np.round(phi_ab.cpu().squeeze(), decimals=3)
                    print_phi_inf = np.round(phi_inf.cpu().squeeze(), decimals=3)
                    log_value_txt = f"\nInitialized values: " \
                                    f"\nphi_ab: [{measure_config['operator']['phi_ab']}], lr: {measure_config['operator']['phi_ab_eta']}" \
                                    f"\nphi_inf: [{measure_config['operator']['phi_inf']}], lr: {measure_config['operator']['phi_inf_eta']}" \
                                    f"\n\nResults values: " \
                                    f"\nphi_ab: {print_phi_ab}" \
                                    f"\nphi_inf: {print_phi_inf}" \
                                    f"\n\nNorm loss: {norm_loss_final}" \
                                    f"\nFinal loss: {np.round(np.array(loss), decimals=5)}"
                    
                    logger.log(log_value_txt)
                
                else:
                    raise NotImplementedError("Operator can be for 'underwater' or 'haze' ")
                
                # Save single images
                if args.save_singles:
                    ref_im_pil = tvtf.to_pil_image(ref_img_01)
                    ref_im_pil.save(pjoin(save_input_path, f'{orig_file_name}.png'))
                    
                    sample_rgb_01_clip_pil = tvtf.to_pil_image(sample_rgb_01_clip)
                    sample_rgb_01_clip_pil.save(pjoin(save_rgb_path, f'{orig_file_name}.png'))
                    
                    sample_depth_vis_pmm_color_pil = tvtf.to_pil_image(sample_depth_vis_pmm_color)
                    sample_depth_vis_pmm_color_pil.save(pjoin(save_depth_pmm_color_path, f'{orig_file_name}.png'))
                    
                    sample_depth_vis_mm_pil = tvtf.to_pil_image(sample_depth_mm)
                    sample_depth_vis_mm_pil.save(pjoin(save_depth_mm_path, f'{orig_file_name}.png'))
                    
                    if use_unidepth and unidepth_depth is not None:
                        unidepth_vis = 0.5 * (unidepth_depth.cpu().squeeze() + 1)
                        unidepth_vis_color = utilso.depth_tensor_to_color_image(unidepth_vis.unsqueeze(0))
                        unidepth_vis_pil = tvtf.to_pil_image(unidepth_vis_color)
                        unidepth_vis_pil.save(pjoin(save_unidepth_path, f'{orig_file_name}.png'))
                
                # Save grid
                if args.save_grids:
                    grid_list = [ref_img_01, sample_rgb_01_clip, sample_depth_vis_pmm_color]
                    
                    if gt_flag:
                        grid_list += [
                            torch.zeros_like(sample_rgb_01, device=torch.device('cpu')),
                            gt_rgb_img_01, gt_depth_img_01
                        ]
                    
                    results_grid = make_grid(grid_list, nrow=3, pad_value=1.)
                    results_grid = utilso.clip_image(results_grid, scale=False, move=False, is_uint8=True)
                    results_pil = tvtf.to_pil_image(results_grid)
                    results_pil.save(pjoin(save_grids_path, f'{orig_file_name}.png'))
                
                if args.save_singles or args.save_grids:
                    logger.log(f"result images was saved into: {out_path}")
                
                logger.log(f"Run time: {datetime.datetime.now() - start_run_time_ii}")
    
    # Close the logger txt file
    logger.get_current().close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./configs/osmosis_unidepth_config.yaml",
                        help="Configurations file")
    parser.add_argument("-d", "--device", default=0, help="GPU Device", type=int)
    
    args = vars(parser.parse_args())
    CONFIG_FILE = os.path.abspath(args["config_file"])
    DEVICE = args["device"]
    
    print(f"\nConfiguration file:\n{CONFIG_FILE}\n")
    
    main()
    print(f"\nFINISH!")
    sys.exit()
