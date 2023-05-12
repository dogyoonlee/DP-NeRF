import os
import time

import cv2
import imageio
from tensorboardX import SummaryWriter

# from NeRF import *
from models.dpnerf import *
from data_utils.load_llff import load_llff_data
from utils.run_dpnerf_helpers import *
from utils.metrics import compute_img_metric
from PIL import Image as PILImage

# np.random.seed(0)
DEBUG = False


def compute_time(dt):
    # train_time = time.time()-start_time
    dt_h = dt//3600
    dt_m = (dt - dt_h*3600)//60
    dt_s = dt - dt_h*3600 - dt_m*60
    return dt_h, dt_m, dt_s

def exponential_scale_fine_loss_weight(N_iters, kernel_start_iter, start_ratio, end_ratio, iter):
    interval_len = N_iters - kernel_start_iter
    scale = (1 / interval_len) * np.log(end_ratio / start_ratio)
    return start_ratio * np.exp(scale * (iter - kernel_start_iter))

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', required=True,
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, required=True,
                        help='input data directory')
    parser.add_argument("--datadownsample", type=float, default=-1,
                        help='if downsample > 0, means downsample the image to scale=datadownsample')
    parser.add_argument("--tbdir", type=str, required=True,
                        help="tensorboard log directory")
    parser.add_argument("--num_gpu", type=int, default=1,
                        help=">1 will use DataParallel")
    parser.add_argument("--torch_hub_dir", type=str, default='',
                        help=">1 will use DataParallel")
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # generate N_rand # of rays, divide into chunk # of batch
    # then generate chunk * N_samples # of points, divide into netchunk # of batch
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000,
                        help='number of iteration')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplue"')

    ####### render option, will not effect training ########
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_rmnearplane", type=int, default=0,
                        help='when render, set the density of nearest plane to 0')
    parser.add_argument("--render_focuspoint_scale", type=float, default=1.,
                        help='scale the focal point when render')
    parser.add_argument("--render_radius_scale", type=float, default=1.,
                        help='scale the radius of the camera path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_epi", action='store_true',
                        help='render the video with epi path')

    ## llff flags
    parser.add_argument("--factor", type=int, default=None,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # ######### Unused params from the original ###########
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')


    ################# logging/saving options ##################
    parser.add_argument("--i_print", type=int, default=200,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_tensorboard", type=int, default=200,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=20000,
                        help='frequency of render_poses video saving')

    
    # =================== DP-NeRF Options =============================
    parser.add_argument("--blur_model_type", type=str, default='dpnerf',
                        help='choose among <none>, <dpnerf>')
    
    parser.add_argument("--kernel_start_iter", type=int, default=0,
                        help='start training kernel after # iteration')
    
    parser.add_argument("--tone_mapping_type", type=str, default='none',
                        help='the tone mapping of linear to LDR color space, <none>, <gamma>, <learn>')
    parser.add_argument("--use_dpnerf", action='store_true',
                        help='use_dpnerf')
    parser.add_argument("--rbk_use_view_embed", action='store_true',
                        help='use_view_embedding in rbk')
    parser.add_argument("--rbk_view_embed_ch", type=int, default=32,
                        help='view embedding ch')
    parser.add_argument("--rbk_use_viewdirs", action='store_true',
                        help='use viewdirs in rbk')
    
    parser.add_argument("--rbk_enc_brc_depth", type=int, default=4,
                        help='rbk encoding network depth')
    parser.add_argument("--rbk_enc_brc_width", type=int, default=64,
                        help='rbk encoding  network width')
    parser.add_argument("--rbk_enc_brc_skips", type=int, default=4,
                        help='rbk encoding  network skip connection')
    parser.add_argument("--rbk_num_motion", type=int, default=4,
                        help='rbk network - number of motion')
    parser.add_argument("--rbk_se_r_depth", type=int, default=1,
                        help='rbk se3 r network depth')
    parser.add_argument("--rbk_se_r_width", type=int, default=32,
                        help='rbk se3 r network width')
    parser.add_argument("--rbk_se_r_output_ch", type=int, default=3,
                        help='rbk se3 r network output channel')
    parser.add_argument("--rbk_se_v_depth", type=int, default=1,
                        help='rbk se3 v network depth')
    parser.add_argument("--rbk_se_v_width", type=int, default=32,
                        help='rbk se3 v network width')
    parser.add_argument("--rbk_se_v_output_ch", type=int, default=3,
                        help='rbk se3 v network output channel')
    parser.add_argument("--rbk_ccw_depth", type=int, default=1,
                        help='rbk ccw network depth')
    parser.add_argument("--rbk_ccw_width", type=int, default=32,
                        help='rbk ccw network width')
    parser.add_argument("--rbk_se_rv_window", type=float, default=0.2,
                        help='rbk se3 rv network output scale window')
    parser.add_argument("--rbk_use_origin", action='store_true',
                        help='use original ray in rbk module')
        
    parser.add_argument("--use_awp", action='store_true',
                        help='use awp module')
    
    parser.add_argument("--awp_sam_emb_depth", type=int, default=4,
                        help='awp sample feature embedding layer depth')
    parser.add_argument("--awp_sam_emb_width", type=int, default=32,
                        help='awp sample feature embedding layer width')
    
    parser.add_argument("--awp_dir_freq", type=int, default=2,
                        help='awp dir fourier embedding freq')
    
    parser.add_argument("--awp_mot_emb_depth", type=int, default=1,
                        help='awp motion feature embedding layer depth')
    parser.add_argument("--awp_mot_emb_width", type=int, default=32,
                        help='awp motion feature embedding layer depth')
    
    parser.add_argument("--awp_rgb_freq", type=int, default=2,
                        help='awp rgb freq')
    parser.add_argument("--awp_depth_freq", type=int, default=2,
                        help='awp depth freq')
    parser.add_argument("--awp_ray_dir_freq", type=int, default=2,
                        help='awp network ray dir freq')
    
    parser.add_argument("--use_coarse_to_fine_opt", action='store_true',
                        help='use_coarse_to_fine_optimization')
    
    parser.add_argument("--save_warped_ray_img", action='store_true',
                        help='save_warped_ray_img')
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    
    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args, args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify,
                                                                  path_epi=args.render_epi)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        print('LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.min(bds) * 0.9
            far = np.max(bds) * 1.0

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses)

    # Create log dir and copy the config file
    basedir = args.basedir
    tensorboardbase = args.tbdir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)

    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")
    
    args.num_images = len(images)
    # Create nerf model
    if args.blur_model_type == 'dpnerf':
        blur_kernel_net = RBK_AWP(num_img=len(images), view_embed_ch=args.rbk_view_embed_ch, 
                            D_rbk=args.rbk_enc_brc_depth, W_rbk=args.rbk_enc_brc_width, skips_rbk=[args.rbk_enc_brc_skips], num_motion_rbk=args.rbk_num_motion,
                            D_rbk_r=args.rbk_se_r_depth, W_rbk_r=args.rbk_se_r_width, output_ch_rbk_r=args.rbk_se_r_output_ch,
                            D_rbk_v=args.rbk_se_v_depth, W_rbk_v=args.rbk_se_v_width, output_ch_rbk_v=args.rbk_se_v_output_ch,
                            D_rbk_w=args.rbk_ccw_depth, W_rbk_w=args.rbk_ccw_width, rbk_se_rv_window=args.rbk_se_rv_window,
                            input_ch_awp=(args.netwidth), n_sample_awp=(args.N_samples + args.N_importance),
                            D_awp_sam=args.awp_sam_emb_depth, W_awp_sam=args.awp_sam_emb_width,
                            D_awp_mot=args.awp_mot_emb_depth, W_awp_mot=args.awp_mot_emb_width,
                            awp_dir_freq=args.awp_dir_freq, awp_rgb_freq=args.awp_rgb_freq, 
                            awp_depth_freq=args.awp_depth_freq, awp_ray_dir_freq=args.awp_ray_dir_freq,
                            use_dpnerf=args.use_dpnerf, use_awp = args.use_awp,
                            rbk_use_origin=args.rbk_use_origin, near=near, far=far, ndc=(not args.no_ndc))
        
    elif args.blur_model_type == 'none':
        blur_kernel_net = None
    else:
        raise RuntimeError(f"blur_model_type {args.blur_model_type} not recognized")
    
    nerf = NeRFAll(args, blur_kernel_net)
    
    # nerf = NeRFAll(args, kernelnet)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))

    optim_params = nerf.parameters()

    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))
    
    start = 0
    # Load Checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        smart_load_state_dict(nerf, ckpt)

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'inference': False,
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:  # args.dataset_type != 'llff' or
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['inference'] = True
    render_kwargs_test['save_warped_ray_img'] = args.save_warped_ray_img

    # visualize_motionposes(H, W, K, nerf, 2)
    # visualize_kernel(H, W, K, nerf, 5)
    # visualize_itsample(H, W, K, nerf)
    # visualize_kmap(H, W, K, nerf, img_idx=1)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses[:, :3, :4]).cuda()
    nerf = nerf.cuda()
    # Short circuit if only rendering out from trained model
    
    if args.save_warped_ray_img:
        print('Save warped rays and imgs')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       f"warped_ray_img"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            
            
            render_warped_poses = torch.tensor(poses[i_train]).cuda()
            images_idx_warped = torch.tensor(images_idx[i_train]).cuda()

            print('save poses shape: ', render_warped_poses.shape)
            
            dummy_num = ((len(render_warped_poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(render_warped_poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_warped_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            
            render_warped_poses = torch.cat([render_warped_poses, dummy_poses], dim=0)
            
            dummy_idx = torch.zeros(dummy_num).type_as(images_idx_warped)
            print(f"Append {dummy_num} # of image_idx to fill all the GPUs")
            
            images_idx_warped = torch.cat([images_idx_warped, dummy_idx], dim=0)
            
            nerf.eval()
            rgbshdr, disps, rays_warped = nerf(
                hwf[0], hwf[1], K, args.chunk,
                poses=render_warped_poses,
                render_kwargs=render_kwargs_test,
                render_factor=args.render_factor,
                rays_info=images_idx_warped,
            )
            
            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rays_warped = rays_warped[:len(rays_warped) - dummy_num]
            
            rgbs = rgbshdr
            rgbs = to8b(rgbs.cpu().numpy())
            disps = to8b(disps / disps.max())
            
            for rgb_idx, rgb8 in enumerate(rgbs):
                for warped_idx, rgb_warped in enumerate(rgb8):
                    imageio.imwrite(os.path.join(testsavedir, f'{i_train[rgb_idx]:03d}_scene_{warped_idx:03d}.png'), rgb_warped)
                    imageio.imwrite(os.path.join(testsavedir, f'{i_train[rgb_idx]:03d}_scene_{warped_idx:03d}_disp.png'), disps[rgb_idx][warped_idx])

            np.save(os.path.join(testsavedir, 'rays_warped.npy'), rays_warped.cpu().numpy())
            print("Warped rays and imgs are saved")
            
            return

    
    
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       f"renderonly"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            nerf.eval()
            rgbshdr, disps = nerf(
                hwf[0], hwf[1], K, args.chunk,
                poses=torch.cat([render_poses, dummy_poses], dim=0),
                render_kwargs=render_kwargs_test,
                render_factor=args.render_factor,
            )
            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = rgbshdr
            rgbs = to8b(rgbs.cpu().numpy())
            disps = to8b(disps / disps.max())
            if args.render_test:
                for rgb_idx, rgb8 in enumerate(rgbs):
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps[rgb_idx])
                
                # evaluation
                rgbs_test = torch.tensor(rgbshdr).cuda()
                imagesf = torch.tensor(imagesf).cuda()
                rgbs_test = rgbs_test[i_test]
                target_rgb_gt = imagesf[i_test]
                test_mse = compute_img_metric(rgbs_test, target_rgb_gt, 'mse')
                test_psnr = compute_img_metric(rgbs_test, target_rgb_gt, 'psnr')
                test_ssim = compute_img_metric(rgbs_test, target_rgb_gt, 'ssim')
                test_lpips = compute_img_metric(rgbs_test, target_rgb_gt, 'lpips')
                if isinstance(test_lpips, torch.Tensor):
                    test_lpips = test_lpips.item()

                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"**[Evaluation]** : PSNR:{test_psnr:.8f} SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")
                    print(f"**[Evaluation]** : PSNR:{test_psnr:.8f} SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}")
            else:
                prefix = 'epi_' if args.render_epi else ''
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

            return

    # ============================================
    # Prepare ray dataset if batching random rays
    # ============================================
    N_rand = args.N_rand
    train_datas = {}
    
    # if downsample, downsample the images
    if args.datadownsample > 0:
        images_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in imagesf], axis=0)
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
                        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
                        0, 0, 1]).reshape(3, 3).astype(K.dtype)
    # K = 
    # [[focal,     0, 0.5*W],
    #  [    0, focal, 0.5*H],
    #  [    0,     0,     1]]
    
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3) # [N*H*W,  ro+rd (2), 3]

    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'], train_datas['rays_y'] = xs[i_train].reshape(-1, 1), ys[i_train].reshape(-1, 1)

    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)

    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)

    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}

    print('done')
    i_batch = 0

    # Move training data to GPU
    images = torch.tensor(images).cuda()
    imagesf = torch.tensor(imagesf).cuda()

    poses = torch.tensor(poses).cuda()
    train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    time_init = time.time()
    fine_loss_weight = 0.1
    for i in range(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()} # rays: [N_rand, ro+rd (2), 3]
        batch_rays = iter_data.pop('rays').permute(0, 2, 1) # [N_rand, 3, ro+rd (2)]

        i_batch += N_rand
        if i_batch >= len(train_datas['rays']):
            print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(len(train_datas['rays']))
            train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
            i_batch = 0

        #####  Core optimization loop  #####
        iter_data['poses'] = poses[iter_data['images_idx']].squeeze(1)
        iter_data['K'] = k_train
        nerf.train()
        if i == args.kernel_start_iter:
            torch.cuda.empty_cache()
        rgb, rgb0, extra_loss = nerf(H, W, K, chunk=args.chunk,
                                     rays=batch_rays, rays_info=iter_data,
                                     retraw=True, force_naive=i < args.kernel_start_iter,
                                     **render_kwargs_train)

        # Compute Losses
        # =====================
        target_rgb = iter_data['rgbsf'].squeeze(-2)
        img_loss = img2mse(rgb, target_rgb)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        img_loss0 = img2mse(rgb0, target_rgb)
        loss = loss + img_loss0
        
        if 'rgb_awp' in extra_loss and extra_loss['rgb_awp'] is not None:
            img_fine_loss = img2mse(extra_loss['rgb_awp'], target_rgb)
            if args.use_coarse_to_fine_opt:
                if i % 10000 == 0:
                    fine_loss_weight = exponential_scale_fine_loss_weight(N_iters=N_iters, kernel_start_iter=args.kernel_start_iter, start_ratio=0.1, end_ratio=0.9, iter=i)
                loss = loss*(1 - fine_loss_weight) + img_fine_loss*fine_loss_weight
            else:
                loss = loss + img_fine_loss
        else:
            img_fine_loss = 0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                nerf.eval()
                rgbs, disps = nerf(H, W, K, args.chunk, poses=render_poses, render_kwargs=render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs = rgbs.cpu().numpy()
            # disps = (1. - disps)
            disps = disps.cpu().numpy()
            # disps_max_idx = idnt(disps.size * 0.9)
            # disps_max = disps.reshape(-1)[np.argpartition(disps.reshape(-1), disps_max_idx)[disps_max_idx]]

            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / disps.max()), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses.shape)
            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            
            with torch.no_grad():
                nerf.eval()
                rgbs, _ = nerf(H, W, K, args.chunk, poses=torch.cat([poses, dummy_poses], dim=0).cuda(),
                               render_kwargs=render_kwargs_test)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
                # saving
                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                    imageio.imwrite(filename, rgb8)

                # evaluation
                rgbs = rgbs[i_test]
                target_rgb_gt = imagesf[i_test]

                test_mse = compute_img_metric(rgbs, target_rgb_gt, 'mse')
                test_psnr = compute_img_metric(rgbs, target_rgb_gt, 'psnr')
                test_ssim = compute_img_metric(rgbs, target_rgb_gt, 'ssim')
                test_lpips = compute_img_metric(rgbs, target_rgb_gt, 'lpips')
                if isinstance(test_lpips, torch.Tensor):
                    test_lpips = test_lpips.item()

                tensorboard.add_scalar("Test MSE", test_mse, global_step)
                tensorboard.add_scalar("Test PSNR", test_psnr, global_step)
                tensorboard.add_scalar("Test SSIM", test_ssim, global_step)
                tensorboard.add_scalar("Test LPIPS", test_lpips, global_step)
                
            with open(test_metric_file, 'a') as outfile:
                outfile.write(f"iter{i}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR:{test_psnr:.8f}"
                              f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")
                print(f"**[Evaluation]** Iter{i}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR:{test_psnr:.8f}"
                              f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}")
            

            print('Saved test set')

        if i % args.i_tensorboard == 0:
            tensorboard.add_scalar("Loss", loss.item(), global_step)
            tensorboard.add_scalar("PSNR", psnr.item(), global_step)
            # for k, v in extra_loss.items():
                # tensorboard.add_scalar(k, v.item(), global_step)

        if i % args.i_print == 0:
            dt_h, dt_m, dt_s = compute_time((time.time() - time_init))
            dt_h, dt_m = int(dt_h), int(dt_m)
            # print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} TIME: {dt_h}h:{dt_m}m:{dt_s:.2f}s")
            
        global_step += 1
        
    with open(test_metric_file, 'a') as outfile:
        outfile.write(f"TRINING TIME: {dt_h}h:{dt_m}m:{dt_s:.2f}s")     

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

