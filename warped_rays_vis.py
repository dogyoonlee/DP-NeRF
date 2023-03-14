import os
import time
import numpy as np
import cv2
import imageio
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from NeRF import *
# from NeRF_cbnet import *
# import pdb;pdb.set_trace()
from data_utils.load_llff import load_llff_data
from utils.run_dpnerf_helpers import *
from utils.metrics import compute_img_metric

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
    parser.add_argument("--render_multipoints", action='store_true',
                        help='render sub image that reconstruct the blur image')
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
    
    parser.add_argument("--warped_rays_path", type=str, required=True,
                        help='input pose ray directory')
    parser.add_argument("--vis_img_idx", type=int, default=1,
                        help='dbk cb feature embedding layer width')
    
    return parser

parser = config_parser()
args = parser.parse_args()


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

imagesf = images
images = (images * 255).astype(np.uint8)
images_idx = np.arange(0, len(images))

H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]

if K is None:
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])


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






    
quota = args.vis_img_idx // args.llffhold
assert quota == 0, "Not blurred image index. Please specify blurred image index."
vis_idx = args.vis_img_idx - (quota + 1)
    
poses_rays= np.load(args.warped_rays_path)
# poses = poses[i_train]
print('get rays')

########################################
############## ndc poses ###############
########################################
# import pdb;pdb.set_trace()
for i, rbk_rays in enumerate(poses_rays):
    if i == vis_idx:
        fig_ndc = plt.figure(figsize = (15, 15))
        ax = plt.axes(projection='3d')
        # for j in range(rbk_rays.shape[0]):
        rbk_rays_o = rbk_rays[:,:,0]
        rbk_rays_d = rbk_rays[:,:,1]
        rbk_rays_o, rbk_rays_d = ndc_rays(H, W, K[0][0], 1., torch.tensor(rbk_rays_o), torch.tensor(rbk_rays_d))
        rbk_rays_o, rbk_rays_d = np.array(rbk_rays_o), np.array(rbk_rays_d)
        print('rbk_rays_o : ',rbk_rays_o)
        print('rbk_rays_d : ',rbk_rays_d)

        rbk_x = rbk_rays_o[:,0]
        rbk_y = rbk_rays_o[:,1]
        rbk_z = rbk_rays_o[:,2]

        rbk_x_d = rbk_x + rbk_rays_d[:,0]*1.1 # default: 0.8, defocuspool: 1.5, blurpool: 0.15
        rbk_y_d = rbk_y + rbk_rays_d[:,1]*1.1
        rbk_z_d = rbk_z + rbk_rays_d[:,2]*1.1

        dir_x = np.concatenate([rbk_x[...,None], rbk_x_d[...,None]], axis=1)
        dir_y = np.concatenate([rbk_y[...,None], rbk_y_d[...,None]], axis=1)
        dir_z = np.concatenate([rbk_z[...,None], rbk_z_d[...,None]], axis=1)


        # for j in range(rbk_x.shape[0]):
            # if j == 0:
            #     ax.plot3D(dir_z[j], dir_y[j], dir_x[j], 'orange')
            # else:
            #     ax.plot3D(dir_z[j], dir_y[j], dir_x[j], 'green')
        ax.plot3D(dir_y[0], dir_x[0], dir_z[0], 'orange')
        ax.plot3D(dir_y[1], dir_x[1], dir_z[1], 'green')
        ax.plot3D(dir_y[2], dir_x[2], dir_z[2], 'blue')
        ax.plot3D(dir_y[3], dir_x[3], dir_z[3], 'red')
        ax.plot3D(dir_y[4], dir_x[4], dir_z[4], 'violet')
        ax.plot3D(dir_y[5], dir_x[5], dir_z[5], 'skyblue')
        ax.plot3D(dir_y[6], dir_x[6], dir_z[6], 'olive')
        ax.plot3D(dir_y[7], dir_x[7], dir_z[7], 'brown')
        ax.plot3D(dir_y[8], dir_x[8], dir_z[8], 'pink')
            
        
        # rbk_labels = np.array(['origin', 'C_1', 'C_2', 'C_3', 'C_4'])
        ax.scatter3D(rbk_y, rbk_x, rbk_z, color='black', s=50, marker='>')

        # scene_name_ndc = 'Blurpool RBK Visualization - num_motion 8'
        scene_name_ndc = 'Defocuspool RBK Visualization - num_motion 8'

        # ax_w.title = scene_name
        # ax.set_xlim3d(-1.3, -1.1)
        # ax.set_zlim3d(-1.2, 1)
        # ax.set_ylim3d(0.6, 0.9)
        ax.set_xlabel('Y axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('Z axis')
        ax.view_init(-20,110)

        # plt.title('{} {} Scene'.format(scene_name_ndc, i))
        plt.title('{}'.format(scene_name_ndc))
        plt.show() 
        # exit()