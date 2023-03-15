import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.run_dpnerf_helpers import *
from models.mam import *
import os
import imageio
import time
import math
import utils.rigid_warping as rigid_warping

HALF_PIX = 0.5

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class View_Embedding(nn.Module):
    def __init__(self, num_embed, embed_dim):
        super(View_Embedding, self).__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.view_embed_layer = nn.Embedding(num_embed, embed_dim)
        
    def forward(self, x):
        return self.view_embed_layer(x)

class Rigid_Blurring_Kernel(nn.Module):
    def __init__(self, D, W, D_r, W_r, D_v, W_v, D_w, W_w,
                 output_ch_r, output_ch_v, input_ch, skips, rv_window, 
                 num_motion=2, near=0.0, far=1.0, ndc=False, warp_field=None, view_embedding_layer=None,
                 use_origin=True, use_awp=False):
        super(Rigid_Blurring_Kernel, self).__init__()
        
        self.use_awp = use_awp
        self.input_ch = input_ch
        self.skips = skips
        self.view_embedding_layer = view_embedding_layer
        self.view_embed_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in
                                        range(D - 1)])
        
        self.use_origin = use_origin
        
        self.num_motion = num_motion
        
        self.output_ch_r = output_ch_r * num_motion
        self.output_ch_v = output_ch_v * num_motion 
        self.output_ch_w = num_motion
        self.rv_window = rv_window
        
        self.r_branch = nn.ModuleList([nn.Linear(W, W_r)] + [nn.Linear(W_r, W_r) for i in range(D_r-1)])
        self.r_linear = nn.Linear(W_r, self.output_ch_r)
        r_gain = 0.00001/(math.sqrt((W_r + self.output_ch_r)/6)) # for Uniform(-1.0e-5, 1.0e-5)
        torch.nn.init.xavier_uniform_(self.r_linear.weight, gain=r_gain) # -1e-5, 1e-5
        
        self.v_branch = nn.ModuleList([nn.Linear(W, W_v)] + [nn.Linear(W_v, W_v) for i in range(D_v-1)])
        self.v_linear = nn.Linear(W_v, self.output_ch_v)
        v_gain = 0.00001/(math.sqrt((W_v + self.output_ch_v)/6)) # for Uniform(-1.0e-5, 1.0e-5)
        torch.nn.init.xavier_uniform_(self.v_linear.weight, gain=v_gain) # -1e-5, 1e-5
        
        self.w_branch = nn.ModuleList([nn.Linear(W, W_w)] + [nn.Linear(W_w, W_w) for i in range(D_w-1)])
        self.w_linear = nn.Linear(W_w, self.output_ch_w + 1)
        
        self.warp_field = warp_field
    
    def rbk_warp(self, rays, r, v):
        r = r.reshape(r.shape[0], 3, self.num_motion)
        v = v.reshape(v.shape[0], 3, self.num_motion)
        rays_o = rays[...,0]
        rays_d = rays[...,1]
        pts_rays_end = rays_o + rays_d
        
    
        if self.use_origin:
            new_rays = torch.cat([rays_o[..., None], rays_d[...,None]], dim=-1).unsqueeze(1).repeat(1, self.num_motion + 1, 1, 1)
        else:
            new_rays = torch.zeros_like(rays.unsqueeze(1).repeat(1,self.num_motion, 1, 1))    
        
        for i in range(self.num_motion):
            warped_rays_o = self.warp_field.warp(rays_o, rot=r[:,:,i], trans=v[:,:,i])
            warped_pts_end = self.warp_field.warp(pts_rays_end, rot=r[:,:,i], trans=v[:,:,i])
            warped_rays_d = warped_pts_end - warped_rays_o
            warped_rays = torch.cat([warped_rays_o[..., None], warped_rays_d[..., None]], dim=-1)
            if self.use_origin:
                new_rays[:,i+1,:,:] = warped_rays
            else:
                new_rays[:,i,:,:] = warped_rays

        return new_rays
    
    def rbk_weighted_sum(self, rgb, depth, acc, extras, ccw):
        if self.use_origin:
            num_motion = self.num_motion + 1
        rgb = torch.sum((rgb.reshape(-1, num_motion, rgb.shape[-1]) * ccw[...,None]), dim=1)
        depth = torch.sum(depth.reshape(-1, num_motion) * ccw, dim=1)
        acc = torch.sum(acc.reshape(-1, num_motion) * ccw, dim=1)
        
        for k, v in extras.items():
            if len(v.shape) == 1:
                v = torch.sum(v.reshape(-1, num_motion) * ccw, dim=1)
            if len(v.shape) == 2:
                v = torch.sum((v.reshape(-1, num_motion, v.shape[-1]) * ccw[...,None]), dim=1)
            if len(v.shape) == 3:
                v = torch.sum((v.reshape(-1, num_motion, v.shape[-2], v.shape[-1]) * ccw[...,None][...,None]), dim=1)
            extras[k] = v
        
        return rgb, depth, acc, extras
    
    def forward(self, rays, rays_info):
        view_embedded = self.view_embedding_layer(rays_info['images_idx'].squeeze(-1))

        input_views_embedded = view_embedded
        h = view_embedded
        for i, l in enumerate(self.view_embed_linears):
            h = self.view_embed_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_views_embedded, h], -1)
        
        view_feature = h
        h_v = h.clone()
        h_w = h.clone()
        for i, _ in enumerate(self.r_branch):
            h = self.r_branch[i](h)
            h = F.relu(h)
        
        for i, _ in enumerate(self.v_branch):
            h_v = self.v_branch[i](h_v)
            h_v = F.relu(h_v)
        
        for i, _ in enumerate(self.w_branch):
            h_w = self.w_branch[i](h_w)
            h_w = F.relu(h_w)
        
        r = self.r_linear(h) * self.rv_window
        v = self.v_linear(h_v) * self.rv_window
        
        w = torch.sigmoid(self.w_linear(h_w))
        w = w/(torch.sum(w, dim=-1, keepdims=True) + 1e-10)
        
        new_rays = self.rbk_warp(rays, r, v)
        new_rays = new_rays.reshape(-1, 3, 2)
        
        if self.use_awp:
            return new_rays, w, view_feature
        else:
            return new_rays, w
    
class Adaptive_Weight_Proposal(nn.Module):
    # DBK - Weight Proposal Network
    def __init__(self, input_ch, num_motion,  
                 D_sam, W_sam, D_mot, W_mot, dir_freq, rgb_freq, depth_freq, ray_dir_freq, 
                 view_feature_ch, view_embedding_layer, view_embed_ch, n_sample, use_origin=True):
        super(Adaptive_Weight_Proposal, self).__init__()
        
        self.input_ch = input_ch
        self.num_motion = num_motion
        self.n_sample = n_sample
        self.rgb_freq = rgb_freq
        self.depth_freq = depth_freq
        self.ray_dir_freq = ray_dir_freq
        self.view_embedding_layer = view_embedding_layer
        self.view_embed_ch = view_embed_ch
        self.view_feature_ch = view_feature_ch
        self.ccw_fine_scale = 0.05
        
        self.use_origin = use_origin
        if use_origin:
            self.output_ch = num_motion + 1
        else:
            self.output_ch = num_motion
        
        self.dropout = nn.Dropout(0.1)
        self.temperature = W_mot**0.5
        
        self.dirs_embed_fn, self.dirs_embed_ch = get_embedder(dir_freq, input_dim=3)
        self.rgb_embed_fn, self.rgb_embed_ch = get_embedder(self.rgb_freq, input_dim=3)
        self.depth_embed_fn, self.depth_embed_ch = get_embedder(self.depth_freq, input_dim=1)
        self.ray_dirs_embed_fn, self.ray_dirs_embed_ch = get_embedder(self.ray_dir_freq, input_dim=3)
        
        self.sample_feature_embed_layer = nn.ModuleList([nn.Linear(self.input_ch, W_sam)] + [nn.Linear(W_sam, W_sam) for i in range(D_sam-1)])
        
        self.motion_feature_embed_layer = nn.ModuleList([nn.Linear((W_sam  + self.view_feature_ch + self.ray_dirs_embed_ch), W_mot)] 
                                                        + [nn.Linear(W_mot, W_mot) for i in range(D_mot)])
        
        
        self.MAM = Motion_Aggregation_Module(in_channels=W_mot, k=3, num_motion=self.num_motion)
        
        self.w_linear = nn.Linear(W_mot, self.output_ch)    
    
    def feature_integration(self, feat, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            feat: [num_rays, num_motion, num_samples along ray, feature_dim]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            feat_integrated: [num_rays, num_motion, feature_dim]. integrated feature of a ray.
        """
        
        N_rays, N_motion, N_sample, N_dim = feat.shape
        feat = feat.reshape(-1, N_sample, N_dim)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        feat_density = feat[..., :-1, :]
        alpha = - torch.exp(-feat_density*dists[...,None]) + 1
        alpha = torch.cat([alpha, torch.zeros_like(alpha[:, 0:1])], dim=-2)

        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, alpha.shape[-1])), - alpha + (1. + 1e-10)], -2), -1)[:, :-1, :]

        feat_integrated = torch.sum(weights*feat, dim=-2)
        
        return feat_integrated.reshape(N_rays, N_motion, N_dim)

    def forward(self, depth_feature, z_vals, rays_d, view_feature):   
        view_embedded = view_feature
        
        N_ray, _, _, _ = depth_feature.reshape(-1, self.output_ch, depth_feature.shape[-2], depth_feature.shape[-1]).shape # N_ray, N_motion, N_samlpe, fearture_dim
        h = depth_feature
        
        
        viewdirs = rays_d.reshape(N_ray, self.output_ch, -1)[:,0,:]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        ray_dirs_embed = self.ray_dirs_embed_fn(viewdirs)
        
        view_embedded = torch.cat([view_embedded, ray_dirs_embed], dim=-1)
        
        for i, l in enumerate(self.sample_feature_embed_layer):
            h = self.sample_feature_embed_layer[i](h)
            h = F.relu(h)
            
        h_local = h
        
        h = self.feature_integration(h.reshape(N_ray, self.output_ch, h.shape[-2], h.shape[-1]), z_vals, rays_d)
        
        view_embedded = view_embedded.unsqueeze(1).repeat(1, self.output_ch, 1)
        h = torch.cat([h, view_embedded], dim=-1)
        
        for i, l in enumerate(self.motion_feature_embed_layer):
            h = self.motion_feature_embed_layer[i](h)
            h = F.relu(h)
        
        h = self.MAM(h, h_local)
        h = F.adaptive_avg_pool1d(h.transpose(1,2),1).squeeze(-1) # 
        
        w = torch.sigmoid(self.w_linear(h))
        out = w/(torch.sum(w, -1, keepdims=True))
        
        return out


# DP-NeRF RBK_AWP
class RBK_AWP(nn.Module):
    def __init__(self, num_img, view_embed_ch,
                D_rbk, W_rbk, D_rbk_r, W_rbk_r, D_rbk_v, W_rbk_v, D_rbk_w, W_rbk_w, D_awp_sam, W_awp_sam, D_awp_mot, W_awp_mot,
                output_ch_rbk_r, output_ch_rbk_v, skips_rbk, rbk_use_origin, rbk_se_rv_window, num_motion_rbk, 
                awp_dir_freq, awp_rgb_freq, awp_depth_freq, awp_ray_dir_freq, n_sample_awp, input_ch_awp, 
                use_dpnerf=False, use_awp=False, near=0.0, far=1.0, ndc=False):
                
        super(RBK_AWP, self).__init__()
        self.use_dpnerf = use_dpnerf
        self.view_embed_ch = view_embed_ch
        self.view_embed_layer = View_Embedding(num_embed=num_img, embed_dim=self.view_embed_ch)
        self.use_awp = use_awp
        
        self.SE3Field = rigid_warping.SE3Field()
        self.RBK = Rigid_Blurring_Kernel(D=D_rbk, W=W_rbk, num_motion=num_motion_rbk,
                           D_r=D_rbk_r, W_r=W_rbk_r, output_ch_r=output_ch_rbk_r,
                           D_v=D_rbk_v, W_v=W_rbk_v, output_ch_v=output_ch_rbk_v,
                           D_w=D_rbk_w, W_w=W_rbk_w, rv_window=rbk_se_rv_window,
                           view_embedding_layer=self.view_embed_layer, use_origin=rbk_use_origin, 
                           input_ch=self.view_embed_ch, skips=skips_rbk, near=near, far=far, ndc=ndc, 
                           warp_field=self.SE3Field, use_awp=self.use_awp)
        
        self.AWPnet = Adaptive_Weight_Proposal(input_ch=input_ch_awp, D_sam=D_awp_sam, W_sam=W_awp_sam, 
                                               D_mot=D_awp_mot, W_mot=W_awp_mot, view_embedding_layer=self.view_embed_layer,
                                               view_embed_ch=view_embed_ch, view_feature_ch=W_rbk, dir_freq=awp_dir_freq,
                                               rgb_freq=awp_rgb_freq, depth_freq=awp_depth_freq, ray_dir_freq=awp_ray_dir_freq, 
                                               num_motion=num_motion_rbk, n_sample=n_sample_awp, use_origin=rbk_use_origin)
        
class NeRFAll(nn.Module):
    def __init__(self, args, blur_kernel_net=None):
        super().__init__()
        self.args = args
        
        
        self.blur_model_type = args.blur_model_type
        self.blur_kernel_net = blur_kernel_net
            
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)
            
        self.input_ch_views = 0
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)
                
        self.output_ch = 5 if args.N_importance > 0 else 4

        self.use_awp = args.use_awp
        
        skips = [4]
        self.mlp_coarse = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
            use_awp=self.use_awp)

        self.mlp_fine = None
        if args.N_importance > 0:
            self.mlp_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
                use_awp=self.use_awp)
            
        # 3 start>=============================================================================
        if blur_kernel_net is not None and self.blur_model_type == 'dpnerf':
            self.dbk_view_embedding = blur_kernel_net.view_embed_layer
            self.mlp_rbk = blur_kernel_net.RBK
        
        if self.use_awp:
            self.AWPnet = self.blur_kernel_net.AWPnet
        # 3 end>=============================================================================
        
        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)
        # self.white_balance = WhiteBalance('white_balance_consistent', args.num_images)

    def mlpforward(self, inputs, viewdirs, mlp, force_naive, inference, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)
        
        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            if self.use_awp and not force_naive and not inference:
                outputs_flat, depth_feature_flat = mlp(embedded)
            else:
                outputs_flat = mlp(embedded)
        else:
            if self.use_awp and not force_naive and not inference:
                outputs_flat = torch.cat([mlp(embedded[i:i + netchunk], force_naive)[0] for i in range(0, embedded.shape[0], netchunk)], 0)
                depth_feature_flat = torch.cat([mlp(embedded[i:i + netchunk], force_naive)[1] for i in range(0, embedded.shape[0], netchunk)], 0)
            else:
                outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)
        
        
        if self.use_awp and not force_naive and not inference:
            outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
            depth_feature = torch.reshape(depth_feature_flat, list(inputs.shape[:-1]) + [depth_feature_flat.shape[-1]])
            return outputs, depth_feature
        else:
            outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
            return outputs


    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.args.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.args.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density
        
        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, density, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    img_idx=None,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False,
                    force_naive=False,
                    inference=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        if self.use_awp and not force_naive and not inference:
            raw, depth_feature = self.mlpforward(pts, viewdirs, self.mlp_coarse, force_naive, inference)
        else:
            raw = self.mlpforward(pts, viewdirs, self.mlp_coarse, force_naive, inference)
            
        rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                             white_bkgd, pytest=pytest)

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0  = rgb_map, depth_map, acc_map, density_map
            if self.use_awp and not force_naive and not inference:
                _ = depth_feature

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals_coarse = z_vals
            z_vals, sample_idx = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]
            
            # 1 start>============================================================================= 
            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            if self.use_awp and not force_naive and not inference:
                raw, depth_feature = self.mlpforward(pts, viewdirs, mlp, force_naive, inference)
            else:
                raw = self.mlpforward(pts, viewdirs, mlp, force_naive, inference)

            rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            if self.use_awp and not force_naive and not inference:
                ret['depth_feature'] = depth_feature
                # ret['awp_feature'] = depth_map.unsqueeze(-1) 
                ret['z_vals'] = z_vals 
                
        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the blur_kernelnet is specified

        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            # force_baseline = kwargs.pop("force_naive", True)
            force_baseline = True if kwargs['force_naive'] else False
            
            if self.blur_kernel_net is not None and not force_baseline and self.blur_model_type == 'dpnerf':
                kwargs['img_idx'] = rays_info['images_idx'].squeeze(-1)
                if self.blur_kernel_net.use_dpnerf:
                    if self.use_awp:
                        rays_info['H'] = H
                        rays_info['W'] = W
                        rays, ccw, view_feature = self.mlp_rbk(rays, rays_info)
                        kwargs['use_awp'] = True
                    else:
                        rays, ccw = self.mlp_rbk(rays, rays_info)
                        
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                    if self.use_awp:    
                        ccw_fine = self.AWPnet(extras['depth_feature'], extras['z_vals'], extras['rays_d'], view_feature)
                        ccw_fine = ccw_fine + ccw_fine * self.AWPnet.ccw_fine_scale
                        ccw_fine = ccw_fine/(torch.sum(ccw_fine, -1, keepdims=True))
                    
                        rgb_fine = rgb.clone()
                        depth_fine = depth.clone()
                        acc_fine = acc.clone()
                        extras_fine = extras.copy()
                        rgb, depth, acc, extras = self.mlp_rbk.rbk_weighted_sum(rgb, depth, acc, extras, ccw)
                        rgb_awp, _, _, _ = self.mlp_rbk.rbk_weighted_sum(rgb_fine, depth_fine, acc_fine, extras_fine, ccw_fine)
                        return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), {'rgb_awp': self.tonemapping(rgb_awp)}
                        
                    else:
                        rgb, depth, acc, extras = self.mlp_rbk.rbk_weighted_sum(rgb, depth, acc, extras, ccw)
                        return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), 
                        
                else:
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                    return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), {}
                    
                
            else:
                kwargs['img_idx'] = rays_info['images_idx'].squeeze(-1)
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), {}
                

        #  evaluation
        else:
            save_warped_ray_img = kwargs['render_kwargs'].pop('save_warped_ray_img', False)
            if save_warped_ray_img:
                # rays, cb_w, view_feature = self.mlp_decompose_cbnet(rays, rays_info)
                rgbs, depths, rays_warped =  self.render_warped_path(H, W, K, chunk, poses, images_idx=rays_info, **kwargs)
                return self.tonemapping(rgbs), depths, rays_warped
            else:
                assert poses is not None, "Please specify poses when in the eval model"
                rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)
                return self.tonemapping(rgbs), depths

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None, 
               use_awp=False,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        
        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        if use_awp:
            ret_dict['rays_d'] = rays_d
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)
            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_warped_path(self, H, W, K, chunk, render_poses, images_idx, render_kwargs, render_factor=0):
        """
        render warped images and depths specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        rays_save = []
        
        rays_info = {}
        
        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            
            ########
            idx = images_idx[i]
            rays_org = rays.reshape(-1, 3, 2)
            idx = idx[None,...].repeat(rays_org.shape[0]).unsqueeze(-1)
            rays_info['images_idx'] = idx
            rays_warped, _, _ = self.mlp_rbk(rays_org, rays_info)
            rays_warped_save = rays_warped.reshape(H, W, self.mlp_rbk.num_motion +1, 3, 2)[int(H/2), int(W/2), ...]
            ########
            
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays_warped, c2w=c2w[:3, :4], **render_kwargs)
            
            ########
            rgb = rgb.reshape(H, W, self.mlp_rbk.num_motion+1, 3).permute(2, 0, 1, 3)
            depth = depth.reshape(H, W, self.mlp_rbk.num_motion+1, 1).permute(2, 0, 1, 3)
            ########
            
            
            rgbs.append(rgb)
            depths.append(depth)
            rays_save.append(rays_warped_save)
            if i == 0:
                print(rgb.shape, depth.shape)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        rays_save = torch.stack(rays_save, 0)

        return rgbs, depths, rays_save