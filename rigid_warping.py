import torch
import torch.nn as nn




class SE3Field():
    """
        warp the given points with rotation(r) and translation(v) value.
        
        input: 
        
        output:
    """
    def __init__(self):
        super(SE3Field, self).__init__()
        self.rigid_body = Rigid_body()
        
    # def warp(self, pts, rot, trans, adj_pts_use=False):
    def warp(self, pts, rot, trans):
        '''
            one sample:
                        pts: N_rays*N_samples x 3
                        rot: N_rays*N_samples x 3
                        trans: N_rays*N_samples x 3
            adj sample:
                        pts: N_rays*N_samples x (N_adjs+1) x 3
                        rot: N_rays*N_samples x 3
                        trans: N_rays*N_samples x 3
        '''
        theta = torch.linalg.norm(rot, axis=-1) # N_rays*N_samples
        theta += 1.0e-10
        rot = rot / theta.unsqueeze(-1) # N_rays*N_samples x 3
        trans = trans / theta.unsqueeze(-1) # N_rays*N_samples x 3
        screw_axis = torch.cat([rot, trans], axis=-1) # N_rays*N_samples x 6
        transform = self.rigid_body.exp_se3(screw_axis, theta) # N_rays*N_samples x 4 x 4
        
        # if adj_pts_use:
        #     warped_pts = torch.matmul(transform.unsqueeze(1), self.rigid_body.to_homogenous(pts)[..., None]).squeeze(-1)
        # else:
        #     warped_pts = torch.matmul(transform, self.rigid_body.to_homogenous(pts)[..., None]).squeeze(-1)
        
        warped_pts = torch.matmul(transform, self.rigid_body.to_homogenous(pts)[..., None]).squeeze(-1)
        warped_pts = self.rigid_body.from_homogenous(warped_pts)
        return warped_pts



class Rigid_body():
    def __init__(self):
        super(Rigid_body, self).__init__()  
          
    def exp_se3(self, S, theta):
        ''' 
            Exponential map from Lie algebra so3 to Lie group SO3.

            Modern Robotics Eqn 3.88.

            Args:
                S: (6,) A screw axis of motion.
                theta: Magnitude of motion.

            Returns:
                a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
                motion of magnitude theta about S for one second.
        '''
        # w, v = torch.split(S, 2)
        w, v = torch.split(S, [3, 3], dim=-1)
        W =self.skew(w)
        R = self.exp_so3(w, theta) 
        theta = theta[..., None, None]
        p = torch.matmul((theta * torch.eye(3)[None, ...] + (1.0 - torch.cos(theta)) * W +
                          (theta - torch.sin(theta)) * torch.matmul(W, W)), v[..., None])
        return self.rp_to_se3(R, p)
    
    def exp_so3(self, w, theta):
        """Exponential map from Lie algebra so3 to Lie group SO3.

            Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

            Args:
                w: (3,) An axis of rotation. This is assumed to be a unit-vector.
                theta: An angle of rotation.

            Returns:
                R: (3, 3) An orthonormal rotation matrix representing a rotation of
                magnitude theta about axis w.
        """
        W = self.skew(w)
        theta = theta[..., None, None]
        return (torch.eye(3)[None, ...] + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * torch.matmul(W, W))
        
    def skew(self, w):
        """Build a skew matrix ("cross product matrix") for vector w.

            Modern Robotics Eqn 3.30.

            Args:
                w: (3,) A 3-vector

            Returns:
                W: (3, 3) A skew matrix such that W @ v == w x v
        """
        w_new = torch.zeros_like(w).unsqueeze(1).repeat(1,3,1) # W_ : Num_rays*Num_samples x 3 x 3
        w_re = torch.reshape(w_new, (w.shape[0], 9))
        w_re[..., 1] =  -w[...,2]
        w_re[..., 2] = w[..., 1]
        w_re[..., 3] = w[..., 2]
        w_re[..., 5] = -w[..., 0]
        w_re[..., 6] = -w[..., 1]
        w_re[..., 7] = w[..., 0]
        
        return w_re.reshape((w.shape[0], 3, 3))
        # w = torch.reshape(w, (3))
        # return torch.tensor([[0.0, -w[2], w[1]], \
                            # [w[2], 0.0, -w[0]], \
                            # [-w[1], w[0], 0.0]])
        
    
    def rp_to_se3(self, R, p):
        """
            Rotation and translation to homogeneous transform.

            Args:
                R: (3, 3) An orthonormal rotation matrix.
                p: (3,) A 3-vector representing an offset.

            Returns:
                X: (4, 4) The homogeneous transformation matrix described by rotating by R
                and translating by p.
        """
        RT_mat = torch.cat([R, p], -1)
        filling_mat = torch.tensor([0.0, 0.0, 0.0, 1.0])[None, None, ...].repeat(RT_mat.shape[0], 1, 1)
        return torch.cat([RT_mat, filling_mat], dim=-2)
        
    def to_homogenous(self, v):
        return torch.cat([v, torch.ones_like(v[..., :1])], axis=-1)
    
    def from_homogenous(self, v):
        return v[..., :3] / v[..., -1:]
    
    
    