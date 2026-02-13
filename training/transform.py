import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import numpy as np

def trilinear_interpolation3_torch(p, ddf):
    
    p = torch.clamp(p, min=0, max=ddf.shape[2]-2)
    p = p[0].T

    x = p[1, :]
    y = p[0, :]
    z = p[2, :]

    const1 = torch.tensor(1)
    const2 = torch.tensor(ddf.shape[2]-3)

    x0 = torch.maximum(torch.floor(x), const1)
    y0 = torch.maximum(torch.floor(y), const1)
    z0 = torch.maximum(torch.floor(z), const1)

    x0 = torch.minimum(x0, const2)
    y0 = torch.minimum(y0, const2)
    z0 = torch.minimum(z0, const2)

    x1 = torch.clone(x0) + 1
    y1 = torch.clone(y0) + 1
    z1 = torch.clone(z0) + 1

    x0 = x0.long()
    y0 = y0.long()
    z0 = z0.long()
    x1 = x1.long()
    y1 = y1.long()
    z1 = z1.long()

    ddf_x = ddf[0, 1, :, :, :]
    ddf_y = ddf[0, 0, :, :, :]
    ddf_z = ddf[0, 2, :, :, :]

    C000_x = ddf_x[y0, x0, z0]
    C100_x = ddf_x[y1, x0, z0]
    C010_x = ddf_x[y0, x1, z0]
    C001_x = ddf_x[y0, x0, z1]
    C110_x = ddf_x[y1, x1, z0]
    C101_x = ddf_x[y1, x0, z1]
    C011_x = ddf_x[y0, x1, z1]
    C111_x = ddf_x[y1, x1, z1]
    
    C000_y = ddf_y[y0, x0, z0]
    C100_y = ddf_y[y1, x0, z0]
    C010_y = ddf_y[y0, x1, z0]
    C001_y = ddf_y[y0, x0, z1]
    C110_y = ddf_y[y1, x1, z0]
    C101_y = ddf_y[y1, x0, z1]
    C011_y = ddf_y[y0, x1, z1]
    C111_y = ddf_y[y1, x1, z1]

    C000_z = ddf_z[y0, x0, z0]
    C100_z = ddf_z[y1, x0, z0]
    C010_z = ddf_z[y0, x1, z0]
    C001_z = ddf_z[y0, x0, z1]
    C110_z = ddf_z[y1, x1, z0]
    C101_z = ddf_z[y1, x0, z1]
    C011_z = ddf_z[y0, x1, z1]
    C111_z = ddf_z[y1, x1, z1]

    xd = x - x0
    yd = y - y0
    zd = z - z0

    C00_x = C000_x*(1-xd) + C100_x*xd
    C01_x = C001_x*(1-xd) + C101_x*xd
    C10_x = C010_x*(1-xd) + C110_x*xd
    C11_x = C011_x*(1-xd) + C111_x*xd
    C0_x = C00_x*(1-yd) + C10_x*yd
    C1_x = C01_x*(1-yd) + C11_x*yd
    Ci_x = C0_x*(1-zd) + C1_x*zd
    C00_y = C000_y*(1-xd) + C100_y*xd
    C01_y = C001_y*(1-xd) + C101_y*xd
    C10_y = C010_y*(1-xd) + C110_y*xd
    C11_y = C011_y*(1-xd) + C111_y*xd
    C0_y = C00_y*(1-yd) + C10_y*yd
    C1_y = C01_y*(1-yd) + C11_y*yd
    Ci_y = C0_y*(1-zd) + C1_y*zd
    C00_z = C000_z*(1-xd) + C100_z*xd
    C01_z = C001_z*(1-xd) + C101_z*xd
    C10_z = C010_z*(1-xd) + C110_z*xd
    C11_z = C011_z*(1-xd) + C111_z*xd
    C0_z = C00_z*(1-yd) + C10_z*yd
    C1_z = C01_z*(1-yd) + C11_z*yd
    Ci_z = C0_z*(1-zd) + C1_z*zd
    
    # change Ci to shape [N 1]
    Ci_x = Ci_x.reshape([-1, 1])
    Ci_y = Ci_y.reshape([-1, 1])
    Ci_z = Ci_z.reshape([-1, 1])

    q = torch.cat([Ci_y, Ci_x, Ci_z], axis=1)
    q = q[None, :]
    p = p.T[None, :]

    return p + q

def trilinear_interpolation3_torch1(p, ddf):
    
    p = torch.clamp(p, min=0, max=ddf.shape[2]-2)
    p = p[0].T

    x = p[0, :]
    y = p[1, :]
    z = p[2, :]

    const1 = torch.tensor(1)
    const2 = torch.tensor(ddf.shape[2]-3)

    x0 = torch.maximum(torch.floor(x), const1)
    y0 = torch.maximum(torch.floor(y), const1)
    z0 = torch.maximum(torch.floor(z), const1)

    x0 = torch.minimum(x0, const2)
    y0 = torch.minimum(y0, const2)
    z0 = torch.minimum(z0, const2)

    x1 = torch.clone(x0) + 1
    y1 = torch.clone(y0) + 1
    z1 = torch.clone(z0) + 1

    x0 = x0.long()
    y0 = y0.long()
    z0 = z0.long()
    x1 = x1.long()
    y1 = y1.long()
    z1 = z1.long()

    ddf_x = ddf[0, 0, :, :, :]
    ddf_y = ddf[0, 1, :, :, :]
    ddf_z = ddf[0, 2, :, :, :]

    C000_x = ddf_x[y0, x0, z0]
    C100_x = ddf_x[y1, x0, z0]
    C010_x = ddf_x[y0, x1, z0]
    C001_x = ddf_x[y0, x0, z1]
    C110_x = ddf_x[y1, x1, z0]
    C101_x = ddf_x[y1, x0, z1]
    C011_x = ddf_x[y0, x1, z1]
    C111_x = ddf_x[y1, x1, z1]
    
    C000_y = ddf_y[y0, x0, z0]
    C100_y = ddf_y[y1, x0, z0]
    C010_y = ddf_y[y0, x1, z0]
    C001_y = ddf_y[y0, x0, z1]
    C110_y = ddf_y[y1, x1, z0]
    C101_y = ddf_y[y1, x0, z1]
    C011_y = ddf_y[y0, x1, z1]
    C111_y = ddf_y[y1, x1, z1]

    C000_z = ddf_z[y0, x0, z0]
    C100_z = ddf_z[y1, x0, z0]
    C010_z = ddf_z[y0, x1, z0]
    C001_z = ddf_z[y0, x0, z1]
    C110_z = ddf_z[y1, x1, z0]
    C101_z = ddf_z[y1, x0, z1]
    C011_z = ddf_z[y0, x1, z1]
    C111_z = ddf_z[y1, x1, z1]

    xd = x - x0
    yd = y - y0
    zd = z - z0

    C00_x = C000_x*(1-xd) + C100_x*xd
    C01_x = C001_x*(1-xd) + C101_x*xd
    C10_x = C010_x*(1-xd) + C110_x*xd
    C11_x = C011_x*(1-xd) + C111_x*xd
    C0_x = C00_x*(1-yd) + C10_x*yd
    C1_x = C01_x*(1-yd) + C11_x*yd
    Ci_x = C0_x*(1-zd) + C1_x*zd
    C00_y = C000_y*(1-xd) + C100_y*xd
    C01_y = C001_y*(1-xd) + C101_y*xd
    C10_y = C010_y*(1-xd) + C110_y*xd
    C11_y = C011_y*(1-xd) + C111_y*xd
    C0_y = C00_y*(1-yd) + C10_y*yd
    C1_y = C01_y*(1-yd) + C11_y*yd
    Ci_y = C0_y*(1-zd) + C1_y*zd
    C00_z = C000_z*(1-xd) + C100_z*xd
    C01_z = C001_z*(1-xd) + C101_z*xd
    C10_z = C010_z*(1-xd) + C110_z*xd
    C11_z = C011_z*(1-xd) + C111_z*xd
    C0_z = C00_z*(1-yd) + C10_z*yd
    C1_z = C01_z*(1-yd) + C11_z*yd
    Ci_z = C0_z*(1-zd) + C1_z*zd
    
    # change Ci to shape [N 1]
    Ci_x = Ci_x.reshape([-1, 1])
    Ci_y = Ci_y.reshape([-1, 1])
    Ci_z = Ci_z.reshape([-1, 1])

    q = torch.cat([Ci_x, Ci_y, Ci_z], axis=1)
    q = q[None, :]
    p = p.T[None, :]

    return p + q

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, integrate=False):
        super().__init__()
        self.integrate = integrate

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.numpy().astype(np.float32)
        grid = torch.from_numpy(grid).to('cuda')
        self.grid = grid
        self.dfs = {}

    def forward(self, src, flow, mode='bilinear'):

        # new locations
        # self.grid shape: torch.Size([1, 3, 64, 76, 80])
        # flow.shape: torch.Size([1, 3, 64, 76, 80])

        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # need to NORMALIZE GRID VALUES to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1) # permute dimension
        new_locs = new_locs[..., [2, 1, 0]] # flip channels
        moved = nnf.grid_sample(src, new_locs, align_corners=True, mode=mode) # from the nnf.grid_sample documentation: When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear

        return moved

class PointsTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        pass

    def forward(self, src, flow):

        # new locations
        # self.grid shape: torch.Size([1, 3, 64, 76, 80])
        # flow.shape: torch.Size([1, 3, 64, 76, 80])

        new_locs = self.grid + flow
        new_df = new_locs.clone()

        shape = flow.shape[2:]

        # need to NORMALIZE GRID VALUES to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1) # permute dimension
        new_locs = new_locs[..., [2, 1, 0]] # flip channels
        moved = nnf.grid_sample(src, new_locs, align_corners=True, mode='bilinear') # from the nnf.grid_sample documentation: When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear

        return moved