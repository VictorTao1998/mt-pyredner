import math
import torch
import torch.nn as nn
from .Regressor import Regressor
from .losses import *
from ..psmnet.build_model import build_model as build_psmnet
from .utils import *

class SMDHead(nn.Module):
    def __init__(self, cfg):
        super(SMDHead, self).__init__()
        self.output_representation = cfg.SMDNet.OUTPUT
        self.maxdisp = cfg.SMDNet.MAXDISP
        self.superes_factor = cfg.SMDNet.SUPERRES
        self.aspect_ratio = cfg.SMDNet.ASPECT
        self.last_dim = {"standard": 1, "unimodal": 2, "bimodal": 5}

        self.stereo_network = build_psmnet(cfg)
        self.mlp = Regressor(filter_channels=[self.stereo_network.init_dim, 1024, 512, 256, 128, self.last_dim[self.output_representation]], \
                             no_sine=cfg.SMDNet.SINE, no_residual=cfg.SMDNet.RESIDUAL)

    def filter(self, left, right, phase='train'):
        # Extract features from the stereo backbone
        batch = {"img_l": left, "img_r": right}
        pred_dict =  self.stereo_network(batch)
        self.feat_list = [pred_dict["cost3"], pred_dict["refimg_fea"]]
        self.height = left.shape[2]
        self.width = left.shape[3]

        # Disparity at arbitrary spatial resolution
        if phase == 'test':
            self.height *= self.superes_factor
            self.width *=  self.superes_factor

    def query(self, points, labels=None):
        if labels is not None:
            # Rescale ground truth between [0, 1] for numerical stability
            self.labels = labels / self.maxdisp
            self.height = self.height / self.aspect_ratio
            self.width = self.width / self.aspect_ratio

        # Coordinated between [-1, 1]
        u = scale_coords(points[:, 0:1, :], self.width)
        v = scale_coords(points[:, 1:2, :], self.height)
        uv = torch.cat([u,v],1)

        # Interpolate features
        for i, im_feat in enumerate(self.feat_list):
            interp_feat = interpolate(im_feat, uv)
            features = interp_feat if not i else torch.cat([features,  interp_feat], 1)

        pred = self.mlp(features)
        activation = nn.Sigmoid()

        # Bimodal output representation
        if self.output_representation == "bimodal":
            eps = 1e-2 #1e-3 in case of gaussian distribution
            self.mu0 = activation(torch.unsqueeze(pred[:,0,:],1))
            self.mu1 = activation(torch.unsqueeze(pred[:,1,:],1))

            self.sigma0 =  torch.clamp(activation(torch.unsqueeze(pred[:,2,:],1)), eps, 1.0)
            self.sigma1 =  torch.clamp(activation(torch.unsqueeze(pred[:,3,:],1)), eps, 1.0)

            self.pi0 = activation(torch.unsqueeze(pred[:,4,:],1))
            self.pi1 = 1. - self.pi0

            # Mode with the highest density value as final prediction
            mask = (self.pi0 / self.sigma0  >   self.pi1 / self.sigma1).float()
            self.disp = self.mu0 * mask + self.mu1 * (1. - mask)

            # Rescale outputs
            self.preds = [self.disp * self.maxdisp,
                          self.mu0 * self.maxdisp,
                          self.mu1 * self.maxdisp ,
                          self.sigma0, self.sigma1,
                          self.pi0, self.pi1]

        # Unimodal output representation
        elif self.output_representation == "unimodal":
            self.disp = activation(torch.unsqueeze(pred[:,0,:],1))
            self.var = activation(torch.unsqueeze(pred[:, 1, :], 1))
            self.preds = [self.disp * self.maxdisp , self.var]

        # Standard regression
        else:
            self.disp = activation(pred)
            self.preds = self.disp * self.maxdisp

    def get_preds(self):
        return self.preds

    def get_error(self):
        mask = torch.mul(self.labels > 0, self.labels <= 1.)

        if self.output_representation == "bimodal":
            loss = bimodal_loss(self.mu0[mask], self.mu1[mask], self.sigma0[mask], self.sigma1[mask],
                                self.pi0[mask], self.pi1[mask], self.labels[mask], dist="laplacian").mean()
            errors = {"log_likelihood_loss": loss}

        elif self.output_representation == "unimodal":
            loss = unimodal_loss(self.disp[mask], self.var[mask], self.labels[mask]).mean()
            errors = {"log_likelihood_loss": loss}
        else:
            loss = torch.abs(self.disp[mask] - self.labels[mask]).mean()
            errors = {"l1_loss": loss}

        return errors

    def forward(self, data_batch):
        left = data_batch["img_l"]
        right = data_batch["img_r"]
        points = data_batch["img_points"]
        labels = data_batch["img_labels"]
        # Get stereo features
        self.filter(left, right)

        # Point query
        self.query(points=points, labels=labels)

        # Get the error
        error = self.get_error()

        return error

    def inference(self, height=1536, width=2048, num_samples=200000, num_out=1):
        nx = np.linspace(0, width - 1, width)
        ny = np.linspace(0, height - 1, height)
        u, v = np.meshgrid(nx, ny)

        coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
        batch_size, n_pts, _= coords.shape
        coords = torch.Tensor(coords).float().cuda()
        output = torch.zeros(num_out, math.ceil(width * height / num_samples), num_samples)

        with torch.no_grad():
            for i, p_split in enumerate(torch.split(coords.reshape(batch_size, -1, 2), int(num_samples / batch_size), dim=1)):
                points = torch.transpose(p_split, 1, 2)
                self.query(points.cuda())
                preds = self.get_preds()
                for k in range(num_out):
                    output[k, i, :p_split.shape[1]] = preds[k].cuda()
        res = []
        for i in range(num_out):
            res.append(output[i].view( 1, -1)[:,:n_pts].reshape(-1, height, width))
        return res

    def predict(self, data, superes_factor=1., compute_entropy=False):
        output = {}
        left = data['img_l']
        right = data['img_r']
        shape = left[0][0].shape

        self.filter(left, right, phase='test')

        num_out = {"standard": 1,
                   "unimodal": 2,
                   "bimodal": 7}
        height = left[0].shape[1]
        width = left[0].shape[2]

        h = shape[1]
        w = shape[0]
        s = superes_factor

        res = self.inference(height=int(height * superes_factor),
                                   width=int(width * superes_factor),
                                   num_out=num_out[self.output_representation])

        output["pred_disp"] = to_numpy(res[0])
        return output
