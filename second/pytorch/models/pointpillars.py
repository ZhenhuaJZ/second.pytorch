"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty, GroupNorm, Sequential

from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PointConvFLN(nn.Module):
    def __init__(self,
                 out_filters = [4, 8, 16],
                 kernel_size = [7, 11, 19],
                 num_features = 4,
                 is_max_pool = True):
        super().__init__()
        Linear = change_default_args(bias=False)(nn.Linear)
        Conv1d = change_default_args(bias=False)(nn.Conv1d)
        BatchNorm1d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm1d)

        self.is_max_pool = is_max_pool

        self.pn_linear = Linear(num_features, 64)
        self.pn_norm = BatchNorm1d(64)
        self.pn_relu = nn.ReLU()

        self.pool_linear = Linear(64, 1)
        self.pool_bn = BatchNorm1d(100)
        self.pool_relu = nn.ReLU()

        self.block1 = Sequential()
        lo = 100
        for idx in range(len(out_filters)):
            in_filter = 1 if idx == 0 else out_filters[idx-1]
            self.block1.add(Conv1d(in_filter, out_filters[idx], kernel_size[0]))
            self.block1.add(BatchNorm1d(out_filters[idx]))
            self.block1.add(nn.ReLU())
            lo = (lo + 2 - (kernel_size[0] - 1) - 1) + 1
        print(lo)
        if is_max_pool:
            self.block1_mp = nn.MaxPool1d(82)

        self.block2 = Sequential()
        for idx in range(len(out_filters)):
            in_filter = 1 if idx == 0 else out_filters[idx-1]
            self.block2.add(Conv1d(in_filter, out_filters[idx], kernel_size[1]))
            self.block2.add(BatchNorm1d(out_filters[idx]))
            self.block2.add(nn.ReLU())
            lo = (lo + 2 - (kernel_size[0] - 1) - 1)/1 + 1


        self.block3 = Sequential()
        for idx in range(len(out_filters)):
            in_filter = 1 if idx == 0 else out_filters[idx-1]
            self.block3.add(Conv1d(in_filter, out_filters[idx], kernel_size[2]))
            self.block3.add(BatchNorm1d(out_filters[idx]))
            self.block3.add(nn.ReLU())
            lo = (lo + 2 - (kernel_size[0] - 1) - 1)/1 + 1

    def forward(self, feature_):
        feature = self.pn_linear(feature_)
        feature = self.pn_norm(feature.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        feature = self.pn_relu(feature)
        print("[debug] pn_relu: ", feature.shape)

        feature = self.pool_linear(feature)
        print("[debug] pool_linear: ", feature.shape)

        feature = self.pool_bn(feature)
        print("[debug] pool_bn: ", feature.shape)

        feature = self.pool_relu(feature)
        print("[debug] pool_relu: ", feature.shape)

        feature = feature.permute(0,2,1).contiguous()
        feat1 = self.block1(feature)
        print("[debug] feat1: ", feat1.shape)
        if self.is_max_pool:
            feat1 = self.block1_mp(feat1)
        print("[debug] feat1: ", feat1.shape)

        feat2 = self.block2(feature)
        print("[debug] feat2: ", feat2.shape)

        feat3 = self.block3(feature)
        print("[debug] feat3: ", feat3.shape)



class PointLinearFLN(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 is_max_pool = True):
        super().__init__()
        Linear = change_default_args(bias=False)(nn.Linear)
        BatchNorm1d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm1d)

        self.pn_linear = Linear(in_filters[0], out_filters[0])
        self.pn_norm = BatchNorm1d(out_filters[0])
        self.pn_relu = nn.ReLU()

        self.is_max_pool = is_max_pool
        if is_max_pool:
            kernel = out_filters[0]
            self.pool = nn.MaxPool1d(kernel)
        else:
            self.pool_linear = Linear(out_filters[0], 1)
            self.pool_bn = BatchNorm1d(100)
            self.pool_relu = nn.ReLU()

        self.block = Linear(in_filters[1], out_filters[1])
        self.bn = BatchNorm1d(out_filters[1])
        self.relu = nn.ReLU()


    def forward(self, feature_):
        # feature = torch.unsqueeze(feature, -1)
        print("#"*50)
        # print("[debug] feature_ shape: ", feature_.shape)
        # print("[debug] feature_: \n", feature_)
        feature = self.pn_linear(feature_)
        # print("[debug] feature shape: ", feature.shape)
        # print("[debug] feature: \n", feature)
        # print("[debug] feature.permute(0, 2, 1).contiguous(): ", feature.permute(0, 2, 1).contiguous().shape)
        feature = self.pn_norm(feature.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        feature = self.pn_relu(feature)
        # Max pool maximum or linear Feature for each point
        if self.is_max_pool:
            feature = self.pool(feature)
        else:
            feature = self.pool_linear(feature)
            feature = self.pool_bn(feature)
            feature = self.pool_relu(feature)

        feature = feature.permute(0, 2, 1).contiguous()
        # apply linear layer to all the points to extract point relationship
        feature = self.block(feature)
        feature = self.bn(feature.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        feature = self.relu(feature)
        return feature


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        self.clus_layer = PointLinearFLN([num_input_features,100], [32, num_filters[0]], is_max_pool = False)
        self.conv_layer = PointConvFLN(num_features = num_input_features)
        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = features[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
        # print("[debug] aft f_center: ",np.unique(np.isnan(f_center.cpu().numpy())))
        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            # print("[debug] aft norm: ",np.unique(np.isnan(features.cpu().numpy())))
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
        # print("[debug] aft cat: ",np.unique(np.isnan(features.cpu().numpy())))

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        # Forward pass through PFNLayers
        # print(np.unique(np.isnan(features.cpu().numpy())))
        # for pfn in self.pfn_layers:
        #     features = pfn(features)
        features = self.clus_layer(features)
        # features = self.conv_layer(features)
        # print(features.shape)
        return features.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas
