
"""
PointNet for 3D Classification and Segmentation.
Based on the paper: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.img_encoder_v2 import ImageEncoder


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim, device=x.device).unsqueeze(0)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class PointNetBackbone(nn.Module):

    def __init__(self, input_channels=3):
        super(PointNetBackbone, self).__init__()
        self.input_channels = input_channels

        self.input_tnet = TransformationNet(input_dim=3, output_dim=3)

        self.conv_1 = nn.Conv1d(input_channels, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)

        self.feature_tnet = TransformationNet(input_dim=64, output_dim=64)

        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 1024, 1)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        xyz = x[:, :, :3]
        input_transform = self.input_tnet(xyz)
        xyz = torch.bmm(xyz, input_transform)

        if x.shape[2] > 3:
            extra_channels = x[:, :, 3:]
            x = torch.cat([xyz, extra_channels], dim=2)
        else:
            x = xyz

        x = x.transpose(2, 1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        feat1 = x
        x = F.relu(self.bn_2(self.conv_2(x)))
        feat2 = x

        x = x.transpose(2, 1)
        feature_transform = self.feature_tnet(x)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)

        point_features = x.clone()

        x = F.relu(self.bn_3(self.conv_3(x)))
        feat3 = x
        x = F.relu(self.bn_4(self.conv_4(x)))
        feat4 = x
        x = F.relu(self.bn_5(self.conv_5(x)))

        global_feature = torch.max(x, 2, keepdim=True)[0]

        return feat1, feat2, feat3, feat4, feature_transform, point_features, global_feature


class PointNetSegmentation(nn.Module):

    def __init__(self, num_classes, input_channels=3, dropout=0.3, skip_conn=False):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.skip_conn = skip_conn
        self.dropout = nn.Dropout(p=dropout)

        self.backbone = PointNetBackbone(input_channels=input_channels)

        conv_size = 1088
        if self.skip_conn:
            conv_size += (64 + 64 + 64 + 128)

        self.conv_1 = nn.Conv1d(conv_size, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

        self.conv_4 = nn.Conv1d(128, 128, 1)
        self.conv_5 = nn.Conv1d(128, num_classes, 1)
        self.bn_4 = nn.BatchNorm1d(128)

    def forward(self, x):
        f1, f2, f3, f4, feature_transform, point_features, global_feature = self.backbone(x)

        num_points = x.shape[1]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)

        if self.skip_conn:
            x = torch.cat([f1, f2, point_features, f3, f4, global_feature_expanded], dim=1)
        else:
            x = torch.cat([point_features, global_feature_expanded], dim=1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = self.dropout(x)
        log_probs = F.log_softmax(self.conv_5(x), dim=1)

        return feature_transform, log_probs


class IPointNetSegmentation(nn.Module):

    def __init__(self, num_classes, input_channels=3, dropout=0.3, skip_conn=False):
        super(IPointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.skip_conn = skip_conn
        self.dropout = nn.Dropout(p=dropout)

        self.backbone = PointNetBackbone(input_channels=input_channels)

        self.img_encoder = ImageEncoder(channels=(4, 64, 128))

        conv_size = 1088 + 128
        if self.skip_conn:
            conv_size += (64 + 64 + 64 + 128)

        self.conv_1 = nn.Conv1d(conv_size, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

        self.conv_4 = nn.Conv1d(128, 128, 1)
        self.conv_5 = nn.Conv1d(128, num_classes, 1)
        self.bn_4 = nn.BatchNorm1d(128)

    def _sample_bev_features(self, bev_feat_map, xy_grid):

        grid = xy_grid.unsqueeze(2)

        sampled = F.grid_sample(
            bev_feat_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        sampled = sampled.squeeze(-1)
        return sampled

    def forward(self, x, img, xy_grid):
        f1, f2, f3, f4, feature_transform, point_features, global_feature = self.backbone(x)

        _, image_feature_map = self.img_encoder(img)

        num_points = x.shape[1]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)

        image_feature_sampled = self._sample_bev_features(image_feature_map, xy_grid)

        if self.skip_conn:
            x = torch.cat(
                [f1, f2, point_features, f3, f4, global_feature_expanded, image_feature_sampled],
                dim=1
            )
        else:
            x = torch.cat(
                [point_features, global_feature_expanded, image_feature_sampled],
                dim=1
            )

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = self.dropout(x)
        log_probs = F.log_softmax(self.conv_5(x), dim=1)

        return feature_transform, log_probs
