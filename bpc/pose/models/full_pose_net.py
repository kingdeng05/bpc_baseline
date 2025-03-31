import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

class FullPoseNet(nn.Module):
    def __init__(self, loss_type="euler", pretrained=True, backbone_type="convnext_tiny"):
        """
        Args:
            loss_type: one of "euler", "quat", or "6d". Determines the number of output neurons.
            pretrained: if True, use pretrained ResNet50 weights.
        """
        super(FullPoseNet, self).__init__()

        # self.backbone = timm.create_model(
        #     backbone_type,
        #     pretrained=pretrained,
        #     features_only=False,
        #     num_classes=0  # we don't want a classifier head
        # )
        # backbone_out_dim = self.backbone.num_features

        # Load a ResNet50 backbone.
        backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        layers = list(backbone.children())[:-1]  # Remove the classification head.
        self.backbone = nn.Sequential(*layers)
        backbone_out_dim = 2048
        
        # Determine rotation output dimension.
        if loss_type == "euler":
            out_dim = 3
        elif loss_type == "quat":
            out_dim = 4
        elif loss_type == "6d":
            out_dim = 6
        else:
            raise ValueError("loss_type must be one of 'euler', 'quat', or '6d'")
        
        self.rot_fc = nn.Linear(backbone_out_dim, out_dim)
        self.trans_fc = nn.Linear(backbone_out_dim, 3)

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        preds_rot = self.rot_fc(feats)
        preds_trans = self.trans_fc(feats)
        return preds_rot, preds_trans
