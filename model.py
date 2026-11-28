"""ResNet backbone pre-trained on ImageNet + a new 1-logit head (cyclone vs non-cyclone)."""
import torch, torch.nn as nn
import torchvision.models as tvm

_ARCH = {"resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
         "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1),
         "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2)}

def build_model(arch="resnet18", pretrained=True, dropout=0.3):
    fn, w = _ARCH[arch]
    m = fn(weights=w if pretrained else None)        # PRE-TRAINING: ImageNet weights (downloaded once)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))   # new head
    return m

def set_stage(m, stage):
    """stage = 'head'     : only the new fc layer learns            (feature extraction)
       stage = 'finetune' : layer3 + layer4 + fc learn              (fine-tuning, low LR)
       Early layers (conv1, layer1, layer2) stay frozen: edges/textures are generic."""
    for p in m.parameters():
        p.requires_grad = False
    train_parts = {"head": [m.fc], "finetune": [m.layer3, m.layer4, m.fc]}[stage]
    for part in train_parts:
        for p in part.parameters():
            p.requires_grad = True

def set_train_mode(m, stage):
    """Frozen blocks keep BatchNorm in eval mode so their statistics are not disturbed."""
    m.train()
    frozen = {"head": [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4],
              "finetune": [m.conv1, m.bn1, m.layer1, m.layer2]}[stage]
    for part in frozen:
        part.eval()

def param_groups(m, stage, lr):
    """Discriminative LRs in fine-tuning: deeper = more task-specific = larger LR."""
    if stage == "head":
        return [{"params": m.fc.parameters(), "lr": lr}]
    return [{"params": m.layer3.parameters(), "lr": lr * 0.3},
            {"params": m.layer4.parameters(), "lr": lr},
            {"params": m.fc.parameters(), "lr": lr * 3}]
