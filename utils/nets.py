import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils.vit as vits


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadEncoder(nn.Module):
    def __init__(
        self,
        arch,
        patch_size,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
        pretrained=None,
        freeze=False,
    ):
        super().__init__()

        # backbone
        self.patch_size = patch_size
        self.encoder = self.set_encoder(arch, low_res, pretrained, freeze)

        self.head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()
    
    @torch.no_grad()
    def set_encoder(self, arch, low_res, pretrained, freeze):
        if 'resnet' in arch:
            model = models.__dict__[arch]()
            self.feat_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
            if pretrained is not None:
                ckpt = torch.load(pretrained, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt, strict=False)
            if freeze:
                for param in model.layer1.parameters():
                    param.requires_grad = False
                for param in model.layer2.parameters():
                    param.requires_grad = False
                for param in model.layer3.parameters():
                    param.requires_grad = False
                for param in model.layer4.parameters():
                    param.requires_grad = False
            # modify the encoder for lower resolution
            if low_res:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
        # elif 'vit_base' in arch:
        #     model = vits.__dict__[arch](patch_size=16)
        #     self.feat_dim = model.num_features
        #     if backbone is not None:
        #         ckpt = torch.load(backbone, map_location=torch.device('cpu'))
        #         if 'sup' in backbone:
        #             del ckpt['pos_embed'], ckpt['patch_embed.proj.weight'], ckpt['patch_embed.proj.bias']
        #         model.load_state_dict(ckpt, strict=False)
        #     if freeze:
        #         for param in model.blocks.parameters():
        #             param.requires_grad = False
        elif 'vit' in arch:
            model = vits.__dict__[arch](
                        image_size=32 if low_res else 224,
                        patch_size=self.patch_size,
                        pretrained=pretrained
                    )
            self.feat_dim = model.hidden_dim
            if freeze:
                for param in model.encoder.parameters():
                    param.requires_grad = False
        if pretrained is None:
            self._reinit_all_layers()
        return model

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out