# factory backbone

import inspect

def get_backbone(args: object):
    from models.backbone.vit1d import ViT1DEncoder

    BACKBONE_REGISTRY = {
        "vit1d": ViT1DEncoder,
    }

    backbone_name = args.model
    backbone_class = BACKBONE_REGISTRY[backbone_name]

    # Convert args -> dict
    args_dict = vars(args)

    # Inspecte la signature du constructeur
    sig = inspect.signature(backbone_class.__init__)
    valid_args = {
        k: v for k, v in args_dict.items()
        if k in sig.parameters and k != "self"
    }

    return backbone_class(**valid_args)
