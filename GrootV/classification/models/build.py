from .grootv import GrootV


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'grootv':
        model = GrootV(
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.GROOTV.CHANNELS,
            depths=config.MODEL.GROOTV.DEPTHS,
            layer_scale=config.MODEL.GROOTV.LAYER_SCALE,
            post_norm=config.MODEL.GROOTV.POST_NORM,
            mlp_ratio=config.MODEL.GROOTV.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
