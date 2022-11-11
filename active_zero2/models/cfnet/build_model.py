from active_zero2.models.cfnet.cfnet import cfnet


def build_model(cfg):
    model = cfnet(
        maxdisp=cfg.CFNet.MAX_DISP,
        use_concat_volume=cfg.CFNet.USE_CONCAT_VOLUME,
    )
    return model
