from active_zero2.models.psmnet_range.psmnet_3 import PSMNetRange


def build_model(cfg):
    model = PSMNetRange(
        min_disp=cfg.PSMNetRange.MIN_DISP,
        max_disp=cfg.PSMNetRange.MAX_DISP,
        num_disp=cfg.PSMNetRange.NUM_DISP,
        set_zero=cfg.PSMNetRange.SET_ZERO,
    )
    return model
