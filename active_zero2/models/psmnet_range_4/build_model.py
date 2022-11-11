from active_zero2.models.psmnet_range_4.psmnet_3 import PSMNetRange4


def build_model(cfg):
    model = PSMNetRange4(
        min_disp=cfg.PSMNetRange4.MIN_DISP,
        max_disp=cfg.PSMNetRange4.MAX_DISP,
        num_disp=cfg.PSMNetRange4.NUM_DISP,
        set_zero=cfg.PSMNetRange4.SET_ZERO,
    )
    return model
