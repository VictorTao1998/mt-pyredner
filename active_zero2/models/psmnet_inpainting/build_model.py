from active_zero2.models.psmnet_inpainting.psmnet_3 import PSMNetInpainting


def build_model(cfg):
    model = PSMNetInpainting(
        min_disp=cfg.PSMNetInpainting.MIN_DISP,
        max_disp=cfg.PSMNetInpainting.MAX_DISP,
        num_disp=cfg.PSMNetInpainting.NUM_DISP,
        set_zero=cfg.PSMNetInpainting.SET_ZERO,
        dilation=cfg.PSMNetInpainting.DILATION,
        epsilon=cfg.PSMNetInpainting.EPSILON,
        grad_threshold=cfg.PSMNetInpainting.GRAD_THRESHOLD,
        sub_avg_size=cfg.PSMNetInpainting.SUB_AVG_SIZE,
        disp_grad_norm=cfg.PSMNetInpainting.DISP_GRAD_NORM,
        use_off=cfg.PSMNetInpainting.USE_OFF,
        use_edge=cfg.PSMNetInpainting.USE_EDGE,
        conf_range=cfg.PSMNetInpainting.CONF_RANGE,
    )
    return model
