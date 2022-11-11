from active_zero2.models.psmnet_grad.psmnet_3 import PSMNetGrad


def build_model(cfg):
    model = PSMNetGrad(
        min_disp=cfg.PSMNetGrad.MIN_DISP,
        max_disp=cfg.PSMNetGrad.MAX_DISP,
        num_disp=cfg.PSMNetGrad.NUM_DISP,
        set_zero=cfg.PSMNetGrad.SET_ZERO,
        dilation=cfg.PSMNetGrad.DILATION,
        epsilon=cfg.PSMNetGrad.EPSILON,
        use_off=cfg.PSMNetGrad.USE_OFF,
        grad_threshold=cfg.PSMNetGrad.GRAD_THRESHOLD,
        disp_grad_norm=cfg.PSMNetGrad.DISP_GRAD_NORM,
    )
    return model
