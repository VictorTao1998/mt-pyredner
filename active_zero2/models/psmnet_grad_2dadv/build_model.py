from active_zero2.models.psmnet_grad_2dadv.psmnet_3 import PSMNetGrad2DADV


def build_model(cfg):
    model = PSMNetGrad2DADV(
        min_disp=cfg.PSMNetGrad2DADV.MIN_DISP,
        max_disp=cfg.PSMNetGrad2DADV.MAX_DISP,
        num_disp=cfg.PSMNetGrad2DADV.NUM_DISP,
        set_zero=cfg.PSMNetGrad2DADV.SET_ZERO,
        dilation=cfg.PSMNetGrad2DADV.DILATION,
        epsilon=cfg.PSMNetGrad2DADV.EPSILON,
        grad_threshold=cfg.PSMNetGrad2DADV.GRAD_THRESHOLD,
        d_channels=cfg.PSMNetGrad2DADV.D_CHANNELS,
        wgangp_norm=cfg.PSMNetGrad2DADV.WGANGP_NORM,
        wgangp_lambda=cfg.PSMNetGrad2DADV.WGANGP_LAMBDA,
        sub_avg_size=cfg.PSMNetGrad2DADV.SUB_AVG_SIZE,
        disp_grad_norm=cfg.PSMNetGrad2DADV.DISP_GRAD_NORM,
    )
    return model
