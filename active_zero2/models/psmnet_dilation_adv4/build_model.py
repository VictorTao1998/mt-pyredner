from active_zero2.models.psmnet_dilation_adv4.psmnet_3 import PSMNetADV4


def build_model(cfg):
    model = PSMNetADV4(
        min_disp=cfg.PSMNetADV4.MIN_DISP,
        max_disp=cfg.PSMNetADV4.MAX_DISP,
        num_disp=cfg.PSMNetADV4.NUM_DISP,
        set_zero=cfg.PSMNetADV4.SET_ZERO,
        dilation=cfg.PSMNetADV4.DILATION,
        epsilon=cfg.PSMNetADV4.EPSILON,
        grad_threshold=cfg.PSMNetADV4.GRAD_THRESHOLD,
        d_channels=cfg.PSMNetADV4.D_CHANNELS,
        disp_encoding=cfg.PSMNetADV4.DISP_ENCODING,
        wgangp_norm=cfg.PSMNetADV4.WGANGP_NORM,
        wgangp_lambda=cfg.PSMNetADV4.WGANGP_LAMBDA,
        d_type=cfg.PSMNetADV4.D_TYPE,
        disp_grad_norm=cfg.PSMNetADV4.DISP_GRAD_NORM,
    )
    return model
