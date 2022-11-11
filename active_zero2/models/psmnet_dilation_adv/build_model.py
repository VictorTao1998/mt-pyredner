from active_zero2.models.psmnet_dilation_adv.psmnet_3 import PSMNetADV


def build_model(cfg):
    model = PSMNetADV(
        min_disp=cfg.PSMNetADV.MIN_DISP,
        max_disp=cfg.PSMNetADV.MAX_DISP,
        num_disp=cfg.PSMNetADV.NUM_DISP,
        set_zero=cfg.PSMNetADV.SET_ZERO,
        dilation=cfg.PSMNetADV.DILATION,
        epsilon=cfg.PSMNetADV.EPSILON,
        d_channels=cfg.PSMNetADV.D_CHANNELS,
        disp_encoding=cfg.PSMNetADV.DISP_ENCODING,
        wgangp_norm=cfg.PSMNetADV.WGANGP_NORM,
        wgangp_lambda=cfg.PSMNetADV.WGANGP_LAMBDA,
    )
    return model
