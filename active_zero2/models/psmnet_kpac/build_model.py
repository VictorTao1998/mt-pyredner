from active_zero2.models.psmnet_kpac.psmnet_3 import PSMNetKPAC


def build_model(cfg):
    model = PSMNetKPAC(
        min_disp=cfg.PSMNetKPAC.MIN_DISP,
        max_disp=cfg.PSMNetKPAC.MAX_DISP,
        num_disp=cfg.PSMNetKPAC.NUM_DISP,
        set_zero=cfg.PSMNetKPAC.SET_ZERO,
        dilation_list=cfg.PSMNetKPAC.DILATION_LIST,
        dilation=cfg.PSMNetKPAC.DILATION,
    )
    return model
