from active_zero2.models.psmnet_clear_grasp.psmnet_3 import PSMNetClearGrasp


def build_model(cfg):
    model = PSMNetClearGrasp(
        min_disp=cfg.PSMNetClearGrasp.MIN_DISP,
        max_disp=cfg.PSMNetClearGrasp.MAX_DISP,
        num_disp=cfg.PSMNetClearGrasp.NUM_DISP,
        set_zero=cfg.PSMNetClearGrasp.SET_ZERO,
        dilation=cfg.PSMNetClearGrasp.DILATION,
        epsilon=cfg.PSMNetClearGrasp.EPSILON,
        grad_threshold=cfg.PSMNetClearGrasp.GRAD_THRESHOLD,
        use_off=cfg.PSMNetClearGrasp.USE_OFF,
        edge_weight=cfg.PSMNetClearGrasp.EDGE_WEIGHT,
    )
    return model
