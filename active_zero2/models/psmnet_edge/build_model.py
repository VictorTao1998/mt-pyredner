from active_zero2.models.psmnet_edge.psmnet_3 import PSMNetEdge


def build_model(cfg):
    model = PSMNetEdge(
        min_disp=cfg.PSMNetEdge.MIN_DISP,
        max_disp=cfg.PSMNetEdge.MAX_DISP,
        num_disp=cfg.PSMNetEdge.NUM_DISP,
        set_zero=cfg.PSMNetEdge.SET_ZERO,
        dilation=cfg.PSMNetEdge.DILATION,
        epsilon=cfg.PSMNetEdge.EPSILON,
        grad_threshold=cfg.PSMNetEdge.GRAD_THRESHOLD,
        use_off=cfg.PSMNetEdge.USE_OFF,
        use_volume=cfg.PSMNetEdge.USE_VOLUME,
        edge_weight=cfg.PSMNetEdge.EDGE_WEIGHT,
    )
    return model
