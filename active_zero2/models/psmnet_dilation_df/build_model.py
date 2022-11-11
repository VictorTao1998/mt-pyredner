from active_zero2.models.psmnet_dilation_df.psmnet_3 import PSMNetDilationDF


def build_model(cfg):
    model = PSMNetDilationDF(
        min_disp=cfg.PSMNetDilationDF.MIN_DISP,
        max_disp=cfg.PSMNetDilationDF.MAX_DISP,
        num_disp=cfg.PSMNetDilationDF.NUM_DISP,
        set_zero=cfg.PSMNetDilationDF.SET_ZERO,
        dilation=cfg.PSMNetDilationDF.DILATION,
        epsilon=cfg.PSMNetDilationDF.EPSILON,
        grad_threshold=cfg.PSMNetDilationDF.GRAD_THRESHOLD,
        df_channels=cfg.PSMNetDilationDF.DF_CHANNELS,
        use_image=cfg.PSMNetDilationDF.USE_IMAGE,
        use_off=cfg.PSMNetDilationDF.USE_OFF,
        use_edge=cfg.PSMNetDilationDF.USE_EDGE,
        use_full_volume=cfg.PSMNetDilationDF.USE_FULL_VOLUME,
        use_conf_map=cfg.PSMNetDilationDF.USE_CONF_MAP,
        mask_to_zero=cfg.PSMNetDilationDF.MASK_TO_ZERO,
        df_res=cfg.PSMNetDilationDF.DF_RES,
        mix=cfg.PSMNetDilationDF.MIX,
        conf_range=cfg.PSMNetDilationDF.CONF_RANGE,
        sim_disp_weight=cfg.PSMNetDilationDF.SIM_DISP_WEIGHT,
    )
    return model
