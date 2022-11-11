from active_zero2.models.smdnet.SMDNet import SMDNet


def build_model(cfg):
    model = SMDNet(
        output_representation=cfg.SMDNet.OUTPUT_REPRESENTATION,
        maxdisp=cfg.SMDNet.MAX_DISP,
        no_sine=cfg.SMDNet.NO_SINE,
        no_residual=cfg.SMDNet.NO_RESIDUAL,
    )
    return model
