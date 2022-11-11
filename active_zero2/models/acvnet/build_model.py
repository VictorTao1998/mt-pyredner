from active_zero2.models.acvnet.acv import ACVNet


def build_model(cfg):
    model = ACVNet(
        num_ir=cfg.DATA.NUM_IR,
        min_disp=cfg.ACVNet.MIN_DISP,
        max_disp=cfg.ACVNet.MAX_DISP,
        num_disp=cfg.ACVNet.NUM_DISP,
    )
    return model
