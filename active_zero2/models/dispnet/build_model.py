from active_zero2.models.dispnet.dispnet import DispNet


def build_model(cfg):
    model = DispNet(max_disp=cfg.DispNet.MAX_DISP)
    return model
