from active_zero2.models.raft.raft_stereo import RAFTStereo


def build_model(cfg):
    model = RAFTStereo(max_disp=cfg.RaftStereo.MAX_DISP)
    return model
