from active_zero2.models.cfnet.build_model import build_model as build_cfnet
from active_zero2.models.psmnet.build_model import build_model as build_psmnet
from active_zero2.models.psmnet_dilation.build_model import build_model as build_psmnetdilation
from active_zero2.models.psmnet_dilation_adv4.build_model import build_model as build_psmnetadv4
from active_zero2.models.psmnet_dilation_adv.build_model import build_model as build_psmnetadv
from active_zero2.models.psmnet_dilation_df.build_model import build_model as build_psmnetdf
from active_zero2.models.psmnet_grad.build_model import build_model as build_psmnetgrad
from active_zero2.models.psmnet_grad_2dadv.build_model import build_model as build_psmnetgrad2dadv
from active_zero2.models.psmnet_inpainting.build_model import build_model as build_psmnetinpaint
from active_zero2.models.psmnet_kpac.build_model import build_model as build_psmnetkpac
from active_zero2.models.psmnet_range.build_model import build_model as build_psmnetrange
from active_zero2.models.psmnet_range_4.build_model import build_model as build_psmnetrange4
from active_zero2.models.smdnet.build_model import build_model as build_smdnet
from active_zero2.models.psmnet_edge.build_model import build_model as build_psmnetedge
from active_zero2.models.psmnet_edge_norm.build_model import build_model as build_psmnetedgenorm
from active_zero2.models.dispnet.build_model import build_model as build_dispnet
from active_zero2.models.raft.build_model import build_model as build_raftstereo
from active_zero2.models.psmnet_clear_grasp.build_model import build_model as build_psmnetclear
from active_zero2.models.acvnet.build_model import build_model as build_acvnet

MODEL_LIST = (
    "PSMNet",
    "CFNet",
    "SMDNet",
    "PSMNetRange",
    "PSMNetRange4",
    "PSMNetDilation",
    "PSMNetKPAC",
    "PSMNetGrad",
    "PSMNetADV",
    "PSMNetADV4",
    "PSMNetGrad2DADV",
    "PSMNetDilationDF",
    "PSMNetInpainting",
    "PSMNetEdge",
    "PSMNetEdgeNormal",
    "DispNet",
    "RaftStereo",
    "PSMNetClearGrasp",
    "ACVNet",
)


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNet":
        model = build_psmnet(cfg)
    elif cfg.MODEL_TYPE == "CFNet":
        model = build_cfnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetRange":
        model = build_psmnetrange(cfg)
    elif cfg.MODEL_TYPE == "PSMNetRange4":
        model = build_psmnetrange4(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilation":
        model = build_psmnetdilation(cfg)
    elif cfg.MODEL_TYPE == "SMDNet":
        model = build_smdnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilation":
        model = build_psmnetdilation(cfg)
    elif cfg.MODEL_TYPE == "PSMNetKPAC":
        model = build_psmnetkpac(cfg)
    elif cfg.MODEL_TYPE == "PSMNetGrad":
        model = build_psmnetgrad(cfg)
    elif cfg.MODEL_TYPE == "PSMNetADV":
        model = build_psmnetadv(cfg)
    elif cfg.MODEL_TYPE == "PSMNetADV4":
        model = build_psmnetadv4(cfg)
    elif cfg.MODEL_TYPE == "PSMNetGrad2DADV":
        model = build_psmnetgrad2dadv(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilationDF":
        model = build_psmnetdf(cfg)
    elif cfg.MODEL_TYPE == "PSMNetInpainting":
        model = build_psmnetinpaint(cfg)
    elif cfg.MODEL_TYPE == "PSMNetEdge":
        model = build_psmnetedge(cfg)
    elif cfg.MODEL_TYPE == "PSMNetEdgeNormal":
        model = build_psmnetedgenorm(cfg)
    elif cfg.MODEL_TYPE == "DispNet":
        model = build_dispnet(cfg)
    elif cfg.MODEL_TYPE == "RaftStereo":
        model = build_raftstereo(cfg)
    elif cfg.MODEL_TYPE == "PSMNetClearGrasp":
        model = build_psmnetclear(cfg)
    elif cfg.MODEL_TYPE == "ACVNet":
        model = build_acvnet(cfg)
    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
