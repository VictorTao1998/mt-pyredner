#!/usr/bin/env python
import os
import os.path as osp
import sys

import tensorboardX

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import argparse
import gc
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.models.build_model import build_model
from active_zero2.utils.cfg_utils import purge_cfg
from active_zero2.utils.checkpoint import CheckpointerV2
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.metric_logger import MetricLogger
from active_zero2.utils.reduce import set_random_seed, synchronize
from active_zero2.utils.sampler import IterationBasedBatchSampler
from active_zero2.utils.solver import build_d_optimizer, build_g_optimizer, build_lr_scheduler
from active_zero2.utils.torch_utils import worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveZero2")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()
    local_rank = 0

    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()
    config_name = args.config_file.split("/")[-1].split(".")[0]

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)

    # Parse the output directory
    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs", "outputs"))
        if osp.isdir(output_dir):
            warnings.warn("Output directory exists.")
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(
        f"ActiveZero2.train [{config_name}]", output_dir, rank=local_rank, filename=f"log.train.{run_name}.txt"
    )
    logger.info(args)
    from active_zero2.utils.collect_env import collect_env_info

    logger.info("Collecting env info (might take some time)\n" + collect_env_info())
    logger.info(f"Loaded config file: '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")
    logger.info(f"Running with {torch.cuda.device_count()} GPUs")

    # Build tensorboard logger
    summary_writer = SummaryWriter(f"{output_dir}/{run_name}")

    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # Build model
    set_random_seed(cfg.RNG_SEED)
    assert cfg.MODEL_TYPE in ("PSMNetADV", "PSMNetADV4")
    model = build_model(cfg)
    logger.info(f"Model: \n{model}")
    model = model.cuda()

    # Build optimizer
    d_optimizer = build_d_optimizer(cfg, model)
    g_optimizer = build_g_optimizer(cfg, model)
    # Build lr_scheduler
    scheduler = build_lr_scheduler(cfg, g_optimizer)

    model_parallel = torch.nn.DataParallel(model)

    # Build checkpointer
    # Note that checkpointer will only load state_dict of model.
    checkpointer = CheckpointerV2(
        model_parallel,
        optimizer=None,
        scheduler=None,
        save_dir=output_dir,
        logger=logger,
        max_to_keep=cfg.TRAIN.MAX_TO_KEEP,
        local_rank=0,
    )
    checkpoint_data = checkpointer.load(
        cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES, strict=cfg.RESUME_STRICT
    )
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    start_iter = checkpoint_data.get("iteration", 0)

    # Build dataloader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_sim_dataset = build_dataset(cfg, mode="train", domain="sim")
    train_real_dataset = build_dataset(cfg, mode="train", domain="real")
    assert (
        train_sim_dataset and train_real_dataset
    ), "Sim and Real dataset should both be valid for adversarial learning."
    val_sim_dataset = build_dataset(cfg, mode="val", domain="sim")
    val_real_dataset = build_dataset(cfg, mode="val", domain="real")
    if train_sim_dataset:
        sampler = RandomSampler(train_sim_dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER, start_iter=start_iter
        )
        train_sim_loader = iter(
            DataLoader(
                train_sim_dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                worker_init_fn=lambda worker_id: worker_init_fn(
                    worker_id, base_seed=cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None
                ),
            )
        )
    else:
        train_sim_loader = None
    if train_real_dataset:
        sampler = RandomSampler(train_real_dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER, start_iter=start_iter
        )
        train_real_loader = iter(
            DataLoader(
                train_real_dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                worker_init_fn=lambda worker_id: worker_init_fn(
                    worker_id, base_seed=cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None
                ),
            )
        )
    else:
        train_real_loader = None

    if val_sim_dataset:
        val_sim_loader = DataLoader(
            val_sim_dataset,
            batch_size=cfg.VAL.BATCH_SIZE,
            num_workers=cfg.VAL.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

    else:
        val_sim_loader = None
    if val_real_dataset:
        val_real_loader = DataLoader(
            val_real_dataset,
            batch_size=cfg.VAL.BATCH_SIZE,
            num_workers=cfg.VAL.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
    else:
        val_real_loader = None

    # ---------------------------------------------------------------------------- #
    # Setup validation
    # ---------------------------------------------------------------------------- #
    val_period = cfg.VAL.PERIOD
    do_validation = val_period > 0
    assert not do_validation, "validation not implemented yet"
    if do_validation:
        best_metric_name = "best_{}".format(cfg.VAL.METRIC)
        best_metric = checkpoint_data.get(best_metric_name, None)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    train_meters = MetricLogger()
    val_meters = MetricLogger()

    def setup_train():
        model.train()
        train_meters.reset()

    def setup_val():
        model.eval()
        val_meters.reset()

    setup_train()
    max_iter = cfg.TRAIN.MAX_ITER
    logger.info("Start training from iteration {}".format(start_iter))
    tic = time.time()
    for iteration in range(start_iter, max_iter):
        cur_iter = iteration + 1
        loss_dict = {}
        time_dict = {}
        # train D...
        for p in model.psmnet.parameters():
            p.requires_grad = False
        for p in model.D.parameters():
            p.requires_grad = True
        model.D.zero_grad()
        loss_D = 0
        # sim data
        sim_data_batch = next(train_sim_loader)
        sim_data_time = time.time() - tic
        time_dict["time_data_sim"] = sim_data_time
        # Copy data from cpu to gpu
        sim_data_batch = {
            k: v.cuda(non_blocking=True) for k, v in sim_data_batch.items() if isinstance(v, torch.Tensor)
        }
        # Forward
        sim_pred_dict = model_parallel(sim_data_batch)
        sim_D_dict = model.D_backward(sim_data_batch, sim_pred_dict)
        for k, v in sim_D_dict.items():
            if "err_" in k:
                loss_dict["sim_" + k] = v
                loss_D += v

        # real data
        real_tic = time.time()
        real_data_batch = next(train_real_loader)
        real_data_time = time.time() - real_tic
        time_dict["time_data_real"] = real_data_time
        # Copy data from cpu to gpu
        real_data_batch = {
            k: v.cuda(non_blocking=True) for k, v in real_data_batch.items() if isinstance(v, torch.Tensor)
        }
        if "img_disp_l" not in real_data_batch:
            if cfg.MODEL_TYPE == "PSMNetADV":
                if cfg.PSMNetADV.USE_SIM_PRED:
                    real_data_batch["gt_prob_volume"] = sim_D_dict["pred_prob_volume"]
                else:
                    real_data_batch["gt_prob_volume"] = sim_D_dict["gt_prob_volume"]
            if cfg.MODEL_TYPE == "PSMNetADV4":
                if cfg.PSMNetADV4.USE_SIM_PRED:
                    real_data_batch["gt_prob_volume"] = sim_D_dict["pred_prob_volume"]
                else:
                    real_data_batch["gt_prob_volume"] = sim_D_dict["gt_prob_volume"]
        real_pred_dict = model_parallel(real_data_batch)
        real_D_dict = model.D_backward(real_data_batch, real_pred_dict)
        for k, v in real_D_dict.items():
            if "err_" in k:
                loss_dict["real_" + k] = v
                loss_D += v
        loss_dict["loss_D_total"] = loss_D
        d_optimizer.step()
        time_dict["time_D"] = time.time() - tic

        # train G...
        G_tic = time.time()
        for p in model.psmnet.parameters():
            p.requires_grad = True
        for p in model.D.parameters():
            p.requires_grad = False
        model.psmnet.zero_grad()
        # sim data
        loss = 0
        # Forward
        sim_pred_dict = model_parallel(sim_data_batch)

        if cfg.LOSS.SIM_REPROJ.WEIGHT > 0:
            sim_reproj = model.compute_reproj_loss(
                sim_data_batch,
                sim_pred_dict,
                use_mask=cfg.LOSS.SIM_REPROJ.USE_MASK,
                patch_size=cfg.LOSS.SIM_REPROJ.PATCH_SIZE,
                only_last_pred=cfg.LOSS.SIM_REPROJ.ONLY_LAST_PRED,
            )
            sim_reproj *= cfg.LOSS.SIM_REPROJ.WEIGHT
            loss += sim_reproj
            loss_dict["loss_sim_reproj"] = sim_reproj
        if cfg.LOSS.SIM_DISP.WEIGHT > 0:
            sim_disp = model.compute_disp_loss(sim_data_batch, sim_pred_dict)
            sim_disp *= cfg.LOSS.SIM_DISP.WEIGHT
            loss += sim_disp
            loss_dict["loss_sim_disp"] = sim_disp
        if cfg.LOSS.SIM_GRAD > 0:
            grad_loss = model.compute_grad_loss(sim_data_batch, sim_pred_dict)
            grad_loss *= cfg.LOSS.SIM_GRAD
            loss += grad_loss
            loss_dict["loss_sim_grad"] = grad_loss
        loss_dict["loss_sim_total"] = loss
        loss.backward()

        # real data
        loss = 0
        # Forward
        real_pred_dict = model_parallel(real_data_batch)

        if cfg.LOSS.REAL_REPROJ.WEIGHT > 0:
            real_reproj = model.compute_reproj_loss(
                real_data_batch,
                real_pred_dict,
                use_mask=cfg.LOSS.REAL_REPROJ.USE_MASK,
                patch_size=cfg.LOSS.REAL_REPROJ.PATCH_SIZE,
                only_last_pred=cfg.LOSS.REAL_REPROJ.ONLY_LAST_PRED,
            )
            real_reproj *= cfg.LOSS.REAL_REPROJ.WEIGHT
            loss += real_reproj
            loss_dict["loss_real_reproj"] = real_reproj
        if cfg.LOSS.REAL_DISP.WEIGHT > 0:
            real_disp = model.compute_disp_loss(real_data_batch, real_pred_dict)
            real_disp *= cfg.LOSS.REAL_DISP.WEIGHT
            loss += real_disp
            loss_dict["loss_real_disp"] = real_disp

        if cfg.LOSS.REAL_GRAD > 0:
            grad_loss = model.compute_grad_loss(real_data_batch, real_pred_dict)
            grad_loss *= cfg.LOSS.REAL_GRAD
            loss += grad_loss
            loss_dict["loss_real_grad"] = grad_loss

        loss.backward()
        # only compute adversarial loss for real data
        if cur_iter > cfg.ADV_ITER:
            real_pred_dict = model_parallel(real_data_batch)
            adv_loss = model.G_backward(real_data_batch, real_pred_dict, cfg.LOSS.ADV)
            loss_dict["loss_real_adv"] = adv_loss
            loss += adv_loss

        loss_dict["loss_real_total"] = loss

        g_optimizer.step()
        time_dict["time_G"] = time.time() - G_tic
        with torch.no_grad():
            train_meters.update(**loss_dict)

        time_dict["time_batch"] = time.time() - tic
        train_meters.update(**time_dict)

        # Logging
        log_period = cfg.TRAIN.LOG_PERIOD
        if log_period > 0 and (cur_iter % log_period == 0 or cur_iter == 1):
            logger.info(
                train_meters.delimiter.join(
                    [
                        "iter: {iter:6d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_meters),
                    lr=g_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0**2),
                )
            )
            keywords = (
                "loss",
                "acc",
                "heading",
            )
            for name, metric in train_meters.metrics.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar("train/" + name, metric.result, global_step=cur_iter)
            summary_writer.add_scalar("train/learning_rate", g_optimizer.param_groups[0]["lr"], global_step=cur_iter)

        # ---------------------------------------------------------------------------- #
        # Validate one epoch
        # ---------------------------------------------------------------------------- #
        if do_validation and (cur_iter % val_period == 0 or cur_iter == max_iter):
            gc.collect()
            setup_val()
            logger.info("Validation begins at iteration {}.".format(cur_iter))

            start_time_val = time.time()
            tic = time.time()
            # sim data
            if val_sim_loader:
                for iteration_val, data_batch in enumerate(val_sim_loader, 1):
                    data_time = time.time() - tic
                    loss = 0
                    loss_dict = {}

                    # copy data from cpu to gpu
                    data_batch = {
                        k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                    }

                    # Forward
                    with torch.no_grad():
                        # Forward
                        pred_dict = model_parallel(data_batch)

                        if cfg.LOSS.SIM_REPROJ.WEIGHT > 0:
                            sim_reproj = model.compute_reproj_loss(
                                data_batch,
                                pred_dict,
                                use_mask=cfg.LOSS.SIM_REPROJ.USE_MASK,
                                patch_size=cfg.LOSS.SIM_REPROJ.PATCH_SIZE,
                                only_last_pred=cfg.LOSS.SIM_REPROJ.ONLY_LAST_PRED,
                            )
                            sim_reproj *= cfg.LOSS.SIM_REPROJ.WEIGHT
                            loss += sim_reproj
                            loss_dict["loss_sim_reproj"] = sim_reproj
                        if cfg.LOSS.SIM_DISP.WEIGHT > 0:
                            sim_disp = model.compute_disp_loss(data_batch, pred_dict)
                            sim_disp *= cfg.LOSS.SIM_DISP.WEIGHT
                            loss += sim_disp
                            loss_dict["loss_sim_disp"] = sim_disp

                        if cfg.MODEL_TYPE == "PSMNetGrad":
                            if cfg.PSMNetGrad.LOSS_WEIGHT > 0:
                                grad_loss = model.compute_grad_loss(data_batch, pred_dict)
                                grad_loss *= cfg.PSMNetGrad.LOSS_WEIGHT
                                loss += grad_loss
                                loss_dict["loss_sim_grad"] = grad_loss

                    batch_time = time.time() - tic
                    val_meters.update(time=batch_time, data=data_time)
                    val_meters.update(**loss_dict)

                    # Logging
                    if cfg.VAL.LOG_PERIOD > 0 and iteration_val % cfg.VAL.LOG_PERIOD == 0:
                        logger.info(
                            "Sim Val "
                            + val_meters.delimiter.join(
                                [
                                    "iter: {iter:6d}",
                                    "{meters}",
                                    "max mem: {memory:.0f}",
                                ]
                            ).format(
                                iter=iteration_val,
                                meters=str(val_meters),
                                memory=torch.cuda.max_memory_allocated() / (1024.0**2),
                            )
                        )

                    tic = time.time()
            if val_real_loader:
                for iteration_val, data_batch in enumerate(val_real_loader, 1):
                    data_time = time.time() - tic
                    loss = 0
                    loss_dict = {}

                    # copy data from cpu to gpu
                    data_batch = {
                        k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                    }

                    # Forward
                    with torch.no_grad():
                        pred_dict = model_parallel(data_batch)

                        if cfg.LOSS.REAL_REPROJ.WEIGHT > 0:
                            real_reproj = model.compute_reproj_loss(
                                data_batch,
                                pred_dict,
                                use_mask=cfg.LOSS.REAL_REPROJ.USE_MASK,
                                patch_size=cfg.LOSS.REAL_REPROJ.PATCH_SIZE,
                                only_last_pred=cfg.LOSS.REAL_REPROJ.ONLY_LAST_PRED,
                            )
                            real_reproj *= cfg.LOSS.REAL_REPROJ.WEIGHT
                            loss += real_reproj
                            loss_dict["loss_real_reproj"] = real_reproj
                        if cfg.LOSS.REAL_DISP.WEIGHT > 0:
                            real_disp = model.compute_disp_loss(data_batch, pred_dict)
                            real_disp *= cfg.LOSS.REAL_DISP.WEIGHT
                            loss += real_disp
                            loss_dict["loss_real_disp"] = real_disp
                        if cfg.MODEL_TYPE == "PSMNetGrad":
                            if cfg.PSMNetGrad.LOSS_WEIGHT > 0:
                                grad_loss = model.compute_grad_loss(data_batch, pred_dict)
                                grad_loss *= cfg.PSMNetGrad.LOSS_WEIGHT
                                loss += grad_loss
                                loss_dict["loss_real_grad"] = grad_loss

                    batch_time = time.time() - tic
                    val_meters.update(time=batch_time, data=data_time)
                    val_meters.update(**loss_dict)

                    # Logging
                    if cfg.VAL.LOG_PERIOD > 0 and iteration_val % cfg.VAL.LOG_PERIOD == 0:
                        logger.info(
                            "Real Val "
                            + val_meters.delimiter.join(
                                [
                                    "iter: {iter:6d}",
                                    "{meters}",
                                    "max mem: {memory:.0f}",
                                ]
                            ).format(
                                iter=iteration_val,
                                meters=str(val_meters),
                                memory=torch.cuda.max_memory_allocated() / (1024.0**2),
                            )
                        )

                    tic = time.time()

            # END: validation loop
            epoch_time_val = time.time() - start_time_val
            logger.info(
                "Iteration[{}]-Val {}  total_time: {:.2f}s".format(cur_iter, val_meters.summary_str, epoch_time_val)
            )
            keywords = (
                "loss",
                "acc",
                "heading",
            )
            for name, metric in val_meters.metrics.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar("val/" + name, metric.result, global_step=cur_iter)
            # best validation
            if cfg.VAL.METRIC in val_meters.metrics:
                cur_metric = val_meters.metrics[cfg.VAL.METRIC].result
                if (
                    best_metric is None
                    or (cfg.VAL.METRIC_ASCEND and cur_metric > best_metric)
                    or (not cfg.VAL.METRIC_ASCEND and cur_metric < best_metric)
                ):
                    best_metric = cur_metric
                    checkpoint_data["iteration"] = cur_iter
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save("model_best", tag=False, **checkpoint_data)

            # restore training
            setup_train()
            torch.cuda.empty_cache()
            gc.collect()

        # ---------------------------------------------------------------------------- #
        # After validation
        # ---------------------------------------------------------------------------- #
        # Checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iter:
            checkpoint_data["iteration"] = cur_iter
            if do_validation and best_metric is not None:
                checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:06d}".format(cur_iter), **checkpoint_data)

        # ---------------------------------------------------------------------------- #
        # Finalize one step
        # ---------------------------------------------------------------------------- #
        # Since pytorch v1.1.0, lr_scheduler is called after optimization.
        if scheduler is not None:
            scheduler.step()
        tic = time.time()

    # END: training loop
    if do_validation and cfg.VAL.METRIC:
        logger.info("Best val-{} = {}".format(cfg.VAL.METRIC, best_metric))
