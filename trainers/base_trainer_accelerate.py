import argparse
import datetime
import itertools
import os
import random
import traceback
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import shutil
from utils.basic import seed_anything, count_parameters

from datasets import create_dataloader
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from transformers.trainer_pt_utils import get_model_param_count
import numpy as np
from torchvision.transforms.functional import to_tensor as pil_to_tensor

try:  # pragma: no cover - exercised in the real training environment
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs
    from accelerate import DistributedType
    from accelerate.utils import (
        DataLoaderConfiguration,
        DynamoBackend,
        GradientAccumulationPlugin,
        ProjectConfiguration,
        TorchDynamoPlugin,
        set_seed,
    )
except ModuleNotFoundError:  # pragma: no cover - used in the test environment
    from contextlib import contextmanager

    class _NoOpState:
        deepspeed_plugin = None

    class Accelerator:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.num_processes = 1
            self.is_main_process = True
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
            self.sync_gradients = True
            self.device = torch.device("cpu")
            self.state = _NoOpState()
            self.trackers = []

        def prepare(self, *objects):
            return objects[0] if len(objects) == 1 else objects

        def wait_for_everyone(self):
            return None

        def init_trackers(self, *args, **kwargs):
            return None

        def save_state(self, *args, **kwargs):
            return None

        def load_state(self, *args, **kwargs):
            return None

        def backward(self, loss):
            loss.backward()

        def gather(self, tensor):
            return tensor

        def clip_grad_norm_(self, parameters, max_norm):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        def log(self, *args, **kwargs):
            return None

        @contextmanager
        def accumulate(self, model):
            del model
            yield

        @contextmanager
        def autocast(self):
            yield

        def end_training(self):
            return None

    class DistributedDataParallelKwargs:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    class DistributedType:  # type: ignore[override]
        NO = "NO"

    class DataLoaderConfiguration:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    class DynamoBackend:  # type: ignore[override]
        NO = "NO"

    class GradientAccumulationPlugin:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    class ProjectConfiguration:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    class TorchDynamoPlugin:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

    def set_seed(seed, device_specific=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging

        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)              # deterministic=True for reproduction

        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        if bool(getattr(cfg.test, "use_train_loader", False)):
            self.log_info("Using train dataloader for validation.")
            self.test_loader = self.train_loader
        else:
            self.log_info("Making test dataloader...")
            self.test_loader = create_dataloader(cfg, 'test')
        self.accelerator.wait_for_everyone()

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self.cfg.get("fsdp_plugin"):
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        self.iters_per_test = self.cfg.test.iters_per_test if self.cfg.test.iters_per_test > 0 else len(self.test_loader)
        self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")

        # (
        #     self.model,
        #     self.train_loader,
        #     self.test_loader,
        #     self.optimizer,
        #     self.lr_scheduler,
        # ) = self.accelerator.prepare(
        #     self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
        # )

        # don't wrap dataloader
        (
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))
        self.tb_writer = None
        if self.cfg.log.use_tensorboard and self.accelerator.is_main_process:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError:
                from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.cfg.log.output_dir)

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.first_epoch = latest_epoch
        self._pending_resume_path = getattr(self, "resume_path", None)

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

        if getattr(self, "_pending_resume_path", None) is not None:
            self.load_training_state(self._pending_resume_path)
            self.log_info(f"Loaded training state from {self._pending_resume_path}")
            self._pending_resume_path = None

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        return model
    
    def before_epoch(self, epoch):
        pass

    def _cleanup_checkpoints(self, keep_paths):
        ckpt_dir = self.cfg.log.ckpt_dir
        if not os.path.isdir(ckpt_dir):
            return
        keep_set = {os.path.normpath(p) for p in keep_paths if os.path.exists(p)}
        for entry in os.listdir(ckpt_dir):
            full = os.path.join(ckpt_dir, entry)
            if os.path.normpath(full) in keep_set:
                continue
            if entry.startswith("checkpoint-"):
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
                self.log_info(f"Removed old checkpoint: {entry}")

    def _save_ckpt(self, tag, epoch):
        save_path = os.path.join(self.cfg.log.ckpt_dir, f"checkpoint-{tag}")
        self.save_training_state(save_path, epoch)
        self.log_info(f"Saved checkpoint: checkpoint-{tag} (epoch {epoch}, step {self.global_step})")
        return save_path

    def train(self):
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        best_val_metrics = []  # list of (loss, path), sorted best first
        latest_path = None

        if self.first_epoch < self.cfg.train.num_epoch:
            self.validate(-1)
            self.accelerator.wait_for_everyone()

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)

            val_stats = self.validate(epoch)

            current_val_loss = val_stats.get("loss", float('inf'))

            # Epoch-end checkpoint
            if self.accelerator.is_main_process:
                epoch_path = self._save_ckpt(f"epoch-{epoch:04d}", epoch)
                latest_path = epoch_path

                # Track best-3 by val loss
                best_val_metrics.append((current_val_loss, epoch_path))
                best_val_metrics.sort(key=lambda x: x[0])
                if len(best_val_metrics) > 3:
                    _, old_path = best_val_metrics.pop()
                    if os.path.exists(old_path) and old_path != latest_path:
                        if os.path.isdir(old_path):
                            shutil.rmtree(old_path)
                        else:
                            os.remove(old_path)

                # Cleanup: keep best 3 + latest 1
                keep = [p for _, p in best_val_metrics]
                if latest_path and latest_path not in keep:
                    keep.append(latest_path)
                self._cleanup_checkpoints(keep)

                self.log_info(
                    f"Epoch {epoch} | val_loss={current_val_loss:.4f} | "
                    f"best_losses={[f'{l:.4f}' for l, _ in best_val_metrics]}"
                )

            self.accelerator.wait_for_everyone()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.close()

    def validate(self, epoch):
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        weighted_metric_logger = MetricLogger(delimiter="  ")

        val_loss = 0.0
        total_samples = 0

        self.log_info(f"Start validation for epoch {epoch}")
        disable_pbar = not self.accelerator.is_main_process
        total_iters = self.iters_per_test if self.iters_per_test > 0 else len(self.test_loader)
        pbar = tqdm(
            total=total_iters,
            disable=disable_pbar,
            desc=f"Val   {epoch:>3d}",
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        test_iter = self.test_loader
        if len(self.test_loader) < total_iters:
            test_iter = itertools.cycle(self.test_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iter):
                if batch_idx >= self.iters_per_test:
                    break
                batch = move_to_device(batch, self.accelerator.device)

                # Forward pass
                forward_outputs = self.forward_batch(batch, mode='test')
                outputs = self.calculate_loss(forward_outputs, batch, mode='test')
                loss = outputs.loss

                self.maybe_export_validation_sample(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    batch=batch,
                    forward_outputs=forward_outputs,
                    loss_outputs=outputs,
                    mode="test",
                )

                # Gather statistics
                loss_value = loss.item()
                batch_size = batch[0]["img"].shape[0] if isinstance(batch, list) and batch and isinstance(batch[0], dict) else len(batch)
                val_loss += loss_value * batch_size
                total_samples += batch_size

                val_outputs = dict(outputs)
                weighted = val_outputs.pop('_weighted_loss_details', None)
                for key, value in val_outputs.items():
                    if value is None:
                        continue
                    if torch.is_tensor(value):
                        value = value.item()
                    metric_logger.meters[key].update(value, n=batch_size)
                if weighted is not None:
                    for key, value in weighted.items():
                        if value is None:
                            continue
                        if torch.is_tensor(value):
                            value = value.item()
                        weighted_metric_logger.meters[key].update(value, n=batch_size)
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})
                pbar.update(1)

        pbar.close()

        # Average the validation loss
        if total_samples > 0:
            val_loss /= total_samples

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        weighted_metric_logger.synchronize_between_processes()
        self.log_info(
            "Validation results: "
            + self._format_metric_logger_with_weighted(
                metric_logger, weighted_metric_logger
            )
        )

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats.update(
            {
                f"{k}_w": meter.global_avg
                for k, meter in weighted_metric_logger.meters.items()
            }
        )
        return stats

    def _format_metric_logger_with_weighted(
        self,
        metric_logger,
        weighted_metric_logger,
    ):
        formatted = []
        for name, meter in metric_logger.meters.items():
            weighted_meter = weighted_metric_logger.meters.get(name)
            weighted_global_avg = (
                weighted_meter.global_avg
                if weighted_meter is not None
                else meter.global_avg
            )
            formatted.append(
                "{}: {}".format(
                    name,
                    meter.fmt.format(
                        median=meter.median,
                        avg=meter.avg,
                        global_avg=weighted_global_avg,
                        max=meter.max,
                        value=meter.value,
                    ),
                )
            )
        return metric_logger.delimiter.join(formatted)

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        loss_details_dict = {}
        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps
        summary_interval = int(getattr(self.cfg.log, "summary_interval", 10))
        summary_interval = max(1, summary_interval)

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        disable_pbar = not self.accelerator.is_main_process
        pbar = tqdm(
            total=self.iters_per_epoch,
            disable=disable_pbar,
            desc=f"Epoch {epoch:>3d}",
            unit="step",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        train_iter = self.train_loader
        if len(self.train_loader) < self.iters_per_epoch:
            train_iter = itertools.cycle(self.train_loader)

        for it, batch in enumerate(train_iter):
            if it >= self.iters_per_epoch:
                break
            with self.accelerator.accumulate(self.model):
                # Perform the forward using the accerlate
                batch = move_to_device(batch, device=self.accelerator.device)
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train')
                batch_output = self.calculate_loss(forward_output, batch, mode='train')
                loss = batch_output.loss
                if loss > self.cfg.train.clip_loss:
                    loss = loss * 0.0

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    rank = get_rank()
                    print(
                        f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                        force=True,
                    )
                    sys.exit(1)

                self.accelerator.backward(loss)

                for item in batch_output:
                    if item == '_weighted_loss_details':
                        continue
                    if 'loss' in item:
                        batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                        if item in loss_details_dict:
                            loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                        else:
                            loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0

                # clip the gradient
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.train.clip_grad
                    )

                    def get_gradient_norm(parameters):
                        norm = 0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        norm = norm**0.5
                        return norm

                    grad_norm = get_gradient_norm(self.model.parameters())

                if self.accelerator.state.deepspeed_plugin is None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()

                if self.accelerator.sync_gradients:
                    start_steps += 1

                    # Report to tensorboard
                    batch_output.update(loss_details_dict)
                    loss_details_dict = {}

                    def _metric_scalar(value):
                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                return value.detach().item()
                            return value.detach().float().mean().item()
                        if np.isscalar(value):
                            return value
                        return None

                    metric_batch_output = {
                        key: scalar
                        for key, value in batch_output.items()
                        if (scalar := _metric_scalar(value)) is not None
                    }

                    if start_steps % summary_interval == 0:
                        self.log_all(batch_output, start_steps, prefix='train')
                    metric_logger.update(**metric_batch_output)

                    min_lr = 10.0
                    max_lr = 0.0
                    for group in self.optimizer.param_groups:
                        min_lr = min(min_lr, group["lr"])
                        max_lr = max(max_lr, group["lr"])

                    metric_logger.update(lr=max_lr)
                    metric_logger.update(min_lr=min_lr)
                    self.log_scalars({"lr": max_lr, "min_lr": min_lr}, step=start_steps)

                    weight_decay_value = None
                    for group in self.optimizer.param_groups:
                        if group["weight_decay"] > 0:
                            weight_decay_value = group["weight_decay"]
                    metric_logger.update(weight_decay=weight_decay_value)
                    metric_logger.update(grad_norm=grad_norm)
                    self.log_scalars({"weight_decay": weight_decay_value, "grad_norm": grad_norm}, step=start_steps)

                    self.global_step = start_steps

                    pbar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        lr=f"{max_lr:.2e}",
                    )
                    pbar.update(1)

                    # Step-based checkpoint
                    ckpt_interval = int(getattr(self.cfg.log, "ckpt_interval", 1000))
                    if ckpt_interval > 0 and start_steps % ckpt_interval == 0 and self.accelerator.is_main_process:
                        self._save_ckpt(f"step-{start_steps:07d}", epoch)

                    # Step-based visualization
                    vis_interval = int(getattr(self.cfg, "vis", {}).get("interval", 1000) if hasattr(self.cfg, "vis") else 1000)
                    if vis_interval > 0 and start_steps % vis_interval == 0 and start_steps > 0:
                        batch_vis = batch
                        fwd_vis = forward_output
                        self.maybe_export_validation_sample(
                            epoch=epoch,
                            batch_idx=it,
                            batch=batch_vis,
                            forward_outputs=fwd_vis,
                            loss_outputs=batch_output,
                            mode="train",
                            global_step=start_steps,
                        )

                del forward_output, batch_output, loss

        pbar.close()
        avg_loss = metric_logger.meters.get("loss")
        if avg_loss is not None:
            self.log_info(f"Epoch {epoch:>3d} | loss={avg_loss.global_avg:.4f}")

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def log_all(self, output, step, prefix=""):        
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        log_scaler = {}
        log_img = {}
        for k in log_keys:
            v = output[k]
            if np.isscalar(v):
                log_scaler[prefix+'/'+k] = v
                continue
            if isinstance(v, Image.Image):
                log_img[prefix+'/'+k] = v

        tensorboard_only = (
            getattr(getattr(self, "cfg", None), "log", None) is not None
            and getattr(self.cfg.log, "use_tensorboard", False)
            and not getattr(self.cfg.log, "use_wandb", False)
        )
        accelerator_scalars = {
            key: value
            for key, value in log_scaler.items()
            if not (tensorboard_only and key.endswith("_w"))
        }

        self.accelerator.log(accelerator_scalars, step)
        if getattr(self, "tb_writer", None) is not None:
            for k, v in accelerator_scalars.items():
                self.tb_writer.add_scalar(k, v, global_step=step)
        log_img_batches = {}
        for tracker in self.accelerator.trackers:
            if log_img:
                for k, v in log_img.items():
                    log_img_batches[k] = pil_to_tensor(v).unsqueeze(0)
                tracker.log_images(log_img_batches, step)
        if getattr(self, "tb_writer", None) is not None:
            for k, v in log_img.items():
                self.tb_writer.add_image(k, pil_to_tensor(v), global_step=step)
            self.tb_writer.flush()

    def log_scalars(self, values, step):
        self.accelerator.log(values, step)
        if getattr(self, "tb_writer", None) is not None:
            for k, v in values.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, global_step=step)
            self.tb_writer.flush()

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def maybe_export_validation_sample(self, **kwargs):
        return None

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = 'all'

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        dynamo_plugin = TorchDynamoPlugin(
            backend=dynamo_backend,
            mode="max-autotune-no-cudagraphs",
            dynamic=self.cfg.train.get("dynamic_compile", True),
        )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True,
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            dynamo_plugin=dynamo_plugin,
        )

        # fsdp
        if self.cfg.get("fsdp_plugin"):
            fsdp_plugin_kwargs = {}
            fsdp_plugin_kwargs[
                "mixed_precision_policy"
            ] = torch.distributed.fsdp.MixedPrecision(
                param_dtype=self.weight_dtype,
                reduce_dtype=self.weight_dtype,
                buffer_dtype=self.weight_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True,
            )

            fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)(**fsdp_plugin_kwargs)
            accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters)
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True
        
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def save_training_state(self, save_path, epoch):
        self.accelerator.save_state(save_path, safe_serialization=False)

    def load_training_state(self, load_path):
        self.accelerator.load_state(load_path)

    def auto_resume(self):
        self.resume_path = None
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            epoch_dirs = [d for d in dirs if d.startswith("checkpoint-epoch-")]
            if epoch_dirs:
                epoch_dirs = sorted(epoch_dirs, key=lambda x: int(x.split("checkpoint-epoch-")[-1]))
                path = epoch_dirs[-1]
            else:
                legacy_dirs = [d for d in dirs if d.startswith("checkpoint_")]
                legacy_dirs = sorted(legacy_dirs, key=lambda x: int(x.split("_")[1]))
                path = legacy_dirs[-1] if len(legacy_dirs) > 0 else None
            if path is not None:
                path = os.path.join(self.cfg.log.ckpt_dir, path)
        else:
            path = None

        if path is None:
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            self.resume_path = path
            # Extract epoch number from checkpoint path
            # Handles both "checkpoint-epoch-NNNN" and legacy "checkpoint_N" formats.
            checkpoint_name = path.rstrip('/').split('/')[-1]
            if checkpoint_name.startswith("checkpoint-epoch-"):
                start_epoch = int(checkpoint_name.split("checkpoint-epoch-")[-1]) + 1
            elif "checkpoint_" in checkpoint_name:
                checkpoint_name = path.rstrip('/').split('/')[-1]
                start_epoch = int(checkpoint_name.split("checkpoint_")[-1]) + 1
            else:
                start_epoch = 0

        return start_epoch

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
