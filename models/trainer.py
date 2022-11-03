import collections
import gc
import inspect
import math
from multiprocessing.spawn import import_main_path
import os
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Counter,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from torch import optim
import nvidia_smi

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_distributed_available,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers import Trainer
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.utils.modeling_auto_mapping import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
)
from transformers import Trainer
from transformers.integrations import WandbCallback, rewrite_logs
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    import fairscale
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

    if version.parse(fairscale.__version__) >= version.parse("0.3"):
        from fairscale.nn.data_parallel import (
            FullyShardedDataParallel as FullyShardedDDP,
        )
        from fairscale.nn.wrap import auto_wrap
    else:
        FullyShardedDDP = None

if is_sagemaker_distributed_available():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import (
        DistributedDataParallel as DDP,
    )
else:
    import torch.distributed as dist

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


class WandbCallbackThreadFix(WandbCallback):
    def setup(self, args, state, model, reinit, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name
            init_args["settings"] = wandb.Settings(start_method="fork")
            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                config=combined_dict,
                name=run_name,
                reinit=reinit,
                **init_args,
            )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, reinit=False)

        is_table = len(logs) == 1

        if state.is_world_process_zero:
            if is_table:
                self._wandb.log(logs)
            else:
                use_global_step = logs.pop("use_global_step", True)
                logs = rewrite_logs(logs)

                if use_global_step:
                    self._wandb.log(logs, step=state.global_step)
                else:
                    self._wandb.log(logs)

class MuxTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError(
                    "`Trainer` requires either a `model` or `model_init` argument"
                )
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. "
                    "`model_init` will overwrite your model when calling the `train` method. This will become a fatal error in the next release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if (
            hasattr(model, "is_parallelizable")
            and model.is_parallelizable
            and model.model_parallel
        ):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # Setup Sharded DDP training
        self.sharded_ddp = None
        if len(args.sharded_ddp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )

            if args.local_rank == -1:
                raise ValueError(
                    "Using sharded DDP only works in distributed training."
                )
            elif not is_fairscale_available():
                raise ImportError(
                    "Sharded DDP training requires fairscale: `pip install fairscale`."
                )
            elif (
                ShardedDDPOption.SIMPLE not in args.sharded_ddp
                and FullyShardedDDP is None
            ):
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3

        # one place to sort out whether to place the model on device or not
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or (args.deepspeed and args.do_train)
            or (args.fp16_full_eval and not args.do_train)
            or (
                self.sharded_ddp
                in [ShardedDDPOption.ZERO_DP_2, ShardedDDPOption.ZERO_DP_3]
            )
        ):
            self.place_model_on_device = False

        default_collator = (
            default_data_collator
            if tokenizer is None
            else DataCollatorWithPadding(tokenizer)
        )
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        if self.place_model_on_device:
            model = model.to(args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        # default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
        #     self.args.report_to
        # )
        default_callbacks = DEFAULT_CALLBACKS + [WandbCallbackThreadFix]
        callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if not callable(self.data_collator) and callable(
            getattr(self.data_collator, "collate_batch", None)
        ):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        # Enforce rules on using datasets with no __len__
        if (
            train_dataset is not None
            and not isinstance(train_dataset, collections.abc.Sized)
            and args.max_steps <= 0
        ):
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        self._signature_columns = None
        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None

        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16 and not args.deepspeed:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = (
                    ShardedGradScaler()
                    if self.sharded_ddp is not None
                    else torch.cuda.amp.GradScaler()
                )
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor
            )
        else:
            self.label_smoother = None

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable for total_flos used to count as tensors (for distributed + TPU), will be sent in the
        # state at each call to self.log.
        self._total_flos = None
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions", "end_positions"]
            if type(self.model).__name__
            in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values()
            else ["labels"]
        )
        self.label_names = (
            default_label_names
            if self.args.label_names is None
            else self.args.label_names
        )
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control
        )

        # very last
        self._memory_tracker.stop_and_update_metrics()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}."
            )
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        ):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = (
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    self.args.max_steps // num_update_steps_per_epoch
                    + int(self.args.max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = math.ceil(
                    self.args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        delay_optimizer_creation = (
            self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        )
        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(
                self, num_training_steps=max_steps
            )
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * world_size
        )
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, "trainer_state.json")
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        tr_task_loss = torch.tensor(0.0).to(self.args.device)
        tr_retrieval_loss = torch.tensor(0.0).to(self.args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(
                    train_dataloader, [self.args.device]
                ).per_device_loader(self.args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        self.args, self.state, self.control
                    )

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and self.args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        (
                            cur_tr_loss,
                            cur_task_loss,
                            cur_retrieval_loss,
                        ) = self.training_step(model, inputs)
                        tr_loss += cur_tr_loss
                        if cur_task_loss is not None:
                            tr_task_loss += cur_task_loss
                        if cur_retrieval_loss is not None:
                            tr_retrieval_loss += cur_retrieval_loss
                else:
                    cur_tr_loss, cur_task_loss, cur_retrieval_loss = self.training_step(
                        model, inputs
                    )
                    tr_loss += cur_tr_loss
                    if cur_task_loss is not None:
                        tr_task_loss += cur_task_loss
                    if cur_retrieval_loss is not None:
                        tr_retrieval_loss += cur_retrieval_loss

                self._total_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if (
                        self.args.max_grad_norm is not None
                        and self.args.max_grad_norm > 0
                        and not self.deepspeed
                    ):
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer)
                                if self.use_apex
                                else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, tr_task_loss, tr_retrieval_loss, model, trial, epoch
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, tr_task_loss, tr_retrieval_loss, model, trial, epoch
            )

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(
                    self.state.best_model_checkpoint
                )
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(
                    os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # force memory release
            # to restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(
            self.state.global_step,
            self._total_loss_scalar / self.state.global_step,
            metrics,
        )

    def _maybe_log_save_evaluate(
        self, tr_loss, task_loss, retrieval_loss, model, trial, epoch
    ):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            task_loss_scalar = task_loss.item() if task_loss is not None else None
            retrieval_loss_scalar = (
                retrieval_loss.item() if retrieval_loss is not None else None
            )
            # reset tr_loss to zero
            tr_loss -= tr_loss
            task_loss -= task_loss
            retrieval_loss -= retrieval_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if task_loss_scalar is not None:
                logs["task_loss"] = round(
                    task_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            if retrieval_loss_scalar is not None:
                logs["retrieval_loss"] = round(
                    retrieval_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                (
                    loss,
                    task_loss,
                    retrieval_loss,
                    retrieval_logits,
                    retrieval_instance_labels,
                ) = self.compute_loss(model, inputs)
        else:
            (
                loss,
                task_loss,
                retrieval_loss,
                retrieval_logits,
                retrieval_instance_labels,
            ) = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            task_loss = task_loss.mean() if task_loss is not None else None
            retrieval_loss = (
                retrieval_loss.mean() if retrieval_loss is not None else None
            )

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            task_loss = (
                task_loss / self.args.gradient_accumulation_steps
                if task_loss is not None
                else None
            )
            retrieval_loss = (
                retrieval_loss / self.args.gradient_accumulation_steps
                if retrieval_loss is not None
                else None
            )

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        task_loss = task_loss.detach() if task_loss is not None else None
        retrieval_loss = retrieval_loss.detach() if retrieval_loss is not None else None
        return loss.detach(), task_loss, retrieval_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        task_loss = None
        retrieval_loss = None

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            task_loss = outputs["task_loss"] if "task_loss" in outputs else None
            retrieval_loss = (
                outputs["retrieval_loss"] if "retrieval_loss" in outputs else None
            )
            retrieval_logits = (
                outputs["retrieval_predictions"]
                if "retrieval_predictions" in outputs
                else None
            )
            retrieval_instance_labels = (
                outputs["retrieval_instance_labels"]
                if "retrieval_instance_labels" in outputs
                else None
            )

        return (
            (
                loss,
                task_loss,
                retrieval_loss,
                retrieval_logits,
                retrieval_instance_labels,
                outputs,
            )
            if return_outputs
            else (
                loss,
                task_loss,
                retrieval_loss,
                retrieval_logits,
                retrieval_instance_labels,
            )
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        speed_metrics=False,
        interference_report=False
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        nvidia_smi.nvmlInit()        
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        # output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        self.log(output.metrics, use_global_step=False)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        if speed_metrics:
            
            train_dataloader =  self.get_train_dataloader()
            model = self._wrap_model(self.model, training=False)
            model.eval()
            tot_samples = 0 
            total_infer_time = 0
            average_gpu_memory = 0
            batch_ctr = 0
            
            with torch.no_grad():
                for _, inputs in enumerate(tqdm(train_dataloader)):

                    inputs = self._prepare_inputs(inputs)
                    start_time = time.time()
                    _ = model(**inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    total_infer_time += (end_time - start_time)
                    tot_samples += inputs['input_ids'].shape[0]
                    batch_ctr += 1
                    # gpu memory calculations
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    average_gpu_memory += info.used / 1e+9
                    # if batch_ctr > 20:
                    #     break
                throughput = tot_samples / total_infer_time
                average_gpu_memory = average_gpu_memory / batch_ctr
                # update metrics
                output.metrics[f"{metric_key_prefix}_throughput"] = throughput
                output.metrics[f"{metric_key_prefix}_speed_tot_samples"] = tot_samples
                output.metrics[f"{metric_key_prefix}_inference_time"] = total_infer_time
                output.metrics[f"{metric_key_prefix}_average_memory"] = average_gpu_memory

        if interference_report:
            num_anchors = 10
            num_batches = 30
            train_dataloader =  self.get_train_dataloader()
            model = self._wrap_model(self.model, training=False)
            model.eval()
            anchors = None
            anchor_representations = {i: [] for i in range(num_anchors)}
            with torch.no_grad():
                for batch_id, inputs in enumerate(tqdm(train_dataloader)):
                    if batch_id > num_batches:
                        break
                    inputs = self._prepare_inputs(inputs)
                    if anchors is None:
                        anchors = inputs["input_ids"][:num_anchors]
                        continue
                    # replace with anchor and get corresponding demux representations
                    for anchor_id in range(num_anchors):
                        anchor = anchors[anchor_id]
                        inputs["input_ids"][0] = anchor
                        inputs["return_dict"] = True
                        outputs = model(**inputs)
                        anchor_representations[anchor_id].append(outputs["hidden_states"][0])
            # t-sne plot
            anchor_representations_stacked = []
            for anchor_id in range(num_anchors):
               anchor_representations_stacked.append(torch.stack(anchor_representations[anchor_id]))
            anchor_representations_stacked = torch.cat(anchor_representations_stacked)
            anchor_representations_stacked = anchor_representations_stacked.cpu().numpy() 
            average_cos_similarity = 0 
            for anchor_i in range(num_anchors):
                for anchor_j in range(anchor_i +1, num_anchors):
                    anchor_i_representations = torch.stack(anchor_representations[anchor_i])
                    anchor_j_representations = torch.stack(anchor_representations[anchor_j])
                    average_cos_similarity += (torch.matmul(anchor_i_representations, anchor_j_representations.t()) / (torch.norm(anchor_i_representations, dim=1) * torch.norm(anchor_j_representations, dim=1))).mean()
            average_cos_similarity /= (num_anchors * (num_anchors - 1) * 0.5)
            pca_50 = PCA(n_components=200)
            pca_result_50 = pca_50.fit_transform(anchor_representations_stacked)
            tsne = TSNE(n_components=2, verbose=1, n_iter=500)
            tsne_pca_results = tsne.fit_transform(pca_result_50)
            df = pd.DataFrame()
            df["tsne_1"] = tsne_pca_results[:, 0]
            df["tsne_2"] = tsne_pca_results[:, 1]
            df["sample"] = np.repeat(np.arange(num_anchors), len(anchor_representations[0]))
            df["sample"] = df["sample"].apply(lambda i: str(i))
            sns.scatterplot(
            x="tsne_1", y="tsne_2",
            hue='sample',
            palette=sns.color_palette("bright", num_anchors),
            data=df,
            legend="full",
            alpha=0.3,
            )
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title(f'Interference analysis: N = {self.model.config.num_instances}', fontsize=16)
            plt.legend(loc="lower right", fontsize=8)
            plt.savefig(f"interference_fig_{self.model.config.num_instances}.png")
            df.to_csv(f"interference_fig_{self.model.config.num_instances}.csv")
            print(f"average cos similarity: {average_cos_similarity}")
        
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(
            test_dataset, collections.abc.Sized
        ):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # output.metrics.update(
        #     speed_metrics(metric_key_prefix, start_time, len(test_dataset))
        # )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info(
                "Detected the deepspeed argument but it will not be used for evaluation"
            )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        num_examples = (
            (num_examples // batch_size) * batch_size
            if self.args.dataloader_drop_last
            else num_examples
        )

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        task_losses_host: torch.Tensor = None
        retrieval_losses_host: torch.Tensor = None
        retrieval_accs_host: torch.Tensor = None

        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        eval_task_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        eval_retrieval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        eval_retrieval_acc_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )

        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, SequentialDistributedSampler
            ):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            labels_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        column_names = ["Token", "Count", "Correct", "Accuracy"]
        table = wandb.Table(columns=column_names)
        token_acc_counter = {}
        for step, inputs in enumerate(dataloader):
            (
                loss,
                logits,
                labels,
                task_loss,
                retrieval_loss,
                retrieval_acc,
                token_acc_counter,
            ) = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                token_acc_counter=token_acc_counter,
            )
            # print("token acc counter", token_acc_counter)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if task_loss is not None:
                task_losses = task_loss.repeat(batch_size)
                task_losses_host = (
                    task_losses
                    if task_losses_host is None
                    else torch.cat((task_losses_host, task_losses), dim=0)
                )
            if retrieval_loss is not None:
                retrieval_losses = retrieval_loss.repeat(batch_size)
                retrieval_losses_host = (
                    retrieval_losses
                    if retrieval_losses_host is None
                    else torch.cat((retrieval_losses_host, retrieval_losses), dim=0)
                )
            if retrieval_acc is not None:
                retrieval_accs = retrieval_acc.repeat(batch_size)
                retrieval_accs_host = (
                    retrieval_accs
                    if retrieval_accs_host is None
                    else torch.cat((retrieval_accs_host, retrieval_accs), dim=0)
                )
            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                if task_losses_host is not None:
                    eval_task_losses_gatherer.add_arrays(
                        self._gather_and_numpify(task_losses_host, "eval_task_losses")
                    )

                if retrieval_losses_host is not None:
                    eval_retrieval_losses_gatherer.add_arrays(
                        self._gather_and_numpify(
                            retrieval_losses_host, "eval_retrieval_losses"
                        )
                    )

                if not prediction_loss_only:
                    preds_gatherer.add_arrays(
                        self._gather_and_numpify(preds_host, "eval_preds")
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )

                # Set back to None to begin a new accumulation
                (
                    losses_host,
                    preds_host,
                    labels_host,
                    task_losses_host,
                    retrieval_losses_host,
                ) = (None, None, None, None, None)

        # add to wandb table
        for token in token_acc_counter:
            table.add_data(
                token,
                token_acc_counter[token]["tot_count"],
                token_acc_counter[token]["correct"]
                if "correct" in token_acc_counter[token]
                else 0,
                token_acc_counter[token]["correct"]
                / token_acc_counter[token]["tot_count"]
                if "correct" in token_acc_counter[token]
                else 0,
            )
        # log table
        self.callback_handler.on_log(
            self.args, self.state, self.control, {str(self.state.global_step): table}
        )
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )
        if task_losses_host is not None:
            eval_task_losses_gatherer.add_arrays(
                self._gather_and_numpify(task_losses_host, "eval_task_losses")
            )

        if retrieval_losses_host is not None:
            eval_retrieval_losses_gatherer.add_arrays(
                self._gather_and_numpify(retrieval_losses_host, "eval_retrieval_losses")
            )
        if retrieval_accs_host is not None:
            eval_retrieval_acc_gatherer.add_arrays(
                self._gather_and_numpify(retrieval_accs_host, "eval_retrieval_accs")
            )

        if not prediction_loss_only:
            preds_gatherer.add_arrays(
                self._gather_and_numpify(preds_host, "eval_preds")
            )
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )

        eval_loss = eval_losses_gatherer.finalize()
        eval_task_loss = (
            eval_task_losses_gatherer.finalize()
            if task_losses_host is not None
            else None
        )
        eval_retrieval_loss = (
            eval_retrieval_losses_gatherer.finalize()
            if retrieval_losses_host is not None
            else None
        )
        eval_retrieval_accs = (
            eval_retrieval_acc_gatherer.finalize()
            if retrieval_accs_host is not None
            else None
        )
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()
        if eval_task_loss is not None:
            metrics[f"{metric_key_prefix}_task_loss"] = eval_task_loss.mean().item()
        if eval_retrieval_loss is not None:
            metrics[
                f"{metric_key_prefix}_retrieval_loss"
            ] = eval_retrieval_loss.mean().item()
        if eval_retrieval_accs is not None:
            metrics[
                f"{metric_key_prefix}_retrieval_acc"
            ] = eval_retrieval_accs.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        token_acc_counter=None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        logits = None
        retrieval_loss = None
        task_loss = None
        retrieval_predictions=None
        retrieval_instance_labels=None
        with torch.no_grad():

            if has_labels:
                (
                    loss,
                    task_loss,
                    retrieval_loss,
                    retrieval_predictions,
                    retrieval_instance_labels,
                    outputs,
                ) = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(
                        v
                        for k, v in outputs.items()
                        if k
                        not in ignore_keys + ["loss", "task_loss", "retrieval_loss"]
                    )
                else:
                    logits = outputs[1:]
                if "retrieval_loss" in outputs:
                    retrieval_loss = outputs["retrieval_loss"]
                    retrieval_loss = retrieval_loss.mean().detach()

                if "task_loss" in outputs:
                    task_loss = outputs["task_loss"]
                    task_loss = task_loss.mean().detach()

            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        retrieval_acc = None

        if retrieval_predictions is not None:
            retrieval_predictions = nested_detach(retrieval_predictions)
            retrieval_instance_labels = retrieval_instance_labels.detach()
            # calculate retrieval acc from the retrieval logits and the sentence labels
            with torch.no_grad():
                ignore_predictions = retrieval_instance_labels == -100
                retrieval_correct_predictions = (
                    retrieval_predictions == retrieval_instance_labels
                ) & ~ignore_predictions
                retrieval_acc = torch.sum(retrieval_correct_predictions) / torch.sum(
                    ~ignore_predictions
                )
                retrieval_acc = retrieval_acc.detach()
                # add token level information in the token counter object
                if token_acc_counter is not None:
                    correct_tokens = retrieval_predictions[
                        retrieval_correct_predictions
                    ]
                    correct_tokens_unique, correct_tokens_unique_counts = torch.unique(
                        correct_tokens, return_counts=True
                    )
                    # iterate through vocab ids and get the exact token
                    for c_token_id, c_token in enumerate(
                        correct_tokens_unique.tolist()
                    ):
                        c_token_decoded = self.tokenizer.decode([c_token])
                        if c_token_decoded not in token_acc_counter:
                            token_acc_counter[c_token_decoded] = {
                                "correct": 0,
                                "tot_count": 0,
                            }
                        token_acc_counter[c_token_decoded][
                            "correct"
                        ] += correct_tokens_unique_counts[c_token_id].item()

                    all_tokens_unique, all_tokens_count = torch.unique(
                        retrieval_instance_labels[~ignore_predictions],
                        return_counts=True,
                    )
                    for a_token_id, a_token in enumerate(all_tokens_unique.tolist()):

                        a_token_decoded = self.tokenizer.decode([a_token])

                        if a_token_decoded not in token_acc_counter:
                            token_acc_counter[a_token_decoded] = {
                                "correct": 0,
                                "tot_count": 0,
                            }
                        token_acc_counter[a_token_decoded][
                            "tot_count"
                        ] += all_tokens_count[a_token_id].item()

        if prediction_loss_only:
            return (
                loss,
                None,
                None,
                task_loss,
                retrieval_loss,
                retrieval_acc,
                token_acc_counter,
            )
        if logits is not None:
            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

        return (
            loss,
            logits,
            labels,
            task_loss,
            retrieval_loss,
            retrieval_acc,
            token_acc_counter,
        )
    def log(self, logs: Dict[str, float], use_global_step=True) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        logs["use_global_step"] = use_global_step
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )
