from __future__ import annotations

from drone_autopilot.cli import build_parser
from drone_autopilot.models import _strip_dataparallel_prefix
from drone_autopilot.training import (
    DistributedContext,
    TrainingConfig,
    _barrier,
    _reduce_training_loss,
    _should_use_data_parallel,
    _unwrap_distributed_model,
)


class _Cuda:
    def __init__(self, count: int) -> None:
        self._count = count

    def device_count(self) -> int:
        return self._count


class _Torch:
    def __init__(self, cuda_count: int) -> None:
        self.cuda = _Cuda(cuda_count)


class _CudaDevice:
    type = "cuda"


class _WrappedModel:
    def __init__(self, module: object) -> None:
        self.module = module


class _Distributed:
    def __init__(self) -> None:
        self.barrier_device_ids: list[int] | None = None

    def barrier(self, *, device_ids: list[int] | None = None) -> None:
        self.barrier_device_ids = device_ids


class _TorchDistributed:
    def __init__(self) -> None:
        self.distributed = _Distributed()


def test_multi_gpu_detection_accepts_string_and_torch_device_like_values() -> None:
    torch_module = _Torch(cuda_count=2)

    assert _should_use_data_parallel(torch_module, "cuda", True)
    assert _should_use_data_parallel(torch_module, "cuda:0", True)
    assert _should_use_data_parallel(torch_module, _CudaDevice(), True)


def test_multi_gpu_detection_requires_cuda_enabled_and_multiple_devices() -> None:
    assert not _should_use_data_parallel(_Torch(cuda_count=2), "cpu", True)
    assert not _should_use_data_parallel(_Torch(cuda_count=2), "cuda", False)
    assert not _should_use_data_parallel(_Torch(cuda_count=1), "cuda", True)


def test_strip_dataparallel_prefix_keeps_checkpoint_keys_portable() -> None:
    state = {
        "module.rgb_encoder.weight": object(),
        "module.head.bias": object(),
    }

    stripped = _strip_dataparallel_prefix(state)

    assert set(stripped) == {"rgb_encoder.weight", "head.bias"}
    assert stripped["rgb_encoder.weight"] is state["module.rgb_encoder.weight"]


def test_cli_keeps_multi_gpu_opt_in() -> None:
    parser = build_parser()

    default_args = parser.parse_args(["train", "manifest.parquet"])
    multi_gpu_args = parser.parse_args(["train", "manifest.parquet", "--multi-gpu"])
    single_gpu_args = parser.parse_args(["train", "manifest.parquet", "--no-multi-gpu"])
    distributed_args = parser.parse_args(["train", "manifest.parquet", "--distributed"])

    assert not default_args.multi_gpu
    assert not default_args.distributed
    assert multi_gpu_args.multi_gpu
    assert not single_gpu_args.multi_gpu
    assert distributed_args.distributed
    assert not distributed_args.multi_gpu


def test_training_config_uses_stable_single_gpu_default() -> None:
    config = TrainingConfig(
        manifest_path="manifest.parquet",  # type: ignore[arg-type]
        data_root=".",  # type: ignore[arg-type]
    )

    assert not config.multi_gpu
    assert not config.distributed


def test_reduce_training_loss_is_noop_without_distributed_context() -> None:
    running_loss, batches = _reduce_training_loss(
        object(),
        running_loss=3.0,
        batches=2,
        device="cpu",
        context=DistributedContext(),
    )

    assert running_loss == 3.0
    assert batches == 2


def test_unwrap_distributed_model_returns_inner_module() -> None:
    inner = object()

    assert _unwrap_distributed_model(_WrappedModel(inner)) is inner
    assert _unwrap_distributed_model(inner) is inner


def test_barrier_uses_local_rank_device_id() -> None:
    torch_module = _TorchDistributed()

    _barrier(
        torch_module,
        DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2),
    )

    assert torch_module.distributed.barrier_device_ids == [1]
