from __future__ import annotations

from drone_autopilot.cli import build_parser
from drone_autopilot.models import _strip_dataparallel_prefix
from drone_autopilot.training import _should_use_data_parallel


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

    assert not default_args.multi_gpu
    assert multi_gpu_args.multi_gpu
    assert not single_gpu_args.multi_gpu
