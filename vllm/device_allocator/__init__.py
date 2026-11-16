# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Collection, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]

# Selective wake may defer only these; any wake restores every other tag.
DEFERRABLE_TAGS: tuple[str, ...] = ("weights", "kv_cache")


@dataclasses.dataclass(slots=True)
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    is_asleep: bool = False
    # True while `cpu_backup_tensor` holds exactly the device contents: set
    # when they were copied in either direction, cleared when the allocation
    # may have been written (`mark_weights_modified`) or the backup released.
    backup_in_sync: bool = False


def discardable_allocations(
    allocator_and_pools: Mapping[str, Iterable[tuple[Any, Any]]],
    tensors: Collection[int],
) -> set[int]:
    """Weights allocations whose live tensors all start at these addresses."""
    if not tensors:
        return set()
    return {
        allocation["address"]
        for pool, _ in allocator_and_pools.get("weights", ())
        for allocation in pool.snapshot()
        if all(
            block["state"] == "inactive" or block["address"] in tensors
            for block in allocation["blocks"]
        )
    }


class MemAllocator(Protocol):
    def use_memory_pool(self, tag: str | None = None) -> AbstractContextManager: ...

    def sleep(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
        discardable: Collection[int] = (),
    ) -> None: ...

    def discard(self, tags: tuple[str, ...] | str) -> None: ...

    def wake_up(self, tags: list[str] | None = None) -> None: ...

    def get_current_usage(self) -> int: ...


def mark_weights_modified() -> None:
    """Record that weights may have been written, so any host backup of them
    is out of date. A no-op when the CUDA sleep allocator was never created."""
    if not current_platform.is_cuda_alike():
        return
    from vllm.device_allocator.cumem import CuMemAllocator

    if CuMemAllocator.instance is not None:
        CuMemAllocator.instance.mark_weights_modified()


def get_mem_allocator_instance() -> MemAllocator:
    if current_platform.is_cuda_alike():
        from vllm.device_allocator.cumem import CuMemAllocator

        return CuMemAllocator.get_instance()

    if current_platform.is_xpu():
        from vllm.device_allocator.xpumem import XpuMemAllocator

        return XpuMemAllocator.get_instance()

    raise RuntimeError(
        "Sleep mode allocator is not available on platform "
        f"{type(current_platform).__name__} "
        f"(device_type={current_platform.device_type})."
    )
