"""Page-locked buffers with explicit lifetime, without retaining retired cache blocks."""

import ctypes
from pathlib import Path
import torch

_runtime = None


def runtime():
    global _runtime
    if _runtime is None:
        dll = ctypes.WinDLL(str(Path(torch.__file__).parent / "lib" / "cudart64_12.dll"))
        dll.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        dll.cudaHostAlloc.restype = ctypes.c_int
        dll.cudaFreeHost.argtypes = [ctypes.c_void_p]
        dll.cudaFreeHost.restype = ctypes.c_int
        _runtime = dll
    return _runtime


class Allocation:
    def __init__(self, size):
        self.pointer = ctypes.c_void_p()
        self.free = runtime().cudaFreeHost
        error = runtime().cudaHostAlloc(ctypes.byref(self.pointer), size, 0)
        if error:
            raise RuntimeError(f"cudaHostAlloc failed: CUDA error {error}")

    def __del__(self):
        if self.pointer and self.pointer.value:
            self.free(self.pointer)
            self.pointer = None


def allocate_bytes(size):
    torch.cuda.init()
    owner = Allocation(size)
    array = (ctypes.c_ubyte * size).from_address(owner.pointer.value)
    # torch.frombuffer retains this object for the lifetime of its storage.
    array._allocation = owner
    tensor = torch.frombuffer(array, dtype=torch.uint8)
    if not tensor.is_pinned():
        raise RuntimeError("CUDA did not recognize the page-locked allocation")
    return tensor


def bundle_bytes(bundle):
    seen, total = set(), 0
    for value in bundle.values():
        storage = value.untyped_storage()
        if storage.data_ptr() not in seen:
            seen.add(storage.data_ptr())
            total += storage.nbytes()
    return total


def pack_tensors(values, limit):
    layout = {}
    cursor = 0
    for name, value in values.items():
        if value.is_cuda:
            cursor = (cursor + 255) // 256 * 256
            layout[name] = cursor
            cursor += value.numel() * value.element_size()
    extra = bundle_bytes({k: v for k, v in values.items() if not v.is_cuda})
    if cursor + extra > limit:
        return None
    raw = allocate_bytes(cursor) if cursor else None
    bundle = {}
    for name, value in values.items():
        if value.is_cuda:
            begin = layout[name]
            size = value.numel() * value.element_size()
            target = raw[begin : begin + size].view(value.dtype).view(value.shape)
            target.copy_(value)
            bundle[name] = target
        else:
            bundle[name] = value
    return bundle
