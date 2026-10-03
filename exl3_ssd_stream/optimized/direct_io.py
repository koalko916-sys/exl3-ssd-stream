"""Local Windows unbuffered reader. Does not alter checkpoint bytes."""

import ctypes
import math
import os
import time
from collections import deque
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes as wt
import torch
from exllamav3.loader.safetensors import convert_dtype
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.fp16 import LinearFP16
from .host_buffer import allocate_bytes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_void_p),
        ("InternalHigh", ctypes.c_void_p),
        ("Offset", wt.DWORD),
        ("OffsetHigh", wt.DWORD),
        ("hEvent", ctypes.c_void_p),
    ]


k32.CreateFileW.argtypes = [
    wt.LPCWSTR,
    wt.DWORD,
    wt.DWORD,
    ctypes.c_void_p,
    wt.DWORD,
    wt.DWORD,
    ctypes.c_void_p,
]
k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateEventW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR]
k32.CreateEventW.restype = ctypes.c_void_p
k32.ReadFile.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    wt.DWORD,
    ctypes.c_void_p,
    ctypes.POINTER(Overlapped),
]
k32.ReadFile.restype = wt.BOOL
k32.GetOverlappedResult.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(Overlapped),
    ctypes.POINTER(wt.DWORD),
    wt.BOOL,
]
k32.GetOverlappedResult.restype = wt.BOOL
k32.CloseHandle.argtypes = [ctypes.c_void_p]
k32.CloseHandle.restype = wt.BOOL
ALIGN = 4096


class DirectReader:
    def __init__(self, chunk_mib=16):
        self.handles = {}
        self.chunk = chunk_mib * 2**20
        self.buffer = allocate_bytes(self.chunk + ALIGN)
        skip = (-self.buffer.data_ptr()) % ALIGN
        self.buffer = self.buffer[skip : skip + self.chunk]
        self.event = k32.CreateEventW(None, True, False, None)
        self.bytes_read = 0
        self.read_seconds = 0.0

    def read_into(self, filename, offset, count, target_ptr):
        started = time.perf_counter()
        handle = self.handles.get(filename)
        if handle is None:
            handle = k32.CreateFileW(str(filename), 0x80000000, 7, None, 3, 0x60000000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            self.handles[filename] = handle
        ov = Overlapped()
        ov.Offset, ov.OffsetHigh, ov.hEvent = (
            offset & 0xFFFFFFFF,
            offset >> 32,
            self.event,
        )
        ok = k32.ReadFile(handle, target_ptr, count, None, ctypes.byref(ov))
        if not ok and ctypes.get_last_error() != 997:
            raise ctypes.WinError(ctypes.get_last_error())
        got = wt.DWORD()
        if not k32.GetOverlappedResult(handle, ctypes.byref(ov), ctypes.byref(got), True):
            raise ctypes.WinError(ctypes.get_last_error())
        self.bytes_read += got.value
        self.read_seconds += time.perf_counter() - started
        return got.value

    def read(self, filename, offset, size, device="cuda:0"):
        start = offset // ALIGN * ALIGN
        prefix = offset - start
        total = math.ceil((prefix + size) / ALIGN) * ALIGN
        out = torch.empty(size, dtype=torch.uint8, device=device)
        for pos in range(0, total, self.chunk):
            n = min(self.chunk, total - pos)
            got = self.read_into(filename, start + pos, n, self.buffer.data_ptr())
            need = min(n, prefix + size - pos)
            if got < need:
                raise IOError(f"Short read: {filename} @{start + pos}, {got} < {need}")
            left, right = max(pos, prefix), min(pos + got, prefix + size)
            out[left - prefix : right - prefix].copy_(
                self.buffer[left - pos : right - pos], non_blocking=True
            )
            # The staging buffer must not be overwritten before H2D completes.
            if out.is_cuda:
                torch.cuda.current_stream().synchronize()
        return out

    def close(self):
        for h in self.handles.values():
            k32.CloseHandle(h)
        self.handles.clear()
        if self.event:
            k32.CloseHandle(self.event)
            self.event = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class BundleLoader:
    def __init__(self, reader):
        self.reader = reader
        self.plans = {}
        self.markers = {}
        self.registry = {}
        self.pending = {}
        self.group_experts = os.environ.get("GLM_GROUP_EXPERTS", "0") == "1"
        self.prefetch_enabled = os.environ.get("GLM_PREFETCH", "0") == "1"
        self.prefetch_queue = deque()
        self.prefetch_pending = {}
        workers = max(1, min(8, int(os.environ.get("GLM_PREFETCH_WORKERS", "2"))))
        self.prefetch_slots = (
            [DirectReader(32) for _ in range(workers)] if self.prefetch_enabled else []
        )
        self.free_slots = deque(range(len(self.prefetch_slots)))
        self.pool = ThreadPoolExecutor(max_workers=workers) if self.prefetch_enabled else None
        self.prefetched_bytes = 0
        self.tracking = False
        self.routing_counts = Counter()
        self.hot_experts = set()
        self.expert_cache = {}
        self.expert_cache_bytes = 0
        self.expert_cache_limit = 0
        self.expert_cache_hits = 0
        self.access_counts = Counter()
        self.expert_cache_evictions = 0
        self.dynamic_cache = os.environ.get("GLM_DYNAMIC_EXPERT_CACHE", "0") == "1"

    @property
    def total_io_bytes(self):
        return self.reader.bytes_read + sum(r.bytes_read for r in self.prefetch_slots)

    @property
    def total_read_seconds(self):
        return self.reader.read_seconds + sum(r.read_seconds for r in self.prefetch_slots)

    def start_prefill(self):
        self.tracking = True
        self.routing_counts.clear()
        if self.dynamic_cache:
            self.access_counts = Counter({k: v * 0.5 for k, v in self.access_counts.items()})

    def finish_prefill(self):
        self.tracking = False
        if not self.prefetch_enabled or os.environ.get("GLM_EXPERT_CACHE", "0") != "1":
            return
        import psutil

        self.expert_cache_limit = min(
            2**30,
            max(
                0,
                psutil.virtual_memory().available
                - int(float(os.environ.get("GLM_RAM_RESERVE_GIB", "2")) * 2**30)
                + self.expert_cache_bytes,
            ),
        )
        layers = defaultdict(list)
        for (prefix, expert), count in self.routing_counts.items():
            layers[prefix].append((count, expert))
        self.hot_experts = {
            f"{prefix}.experts.{max(values)[1]}.up_proj" for prefix, values in layers.items()
        }
        for key in list(self.expert_cache):
            if key not in self.hot_experts and not self.dynamic_cache:
                self.expert_cache_bytes -= self.expert_cache.pop(key)[0].numel()
        while self.expert_cache_bytes > self.expert_cache_limit:
            victim = min(self.expert_cache, key=lambda k: self.access_counts[k])
            self.expert_cache_bytes -= self.expert_cache.pop(victim)[0].numel()
        print(
            f"Hot expert RAM cache: {self.expert_cache_limit / 2**30:.2f} GiB",
            flush=True,
        )

    def group_plan(self, linear):
        names = [
            linear.key,
            linear.key[:-7] + "gate_proj",
            linear.key[:-7] + "down_proj",
        ]
        modules = [self.registry.get(k) for k in names]
        if not all(m is not None for m in modules):
            return None
        plans = [self.plan(m) for m in modules]
        if not all(p is not None for p in plans):
            return None
        files = {p[0] for p in plans}
        begin = min(p[1] for p in plans)
        end = max(p[1] + p[2] for p in plans)
        if len(files) != 1 or end - begin != sum(p[2] for p in plans):
            return None
        return names, modules, plans, begin, end

    def begin_prefetch(self, prefix, selected):
        if not self.prefetch_enabled:
            return
        if self.prefetch_pending or self.prefetch_queue:
            raise RuntimeError("Previous expert prefetch queue was not consumed")
        flat = selected.reshape(-1).tolist()
        if self.tracking:
            self.routing_counts.update((prefix, expert) for expert in flat)
        elif self.dynamic_cache:
            self.access_counts.update(f"{prefix}.experts.{expert}.up_proj" for expert in set(flat))
        for expert in sorted(set(flat)):
            key = f"{prefix}.experts.{expert}.up_proj"
            if key in self.expert_cache:
                continue
            linear = self.registry[key]
            group = self.group_plan(linear)
            if (
                group is not None
                and group[4] - group[3] + 2 * ALIGN <= self.prefetch_slots[0].chunk
            ):
                self.prefetch_queue.append((key, group))
        self.schedule()

    def schedule(self):
        while self.free_slots and self.prefetch_queue:
            slot = self.free_slots.popleft()
            key, group = self.prefetch_queue.popleft()
            future = self.pool.submit(self.read_group, slot, group)
            self.prefetch_pending[key] = future, slot, group

    def read_group(self, slot, group):
        _, _, plans, begin, end = group
        reader = self.prefetch_slots[slot]
        start = begin // ALIGN * ALIGN
        prefix = begin - start
        size = end - begin
        count = math.ceil((prefix + size) / ALIGN) * ALIGN
        got = reader.read_into(plans[0][0], start, count, reader.buffer.data_ptr())
        if got < prefix + size:
            raise IOError("Short expert prefetch read")
        return reader.buffer[prefix : prefix + size]

    def install_prefetch(self, model):
        if not self.prefetch_enabled:
            return
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

        for module in model:
            if isinstance(module, BlockSparseMLP):
                original = module.routing_fn
                prefix = module.key

                def routing(bsz, cfg, z, params, _original=original, _prefix=prefix):
                    selected, weights = _original(bsz, cfg, z, params)
                    self.begin_prefetch(_prefix, selected)
                    return selected, weights

                module.routing_fn = routing

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.prefetch_pending.clear()
        self.prefetch_queue.clear()
        self.pending.clear()
        self.expert_cache.clear()
        self.expert_cache_bytes = 0
        for reader in self.prefetch_slots:
            reader.close()
        self.prefetch_slots.clear()

    def plan(self, linear):
        key = linear.key
        if key in self.plans:
            return self.plans[key]
        stc = linear.config.stc
        fields = {}
        for name in ("suh", "svh", "trellis", "bias", "mcg", "mul1"):
            tk = key + "." + name
            source = stc.find_stc(tk)
            filename = source.tensor_file_map.get(tk)
            if filename:
                header = source.file_headers[filename]
                fields[name] = (filename, header["_header_offset"], header[tk])
        if not all(x in fields for x in ("suh", "svh", "trellis")):
            return None
        filenames = {v[0] for v in fields.values()}
        if len(filenames) != 1:
            return None
        begin = min(v[2]["data_offsets"][0] for v in fields.values())
        end = max(v[2]["data_offsets"][1] for v in fields.values())
        # Only coalesce a matrix's own contiguous fields, never unrelated tensors.
        if end - begin != sum(
            v[2]["data_offsets"][1] - v[2]["data_offsets"][0] for v in fields.values()
        ):
            return None
        plan = (
            next(iter(filenames)),
            next(iter(fields.values()))[1] + begin,
            end - begin,
            begin,
            fields,
        )
        self.plans[key] = plan
        return plan

    def load(self, linear, device=None):
        ready = self.pending.pop(linear.key, None)
        if ready is not None:
            return ready
        expert = self.expert_cache.get(linear.key)
        if expert is not None:
            raw_cpu, group = expert
            self.expert_cache_hits += 1
            raw = raw_cpu.to(device or linear.device, non_blocking=True)
            names, modules, plans, begin, end = group
            natives = [
                self.make(
                    m,
                    p,
                    raw[p[1] - begin : p[1] + p[2] - begin],
                    device or linear.device,
                )
                for m, p in zip(modules, plans)
            ]
            self.pending.update(zip(names[1:], natives[1:]))
            return natives[0]
        prefetched = self.prefetch_pending.pop(linear.key, None)
        if prefetched is not None:
            future, slot, group = prefetched
            raw_cpu = future.result()
            eligible = linear.key in self.hot_experts or (
                self.dynamic_cache and self.access_counts[linear.key] >= 2
            )
            if eligible and self.dynamic_cache:
                score = self.access_counts[linear.key] * raw_cpu.numel()
                while (
                    self.expert_cache_bytes + raw_cpu.numel() > self.expert_cache_limit
                    and self.expert_cache
                ):
                    victim = min(
                        self.expert_cache,
                        key=lambda k: self.access_counts[k] * self.expert_cache[k][0].numel(),
                    )
                    if score <= self.access_counts[victim] * self.expert_cache[victim][0].numel():
                        break
                    self.expert_cache_bytes -= self.expert_cache.pop(victim)[0].numel()
                    self.expert_cache_evictions += 1
            if eligible and self.expert_cache_bytes + raw_cpu.numel() <= self.expert_cache_limit:
                stored = allocate_bytes(raw_cpu.numel())
                stored.copy_(raw_cpu)
                self.expert_cache[linear.key] = stored, group
                self.expert_cache_bytes += stored.numel()
            raw = raw_cpu.to(device or linear.device, non_blocking=True)
            # Release the pinned slot only after its upload has completed.
            torch.cuda.current_stream().synchronize()
            self.prefetched_bytes += raw.numel()
            self.free_slots.append(slot)
            self.schedule()
            names, modules, plans, begin, end = group
            natives = [
                self.make(
                    m,
                    p,
                    raw[p[1] - begin : p[1] + p[2] - begin],
                    device or linear.device,
                )
                for m, p in zip(modules, plans)
            ]
            self.pending.update(zip(names[1:], natives[1:]))
            return natives[0]
        if self.group_experts and ".experts." in linear.key and linear.key.endswith(".up_proj"):
            names = [
                linear.key,
                linear.key[:-7] + "gate_proj",
                linear.key[:-7] + "down_proj",
            ]
            modules = [self.registry.get(k) for k in names]
            if all(m is not None for m in modules):
                plans = [self.plan(m) for m in modules]
                if all(p is not None for p in plans):
                    files = {p[0] for p in plans}
                    begin = min(p[1] for p in plans)
                    end = max(p[1] + p[2] for p in plans)
                    if len(files) == 1 and end - begin == sum(p[2] for p in plans):
                        raw = self.reader.read(
                            plans[0][0], begin, end - begin, device or linear.device
                        )
                        natives = [
                            self.make(
                                m,
                                p,
                                raw[p[1] - begin : p[1] + p[2] - begin],
                                device or linear.device,
                            )
                            for m, p in zip(modules, plans)
                        ]
                        self.pending.update(zip(names[1:], natives[1:]))
                        return natives[0]
        plan = self.plan(linear)
        if plan is None:
            return self.load_fp16(linear, device or linear.device)
        filename, offset, size, begin, fields = plan
        target = device or linear.device
        raw = self.reader.read(filename, offset, size, target) if size <= 128 * 2**20 else None
        return self.make(linear, plan, raw, target)

    def read_tensor(self, stc, key, target):
        source = stc.find_stc(key)
        filename = source.tensor_file_map.get(key)
        if not filename:
            return None
        header = source.file_headers[filename]
        meta = header[key]
        a, b = meta["data_offsets"]
        raw = self.reader.read(filename, header["_header_offset"] + a, b - a, target)
        value = raw.view(convert_dtype(meta["dtype"])[0]).view(meta["shape"])
        if value.dtype in (torch.float32, torch.bfloat16):
            value = value.half()
        return value

    def load_fp16(self, linear, target):
        key, stc = linear.key, linear.config.stc
        source = stc.find_stc(key + ".weight")
        filename = source.tensor_file_map.get(key + ".weight")
        if not filename or linear.is_sliced or linear.weight_scale != 1.0:
            return None
        meta = source.file_headers[filename][key + ".weight"]
        if meta["dtype"] not in ("F16", "BF16", "F32"):
            return None
        if any(
            stc.has_tensor(key + "." + name)
            for name in ("scale", "weight_scale", "weight_scale_inv")
        ):
            return None
        weight = self.read_tensor(stc, key + ".weight", target)
        bias = self.read_tensor(stc, key + ".bias", target)
        if linear.transposed_load:
            weight = weight.T.contiguous()
        weight, bias = linear.pad_out(weight), linear.pad_out(bias)
        return LinearFP16(
            linear.in_features,
            linear.out_features,
            weight,
            bias,
            linear.full_in_features,
            linear.full_out_features,
            linear.first_in_feature,
            linear.first_out_feature,
            out_dtype=linear.out_dtype,
            key=key,
        )

    def make(self, linear, plan, raw, target):
        filename, offset, size, begin, fields = plan
        args = {}
        for name, (_, _, meta) in fields.items():
            a, b = meta["data_offsets"]
            dtype = convert_dtype(meta["dtype"])[0]
            if raw is None:
                field = self.reader.read(filename, offset + a - begin, b - a, target)
            else:
                # Quant kernels require naturally aligned allocations. Header offsets
                # need not be aligned, and the 4-byte marker precedes the trellis.
                field = raw[a - begin : b - begin].clone()
            value = field.view(dtype).view(meta["shape"])
            if name in ("mul1", "mcg"):
                # Marker contents are retained exactly; the native kernel tests presence.
                marker_key = linear.key + "." + name
                if marker_key not in self.markers:
                    self.markers[marker_key] = value.cpu().clone()
                value = self.markers[marker_key]
            args[name] = value
        return LinearEXL3(
            linear.config,
            linear.in_features,
            linear.out_features,
            out_dtype=linear.out_dtype,
            key=linear.key,
            **args,
        )


class DiskEmbedding:
    def __init__(self, module, reader):
        self.module, self.reader = module, reader
        self.rows = {}
        tk = module.key + ".weight"
        source = module.config.stc.find_stc(tk)
        self.filename = source.tensor_file_map[tk]
        header = source.file_headers[self.filename]
        meta = header[tk]
        self.offset = header["_header_offset"] + meta["data_offsets"][0]
        self.dtype = convert_dtype(meta["dtype"])[0]
        self.width = meta["shape"][1]
        self.vocab = meta["shape"][0]
        self.row_bytes = self.width * self.dtype.itemsize

    def load(self, device, **kwargs):
        self.module.device = torch.device("cpu")

    def forward(self, ids, params, out_dtype=None):
        if params.get("indexed_embeddings"):
            raise ValueError("Disk embedding only supports text input")
        params.setdefault("input_ids", ids)
        result = []
        for token in ids.reshape(-1).tolist():
            if not 0 <= token < self.vocab:
                raise ValueError(f"Invalid embedding token {token}")
            row = self.rows.get(token)
            if row is None:
                raw = self.reader.read(
                    self.filename,
                    self.offset + token * self.row_bytes,
                    self.row_bytes,
                    "cpu",
                )
                row = raw.view(self.dtype).view(self.width).clone()
                if len(self.rows) >= 4096:
                    self.rows.pop(next(iter(self.rows)))
                self.rows[token] = row
            result.append(row)
        x = torch.stack(result).reshape(*ids.shape, self.width)
        if self.module.multiplier != 1.0:
            x *= self.module.multiplier
        x = x.to(out_dtype or self.module.out_dtype or x.dtype)
        if self.module.normalize:
            x *= self.width**0.5
        return x

    def unload(self):
        self.rows.clear()
        self.module.device = None
