"""Framework-ready WY recompute with wrapper-owned beta layout conversion.

The public ``recompute_w_u_fwd`` retains
the original contiguous ``beta[B,T,H]`` interface and materializes a contiguous
head-major copy before launch. The kernel body is adapted from
``sglang.srt.layers.attention.fla.kda.recompute_w_u_fwd_kernel``.

Framework contract:

* ``k``: ``[B, T, H, K]``, contiguous.
* ``v``: ``[B, T, H, V]``, contiguous.
* public-wrapper ``beta``: ``[B, T, H]``, contiguous.
* kernel/direct-wrapper ``beta``: ``[B, H, T]``, contiguous.
* ``A``: ``[B, T, H, BT]``, contiguous.
* ``gk``: ``[B, T, H, K]``, contiguous FP32 chunk-local cumulative gate.
* varlen input is packed with physical ``B == 1`` and sequence boundaries in
  ``cu_seqlens``; ``beta`` covers the complete packed ``T`` axis.
"""

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from sgl_kernel_npu.fla.utils import exp2, prepare_chunk_indices


# Keep the delivered operator self-contained.  These are the small utilities
# that the original SGLang file imports from fla.op/fla.index.  Framework-side
# integration may replace them with the existing SGLang imports, but it does
# not need another file from this handoff package.
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
else:
    exp = tl.exp

cdiv = triton.cdiv


def _tensor_cache(fn):
    cache_entries = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for index, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                same_args = all(a is b for a, b in zip(args, last_args))
                same_kwargs = all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
                if same_args and same_kwargs:
                    del cache_entries[index]
                    cache_entries.append((args, kwargs, last_result))
                    return last_result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries.pop(0)
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@_tensor_cache
def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int):
    """Build ``[sequence_id, chunk_id]`` metadata for packed varlen input."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lengths, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV})
        for BK in (64, 128)
        for BV in (64, 128)
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_head_major_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # T_TOTAL remains the physical packed-token extent even when the varlen
    # branch replaces T with the current logical sequence length.
    T_TOTAL = T
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # beta is physical [B,H,T_TOTAL].  Fixed mode selects the physical batch
    # through i_b.  Packed varlen requires B=1, and bos selects the logical
    # sequence inside the shared T_TOTAL axis.
    beta_bos = bos if IS_VARLEN else 0
    p_b = tl.make_block_ptr(
        beta + (i_b * H + i_h) * T_TOTAL + beta_bos,
        (T,),
        (1,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp(b_gk)
            tl.store(
                p_qg,
                b_qg.to(p_qg.dtype.element_ty),
                boundary_check=(0, 1),
            )
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k,
                other=0.0,
            )
            b_kg = b_k * exp(b_gn - b_gk)
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(
                p_kg,
                b_kg.to(p_kg.dtype.element_ty),
                boundary_check=(0, 1),
            )

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
        
def _validate_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    """Validate pointer-layout invariants without synchronizing the device."""
    if k.ndim != 4:
        raise ValueError(f"k must be [B,T,H,K], got shape={tuple(k.shape)}")
    B, T, H, K = k.shape
    if min(B, T, H, K) <= 0:
        raise ValueError(f"k dimensions must be positive, got shape={tuple(k.shape)}")
    if v.ndim != 4 or tuple(v.shape[:3]) != (B, T, H):
        raise ValueError(
            f"v must be [B,T,H,V] matching k; k={tuple(k.shape)}, "
            f"v={tuple(v.shape)}"
        )
    V = v.shape[-1]
    if V <= 0:
        raise ValueError(f"V must be positive, got V={V}")
    if tuple(beta.shape) != (B, H, T):
        raise ValueError(
            "beta must use the framework head-major contract [B,H,T]: "
            f"expected={(B, H, T)}, got={tuple(beta.shape)}"
        )
    if A.ndim != 4 or tuple(A.shape[:3]) != (B, T, H):
        raise ValueError(
            f"A must be [B,T,H,BT] matching k; k={tuple(k.shape)}, "
            f"A={tuple(A.shape)}"
        )
    BT = A.shape[-1]
    if BT != 64:
        raise ValueError(
            "this KDA recompute candidate is validated for chunk size BT=64; "
            f"got BT={BT}"
        )
    if gk is None:
        raise ValueError("gk is required by the current WY recompute kernel")
    if tuple(gk.shape) != (B, T, H, K):
        raise ValueError(
            f"gk must be [B,T,H,K]={tuple(k.shape)}, got={tuple(gk.shape)}"
        )

    tensors = {"k": k, "v": v, "beta": beta, "A": A, "gk": gk}
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be physically contiguous; "
                f"shape={tuple(tensor.shape)}, stride={tensor.stride()}"
            )
        if tensor.device != k.device:
            raise ValueError(
                f"all inputs must be on {k.device}; {name} is on {tensor.device}"
            )
    if not (k.dtype == v.dtype == A.dtype):
        raise ValueError(
            "k, v, and A must share the dot/output dtype; "
            f"got k={k.dtype}, v={v.dtype}, A={A.dtype}"
        )
    if gk.dtype != torch.float32:
        raise ValueError(f"gk must be FP32, got {gk.dtype}")
    if beta.dtype not in (k.dtype, torch.float32):
        raise ValueError(
            "beta must either match the data dtype or use production FP32; "
            f"data={k.dtype}, beta={beta.dtype}"
        )

    if cu_seqlens is None:
        if chunk_indices is not None:
            raise ValueError("chunk_indices requires cu_seqlens")
    else:
        if B != 1:
            raise ValueError(
                "packed varlen mode requires physical B=1; "
                f"got B={B}, cu_seqlens shape={tuple(cu_seqlens.shape)}"
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "cu_seqlens must be a 1D cumulative-boundary tensor with "
                f"at least two entries, got shape={tuple(cu_seqlens.shape)}"
            )
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}"
            )
        if cu_seqlens.device != k.device or not cu_seqlens.is_contiguous():
            raise ValueError(
                "cu_seqlens must be contiguous and on the same device as k; "
                f"device={cu_seqlens.device}, stride={cu_seqlens.stride()}"
            )
        if chunk_indices is not None:
            if (
                chunk_indices.ndim != 2
                or chunk_indices.shape[1] != 2
                or chunk_indices.dtype != cu_seqlens.dtype
                or chunk_indices.device != k.device
                or not chunk_indices.is_contiguous()
            ):
                raise ValueError(
                    "chunk_indices must be contiguous [num_chunks,2] on the "
                    f"same device as k, got shape={tuple(chunk_indices.shape)}, "
                    f"device={chunk_indices.device}, "
                    f"dtype={chunk_indices.dtype}, stride={chunk_indices.stride()}"
                )
    return B, T, H, K, V, BT


def recompute_w_u_fwd_head_major(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Run WY recompute using caller-owned contiguous ``beta[B,H,T]``.

    No beta view, transpose, allocation, or copy occurs in this wrapper.  ``q``
    is retained for call-site compatibility with the source KDA wrapper; the
    current path keeps ``STORE_QG=False`` and therefore does not read it.
    """
    B, T, H, K, V, BT = _validate_inputs(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    recompute_w_u_fwd_head_major_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STORE_QG=False,
        STORE_KG=True,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION="ieee",
    )
    return w, u, None, kg


def prepare_recompute_beta_head_major(
    beta: torch.Tensor,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Materialize contiguous ``beta[B,T,H]`` as physical ``[B,H,T]``."""
    if beta.ndim != 3:
        raise ValueError(f"beta must be [B,T,H], got shape={tuple(beta.shape)}")
    if expected_shape is not None and tuple(beta.shape) != expected_shape:
        raise ValueError(
            "beta[B,T,H] must match k's first three dimensions: "
            f"expected={expected_shape}, got={tuple(beta.shape)}"
        )
    if not beta.is_contiguous():
        raise ValueError(
            "framework beta[B,T,H] must be physically contiguous before the "
            f"wrapper conversion; stride={beta.stride()}"
        )
    return beta.permute(0, 2, 1).contiguous()


@triton.jit(do_not_specialize=["T"])
def _chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    H_LAYOUT_VK: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        if H_LAYOUT_VK:
            p_h = tl.make_block_ptr(
                h + (i_tg * H + i_h) * V * K,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            # CUDA keeps h as [V, K].
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
        else:
            p_h = tl.make_block_ptr(
                h + (i_tg * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            # The 0728 NPU producer stores h directly as [K, V].
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Keep the original BTH API and convert beta inside this wrapper."""
    if k.ndim != 4:
        raise ValueError(f"k must be [B,T,H,K], got shape={tuple(k.shape)}")
    beta_head_major = prepare_recompute_beta_head_major(
        beta,
        expected_shape=tuple(k.shape[:3]),
    )
    return recompute_w_u_fwd_head_major(
        k=k,
        v=v,
        beta=beta_head_major,
        A=A,
        q=q,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def chunk_gla_fwd_o_gk_npu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Consume Ascend's KxV chunk-state layout without a transpose."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = (
        triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    )
    grid = (triton.cdiv(V, 64), num_chunks, B * H)
    _chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=out,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=64,
        BV=64,
        IS_VARLEN=cu_seqlens is not None,
        H_LAYOUT_VK=False,
    )
    return out
