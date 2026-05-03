#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct norm_abs_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(::precise::abs(x));
  }

  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline float operator()(const T x) {
    const auto abs_2 = ::precise::abs(float2(x));
    return c10::metal::hypot(abs_2.x, abs_2.y);
  }
};

// `reduction_idx` is the index of a particular batch of input elements that all
// get reduced to one output element. `reduction_element_idx` is the index of
// just one input element within its batch.
static uint32_t get_input_offset(
    uint32_t reduction_element_idx,
    uint32_t reduction_idx,
    constant NormParams<>& params) {
  uint32_t input_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto input_dim_size = params.input_sizes[dim];
    auto output_dim_size = params.output_sizes[dim];

    // If the the input and output have the same size for this dim, then this
    // dim is not being reduced, so we index by `reduction_idx`
    if (input_dim_size == output_dim_size) {
      auto index_in_dim = reduction_idx % input_dim_size;
      reduction_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];

      // Otherwise, this dim is being reduced, so we index by
      // `reduction_element_idx`
    } else {
      auto index_in_dim = reduction_element_idx % input_dim_size;
      reduction_element_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];
    }
  }
  return input_offset;
}

// In this kernel, each threadgroup is responsible for calculating one element
// of the output.
// TI - dtype of the input tensor.
// TO - dtype of the output tensor.
template <typename TI, typename TO>
kernel void norm(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = opmath_t<TO>;
  TA output_val = 0;
  const auto p = static_cast<TA>(params.p);

  if (p == INFINITY) {
    output_val = -INFINITY;
  } else if (p == -INFINITY) {
    output_val = INFINITY;
  }

  // First, all the input elements assigned to the threadgroup are divided
  // between all the threads in the threadgroup, and each thread reduces those
  // elements down to one partial `output_val`.
  for (uint32_t reduction_element_idx = tid;
       reduction_element_idx < params.reduction_size;
       reduction_element_idx += tptg) {
    auto input_elem =
        input[get_input_offset(reduction_element_idx, tgid, params)];
    auto input_abs = static_cast<TA>(norm_abs_functor()(input_elem));

    if (p == INFINITY) {
      output_val = max(input_abs, output_val);

    } else if (p == -INFINITY) {
      output_val = min(input_abs, output_val);

    } else if (p == 0) {
      output_val += (input_abs == 0) ? 0 : 1;

    } else {
      output_val += static_cast<TA>(::precise::pow(input_abs, p));
    }
  }

  // Next, all the threads in a threadgroup reduce their `output_val`s together
  // with a series of SIMD group reductions.
  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    if (p == INFINITY) {
      output_val = simd_max(output_val);
    } else if (p == -INFINITY) {
      output_val = simd_min(output_val);
    } else {
      output_val = simd_sum(output_val);
    }

    threads_remaining = ceil_div(threads_remaining, simdgroup_size);

    if (threads_remaining > 1) {
      // One thread from each SIMD group writes to a shared buffer
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // The remaining threads each read one of the partial outputs from the
      // shared buffer
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }

  // Finally, one thread in the threadgroup writes the final output
  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];

      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }

    if (p != 0 && p != 1 && p != INFINITY && p != -INFINITY) {
      output_val = static_cast<TA>(::precise::pow(output_val, 1 / p));
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

#define REGISTER_NORM(TI, TO)                               \
  template [[host_name("norm_" #TI "_" #TO)]]               \
  kernel void norm<TI, TO>(                                 \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

REGISTER_NORM(float, float);
REGISTER_NORM(half, half);
REGISTER_NORM(bfloat, bfloat);
REGISTER_NORM(float2, float);
REGISTER_NORM(half2, half);

#include <c10/metal/reduction_utils.h>

// Load modes for sum_reduction: identity (sum), nan-to-zero (nansum),
// or nonzero-as-one (count_nonzero).
enum LoadMode : uint {
  LOAD_IDENTITY = 0,
  LOAD_NAN_TO_ZERO = 1,
  LOAD_NONZERO = 2
};

template <typename T, ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline bool load_is_nonzero(T v) {
  return v != T(0);
}

template <typename T, ::metal::enable_if_t<is_complex_v<T>, bool> = true>
inline bool load_is_nonzero(T v) {
  return v.x != 0 || v.y != 0;
}

// Load helper: cast to opmath_t, optionally replacing NaN with zero,
// or map nonzero to 1 for count_nonzero semantics.
template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_IDENTITY, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  return static_cast<opmath_t<TI>>(v);
}

template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_NAN_TO_ZERO, bool> = true>
inline opmath_t<TI> load_val(TI v) {
  auto r = static_cast<opmath_t<TI>>(v);
  if (::metal::isnan(static_cast<float>(r)))
    r = 0;
  return r;
}

// LOAD_NONZERO returns uint: MPS tensor numel fits in uint32, so per-TG
// (and per-output-element) non-zero counts cannot overflow. This lets
// count_nonzero accumulate in 32-bit integer instead of 64-bit, which is a
// meaningful speedup for small inputs (especially bool) where compute
// overhead dominates. The final cast back to long happens at the output
// store in the kernel.
template <
    LoadMode MODE,
    typename TI,
    ::metal::enable_if_t<MODE == LOAD_NONZERO, bool> = true>
inline uint load_val(TI v) {
  return load_is_nonzero(v) ? 1u : 0u;
}

// Sum reduction kernel with multiple independent accumulation chains (ILP).
// Each thread maintains NCHAINS independent accumulators to hide ALU latency
// and keep the memory pipeline saturated.
//
// Two internal paths selected per-threadgroup (not per-element):
//   - Single reduced dim (or full reduction): compute input_base + k * stride
//     once per TG, then direct indexing — no per-element dim loop.
//   - Multiple reduced dims: fall back to get_input_offset per element.
// MODE: LOAD_IDENTITY (sum), LOAD_NAN_TO_ZERO (nansum),
// LOAD_NONZERO (count_nonzero — contributes 1 per nonzero element).
// The compiler eliminates dead branches per instantiation.
template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;

  // Compute input_base (once per TG) and detect reduction pattern.
  // For single reduced dim: input_base + k * reduction_stride gives
  // the k-th reduction element — no per-element dim loop needed.
  uint32_t input_base = 0;
  uint32_t reduction_stride = 1;
  uint32_t num_reduced_dims = 0;
  {
    uint32_t out_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      if (params.input_sizes[dim] != params.output_sizes[dim]) {
        num_reduced_dims++;
        reduction_stride = params.input_strides[dim];
      } else {
        auto idx = out_idx % params.output_sizes[dim];
        out_idx /= params.output_sizes[dim];
        input_base += idx * params.input_strides[dim];
      }
    }
  }

  // Load helper: cast to accumulator type, optionally replacing NaN with zero

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) {
    acc[j] = 0;
  }

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    // Fast path: direct indexing with base + k * reduction_stride
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] +=
            load_val<MODE>(input[input_base + (base + j) * reduction_stride]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<MODE>(input[input_base + idx * reduction_stride]);
    }
  } else {
    // Generic path: per-element strided offset for multi-dim reductions
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] +=
            load_val<MODE>(input[get_input_offset(base + j, tgid, params)]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] +=
          load_val<MODE>(input[get_input_offset(idx, tgid, params)]);
    }
  }

  // Collapse chains into a single value
  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) {
    output_val += acc[j];
  }

  // SIMD + threadgroup tree reduction
  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    output_val = c10::metal::simd_sum(output_val);
    threads_remaining = ceil_div(threads_remaining, simdgroup_size);

    if (threads_remaining > 1) {
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }

  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }
    // params.p > 0 means "divide the accumulator by p before casting"
    // (used by mean to keep the division in opmath_t precision so the
    // fp32 accumulation isn't lost when TO is fp16/bf16/half2).
    if (params.p > 0) {
      output_val /= static_cast<TA>(params.p);
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

// Specialized kernel for reducing a non-innermost dim of a contiguous 2D
// tensor. Each thread handles one column, iterating over all rows with
// coalesced reads. Multiple row-workers per threadgroup reduce via shared
// memory. This avoids the strided-access penalty of the generic kernel for
// dim=0.
//
// Grid: (ceil(N/TG_X), 1) threadgroups, each (TG_X, TG_Y) threads.
// TG_X threads cover adjacent columns (coalesced), TG_Y threads split rows.
template <
    typename TI,
    typename TO,
    uint TG_X = 32,
    uint TG_Y = 32,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]], // [M, N, output_stride]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N)
    return;

  // Split rows among TG_Y workers
  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  // Multiple accumulation chains for ILP
  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 0;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<MODE>(input[(row + j) * N + col]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] += load_val<MODE>(input[row * N + col]);
  }

  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    sum += acc[j];

  // Reduce across TG_Y row-workers via shared memory
  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = TG_Y / 2; stride > 0; stride >>= 1) {
    if (tid_tg.y < stride)
      shmem[tid_tg.y][tid_tg.x] += shmem[tid_tg.y + stride][tid_tg.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    TA final_val = shmem[0][tid_tg.x];
    if (divisor > 0) {
      final_val /= static_cast<TA>(divisor);
    }
    output[col * out_stride] = static_cast<TO>(final_val);
  }
}

#define REGISTER_SUM_OUTER_IMPL(TI, TO, PREFIX, MODE)                 \
  template [[host_name(PREFIX "reduction_outer_" #TI "_" #TO)]]       \
  kernel void sum_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS, MODE>( \
      constant TI * input [[buffer(0)]],                              \
      device TO * output [[buffer(1)]],                               \
      constant uint3 & sizes [[buffer(2)]],                           \
      constant float& divisor [[buffer(3)]],                          \
      uint2 tid_tg [[thread_position_in_threadgroup]],                \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_SUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM_OUTER(TI, TO) \
  REGISTER_SUM_OUTER_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO_OUTER(TI) \
  REGISTER_SUM_OUTER_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

REGISTER_SUM_OUTER(float, float);
REGISTER_SUM_OUTER(half, half);
REGISTER_SUM_OUTER(half, float);
REGISTER_SUM_OUTER(bfloat, bfloat);
REGISTER_SUM_OUTER(bfloat, float);
REGISTER_SUM_OUTER(int, int);
REGISTER_SUM_OUTER(int, long);
REGISTER_SUM_OUTER(long, long);
REGISTER_SUM_OUTER(short, short);
REGISTER_SUM_OUTER(short, long);
REGISTER_SUM_OUTER(char, char);
REGISTER_SUM_OUTER(char, long);
REGISTER_SUM_OUTER(uchar, uchar);
REGISTER_SUM_OUTER(uchar, long);
REGISTER_SUM_OUTER(bool, long);
REGISTER_SUM_OUTER(bool, int);
REGISTER_SUM_OUTER(float2, float2);
REGISTER_SUM_OUTER(half2, half2);

// Specialized kernel for reducing the innermost dim of a contiguous tensor.
// Input [M, N] -> output [M], each SIMD group reduces one row of N elements.
// Multiple SIMD groups per TG handle different rows for occupancy.
// No shared memory needed — simd_sum suffices for intra-row reduction.
template <
    typename TI,
    typename TO,
    uint NCHAINS = SUM_NCHAINS,
    LoadMode MODE = LOAD_IDENTITY>
kernel void sum_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], // [M, N]
    constant float& divisor [[buffer(3)]], // >0 divides accumulator before cast
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = ::metal::conditional_t<MODE == LOAD_NONZERO, uint, opmath_t<TO>>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  // Each SIMD group handles a different row
  uint row = tgid * num_simd_groups + simdgroup_id;
  if (row >= M)
    return;

  constant TI* row_ptr = input + row * N;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++)
    acc[j] = 0;

  // Each of 32 lanes reads elements at stride 32, NCHAINS at a time.
  // Align down to full blocks of stride = 32 * NCHAINS elements.
  const uint stride = 32 * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = simd_lane_id * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] += load_val<MODE>(row_ptr[base + j]);
    }
  }
  // Tail: remaining elements after last full block, one per lane
  for (uint i = aligned_N + simd_lane_id; i < N; i += 32) {
    acc[0] += load_val<MODE>(row_ptr[i]);
  }

  TA sum = acc[0];
  for (uint j = 1; j < NCHAINS; j++)
    sum += acc[j];

  sum = c10::metal::simd_sum(sum);

  if (simd_lane_id == 0) {
    if (divisor > 0) {
      sum /= static_cast<TA>(divisor);
    }
    output[row] = static_cast<TO>(sum);
  }
}

#define REGISTER_SUM_INNER_IMPL(TI, TO, PREFIX, MODE)           \
  template [[host_name(PREFIX "reduction_inner_" #TI "_" #TO)]] \
  kernel void sum_reduction_inner<TI, TO, SUM_NCHAINS, MODE>(   \
      constant TI * input [[buffer(0)]],                        \
      device TO * output [[buffer(1)]],                         \
      constant uint2 & sizes [[buffer(2)]],                     \
      constant float& divisor [[buffer(3)]],                    \
      uint tptg [[threads_per_threadgroup]],                    \
      uint tgid [[threadgroup_position_in_grid]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],          \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM_INNER(TI, TO) \
  REGISTER_SUM_INNER_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO_INNER(TI) \
  REGISTER_SUM_INNER_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

REGISTER_SUM_INNER(float, float);
REGISTER_SUM_INNER(half, half);
REGISTER_SUM_INNER(half, float);
REGISTER_SUM_INNER(bfloat, bfloat);
REGISTER_SUM_INNER(bfloat, float);
REGISTER_SUM_INNER(int, int);
REGISTER_SUM_INNER(int, long);
REGISTER_SUM_INNER(long, long);
REGISTER_SUM_INNER(short, short);
REGISTER_SUM_INNER(short, long);
REGISTER_SUM_INNER(char, char);
REGISTER_SUM_INNER(char, long);
REGISTER_SUM_INNER(uchar, uchar);
REGISTER_SUM_INNER(uchar, long);
REGISTER_SUM_INNER(bool, long);
REGISTER_SUM_INNER(bool, int);
REGISTER_SUM_INNER(float2, float2);
REGISTER_SUM_INNER(half2, half2);

#define REGISTER_SUM_IMPL(TI, TO, PREFIX, MODE)             \
  template [[host_name(PREFIX "reduction_" #TI "_" #TO)]]   \
  kernel void sum_reduction<TI, TO, SUM_NCHAINS, MODE>(     \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

#define REGISTER_SUM(TI, TO) REGISTER_SUM_IMPL(TI, TO, "sum_", LOAD_IDENTITY)
#define REGISTER_NANSUM(TI, TO) \
  REGISTER_SUM_IMPL(TI, TO, "nansum_", LOAD_NAN_TO_ZERO)
#define REGISTER_COUNT_NONZERO(TI) \
  REGISTER_SUM_IMPL(TI, long, "count_nonzero_", LOAD_NONZERO)

REGISTER_SUM(float, float);
REGISTER_SUM(float, half);
REGISTER_SUM(float, bfloat);
REGISTER_SUM(half, half);
REGISTER_SUM(half, float);
REGISTER_SUM(bfloat, bfloat);
REGISTER_SUM(bfloat, float);
REGISTER_SUM(int, int);
REGISTER_SUM(int, long);
REGISTER_SUM(long, long);
REGISTER_SUM(short, short);
REGISTER_SUM(short, long);
REGISTER_SUM(char, char);
REGISTER_SUM(char, long);
REGISTER_SUM(uchar, uchar);
REGISTER_SUM(uchar, long);
REGISTER_SUM(bool, long);
REGISTER_SUM(bool, int);
REGISTER_SUM(float2, float2);
REGISTER_SUM(half2, half2);

// nansum variants (floating-point only — integers can't have NaN)
REGISTER_NANSUM(float, float);
REGISTER_NANSUM(half, half);
REGISTER_NANSUM(half, float);
REGISTER_NANSUM(bfloat, bfloat);
REGISTER_NANSUM(bfloat, float);

REGISTER_NANSUM_OUTER(float, float);
REGISTER_NANSUM_OUTER(half, half);
REGISTER_NANSUM_OUTER(half, float);
REGISTER_NANSUM_OUTER(bfloat, bfloat);
REGISTER_NANSUM_OUTER(bfloat, float);

REGISTER_NANSUM_INNER(float, float);
REGISTER_NANSUM_INNER(half, half);
REGISTER_NANSUM_INNER(half, float);
REGISTER_NANSUM_INNER(bfloat, bfloat);
REGISTER_NANSUM_INNER(bfloat, float);

// count_nonzero: output is always long; reuses sum-reduction machinery
// with LOAD_NONZERO mode (1 per nonzero element, 0 otherwise).
REGISTER_COUNT_NONZERO(float);
REGISTER_COUNT_NONZERO(half);
REGISTER_COUNT_NONZERO(bfloat);
REGISTER_COUNT_NONZERO(long);
REGISTER_COUNT_NONZERO(int);
REGISTER_COUNT_NONZERO(short);
REGISTER_COUNT_NONZERO(char);
REGISTER_COUNT_NONZERO(uchar);
REGISTER_COUNT_NONZERO(bool);
REGISTER_COUNT_NONZERO(float2);
REGISTER_COUNT_NONZERO(half2);

REGISTER_COUNT_NONZERO_OUTER(float);
REGISTER_COUNT_NONZERO_OUTER(half);
REGISTER_COUNT_NONZERO_OUTER(bfloat);
REGISTER_COUNT_NONZERO_OUTER(long);
REGISTER_COUNT_NONZERO_OUTER(int);
REGISTER_COUNT_NONZERO_OUTER(short);
REGISTER_COUNT_NONZERO_OUTER(char);
REGISTER_COUNT_NONZERO_OUTER(uchar);
REGISTER_COUNT_NONZERO_OUTER(bool);
REGISTER_COUNT_NONZERO_OUTER(float2);
REGISTER_COUNT_NONZERO_OUTER(half2);

REGISTER_COUNT_NONZERO_INNER(float);
REGISTER_COUNT_NONZERO_INNER(half);
REGISTER_COUNT_NONZERO_INNER(bfloat);
REGISTER_COUNT_NONZERO_INNER(long);
REGISTER_COUNT_NONZERO_INNER(int);
REGISTER_COUNT_NONZERO_INNER(short);
REGISTER_COUNT_NONZERO_INNER(char);
REGISTER_COUNT_NONZERO_INNER(uchar);
REGISTER_COUNT_NONZERO_INNER(bool);
REGISTER_COUNT_NONZERO_INNER(float2);
REGISTER_COUNT_NONZERO_INNER(half2);

// ============================================================================
// Product reduction kernels
// ============================================================================

template <typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = opmath_t<TO>;

  uint32_t input_base = 0;
  uint32_t reduction_stride = 1;
  uint32_t num_reduced_dims = 0;
  {
    uint32_t out_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      if (params.input_sizes[dim] != params.output_sizes[dim]) {
        num_reduced_dims++;
        reduction_stride = params.input_strides[dim];
      } else {
        auto idx = out_idx % params.output_sizes[dim];
        out_idx /= params.output_sizes[dim];
        input_base += idx * params.input_strides[dim];
      }
    }
  }

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) acc[j] = 1;

  const uint32_t rsize = params.reduction_size;
  const uint32_t stride = tptg * NCHAINS;
  uint32_t base = tid * NCHAINS;

  if (num_reduced_dims <= 1) {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] *= static_cast<TA>(input[input_base + (base + j) * reduction_stride]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] *= static_cast<TA>(input[input_base + idx * reduction_stride]);
    }
  } else {
    for (; base + NCHAINS <= rsize; base += stride) {
      for (uint j = 0; j < NCHAINS; j++) {
        acc[j] *= static_cast<TA>(input[get_input_offset(base + j, tgid, params)]);
      }
    }
    for (uint32_t idx = base; idx < rsize; idx++) {
      acc[idx % NCHAINS] *= static_cast<TA>(input[get_input_offset(idx, tgid, params)]);
    }
  }

  TA output_val = acc[0];
  for (uint j = 1; j < NCHAINS; j++) output_val *= acc[j];

  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    output_val = c10::metal::simd_prod(output_val);
    threads_remaining = ceil_div(threads_remaining, simdgroup_size);
    if (threads_remaining > 1) {
      if (simd_lane_id == 0) shared_outputs[simdgroup_id] = output_val;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        output_val = static_cast<TA>(1);
      }
    }
  }

  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

template <typename TI, typename TO, uint TG_X = 32, uint TG_Y = 32, uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint3& sizes [[buffer(2)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) return;

  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) acc[j] = 1;

  uint row = row_start;
  for (; row + NCHAINS <= row_end; row += NCHAINS) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] *= static_cast<TA>(input[(row + j) * N + col]);
    }
  }
  for (; row < row_end; row++) {
    acc[row % NCHAINS] *= static_cast<TA>(input[row * N + col]);
  }

  TA prod = acc[0];
  for (uint j = 1; j < NCHAINS; j++) prod *= acc[j];

  threadgroup TA shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = prod;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = TG_Y / 2; s > 0; s >>= 1) {
    if (tid_tg.y < s)
      shmem[tid_tg.y][tid_tg.x] *= shmem[tid_tg.y + s][tid_tg.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    output[col * out_stride] = static_cast<TO>(shmem[0][tid_tg.x]);
  }
}

template <typename TI, typename TO, uint NCHAINS = SUM_NCHAINS>
kernel void prod_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  using TA = opmath_t<TO>;
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  uint row = tgid;
  if (row >= M) return;

  constant TI* row_ptr = input + row * N;

  metal::array<TA, NCHAINS> acc;
  for (uint j = 0; j < NCHAINS; j++) acc[j] = 1;

  const uint stride = tptg * NCHAINS;
  const uint aligned_N = (N / stride) * stride;
  uint base = tid * NCHAINS;
  for (; base < aligned_N; base += stride) {
    for (uint j = 0; j < NCHAINS; j++) {
      acc[j] *= static_cast<TA>(row_ptr[base + j]);
    }
  }
  for (uint i = aligned_N + tid; i < N; i += tptg) {
    acc[0] *= static_cast<TA>(row_ptr[i]);
  }

  TA prod = acc[0];
  for (uint j = 1; j < NCHAINS; j++) prod *= acc[j];
  prod = c10::metal::simd_prod(prod);

  threadgroup TA shared_prod[32];
  if (simd_lane_id == 0) {
    shared_prod[simdgroup_id] = prod;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_id == 0) {
    prod = (simd_lane_id < num_simd_groups)
        ? shared_prod[simd_lane_id] : static_cast<TA>(1);
    prod = c10::metal::simd_prod(prod);

    if (simd_lane_id == 0) {
      output[row] = static_cast<TO>(prod);
    }
  }
}

#define REGISTER_PROD(TI, TO) \
  template [[host_name("prod_reduction_" #TI "_" #TO)]] \
  kernel void prod_reduction<TI, TO, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      constant NormParams<> & params [[buffer(2)]], \
      uint tid [[thread_position_in_threadgroup]], \
      uint tptg [[threads_per_threadgroup]], \
      uint tgid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

#define REGISTER_PROD_OUTER(TI, TO) \
  template [[host_name("prod_reduction_outer_" #TI "_" #TO)]] \
  kernel void prod_reduction_outer<TI, TO, 32, 32, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      constant uint3 & sizes [[buffer(2)]], \
      uint2 tid_tg [[thread_position_in_threadgroup]], \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_PROD_INNER(TI, TO) \
  template [[host_name("prod_reduction_inner_" #TI "_" #TO)]] \
  kernel void prod_reduction_inner<TI, TO, SUM_NCHAINS>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      constant uint2 & sizes [[buffer(2)]], \
      uint tid [[thread_index_in_threadgroup]], \
      uint tptg [[threads_per_threadgroup]], \
      uint tgid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_PROD(float, float);
REGISTER_PROD(half, half);
REGISTER_PROD(half, float);
REGISTER_PROD(bfloat, bfloat);
REGISTER_PROD(bfloat, float);
REGISTER_PROD(int, int);
REGISTER_PROD(int, long);
REGISTER_PROD(long, long);
REGISTER_PROD(short, short);
REGISTER_PROD(short, long);
REGISTER_PROD(char, char);
REGISTER_PROD(char, long);
REGISTER_PROD(uchar, uchar);
REGISTER_PROD(uchar, long);
REGISTER_PROD(bool, long);
REGISTER_PROD(bool, int);

REGISTER_PROD_OUTER(float, float);
REGISTER_PROD_OUTER(half, half);
REGISTER_PROD_OUTER(half, float);
REGISTER_PROD_OUTER(bfloat, bfloat);
REGISTER_PROD_OUTER(bfloat, float);
REGISTER_PROD_OUTER(int, int);
REGISTER_PROD_OUTER(int, long);
REGISTER_PROD_OUTER(long, long);
REGISTER_PROD_OUTER(short, short);
REGISTER_PROD_OUTER(short, long);
REGISTER_PROD_OUTER(char, char);
REGISTER_PROD_OUTER(char, long);
REGISTER_PROD_OUTER(uchar, uchar);
REGISTER_PROD_OUTER(uchar, long);
REGISTER_PROD_OUTER(bool, long);
REGISTER_PROD_OUTER(bool, int);

REGISTER_PROD_INNER(float, float);
REGISTER_PROD_INNER(half, half);
REGISTER_PROD_INNER(half, float);
REGISTER_PROD_INNER(bfloat, bfloat);
REGISTER_PROD_INNER(bfloat, float);
REGISTER_PROD_INNER(int, int);
REGISTER_PROD_INNER(int, long);
REGISTER_PROD_INNER(long, long);
REGISTER_PROD_INNER(short, short);
REGISTER_PROD_INNER(short, long);
REGISTER_PROD_INNER(char, char);
REGISTER_PROD_INNER(char, long);
REGISTER_PROD_INNER(uchar, uchar);
REGISTER_PROD_INNER(uchar, long);
REGISTER_PROD_INNER(bool, long);
REGISTER_PROD_INNER(bool, int);

// ============================================================================
// Welford reduction kernels (var / std / var_mean / std_mean)
// ============================================================================

inline float3 simd_welford_combine(float3 stats) {
  for (ushort i = simdgroup_size / 2; i > 0; i /= 2) {
    float3 other;
    other.x = ::metal::simd_shuffle_and_fill_down(stats.x, 0.0f, i);
    other.y = ::metal::simd_shuffle_and_fill_down(stats.y, 0.0f, i);
    other.z = ::metal::simd_shuffle_and_fill_down(stats.z, 0.0f, i);
    stats = welford_combine(stats, other);
  }
  return stats;
}

struct WelfordConfig {
  float correction;
  float compute_std;
  float write_mean;
};

template <typename TI, typename TO>
kernel void welford_reduction(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant NormParams<>& params [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size_val [[threads_per_simdgroup]]) {

  uint32_t input_base = 0;
  uint32_t reduction_stride = 1;
  uint32_t num_reduced_dims = 0;
  {
    uint32_t out_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      if (params.input_sizes[dim] != params.output_sizes[dim]) {
        num_reduced_dims++;
        reduction_stride = params.input_strides[dim];
      } else {
        auto idx = out_idx % params.output_sizes[dim];
        out_idx /= params.output_sizes[dim];
        input_base += idx * params.input_strides[dim];
      }
    }
  }

  float w_mean = 0, w_m2 = 0, w_count = 0;
  const uint32_t rsize = params.reduction_size;

  if (num_reduced_dims <= 1) {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      float val = static_cast<float>(input[input_base + k * reduction_stride]);
      w_count += 1;
      float delta = val - w_mean;
      w_mean += delta / w_count;
      w_m2 += delta * (val - w_mean);
    }
  } else {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      float val = static_cast<float>(input[get_input_offset(k, tgid, params)]);
      w_count += 1;
      float delta = val - w_mean;
      w_mean += delta / w_count;
      w_m2 += delta * (val - w_mean);
    }
  }

  float3 stats = simd_welford_combine(float3(w_mean, w_m2, w_count));

  threadgroup float3 shared_stats[MAX_THREADGROUP_SIZE / 32];
  uint num_simdgroups = ceil_div(tptg, simdgroup_size_val);

  if (num_simdgroups > 1) {
    if (simd_lane_id == 0) {
      shared_stats[simdgroup_id] = stats;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_simdgroups) {
      stats = shared_stats[tid];
    } else {
      stats = float3(0, 0, 0);
    }
    stats = simd_welford_combine(stats);
  }

  if (tid == 0) {
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : NAN;

    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }

    output[output_offset] = static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[output_offset] = static_cast<TO>(stats.x);
    }
  }
}

template <typename TI, typename TO, uint TG_X = 32, uint TG_Y = 32>
kernel void welford_reduction_outer(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint3& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint2 tid_tg [[thread_position_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint out_stride = sizes.z;

  uint col = tg_pos.x * TG_X + tid_tg.x;
  if (col >= N) return;

  uint rows_per_y = ceil_div(M, TG_Y);
  uint row_start = tid_tg.y * rows_per_y;
  uint row_end = min(row_start + rows_per_y, M);

  float w_mean = 0, w_m2 = 0, w_count = 0;
  for (uint row = row_start; row < row_end; row++) {
    float val = static_cast<float>(input[row * N + col]);
    w_count += 1;
    float delta = val - w_mean;
    w_mean += delta / w_count;
    w_m2 += delta * (val - w_mean);
  }

  threadgroup float3 shmem[TG_Y][TG_X];
  shmem[tid_tg.y][tid_tg.x] = float3(w_mean, w_m2, w_count);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = TG_Y / 2; s > 0; s >>= 1) {
    if (tid_tg.y < s)
      shmem[tid_tg.y][tid_tg.x] = welford_combine(shmem[tid_tg.y][tid_tg.x], shmem[tid_tg.y + s][tid_tg.x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid_tg.y == 0) {
    float3 stats = shmem[0][tid_tg.x];
    float denom = max(stats.z - config.correction, 0.0f);
    float var = (denom > 0) ? stats.y / denom : NAN;
    output[col * out_stride] = static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
    if (config.write_mean > 0) {
      output_mean[col * out_stride] = static_cast<TO>(stats.x);
    }
  }
}

template <typename TI, typename TO>
kernel void welford_reduction_inner(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    device TO* output_mean [[buffer(2)]],
    constant uint2& sizes [[buffer(3)]],
    constant WelfordConfig& config [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  const uint M = sizes.x;
  const uint N = sizes.y;
  const uint num_simd_groups = tptg / 32;

  uint row = tgid;
  if (row >= M) return;

  constant TI* row_ptr = input + row * N;

  float w_mean = 0, w_m2 = 0, w_count = 0;
  for (uint i = tid; i < N; i += tptg) {
    float val = static_cast<float>(row_ptr[i]);
    w_count += 1;
    float delta = val - w_mean;
    w_mean += delta / w_count;
    w_m2 += delta * (val - w_mean);
  }

  float3 stats = simd_welford_combine(float3(w_mean, w_m2, w_count));

  threadgroup float3 shared_stats[32];
  if (simd_lane_id == 0) {
    shared_stats[simdgroup_id] = stats;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_id == 0) {
    stats = (simd_lane_id < num_simd_groups)
        ? shared_stats[simd_lane_id] : float3(0, 0, 0);
    stats = simd_welford_combine(stats);

    if (simd_lane_id == 0) {
      float denom = max(stats.z - config.correction, 0.0f);
      float var = (denom > 0) ? stats.y / denom : NAN;
      output[row] = static_cast<TO>(config.compute_std > 0 ? ::precise::sqrt(var) : var);
      if (config.write_mean > 0) {
        output_mean[row] = static_cast<TO>(stats.x);
      }
    }
  }
}

#define REGISTER_WELFORD(TI, TO) \
  template [[host_name("welford_" #TI "_" #TO)]] \
  kernel void welford_reduction<TI, TO>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      device TO * output_mean [[buffer(2)]], \
      constant NormParams<> & params [[buffer(3)]], \
      constant WelfordConfig & config [[buffer(4)]], \
      uint tid [[thread_position_in_threadgroup]], \
      uint tptg [[threads_per_threadgroup]], \
      uint tgid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size_val [[threads_per_simdgroup]]);

#define REGISTER_WELFORD_OUTER(TI, TO) \
  template [[host_name("welford_outer_" #TI "_" #TO)]] \
  kernel void welford_reduction_outer<TI, TO, 32, 32>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      device TO * output_mean [[buffer(2)]], \
      constant uint3 & sizes [[buffer(3)]], \
      constant WelfordConfig & config [[buffer(4)]], \
      uint2 tid_tg [[thread_position_in_threadgroup]], \
      uint2 tg_pos [[threadgroup_position_in_grid]]);

#define REGISTER_WELFORD_INNER(TI, TO) \
  template [[host_name("welford_inner_" #TI "_" #TO)]] \
  kernel void welford_reduction_inner<TI, TO>( \
      constant TI * input [[buffer(0)]], \
      device TO * output [[buffer(1)]], \
      device TO * output_mean [[buffer(2)]], \
      constant uint2 & sizes [[buffer(3)]], \
      constant WelfordConfig & config [[buffer(4)]], \
      uint tid [[thread_index_in_threadgroup]], \
      uint tptg [[threads_per_threadgroup]], \
      uint tgid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

REGISTER_WELFORD(float, float);
REGISTER_WELFORD(half, half);
REGISTER_WELFORD(half, float);
REGISTER_WELFORD(bfloat, bfloat);
REGISTER_WELFORD(bfloat, float);

REGISTER_WELFORD_OUTER(float, float);
REGISTER_WELFORD_OUTER(half, half);
REGISTER_WELFORD_OUTER(half, float);
REGISTER_WELFORD_OUTER(bfloat, bfloat);
REGISTER_WELFORD_OUTER(bfloat, float);

REGISTER_WELFORD_INNER(float, float);
REGISTER_WELFORD_INNER(half, half);
REGISTER_WELFORD_INNER(half, float);
REGISTER_WELFORD_INNER(bfloat, bfloat);
REGISTER_WELFORD_INNER(bfloat, float);

// ============================================================================
// Arg-reduce kernels (argmax / argmin / max / min with indices)
// ============================================================================

template <typename TI, bool IS_MAX>
kernel void argreduce(
    constant TI* input [[buffer(0)]],
    device long* output_indices [[buffer(1)]],
    device TI* output_values [[buffer(2)]],
    constant NormParams<>& params [[buffer(3)]],
    constant uchar& write_values [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size_val [[threads_per_simdgroup]]) {
  using TA = opmath_t<TI>;

  TA best_val;
  if (IS_MAX) {
    best_val = ::metal::numeric_limits<TA>::lowest();
  } else {
    best_val = ::metal::numeric_limits<TA>::max();
  }
  long best_idx = 0;

  uint32_t input_base = 0;
  uint32_t reduction_stride = 1;
  uint32_t num_reduced_dims = 0;
  {
    uint32_t out_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      if (params.input_sizes[dim] != params.output_sizes[dim]) {
        num_reduced_dims++;
        reduction_stride = params.input_strides[dim];
      } else {
        auto idx = out_idx % params.output_sizes[dim];
        out_idx /= params.output_sizes[dim];
        input_base += idx * params.input_strides[dim];
      }
    }
  }

  const uint32_t rsize = params.reduction_size;

  if (num_reduced_dims <= 1) {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      TA val = static_cast<TA>(input[input_base + k * reduction_stride]);
      bool better = IS_MAX ? (val > best_val) : (val < best_val);
      if (better || ::metal::isnan(static_cast<float>(val))) {
        best_val = val;
        best_idx = k;
      }
    }
  } else {
    for (uint32_t k = tid; k < rsize; k += tptg) {
      TA val = static_cast<TA>(input[get_input_offset(k, tgid, params)]);
      bool better = IS_MAX ? (val > best_val) : (val < best_val);
      if (better || ::metal::isnan(static_cast<float>(val))) {
        best_val = val;
        best_idx = k;
      }
    }
  }

  threadgroup TA arg_data[32];
  threadgroup long idx_data[32];

  long result_idx;
  if (IS_MAX) {
    result_idx = c10::metal::threadgroup_argmax(arg_data, idx_data, best_val, best_idx, tid, tptg);
  } else {
    result_idx = c10::metal::threadgroup_argmin(arg_data, idx_data, best_val, best_idx, tid, tptg);
  }

  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;
    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];
      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }
    output_indices[output_offset] = result_idx;
    if (write_values) {
      uint32_t val_input_offset;
      if (num_reduced_dims <= 1) {
        val_input_offset = input_base + result_idx * reduction_stride;
      } else {
        val_input_offset = get_input_offset(result_idx, tgid, params);
      }
      output_values[output_offset] = input[val_input_offset];
    }
  }
}

#define REGISTER_ARGREDUCE(TI, NAME, IS_MAX) \
  template [[host_name(NAME "_" #TI)]] \
  kernel void argreduce<TI, IS_MAX>( \
      constant TI * input [[buffer(0)]], \
      device long * output_indices [[buffer(1)]], \
      device TI * output_values [[buffer(2)]], \
      constant NormParams<> & params [[buffer(3)]], \
      constant uchar & write_values [[buffer(4)]], \
      uint tid [[thread_position_in_threadgroup]], \
      uint tptg [[threads_per_threadgroup]], \
      uint tgid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size_val [[threads_per_simdgroup]]);

REGISTER_ARGREDUCE(float, "argmax", true);
REGISTER_ARGREDUCE(float, "argmin", false);
REGISTER_ARGREDUCE(half, "argmax", true);
REGISTER_ARGREDUCE(half, "argmin", false);
REGISTER_ARGREDUCE(bfloat, "argmax", true);
REGISTER_ARGREDUCE(bfloat, "argmin", false);
REGISTER_ARGREDUCE(int, "argmax", true);
REGISTER_ARGREDUCE(int, "argmin", false);
REGISTER_ARGREDUCE(long, "argmax", true);
REGISTER_ARGREDUCE(long, "argmin", false);
REGISTER_ARGREDUCE(short, "argmax", true);
REGISTER_ARGREDUCE(short, "argmin", false);
REGISTER_ARGREDUCE(char, "argmax", true);
REGISTER_ARGREDUCE(char, "argmin", false);
REGISTER_ARGREDUCE(uchar, "argmax", true);
REGISTER_ARGREDUCE(uchar, "argmin", false);
