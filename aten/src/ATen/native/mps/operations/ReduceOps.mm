//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_forward_native.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/median.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReduceOps_metallib.h>
#endif

enum StdVarType { STANDARD_VARIANCE, STANDARD_DEVIATION };

enum MPSReductionType {
  MAX,
  MIN,
  AMAX,
  AMIN,
  PROD,
  MEAN,
  TRACE,
};

static void set_apparent_shapes(NSMutableArray<NSNumber*>*& apparent_out_shape,
                                NSMutableArray<NSNumber*>*& apparent_in_shape,
                                int64_t num_reduce_dims,
                                int64_t num_output_dims,
                                const IntArrayRef& input_shape,
                                NSMutableArray<NSNumber*>*& axes) {
  if (num_reduce_dims == 0) {
    /* Output shape becomes a one
     * Input shape becomes flattened
     * Because 0 reduce dims means all dims are reduced
     */
    apparent_in_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = c10::multiply_integers(input_shape);
    apparent_in_shape[0] = [NSNumber numberWithInt:num_in_elements];

    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_out_shape[0] = @1;
  } else {
    // num_output_dims in this case is number of input dims
    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_output_dims];
    for (const auto i : c10::irange(num_output_dims)) {
      int64_t current_input_dim = input_shape[i];

      // If the current dim is to be reduced
      bool is_reduce_dim = false;

      for (const auto j : c10::irange(num_reduce_dims)) {
        if (i == [axes[j] intValue]) {
          is_reduce_dim = true;
          break;
        }
      }

      apparent_out_shape[i] = is_reduce_dim ? @1 : [NSNumber numberWithInt:current_input_dim];
    }
  }
}

// Helper function to set the axes of reduction
static void set_axes(NSMutableArray<NSNumber*>*& axes,
                     int64_t num_reduce_dims,
                     OptionalIntArrayRef opt_dim,
                     int64_t num_input_dims) {
  if (num_reduce_dims == 0) {
    axes = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    axes[0] = @0;
  } else {
    TORCH_INTERNAL_ASSERT(opt_dim.has_value());
    IntArrayRef dim = opt_dim.value();
    axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_reduce_dims];
    for (const auto i : c10::irange(num_reduce_dims)) {
      axes[i] = [NSNumber numberWithInt:maybe_wrap_dim(dim[i], num_input_dims)];
    }
  }
}

// Helper function to prepare axes and tensor shapes
static void set_axes_and_shapes(const IntArrayRef& input_shape,
                                OptionalIntArrayRef opt_dims,
                                NSMutableArray<NSNumber*>*& axes,
                                NSMutableArray<NSNumber*>*& apparent_input_shape,
                                NSMutableArray<NSNumber*>*& apparent_output_shape,
                                NSMutableArray<NSNumber*>*& output_shape) {
  int64_t num_input_dims = input_shape.size();
  int64_t num_reduce_dims = opt_dims.has_value() ? opt_dims.value().size() : 0;
  int64_t num_output_dims;

  num_output_dims = num_reduce_dims == 0 ? 1 : num_input_dims;

  // Reduction axes
  set_axes(axes, num_reduce_dims, opt_dims, input_shape.size());

  // Shapes
  set_apparent_shapes(apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);

  // Squeeze dims for output shape
  output_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:0];
  for (const auto i : c10::irange(num_output_dims)) {
    if ([apparent_output_shape[i] longValue] != 1) {
      [output_shape addObject:apparent_output_shape[i]];
    }
  }
}

static void reduction_out_mps(const Tensor& input_t,
                              OptionalIntArrayRef opt_dim,
                              bool keepdim,
                              std::optional<ScalarType> dtype,
                              const Tensor& output_t,
                              MPSReductionType reduction_type,
                              const std::string& func_name) {
  // NS: TODO: get rid of all those shenanigans and just call reduction_op with view tensor
  bool canSqueezeLastDim = true;
  IntArrayRef input_shape = input_t.sizes();
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    for (const auto dim_val : dim) {
      auto wrap_dim = maybe_wrap_dim(dim_val, input_shape.size());
      // canSqueeze logic is broken when dim is negative, it introduces off-by-one-errors or crashes
      // See https://github.com/pytorch/pytorch/issues/136132#issuecomment-2354482608
      if (wrap_dim >= 4 || dim_val < 0) {
        canSqueezeLastDim = false;
      }
      TORCH_CHECK(
          wrap_dim < static_cast<decltype(wrap_dim)>(input_shape.size() == 0 ? input_t.numel() : input_shape.size()),
          func_name + ": reduction dim must be in the range of input shape")
    }
  }

  if (input_shape.size() >= 5 && canSqueezeLastDim) {
    for (const auto i : c10::irange(4, input_shape.size())) {
      if (input_shape[i] != 1) {
        canSqueezeLastDim = false;
      }
    }
  } else {
    canSqueezeLastDim = false;
  }

  MPSShape* mpsShape = getMPSShape(input_t);
  if (canSqueezeLastDim) {
    mpsShape = @[ @(input_shape[0]), @(input_shape[1]), @(input_shape[2]), @(input_shape[3]) ];
    input_shape = makeArrayRef(input_shape.begin(), input_shape.end() - (input_t.dim() - 4));
  }

  NSMutableArray<NSNumber*>* axes = nil;
  NSMutableArray<NSNumber*>* apparent_input_shape = nil;
  NSMutableArray<NSNumber*>* apparent_output_shape = nil;
  NSMutableArray<NSNumber*>* output_shape = nil;

  set_axes_and_shapes(input_shape, opt_dim, axes, apparent_input_shape, apparent_output_shape, output_shape);
  NSArray<NSNumber*>* wrappedAxes = getTensorAxes(input_shape, opt_dim);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    switch (reduction_type) {
      case MPSReductionType::PROD:
        output_t.fill_(1);
        break;
      case MPSReductionType::MEAN:
        output_t.fill_(std::numeric_limits<float>::quiet_NaN());
        break;
      case MPSReductionType::AMAX:
      case MPSReductionType::AMIN:
      case MPSReductionType::MAX:
      case MPSReductionType::MIN:
        TORCH_CHECK(opt_dim.has_value(), "Expected reduction dim to be specified for input.numel() == 0");
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected reduction type ", reduction_type);
        break;
    }
    return;
  }
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    std::string dtype_str = dtype.has_value() ? getMPSTypeString(dtype.value()) : "";
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    std::string key = func_name + ":" + std::string([ns_key UTF8String]) + ":" + getTensorsStringKey(input_t) + ":" +
        std::to_string(keepdim) + ":" + std::to_string(reduction_type) + ":" + getTensorsStringKey(output_t) + ":" +
        dtype_str;
    using CachedGraph = MPSUnaryCachedGraph;
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputScalarType = input_t.scalar_type();

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input_t), mpsShape);
      MPSGraphTensor* castInputTensor = inputTensor;
      MPSDataType inputCastType = MPSDataTypeInvalid;
      if (dtype.has_value() &&
          (dtype.value() == kFloat || dtype.value() == kHalf || dtype.value() == kInt || dtype.value() == kLong)) {
        inputCastType = getMPSDataType(dtype.value());
      } else if (inputScalarType != kInt && inputScalarType != kHalf && inputScalarType != kFloat &&
                 inputScalarType != kComplexFloat && inputScalarType != kComplexHalf && inputScalarType != kLong) {
        inputCastType = getMPSDataType(kFloat);
      }

      if (inputCastType != MPSDataTypeInvalid) {
        castInputTensor = castMPSTensor(mpsGraph, inputTensor, inputCastType);
      }

      MPSGraphTensor* castOutputTensor = nil;

      if (reduction_type == MPSReductionType::PROD) {
        castOutputTensor = [mpsGraph reductionProductWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::MEAN) {
        castOutputTensor = [mpsGraph meanOfTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::AMAX) {
        castOutputTensor = [mpsGraph reductionMaximumPropagateNaNWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::AMIN) {
        castOutputTensor = [mpsGraph reductionMinimumPropagateNaNWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::TRACE) {
        MPSGraphTensor* bandPartWithTensor = [mpsGraph bandPartWithTensor:castInputTensor
                                                                 numLower:0
                                                                 numUpper:0
                                                                     name:nil];
        castOutputTensor = [mpsGraph reductionSumWithTensor:bandPartWithTensor axes:@[ @0, @1 ] name:nil];
      }

      MPSGraphTensor* outputTensor = castOutputTensor;
      if (getMPSDataType(output_t) != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, output_t.scalar_type());
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t, mpsShape);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_output_shape);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void norm_kernel_mps(TensorIterator& iter, const Scalar& p_scalar) {
  const Tensor& output = iter.output(0);
  const Tensor& input = iter.input(0);
  auto p = p_scalar.to<double>();

  if (input.numel() == 0) {
    output.fill_((p < 0) ? INFINITY : 0);
    return;
  }

  if (output.numel() == 0) {
    return;
  }

  // Number of input elements that are reduced into one output element
  uint32_t reduction_size = input.numel() / output.numel();

  TORCH_INTERNAL_ASSERT(output.dim() == input.dim());

  NormParams params;

  params.ndim = input.dim();
  params.p = static_cast<float>(p);
  params.reduction_size = reduction_size;

  for (const auto dim_idx : c10::irange(input.dim())) {
    params.input_sizes[dim_idx] = input.size(dim_idx);
    params.input_strides[dim_idx] = input.stride(dim_idx);
    params.output_sizes[dim_idx] = output.size(dim_idx);
    params.output_strides[dim_idx] = output.stride(dim_idx);
  }

  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("norm_{}_{}", scalarToMetalTypeString(input), scalarToMetalTypeString(output)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "norm", {input});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, input, output, params);

      auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, reduction_size);
      uint32_t num_threads = output.numel() * threads_per_group;

      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}


// ============================================================================
// Metal kernel dispatch helpers for prod, welford, argreduce
// ============================================================================

struct WelfordConfig {
  float correction;
  float compute_std;
  float write_mean;
};

static std::vector<int64_t> get_reduce_dims(const Tensor& input, OptionalIntArrayRef opt_dim) {
  std::vector<int64_t> dims;
  if (opt_dim.has_value() && !opt_dim.value().empty()) {
    for (auto d : opt_dim.value()) {
      dims.push_back(maybe_wrap_dim(d, input.dim()));
    }
  } else {
    for (int64_t d = 0; d < input.dim(); d++) {
      dims.push_back(d);
    }
  }
  return dims;
}

static NormParams<> build_reduce_params(
    const Tensor& input,
    const std::vector<int64_t>& reduce_dims,
    const Tensor& output,
    bool keepdim) {
  NormParams params;
  params.ndim = input.dim();
  params.p = 0;
  params.reduction_size = input.numel() / std::max<int64_t>(output.numel(), 1);

  bool is_reduced[c10::metal::max_ndim] = {};
  for (auto d : reduce_dims) is_reduced[d] = true;

  if (keepdim || output.dim() == input.dim()) {
    for (uint32_t d = 0; d < params.ndim; d++) {
      params.input_sizes[d] = input.size(d);
      params.input_strides[d] = input.stride(d);
      params.output_sizes[d] = output.size(d);
      params.output_strides[d] = output.stride(d);
    }
  } else {
    uint32_t out_d = 0;
    for (uint32_t d = 0; d < params.ndim; d++) {
      params.input_sizes[d] = input.size(d);
      params.input_strides[d] = input.stride(d);
      if (is_reduced[d]) {
        params.output_sizes[d] = 1;
        params.output_strides[d] = 0;
      } else {
        params.output_sizes[d] = output.size(out_d);
        params.output_strides[d] = output.stride(out_d);
        out_d++;
      }
    }
  }

  return params;
}

static void prod_kernel_mps(
    const Tensor& input,
    const std::vector<int64_t>& reduce_dims,
    bool keepdim,
    const Tensor& output) {
  if (input.numel() == 0) {
    output.fill_(1);
    return;
  }
  if (output.numel() == 0) return;

  uint32_t reduction_size = input.numel() / output.numel();
  auto type_str = scalarToMetalTypeString(input);
  auto out_str = scalarToMetalTypeString(output);

  MPSStream* stream = getCurrentMPSStream();

  bool is_single = reduce_dims.size() == 1;
  bool is_outer = is_single && reduce_dims[0] == 0 &&
                  input.is_contiguous() && output.is_contiguous();
  bool is_inner = is_single && reduce_dims[0] == input.dim() - 1 &&
                  input.is_contiguous() && output.is_contiguous();

  if (is_outer) {
    uint32_t M = input.size(0);
    uint32_t N = input.numel() / M;
    auto kernel = fmt::format("prod_reduction_outer_{}_{}", type_str, out_str);
    constexpr uint32_t TG_X = 32, TG_Y = 32;
    auto num_tg_x = c10::metal::ceil_div(N, TG_X);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "prod_reduction_outer", {input});
        [enc setComputePipelineState:ps];
        struct { uint32_t M, N, out_stride; } sizes_s = {M, N, 1};
        mtl_setArgs(enc, input, output, sizes_s);
        [enc dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  if (is_inner) {
    uint32_t N = input.size(input.dim() - 1);
    uint32_t M = input.numel() / N;
    auto kernel = fmt::format("prod_reduction_inner_{}_{}", type_str, out_str);
    constexpr uint32_t NCHAINS = 8;
    uint32_t tg_size = c10::metal::ceil_div(N / (NCHAINS * 16), 32u) * 32u;
    tg_size = std::clamp(tg_size, 32u, 1024u);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "prod_reduction_inner", {input});
        [enc setComputePipelineState:ps];
        struct { uint32_t M, N; } sizes_s = {M, N};
        mtl_setArgs(enc, input, output, sizes_s);
        [enc dispatchThreads:MTLSizeMake(M * tg_size, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  auto kernel = fmt::format("prod_reduction_{}_{}", type_str, out_str);
  auto params = build_reduce_params(input, reduce_dims, output, keepdim);

  auto threads_per_group = std::min(MAX_THREADGROUP_SIZE,
      c10::metal::ceil_div(reduction_size, 32u) * 32u);
  uint32_t num_threads = output.numel() * threads_per_group;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(ps, "prod_reduction", {input});
      [enc setComputePipelineState:ps];
      mtl_setArgs(enc, input, output, params);
      [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}

static void welford_kernel_mps(
    const Tensor& input,
    const std::vector<int64_t>& reduce_dims,
    bool keepdim,
    double correction_value,
    bool compute_std,
    const Tensor& output,
    const Tensor* output_mean = nullptr) {
  if (input.numel() == 0 || output.numel() == 0) return;

  auto in_str = scalarToMetalTypeString(input);
  auto out_str = scalarToMetalTypeString(output);
  uint32_t reduction_size = input.numel() / output.numel();

  WelfordConfig config;
  config.correction = static_cast<float>(correction_value);
  config.compute_std = compute_std ? 1.0f : 0.0f;
  config.write_mean = output_mean ? 1.0f : 0.0f;

  Tensor mean_placeholder;
  const Tensor& mean_tensor = output_mean
      ? *output_mean
      : (mean_placeholder = at::empty({1}, output.options()));

  MPSStream* stream = getCurrentMPSStream();

  bool is_single = reduce_dims.size() == 1;
  bool is_outer = is_single && reduce_dims[0] == 0 &&
                  input.is_contiguous() && output.is_contiguous();
  bool is_inner = is_single && reduce_dims[0] == input.dim() - 1 &&
                  input.is_contiguous() && output.is_contiguous();

  if (is_outer) {
    uint32_t M = input.size(0);
    uint32_t N = input.numel() / M;
    auto kernel = fmt::format("welford_outer_{}_{}", in_str, out_str);
    constexpr uint32_t TG_X = 32, TG_Y = 32;
    auto num_tg_x = c10::metal::ceil_div(N, TG_X);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "welford_outer", {input});
        [enc setComputePipelineState:ps];
        struct { uint32_t M, N, out_stride; } sizes_s = {M, N, 1};
        mtl_setArgs(enc, input, output, mean_tensor, sizes_s, config);
        [enc dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  if (is_inner) {
    uint32_t N = input.size(input.dim() - 1);
    uint32_t M = input.numel() / N;
    auto kernel = fmt::format("welford_inner_{}_{}", in_str, out_str);
    uint32_t tg_size = std::min(1024u, c10::metal::ceil_div(N, 32u) * 32u);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "welford_inner", {input});
        [enc setComputePipelineState:ps];
        struct { uint32_t M, N; } sizes_s = {M, N};
        mtl_setArgs(enc, input, output, mean_tensor, sizes_s, config);
        [enc dispatchThreads:MTLSizeMake(M * tg_size, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  auto kernel = fmt::format("welford_{}_{}", in_str, out_str);
  auto params = build_reduce_params(input, reduce_dims, output, keepdim);

  auto threads_per_group = std::min(MAX_THREADGROUP_SIZE,
      c10::metal::ceil_div(reduction_size, 32u) * 32u);
  uint32_t num_threads = output.numel() * threads_per_group;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(ps, "welford_reduction", {input});
      [enc setComputePipelineState:ps];
      mtl_setArgs(enc, input, output, mean_tensor, params, config);
      [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}

static void argreduce_kernel_mps(
    const Tensor& input,
    const std::vector<int64_t>& reduce_dims,
    bool keepdim,
    bool is_max,
    const Tensor& output_indices,
    const Tensor* output_values = nullptr) {
  if (output_indices.numel() == 0) return;

  auto in_str = scalarToMetalTypeString(input);
  auto kernel_name = fmt::format("{}_{}", is_max ? "argmax" : "argmin", in_str);

  uint32_t reduction_size = input.numel() / std::max<int64_t>(output_indices.numel(), 1);
  auto params = build_reduce_params(input, reduce_dims, output_indices, keepdim);

  uint8_t write_values = output_values ? 1 : 0;

  Tensor values_placeholder;
  const Tensor& values_tensor = output_values
      ? *output_values
      : (values_placeholder = at::empty({1}, input.options()));

  auto threads_per_group = std::min(MAX_THREADGROUP_SIZE,
      c10::metal::ceil_div(reduction_size, 32u) * 32u);
  uint32_t num_threads = output_indices.numel() * threads_per_group;

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(ps, is_max ? "argmax" : "argmin", {input});
      [enc setComputePipelineState:ps];
      mtl_setArgs(enc, input, output_indices, values_tensor, params, write_values);
      [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}
static Tensor std_var_common_impl_mps(const Tensor& input_t,
                                      at::OptionalIntArrayRef dim,
                                      const std::optional<Scalar>& correction,
                                      bool keepdim,
                                      StdVarType stdVarType) {
  TORCH_CHECK_NOT_IMPLEMENTED(input_t.scalar_type() != kLong, "Not implemented for MPS");

  if (input_t.dim() == 0) {
    auto input_1d = input_t.unsqueeze(0);
    auto result = std_var_common_impl_mps(input_1d, IntArrayRef({0}), correction, false, stdVarType);
    return result.squeeze();
  }

  auto reduce_dims = get_reduce_dims(input_t, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < input_t.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) { reduced = true; break; }
    }
    if (reduced) {
      if (keepdim) output_shape.push_back(1);
    } else {
      output_shape.push_back(input_t.size(d));
    }
  }

  Tensor output_t = at::empty(output_shape, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return output_t;
  }

  welford_kernel_mps(input_t, reduce_dims, keepdim, correction_value,
                     stdVarType == STANDARD_DEVIATION, output_t);

  return output_t;
}

static Tensor median_common_mps(const Tensor& input_t, bool nanmedian) {
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_in_elements = c10::multiply_integers(input_shape);

  // we allocate 1 here due to MacOS13 bug for gather MPSGraph op, look below for the error
  Tensor output_t = at::empty({1}, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  if (output_t.numel() == 0 || num_in_elements == 0) {
    output_t.fill_(std::numeric_limits<float>::quiet_NaN());
    return output_t;
  }

  std::string medianKey = "median_mps:" + getMPSTypeString(input_t) + getTensorsStringKey(input_t) +
      std::to_string(num_in_elements) + (nanmedian ? ":nan" : "");

  using MedianCachedGraph = MPSUnaryCachedGraph;
  auto medianCachedGraph =
      LookUpOrCreateCachedGraph<MedianCachedGraph>(medianKey, [&](auto mpsGraph, auto newCachedGraph) {
        MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
        MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);

        MPSGraphTensor* reshapedTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @-1 ] name:nil];

        MPSGraphTensor* effectiveLengthTensor = nil;
        if (nanmedian) {
          MPSGraphTensor* isNanTensor = [mpsGraph isNaNWithTensor:reshapedTensor name:nil];
          MPSGraphTensor* nanCountTensor = [mpsGraph reductionSumWithTensor:isNanTensor axis:-1 name:nil];

          MPSGraphTensor* nanCountTensorFloat = [mpsGraph castTensor:nanCountTensor toType:MPSDataTypeInt32 name:nil];

          MPSGraphTensor* totalElementsTensor = [mpsGraph constantWithScalar:num_in_elements
                                                                       shape:@[]
                                                                    dataType:MPSDataTypeInt32];

          effectiveLengthTensor = [mpsGraph subtractionWithPrimaryTensor:totalElementsTensor
                                                         secondaryTensor:nanCountTensor
                                                                    name:nil];
        } else {
          effectiveLengthTensor = [mpsGraph constantWithScalar:num_in_elements shape:@[] dataType:MPSDataTypeInt32];
        }

        // get median index: medianIdx = ((effectiveLength + 1) / 2) - 1
        MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1 shape:@[ @1 ] dataType:MPSDataTypeInt32];
        MPSGraphTensor* twoTensor = [mpsGraph constantWithScalar:2 shape:@[ @1 ] dataType:MPSDataTypeInt32];
        MPSGraphTensor* effectivePlusOne = [mpsGraph additionWithPrimaryTensor:effectiveLengthTensor
                                                               secondaryTensor:oneTensor
                                                                          name:nil];
        MPSGraphTensor* halfEffective = [mpsGraph divisionWithPrimaryTensor:effectivePlusOne
                                                            secondaryTensor:twoTensor
                                                                       name:nil];
        MPSGraphTensor* medianIdxTensor = [mpsGraph subtractionWithPrimaryTensor:halfEffective
                                                                 secondaryTensor:oneTensor
                                                                            name:nil];

        MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:reshapedTensor axis:0 name:nil];

        MPSGraphTensor* medianTensor = [mpsGraph gatherWithUpdatesTensor:sortedTensor
                                                           indicesTensor:medianIdxTensor
                                                                    axis:0
                                                         batchDimensions:0
                                                                    name:nil];
        // MACOS 13 error: Rank of destination array must be greater than 0
        // which is why we initialize @1 here
        MPSGraphTensor* outputTensor = [mpsGraph reshapeTensor:medianTensor withShape:@[ @1 ] name:nil];

        newCachedGraph->inputTensor_ = inputTensor;
        newCachedGraph->outputTensor_ = outputTensor;
      });
  auto inputPlaceholder = Placeholder(medianCachedGraph->inputTensor_, input_t);
  auto outputPlaceHolder = Placeholder(medianCachedGraph->outputTensor_, output_t);
  auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
  runMPSGraph(getCurrentMPSStream(), medianCachedGraph->graph(), feeds, outputPlaceHolder);

  return output_t.squeeze();
}

static Tensor min_max_mps_impl(const Tensor& input_t, MPSReductionType reduction_type, const std::string& func_name) {
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_in_elements = c10::multiply_integers(input_shape);

  Tensor output_t = at::empty({}, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (output_t.numel() == 0 || num_in_elements == 0) {
    return output_t;
  }

  bool is_max = (reduction_type == MPSReductionType::MAX);
  Tensor indices_placeholder = at::empty({}, ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto reduce_dims = get_reduce_dims(input_t, std::nullopt);

  Tensor input_for_kernel = input_t;
  if (input_t.scalar_type() == kBool) {
    input_for_kernel = input_t.to(kInt);
    auto values_buf = at::empty({}, kInt, std::nullopt, kMPS, std::nullopt, std::nullopt);
    argreduce_kernel_mps(input_for_kernel, reduce_dims, false, is_max, indices_placeholder, &values_buf);
    output_t.copy_(values_buf.to(kBool));
  } else {
    argreduce_kernel_mps(input_for_kernel, reduce_dims, false, is_max, indices_placeholder, &output_t);
  }

  return output_t;
}

static void min_max_out_mps(const Tensor& input_t,
                            int64_t dim,
                            bool keepdim,
                            const Tensor& output_t,
                            const Tensor& indices_t,
                            MPSReductionType reduction_type,
                            const std::string& func_name) {
  if (output_t.numel() == 0) {
    return;
  }
  if (input_t.numel() == 1 && input_t.dim() == 0) {
    output_t.fill_(input_t);
    indices_t.fill_(0);
    return;
  }

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  bool is_max = (reduction_type == MPSReductionType::MAX);
  std::vector<int64_t> reduce_dims = {dim_};

  Tensor input_for_kernel = input_t;
  if (input_t.scalar_type() == kBool) {
    input_for_kernel = input_t.to(kInt);
    auto values_buf = at::empty(output_t.sizes(), kInt, std::nullopt, kMPS, std::nullopt, std::nullopt);
    argreduce_kernel_mps(input_for_kernel, reduce_dims, keepdim, is_max, indices_t, &values_buf);
    output_t.copy_(values_buf.to(kBool));
  } else {
    argreduce_kernel_mps(input_for_kernel, reduce_dims, keepdim, is_max, indices_t, &output_t);
  }
}

// Min/Max with dim
static std::tuple<Tensor, Tensor> min_max_mps_impl(const Tensor& input_t,
                                                   int64_t dim,
                                                   bool keepdim,
                                                   MPSReductionType reduction_type,
                                                   const std::string& func_name) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  // Use this if keepdim is false
  int64_t num_output_dims = num_input_dims - 1;

  std::vector<int64_t> vec_apparent_out_shape(num_input_dims);
  std::vector<int64_t> vec_out_shape(num_output_dims);

  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  // Counter for shape when keepdim is false
  int out_i = 0;
  for (const auto i : c10::irange(num_input_dims)) {
    if (dim_ == i) {
      apparent_out_shape[i] = @1;
      vec_apparent_out_shape[i] = 1;
    } else {
      apparent_out_shape[i] = [NSNumber numberWithInt:input_shape[i]];
      vec_apparent_out_shape[i] = input_shape[i];
      vec_out_shape[out_i] = input_shape[i];
      out_i++;
    }
  }

  Tensor output_t;
  Tensor indices_t;
  if (!keepdim) {
    output_t =
        at::empty(IntArrayRef(vec_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices_t = at::empty(IntArrayRef(vec_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  } else {
    output_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  }

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return std::tuple<Tensor, Tensor>{output_t, indices_t};
  }

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, reduction_type, func_name);

  return std::tuple<Tensor, Tensor>{output_t, indices_t};
}

static void argmax_argmin_out_mps(const Tensor& input_t,
                                  std::optional<int64_t> dim,
                                  bool keepdim,
                                  const Tensor& output_t,
                                  MPSReductionType reduction_type,
                                  const std::string& func_name) {
  bool is_max = (reduction_type == MPSReductionType::MAX);

  if (dim.has_value()) {
    int64_t dim_ = maybe_wrap_dim(dim.value(), input_t.dim());
    zero_numel_check_dims(input_t, dim_, is_max ? "argmax()" : "argmin()");
  } else {
    TORCH_CHECK_INDEX(input_t.numel() != 0,
                      is_max ? "argmax()" : "argmin()",
                      ": Expected reduction dim to be specified for input.numel() == 0.");
  }

  if (output_t.numel() == 0) return;

  std::vector<int64_t> reduce_dims;
  if (dim.has_value()) {
    reduce_dims.push_back(maybe_wrap_dim(dim.value(), input_t.dim()));
  } else {
    for (int64_t d = 0; d < input_t.dim(); d++) reduce_dims.push_back(d);
  }

  Tensor input_for_kernel = input_t;
  if (input_t.scalar_type() == kBool) {
    input_for_kernel = input_t.to(kInt);
  }

  argreduce_kernel_mps(input_for_kernel, reduce_dims, keepdim, is_max, output_t);
}

// Shared implementation for sum/nansum/count_nonzero/mean Metal kernels.
// `kernel_prefix` is "sum_", "nansum_" or "count_nonzero_" — selects the
// kernel variant to dispatch.  `divisor` > 0 divides the accumulator (in
// opmath_t) before casting to output, enabling fused mean without losing the
// fp32 accumulation precision for fp16/bf16/half2 outputs.
static void sum_nansum_kernel_mps(TensorIterator& iter, const std::string& kernel_prefix, float divisor = 0.0f) {
  const Tensor& output = iter.output(0);
  const Tensor& input = iter.input(0);

  if (input.numel() == 0) {
    output.zero_();
    return;
  }

  if (output.numel() == 0) {
    return;
  }

  uint32_t reduction_size = input.numel() / output.numel();

  // TensorIterator ensures input and output have matching ndim
  // (reduced dims have size 1 in output)
  TORCH_INTERNAL_ASSERT(output.dim() == input.dim());

  constexpr uint32_t NCHAINS = SUM_NCHAINS;

  auto kernel_name =
      fmt::format("{}reduction_{}_{}", kernel_prefix, scalarToMetalTypeString(input), scalarToMetalTypeString(output));

  MPSStream* stream = getCurrentMPSStream();

  // For large full reductions (output is scalar), use multi-TG with a
  // two-pass approach: first pass splits work across num_groups TGs writing
  // partial sums, second pass reduces the partials to the final scalar.
  if (output.numel() == 1 && reduction_size > MAX_THREADGROUP_SIZE * NCHAINS) {
    auto num_groups = std::min(512u, c10::metal::ceil_div(reduction_size, MAX_THREADGROUP_SIZE * NCHAINS));

    // elems_per_group * num_groups must equal reduction_size exactly,
    // otherwise pass 1's last TG reads past the input's logical end.
    // Reduce num_groups down to a divisor of reduction_size (falling back
    // to 1 is always safe — the inner loop still parallelizes via threads).
    while (num_groups > 1 && reduction_size % num_groups != 0) {
      num_groups--;
    }

    auto partials = at::empty({num_groups}, output.options());
    const auto elems_per_group = reduction_size / num_groups;

    auto out_metal = scalarToMetalTypeString(output);
    auto p1_kernel = fmt::format("{}reduction_{}_{}", kernel_prefix, scalarToMetalTypeString(input), out_metal);
    // Pass 2 combines partials by summing them regardless of pass-1 mode.
    // For count_nonzero the partials are already per-block counts (long);
    // counting them again would be wrong, so always use "sum_" here.
    auto p2_kernel = fmt::format("sum_reduction_{}_{}", out_metal, out_metal);

    // Model as 2D: input is [num_groups, elems_per_group], reduce dim=1
    // Dim 0 (non-reduced): size=num_groups, input_stride=elems_per_group, output_stride=1
    // Dim 1 (reduced):     size=elems_per_group, input_stride=1
    NormParams params1;
    params1.ndim = 2;
    params1.p = 0;
    params1.reduction_size = elems_per_group;
    params1.input_sizes[0] = num_groups;
    params1.input_strides[0] = elems_per_group;
    params1.output_sizes[0] = num_groups;
    params1.output_strides[0] = 1;
    params1.input_sizes[1] = elems_per_group;
    params1.input_strides[1] = 1;
    params1.output_sizes[1] = 1;
    params1.output_strides[1] = 0;

    // Pass 2: partials[num_groups] -> output[1], reduce dim=0.
    // divisor applies here (not on pass 1), so pass 2 produces
    // accumulator/divisor before the final cast to output dtype.
    NormParams params2;
    params2.ndim = 1;
    params2.p = divisor;
    params2.reduction_size = num_groups;
    params2.input_sizes[0] = num_groups;
    params2.input_strides[0] = 1;
    params2.output_sizes[0] = 1;
    params2.output_strides[0] = 0;

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();

        // Pass 1: input -> partials
        auto ps1 = lib.getPipelineStateForFunc(p1_kernel);
        getMPSProfiler().beginProfileKernel(ps1, "sum_reduction_pass1", {input});
        [compute_encoder setComputePipelineState:ps1];
        mtl_setArgs(compute_encoder, input, partials, params1);
        auto tpg1 = std::min(MAX_THREADGROUP_SIZE, elems_per_group);
        [compute_encoder dispatchThreads:MTLSizeMake(num_groups * tpg1, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];
        getMPSProfiler().endProfileKernel(ps1);

        // Pass 2: partials -> output
        auto ps2 = lib.getPipelineStateForFunc(p2_kernel);
        getMPSProfiler().beginProfileKernel(ps2, "sum_reduction_pass2", {partials});
        [compute_encoder setComputePipelineState:ps2];
        mtl_setArgs(compute_encoder, partials, output, params2);
        auto tpg2 = std::min(MAX_THREADGROUP_SIZE, num_groups);
        [compute_encoder dispatchThreads:MTLSizeMake(tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
        getMPSProfiler().endProfileKernel(ps2);
      }
    });
    return;
  }

  // Detect outer-dim (non-innermost) reduction on contiguous 2D tensor.
  // For this case, use a specialized kernel with coalesced column reads.
  // Condition: exactly one reduced dim, it's not the last dim, input is contiguous.
  {
    int num_reduced = 0;
    int reduced_dim = -1;
    for (int64_t d = 0; d < input.dim(); d++) {
      if (input.size(d) != output.size(d)) {
        num_reduced++;
        reduced_dim = d;
      }
    }
    bool is_outer_reduction = (num_reduced == 1 && reduced_dim < input.dim() - 1 && input.is_contiguous());
    bool is_inner_reduction = (num_reduced == 1 && reduced_dim == input.dim() - 1 && input.is_contiguous());

    if (is_outer_reduction && reduced_dim == 0 && output.is_contiguous()) {
      uint32_t M = input.size(0);
      uint32_t N = input.numel() / M;

      auto outer_kernel = fmt::format(
          "{}reduction_outer_{}_{}", kernel_prefix, scalarToMetalTypeString(input), scalarToMetalTypeString(output));
      constexpr uint32_t TG_X = 32, TG_Y = 32;
      const auto num_tg_x = c10::metal::ceil_div(N, TG_X);

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(outer_kernel);
          getMPSProfiler().beginProfileKernel(ps, "sum_reduction_outer", {input});
          struct {
            uint32_t M, N, out_stride;
          } sizes_s = {M, N, 1};
          [compute_encoder setComputePipelineState:ps];
          mtl_setArgs(compute_encoder, input, output, sizes_s, divisor);
          [compute_encoder dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
          getMPSProfiler().endProfileKernel(ps);
        }
      });
      return;
    }

    if (is_inner_reduction && output.is_contiguous()) {
      // M = product of all non-reduced dims, N = size of last dim
      uint32_t N = input.size(input.dim() - 1);
      uint32_t M = input.numel() / N;

      auto inner_kernel = fmt::format(
          "{}reduction_inner_{}_{}", kernel_prefix, scalarToMetalTypeString(input), scalarToMetalTypeString(output));
      // Pack multiple rows per TG: each SIMD group (32 threads) handles one row
      constexpr uint32_t TG_SIZE = 256; // 8 SIMD groups = 8 rows per TG
      constexpr uint32_t rows_per_tg = TG_SIZE / 32;
      const auto num_tgs = c10::metal::ceil_div(M, rows_per_tg);

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(inner_kernel);
          getMPSProfiler().beginProfileKernel(ps, "sum_reduction_inner", {input});
          struct {
            uint32_t M, N;
          } sizes_s = {M, N};
          [compute_encoder setComputePipelineState:ps];
          mtl_setArgs(compute_encoder, input, output, sizes_s, divisor);
          [compute_encoder dispatchThreads:MTLSizeMake(num_tgs * TG_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
          getMPSProfiler().endProfileKernel(ps);
        }
      });
      return;
    }
  }

  NormParams params;
  params.ndim = input.dim();
  params.p = divisor;
  params.reduction_size = reduction_size;

  for (const auto dim_idx : c10::irange(input.dim())) {
    params.input_sizes[dim_idx] = input.size(dim_idx);
    params.input_strides[dim_idx] = input.stride(dim_idx);
    params.output_sizes[dim_idx] = output.size(dim_idx);
    params.output_strides[dim_idx] = output.stride(dim_idx);
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pipeline_state, "sum_reduction", {input});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, input, output, params);

      auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, reduction_size);
      uint32_t num_threads = output.numel() * threads_per_group;

      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

static void sum_kernel_mps(TensorIterator& iter) {
  sum_nansum_kernel_mps(iter, "sum_");
}

static void nansum_kernel_mps(TensorIterator& iter) {
  auto in_dtype = iter.input(0).scalar_type();
  bool is_float = c10::isFloatingType(in_dtype) || c10::isComplexType(in_dtype);
  sum_nansum_kernel_mps(iter, is_float ? "nansum_" : "sum_");
}

static void mean_kernel_mps(TensorIterator& iter) {
  auto output = iter.output(0);
  auto input = iter.input(0);
  if (input.numel() == 0 || output.numel() == 0) {
    sum_nansum_kernel_mps(iter, "sum_");
    return;
  }
  int64_t reduction_size = input.numel() / output.numel();
  // Fused divide: the sum kernel divides the accumulator (in opmath_t)
  // before casting to output, so fp32 accumulation precision is preserved
  // for fp16/bf16/half2 without an intermediate tensor.
  sum_nansum_kernel_mps(iter, "sum_", static_cast<float>(reduction_size));
}

static void count_nonzero_kernel_mps(TensorIterator& iter) {
  sum_nansum_kernel_mps(iter, "count_nonzero_");
}

Tensor trace_mps(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

  Tensor output_t =
      at::empty({}, get_dtype_from_self(self, std::nullopt, true), std::nullopt, kMPS, std::nullopt, std::nullopt);

  std::vector<int64_t> dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);

  reduction_out_mps(self,
                    IntArrayRef(dims),
                    false,
                    std::nullopt,
                    const_cast<Tensor&>(output_t),
                    MPSReductionType::TRACE,
                    "trace_mps");

  return output_t;
}

TORCH_IMPL_FUNC(prod_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, std::optional<ScalarType> dtype, const Tensor& output_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  std::vector<int64_t> reduce_dims = {dim_};
  prod_kernel_mps(input_t, reduce_dims, keepdim, output_t);
}

TORCH_IMPL_FUNC(amax_out_mps)(const Tensor& input_t, IntArrayRef dim, bool keepdim, const Tensor& output_t) {
  TORCH_CHECK(!c10::isComplexType(input_t.scalar_type()), "amax is not defined for complex types");
  reduction_out_mps(input_t, dim, keepdim, std::nullopt, output_t, MPSReductionType::AMAX, "amax_out_mps");
}

TORCH_IMPL_FUNC(amin_out_mps)(const Tensor& input_t, IntArrayRef dim, bool keepdim, const Tensor& output_t) {
  TORCH_CHECK(!c10::isComplexType(input_t.scalar_type()), "amin is not defined for complex types");
  reduction_out_mps(input_t, dim, keepdim, std::nullopt, output_t, MPSReductionType::AMIN, "amin_out_mps");
}

TORCH_IMPL_FUNC(aminmax_out_mps)
(const Tensor& input_t, std::optional<int64_t> dim_opt, bool keepdim, const Tensor& min_t, const Tensor& max_t) {
  TORCH_CHECK(!c10::isComplexType(input_t.scalar_type()), "aminmax is not defined for complex types");
  reduction_out_mps(input_t,
                    dim_opt.has_value() ? OptionalIntArrayRef({*dim_opt}) : std::nullopt,
                    keepdim,
                    std::nullopt,
                    min_t,
                    MPSReductionType::AMIN,
                    "aminmax_out_mps_min");
  reduction_out_mps(input_t,
                    dim_opt.has_value() ? OptionalIntArrayRef({*dim_opt}) : std::nullopt,
                    keepdim,
                    std::nullopt,
                    max_t,
                    MPSReductionType::AMAX,
                    "aminmax_out_mps_max");
}

Tensor prod_mps(const Tensor& self, std::optional<ScalarType> opt_dtype) {
  auto reduce_dims = get_reduce_dims(self, std::nullopt);

  Tensor output_t =
      at::empty({}, get_dtype_from_self(self, opt_dtype, true), std::nullopt, kMPS, std::nullopt, std::nullopt);

  prod_kernel_mps(self, reduce_dims, false, output_t);

  return output_t;
}

Tensor count_nonzero_mps(const Tensor& self, IntArrayRef dims) {
  Tensor result = create_reduction_result(self, dims, /*keepdim=*/false, ScalarType::Long);
  auto iter =
      make_reduction("count_nonzero_mps", result, self, dims, /*keepdim=*/false, self.scalar_type(), ScalarType::Long);
  count_nonzero_kernel_mps(iter);
  return result;
}

Tensor _cdist_forward_mps(const Tensor& x1, const Tensor& x2, const double p, std::optional<int64_t> compute_mode) {
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(x1.size(-1) == x2.size(-1),
              "X1 and X2 must have the same number of columns. X1: ",
              x1.size(-1),
              " X2: ",
              x2.size(-1));
  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");

  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

  Tensor x1_ = x1.unsqueeze(-2);
  Tensor x2_ = x2.unsqueeze(-3);
  Tensor diff = x1_.sub(x2_);
  IntArrayRef output_shape(diff.sizes().data(), diff.dim() - 1);
  Tensor result = at::empty(output_shape, x1.options());
  linalg_vector_norm_out(result, diff, p, makeArrayRef<int64_t>(-1), /*keepdim=*/false, /*dtype=*/std::nullopt);

  return result;
}

Tensor var_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_common_impl_mps(input_t, dim, correction, keepdim, STANDARD_VARIANCE);
}

Tensor std_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_common_impl_mps(input_t, dim, correction, keepdim, STANDARD_DEVIATION);
}

typedef MPSGraphTensor* (^ReductionOpBlock)(MPSGraph*, MPSGraphTensor*, int64_t);
static void all_any_common_impl_mps(const Tensor& input_t,
                                    int64_t dim,
                                    bool keepdim,
                                    const Tensor& output_t,
                                    ReductionOpBlock reduction_op,
                                    const std::string& op_name) {
  using CachedGraph = MPSUnaryCachedGraph;
  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return;
  }
  if (input_t.numel() == 1) {
    output_t.copy_(input_t.view_as(output_t).to(at::kBool));
    return;
  }

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, op_name.c_str());

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  for (const auto i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }

  @autoreleasepool {
    std::string key = op_name + "_out_mps:" + getTensorsStringKey(input_t) + ":" + std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);

      auto castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);
      // reductionOrWithTensor:axis: will throw an internal assert if number of dimensions is more than 4
      // See https://github.com/pytorch/pytorch/issues/95538
      MPSGraphTensor* outputTensor = nil;
      if (input_t.ndimension() > 4) {
        auto reduceDimLen = input_t.size(dim_);
        if (dim_ == 0) {
          castInputTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @(reduceDimLen), @-1 ] name:nil];
          outputTensor = reduction_op(mpsGraph, castInputTensor, 0);
        } else {
          if (dim_ == input_t.dim() - 1) {
            castInputTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @-1, @(reduceDimLen) ] name:nil];
          } else {
            auto beforeNumel = 1;
            for (auto i : c10::irange(dim_)) {
              beforeNumel *= input_t.size(i);
            }
            castInputTensor = [mpsGraph reshapeTensor:castInputTensor
                                            withShape:@[ @(beforeNumel), @(reduceDimLen), @-1 ]
                                                 name:nil];
          }
          outputTensor = reduction_op(mpsGraph, castInputTensor, 1);
        }
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:apparent_out_shape name:nil];
      } else {
        outputTensor = reduction_op(mpsGraph, castInputTensor, dim_);
      }
      if (MPSDataTypeBool != [outputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, outputTensor, MPSDataTypeBool);
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(any_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t) {
  all_any_common_impl_mps(
      input_t,
      dim,
      keepdim,
      output_t,
      ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* tensor, int64_t dim_) {
        return [graph reductionOrWithTensor:tensor axis:dim_ name:nil];
      },
      "any");
}

TORCH_IMPL_FUNC(any_all_out_mps)(const Tensor& input_t, const Tensor& output_t) {
  using CachedGraph = MPSUnaryCachedGraph;
  if (input_t.numel() == 0) {
    output_t.zero_();
    return;
  } else if (input_t.numel() == 1) {
    output_t.copy_(input_t.view_as(output_t).to(at::kBool));
    return;
  } else if (output_t.numel() == 0) {
    return;
  }

  @autoreleasepool {
    std::string key = std::string("any_all_out_mps:") + getTensorsStringKey(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      auto castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);
      // reductionOrWithTensor:axes: will throw an internal assert if number of dimensions is more than 4
      // See https://github.com/pytorch/pytorch/issues/95538
      if (input_t.dim() > 4) {
        castInputTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @-1 ] name:nil];
      }
      auto outputTensor = [mpsGraph reductionOrWithTensor:castInputTensor axes:nil name:nil];

      if (getMPSDataType(output_t) != [outputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, outputTensor, output_t.scalar_type());
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(all_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t) {
  all_any_common_impl_mps(
      input_t,
      dim,
      keepdim,
      output_t,
      ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* tensor, int64_t dim_) {
        return [graph reductionAndWithTensor:tensor axis:dim_ name:nil];
      },
      "all");
}

TORCH_IMPL_FUNC(all_all_out_mps)(const Tensor& input_t, const Tensor& output_t) {
  using CachedGraph = MPSUnaryCachedGraph;
  if (output_t.numel() == 0 || input_t.numel() == 0) {
    // in line with cpu behaviour and numpy, an empty tensor should return true.
    // specifying ones forces the output to be true for this case.
    output_t.fill_(1);
    return;
  }

  @autoreleasepool {
    std::string key = std::string("all_all_out_mps:") + getTensorsStringKey(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      auto castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);
      // reductionAndWithTensor:axes: will throw an internal assert if number of dimensions is more than 4
      // See https://github.com/pytorch/pytorch/issues/95538
      if (input_t.ndimension() > 4) {
        castInputTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @-1 ] name:nil];
      }
      auto outputTensor = [mpsGraph reductionAndWithTensor:castInputTensor axes:nil name:nil];
      if (MPSDataTypeBool != [outputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, outputTensor, MPSDataTypeBool);
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

//-----------------------------------------------------------------------
// Min and max functions

// Max entire tensor into scalar result
Tensor max_mps(const Tensor& input_t) {
  return min_max_mps_impl(input_t, MPSReductionType::MAX, "max_mps");
}

// Min entire tensor into scalar result
Tensor min_mps(const Tensor& input_t) {
  return min_max_mps_impl(input_t, MPSReductionType::MIN, "min_mps");
}

// Max out with dim
TORCH_IMPL_FUNC(max_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, MPSReductionType::MAX, "max_out_mps");
}

// Min out with dim
TORCH_IMPL_FUNC(min_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "min()");

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, MPSReductionType::MIN, "min_out_mps");
}

TORCH_IMPL_FUNC(argmax_out_mps)
(const Tensor& input_t, std::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  argmax_argmin_out_mps(input_t, dim, keepdim, output_t, MPSReductionType::MAX, "argmax_out_mps");
}

TORCH_IMPL_FUNC(argmin_out_mps)
(const Tensor& input_t, std::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  argmax_argmin_out_mps(input_t, dim, keepdim, output_t, MPSReductionType::MIN, "argmin_out_mps");
}

// Max with dim
static std::tuple<Tensor, Tensor> max_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return min_max_mps_impl(input_t, dim, keepdim, MPSReductionType::MAX, "max_mps");
}

// Min with dim
static std::tuple<Tensor, Tensor> min_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return min_max_mps_impl(input_t, dim, keepdim, MPSReductionType::MIN, "min_mps");
}

// Median of entire tensor into scalar result
Tensor median_mps(const Tensor& input_t) {
  return median_common_mps(input_t, /*nanmedian=*/false);
}

static void median_out_mps_common(const Tensor& input_t,
                                  int64_t dim,
                                  bool keepdim,
                                  Tensor& values,
                                  Tensor& indices,
                                  const std::string& func_name,
                                  bool nanmedian) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  // Use this if keepdim is false
  int64_t num_output_dims = num_input_dims - 1 < 0 ? 0 : num_input_dims - 1;

  std::vector<int64_t> vec_apparent_out_shape(num_input_dims);
  std::vector<int64_t> vec_out_shape(num_output_dims);

  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  // Counter for shape when keepdim is false
  int out_i = 0;
  for (const auto i : c10::irange(num_input_dims)) {
    if (dim_ == i) {
      apparent_out_shape[i] = @1;
      vec_apparent_out_shape[i] = 1;
    } else {
      apparent_out_shape[i] = [NSNumber numberWithInt:input_shape[i]];
      vec_apparent_out_shape[i] = input_shape[i];
      vec_out_shape[out_i] = input_shape[i];
      out_i++;
    }
  }

  if (!keepdim) {
    values =
        at::empty(IntArrayRef(vec_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices = at::empty(IntArrayRef(vec_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  } else {
    values = at::empty(
        IntArrayRef(vec_apparent_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices = at::empty(
        IntArrayRef(vec_apparent_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  }

  if (values.numel() == 0 || input_t.numel() == 0) {
    return;
  }

  if (input_t.numel() == 1 && input_t.dim() == 0) {
    values.fill_(input_t);
    indices.fill_(0);
    return;
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* indicesTensor_ = nil;
  };

  for (const int i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }
  int dim_total_elements = input_shape[dim_];

  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = func_name + ":" + std::to_string(dim_) + ":" + getTensorsStringKey(input_t) + ":" +
        getTensorsStringKey(indices);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);

      MPSGraphTensor* effectiveLengthTensor = nil;
      if (nanmedian) {
        MPSGraphTensor* isNanTensor = [mpsGraph isNaNWithTensor:castInputTensor name:nil];
        MPSGraphTensor* nanCountTensor = [mpsGraph reductionSumWithTensor:isNanTensor
                                                                     axis:(NSInteger)dim_
                                                                     name:@"nanCount"];
        MPSGraphTensor* nanCountTensorInt = [mpsGraph castTensor:nanCountTensor
                                                          toType:MPSDataTypeInt32
                                                            name:@"nanCountInt"];
        MPSGraphTensor* dimSizeTensor = [mpsGraph constantWithScalar:dim_total_elements
                                                               shape:@[]
                                                            dataType:MPSDataTypeInt32];
        // effective count: effectiveLength = dim_size - nan_count.
        effectiveLengthTensor = [mpsGraph subtractionWithPrimaryTensor:dimSizeTensor
                                                       secondaryTensor:nanCountTensorInt
                                                                  name:@"effectiveLength"];
      } else {
        effectiveLengthTensor = [mpsGraph constantWithScalar:dim_total_elements
                                                       shape:apparent_out_shape
                                                    dataType:MPSDataTypeInt32];
      }
      // median index = ((effectiveLength + 1) / 2) - 1.
      MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1 shape:@[] dataType:MPSDataTypeInt32];
      MPSGraphTensor* twoTensor = [mpsGraph constantWithScalar:2 shape:@[] dataType:MPSDataTypeInt32];
      MPSGraphTensor* effectivePlusOne = [mpsGraph additionWithPrimaryTensor:effectiveLengthTensor
                                                             secondaryTensor:oneTensor
                                                                        name:@"effectivePlusOne"];
      MPSGraphTensor* halfEffective = [mpsGraph divisionWithPrimaryTensor:effectivePlusOne
                                                          secondaryTensor:twoTensor
                                                                     name:@"halfEffective"];
      MPSGraphTensor* medianIdxTensor = [mpsGraph subtractionWithPrimaryTensor:halfEffective
                                                               secondaryTensor:oneTensor
                                                                          name:@"medianIdx"];

      MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor axis:((NSUInteger)(int)dim_)name:nil];
      MPSGraphTensor* sortedIndicesTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                                   axis:(NSInteger)dim_
                                                                   name:@"argsort_out"];

      MPSGraphTensor* medianValueTensor = [mpsGraph gatherAlongAxis:dim_
                                                  withUpdatesTensor:sortedTensor
                                                      indicesTensor:medianIdxTensor
                                                               name:@"gather_medianValue"];
      MPSGraphTensor* medianIndexTensor = [mpsGraph gatherAlongAxis:dim_
                                                  withUpdatesTensor:sortedIndicesTensor
                                                      indicesTensor:medianIdxTensor
                                                               name:@"gather_medianValue"];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = medianValueTensor;
      newCachedGraph->indicesTensor_ = medianIndexTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, values, apparent_out_shape);
    auto indicesPlaceholder = Placeholder(cachedGraph->indicesTensor_, indices, apparent_out_shape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(outputPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

// in case mps sortWithTensor do not supported on macOS
static std::tuple<Tensor&, Tensor&> median_from_cpu(const Tensor& self,
                                                    int64_t dim,
                                                    bool keepdim,
                                                    Tensor& valuesI,
                                                    Tensor& indicesI,
                                                    IntArrayRef vec_out_shape,
                                                    IntArrayRef vec_apparent_out_shape) {
  Tensor values;
  Tensor indices;
  if (!keepdim) {
    values = at::empty({vec_out_shape}, self.options());
    indices = at::empty({vec_out_shape}, self.options().dtype(kLong));
  } else {
    values = at::empty({vec_apparent_out_shape}, self.options());
    indices = at::empty({vec_apparent_out_shape}, self.options().dtype(kLong));
  }
  at::median_out(values, indices, self, dim, keepdim);

  valuesI.copy_(values);
  indicesI.copy_(indices);
  return std::forward_as_tuple(valuesI, indicesI);
}

TORCH_API ::std::tuple<at::Tensor&, at::Tensor&> median_out_mps(const at::Tensor& input_t,
                                                                int64_t dim,
                                                                bool keepdim,
                                                                at::Tensor& values,
                                                                at::Tensor& indices) {
  median_out_mps_common(input_t, dim, keepdim, values, indices, "median_out_mps", false);
  return std::tuple<Tensor&, Tensor&>{values, indices};
}

std::tuple<Tensor&, Tensor&> nanmedian_out_mps(const at::Tensor& self,
                                               int64_t dim,
                                               bool keepdim,
                                               at::Tensor& values,
                                               at::Tensor& indices) {
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return median_out_mps(self, dim, keepdim, values, indices);
  }
  median_out_mps_common(self, dim, keepdim, values, indices, "nanmedian_out_mps", true);
  return std::tie(values, indices);
}

Tensor nanmedian_mps(const Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return median_mps(self);
  }
  return median_common_mps(self, /*nanmedian=*/true);
}

std::tuple<Tensor, Tensor> std_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const std::optional<Scalar>& correction,
                                        bool keepdim) {
  if (self.dim() == 0) {
    auto self_1d = self.unsqueeze(0);
    auto [s, m] = std_mean_mps(self_1d, IntArrayRef({0}), correction, false);
    return {s.squeeze(), m.squeeze()};
  }
  auto reduce_dims = get_reduce_dims(self, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < self.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) { reduced = true; break; }
    }
    if (reduced) {
      if (keepdim) output_shape.push_back(1);
    } else {
      output_shape.push_back(self.size(d));
    }
  }

  auto std_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto mean_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);

  if (std_out.numel() > 0 && self.numel() > 0) {
    welford_kernel_mps(self, reduce_dims, keepdim, correction_value, true, std_out, &mean_out);
  }

  return {std_out, mean_out};
}

std::tuple<Tensor, Tensor> var_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const std::optional<Scalar>& correction,
                                        bool keepdim) {
  if (self.dim() == 0) {
    auto self_1d = self.unsqueeze(0);
    auto [v, m] = var_mean_mps(self_1d, IntArrayRef({0}), correction, false);
    return {v.squeeze(), m.squeeze()};
  }
  auto reduce_dims = get_reduce_dims(self, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < self.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) { reduced = true; break; }
    }
    if (reduced) {
      if (keepdim) output_shape.push_back(1);
    } else {
      output_shape.push_back(self.size(d));
    }
  }

  auto var_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto mean_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);

  if (var_out.numel() > 0 && self.numel() > 0) {
    welford_kernel_mps(self, reduce_dims, keepdim, correction_value, false, var_out, &mean_out);
  }

  return {var_out, mean_out};
}

REGISTER_DISPATCH(norm_stub, &norm_kernel_mps)
REGISTER_DISPATCH(sum_stub, &sum_kernel_mps)
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_mps)
REGISTER_DISPATCH(mean_stub, &mean_kernel_mps)

} // namespace at::native
