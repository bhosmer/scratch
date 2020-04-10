// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/QuantizedCPUType.h>

// @generated by aten/src/ATen/gen.py from TypeDerived.cpp

#include <c10/core/TensorImpl.h>
#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>



namespace {
static const char* named_tensors_unsupported_error =
  " is not yet supported with named tensors. Please drop names via "
  "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
  "and set names on the result of the operation.";
}

namespace at {

/* example
Tensor * QuantizedCPUType::add(Tensor & a, Tensor & b) {
  std::cout << "add Tensor with backend QuantizedCPU\n";
  return &a;
}
*/

namespace QuantizedCPUType {

Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {

    // DeviceGuard omitted
    return at::native::as_strided_qtensorimpl(self, size, stride, storage_offset);
}
Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
    if (input.has_names() || weight.has_names() || bias.has_names() || mean.has_names() || var.has_names()) {
        AT_ERROR("quantized_batch_norm", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}
Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_clamp(self, min, max);
}
Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {

    const DeviceGuard device_guard(options.device());
    return at::native::empty_affine_quantized_cpu(size, options, scale, zero_point, memory_format);
}
Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
    if (scales.has_names() || zero_points.has_names()) {
        AT_ERROR("_empty_per_channel_affine_quantized", named_tensors_unsupported_error);
    }
    const DeviceGuard device_guard(options.device());
    return at::native::empty_per_channel_affine_quantized_cpu(size, scales, zero_points, axis, options, memory_format);
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    if (self.has_names()) {
        AT_ERROR("quantized_max_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_mean_cpu(self, dtype);
}
Tensor mean_dim(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_mean_cpu(self, dim, keepdim, dtype);
}
Tensor & mean_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_mean_out_cpu(out, self, dim, keepdim, dtype);
}
Tensor relu(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_relu(self);
}
Tensor & relu_(Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_relu_(self);
}
Tensor sigmoid(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_sigmoid(self);
}
Tensor tanh(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_tanh(self);
}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_clone(self, memory_format);
}
Tensor dequantize_self(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("dequantize", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::dequantize_quant(self);
}
std::vector<Tensor> dequantize_tensors(TensorList tensors) {
    if (at::has_names(tensors)) {
        AT_ERROR("dequantize", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(tensors));
    return at::native::dequantize_tensors_quant(tensors);
}
double q_scale(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("q_scale", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_scale_quant(self);
}
int64_t q_zero_point(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("q_zero_point", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_zero_point_quant(self);
}
Tensor q_per_channel_scales(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("q_per_channel_scales", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_scales_quant(self);
}
Tensor q_per_channel_zero_points(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("q_per_channel_zero_points", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_zero_points_quant(self);
}
int64_t q_per_channel_axis(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("q_per_channel_axis", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_axis_quant(self);
}
Tensor int_repr(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("int_repr", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::int_repr_quant(self);
}
QScheme qscheme(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("qscheme", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::qscheme_quant(self);
}
Tensor & set__source_Storage_storage_offset(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    if (self.has_names()) {
        AT_ERROR("set_", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::set_storage_quantized_cpu_(self, source, storage_offset, size, stride);
}
Tensor & set_quantizer_(Tensor & self, ConstQuantizerPtr quantizer) {
    if (self.has_names()) {
        AT_ERROR("set_quantizer_", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::set_quantizer_(self, quantizer);
}
Tensor view(const Tensor & self, IntArrayRef size) {
    if (self.has_names()) {
        AT_ERROR("view", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::view(self, size);
}
Tensor & ne_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_out_quantized_cpu(out, self, other);
}
Tensor ne_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_quantized_cpu(self, other);
}
Tensor & ne_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_out_quantized_cpu(out, self, other);
}
Tensor ne_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_quantized_cpu(self, other);
}
Tensor & eq_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_out_quantized_cpu(out, self, other);
}
Tensor eq_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_quantized_cpu(self, other);
}
Tensor & eq_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_out_quantized_cpu(out, self, other);
}
Tensor eq_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_quantized_cpu(self, other);
}
Tensor & ge_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_out_quantized_cpu(out, self, other);
}
Tensor ge_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_quantized_cpu(self, other);
}
Tensor & ge_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_out_quantized_cpu(out, self, other);
}
Tensor ge_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_quantized_cpu(self, other);
}
Tensor & le_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_out_quantized_cpu(out, self, other);
}
Tensor le_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_quantized_cpu(self, other);
}
Tensor & le_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_out_quantized_cpu(out, self, other);
}
Tensor le_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_quantized_cpu(self, other);
}
Tensor & gt_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_out_quantized_cpu(out, self, other);
}
Tensor gt_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_quantized_cpu(self, other);
}
Tensor & gt_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_out_quantized_cpu(out, self, other);
}
Tensor gt_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_quantized_cpu(self, other);
}
Tensor & lt_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_out_quantized_cpu(out, self, other);
}
Tensor lt_Scalar(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_quantized_cpu(self, other);
}
Tensor & lt_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_out_quantized_cpu(out, self, other);
}
Tensor lt_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_quantized_cpu(self, other);
}
Tensor min(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::min_quant(self);
}
Tensor max(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::max_quant(self);
}
std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
    if (self.has_names()) {
        AT_ERROR("sort", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::sort_quant(self, dim, descending);
}
std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    if (self.has_names()) {
        AT_ERROR("topk", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_topk_cpu(self, k, dim, largest, sorted);
}
bool equal(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_equal(self, other);
}
Tensor _cat(TensorList tensors, int64_t dim) {
    if (at::has_names(tensors)) {
        AT_ERROR("_cat", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(tensors));
    return at::native::quantized_cat(tensors, dim);
}
Tensor & _cat_out_out(Tensor & out, TensorList tensors, int64_t dim) {
    if (out.has_names() || at::has_names(tensors)) {
        AT_ERROR("_cat_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(out));
    return at::native::quantized_cat_out(out, tensors, dim);
}
Tensor & elu_out_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
    if (out.has_names() || self.has_names()) {
        AT_ERROR("elu_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_elu_out(out, self, alpha, scale, input_scale);
}
Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
    if (self.has_names()) {
        AT_ERROR("elu", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_elu(self, alpha, scale, input_scale);
}
Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
    if (self.has_names()) {
        AT_ERROR("elu_", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_elu_(self, alpha, scale, input_scale);
}
Tensor hardsigmoid(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("hardsigmoid", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardsigmoid(self);
}
Tensor & hardtanh_out_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val) {
    if (out.has_names() || self.has_names()) {
        AT_ERROR("hardtanh_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardtanh_out(out, self, min_val, max_val);
}
Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
    if (self.has_names()) {
        AT_ERROR("hardtanh", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardtanh(self, min_val, max_val);
}
Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
    if (self.has_names()) {
        AT_ERROR("hardtanh_", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardtanh_(self, min_val, max_val);
}
Tensor hardswish(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("hardswish", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardswish(self);
}
Tensor & hardswish_(Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("hardswish_", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_hardswish_(self);
}
Tensor & leaky_relu_out_out(Tensor & out, const Tensor & self, Scalar negative_slope) {
    if (out.has_names() || self.has_names()) {
        AT_ERROR("leaky_relu_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_leaky_relu_out(out, self, negative_slope);
}
Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
    if (self.has_names()) {
        AT_ERROR("leaky_relu", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_leaky_relu(self, negative_slope);
}
Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
    if (self.has_names()) {
        AT_ERROR("leaky_relu_", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_leaky_relu_(self, negative_slope);
}
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
    if (self.has_names()) {
        AT_ERROR("_adaptive_avg_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_adaptive_avg_pool2d(self, output_size);
}
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    if (self.has_names()) {
        AT_ERROR("avg_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    if (self.has_names()) {
        AT_ERROR("avg_pool3d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    if (self.has_names()) {
        AT_ERROR("upsample_bilinear2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_upsample_bilinear2d_cpu(self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    if (self.has_names()) {
        AT_ERROR("upsample_nearest2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_upsample_nearest2d_cpu(self, output_size, scales_h, scales_w);
}
Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    if (self.has_names()) {
        AT_ERROR("upsample_nearest3d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_upsample_nearest3d_cpu(self, output_size, scales_d, scales_h, scales_w);
}

}  // namespace QuantizedCPUType

namespace {
auto registerer = torch::import()
  .impl_UNBOXED("aten::as_strided",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::as_strided)
  .impl_UNBOXED("aten::quantized_batch_norm",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::quantized_batch_norm)
  .impl("aten::clamp",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::clamp)
  .impl_UNBOXED("aten::_empty_affine_quantized",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::_empty_affine_quantized)
  .impl_UNBOXED("aten::_empty_per_channel_affine_quantized",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::_empty_per_channel_affine_quantized)
  .impl_UNBOXED("aten::quantized_max_pool2d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::quantized_max_pool2d)
  .impl_UNBOXED("aten::mean",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::mean)
  .impl_UNBOXED("aten::mean.dim",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::mean_dim)
  .impl_UNBOXED("aten::mean.out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::mean_out_out)
  .impl("aten::relu",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::relu)
  .impl_UNBOXED("aten::relu_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::relu_)
  .impl("aten::sigmoid",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::sigmoid)
  .impl("aten::tanh",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::tanh)
  .impl_UNBOXED("aten::clone",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::clone)
  .impl("aten::dequantize.self",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::dequantize_self)
  .impl_UNBOXED("aten::dequantize.tensors",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::dequantize_tensors)
  .impl("aten::q_scale",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::q_scale)
  .impl("aten::q_zero_point",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::q_zero_point)
  .impl_UNBOXED("aten::q_per_channel_scales",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::q_per_channel_scales)
  .impl_UNBOXED("aten::q_per_channel_zero_points",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::q_per_channel_zero_points)
  .impl_UNBOXED("aten::q_per_channel_axis",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::q_per_channel_axis)
  .impl("aten::int_repr",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::int_repr)
  .impl("aten::qscheme",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::qscheme)
  .impl_UNBOXED("aten::set_.source_Storage_storage_offset",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::set__source_Storage_storage_offset)
  .impl_UNBOXED("aten::set_quantizer_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::set_quantizer_)
  .impl_UNBOXED("aten::view",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::view)
  .impl_UNBOXED("aten::ne.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::ne_out_Scalar_out)
  .impl("aten::ne.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ne_Scalar)
  .impl_UNBOXED("aten::ne.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::ne_out_Tensor_out)
  .impl("aten::ne.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ne_Tensor)
  .impl_UNBOXED("aten::eq.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::eq_out_Scalar_out)
  .impl("aten::eq.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::eq_Scalar)
  .impl_UNBOXED("aten::eq.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::eq_out_Tensor_out)
  .impl("aten::eq.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::eq_Tensor)
  .impl_UNBOXED("aten::ge.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::ge_out_Scalar_out)
  .impl("aten::ge.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ge_Scalar)
  .impl_UNBOXED("aten::ge.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::ge_out_Tensor_out)
  .impl("aten::ge.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ge_Tensor)
  .impl_UNBOXED("aten::le.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::le_out_Scalar_out)
  .impl("aten::le.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::le_Scalar)
  .impl_UNBOXED("aten::le.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::le_out_Tensor_out)
  .impl("aten::le.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::le_Tensor)
  .impl_UNBOXED("aten::gt.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::gt_out_Scalar_out)
  .impl("aten::gt.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::gt_Scalar)
  .impl_UNBOXED("aten::gt.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::gt_out_Tensor_out)
  .impl("aten::gt.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::gt_Tensor)
  .impl_UNBOXED("aten::lt.Scalar_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::lt_out_Scalar_out)
  .impl("aten::lt.Scalar",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::lt_Scalar)
  .impl_UNBOXED("aten::lt.Tensor_out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::lt_out_Tensor_out)
  .impl("aten::lt.Tensor",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::lt_Tensor)
  .impl("aten::min",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::min)
  .impl("aten::max",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::max)
  .impl_UNBOXED("aten::sort",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::sort)
  .impl_UNBOXED("aten::topk",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::topk)
  .impl("aten::equal",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::equal)
  .impl_UNBOXED("aten::_cat",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::_cat)
  .impl_UNBOXED("aten::_cat.out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::_cat_out_out)
  .impl_UNBOXED("aten::elu.out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::elu_out_out)
  .impl("aten::elu",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::elu)
  .impl_UNBOXED("aten::elu_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::elu_)
  .impl("aten::hardsigmoid",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::hardsigmoid)
  .impl_UNBOXED("aten::hardtanh.out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::hardtanh_out_out)
  .impl("aten::hardtanh",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::hardtanh)
  .impl_UNBOXED("aten::hardtanh_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::hardtanh_)
  .impl("aten::hardswish",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::hardswish)
  .impl_UNBOXED("aten::hardswish_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::hardswish_)
  .impl_UNBOXED("aten::leaky_relu.out",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::leaky_relu_out_out)
  .impl("aten::leaky_relu",
        DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::leaky_relu)
  .impl_UNBOXED("aten::leaky_relu_",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::leaky_relu_)
  .impl_UNBOXED("aten::_adaptive_avg_pool2d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::_adaptive_avg_pool2d)
  .impl_UNBOXED("aten::avg_pool2d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::avg_pool2d)
  .impl_UNBOXED("aten::avg_pool3d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::avg_pool3d)
  .impl_UNBOXED("aten::upsample_bilinear2d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::upsample_bilinear2d)
  .impl_UNBOXED("aten::upsample_nearest2d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::upsample_nearest2d)
  .impl_UNBOXED("aten::upsample_nearest3d",
                DispatchKey::QuantizedCPUTensorId,
                QuantizedCPUType::upsample_nearest3d);
}

}
