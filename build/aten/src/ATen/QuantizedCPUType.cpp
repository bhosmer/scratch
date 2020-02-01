// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/QuantizedCPUType.h>

// @generated by aten/src/ATen/gen.py

#include <c10/core/TensorImpl.h>
#include <ATen/CPUGenerator.h>
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
        AT_ERROR(
            "_empty_per_channel_affine_quantized is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const DeviceGuard device_guard(options.device());
    return at::native::empty_per_channel_affine_quantized_cpu(size, scales, zero_points, axis, options, memory_format);
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    if (self.has_names()) {
        AT_ERROR(
            "quantized_max_pool2d is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_mean_cpu(self, dtype);
}
Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_mean_cpu(self, dim, keepdim, dtype);
}
Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {

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
Tensor dequantize(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "dequantize is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::dequantize_quant(self);
}
double q_scale(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "q_scale is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_scale_quant(self);
}
int64_t q_zero_point(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "q_zero_point is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_zero_point_quant(self);
}
Tensor q_per_channel_scales(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "q_per_channel_scales is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_scales_quant(self);
}
Tensor q_per_channel_zero_points(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "q_per_channel_zero_points is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_zero_points_quant(self);
}
int64_t q_per_channel_axis(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "q_per_channel_axis is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::q_per_channel_axis_quant(self);
}
Tensor int_repr(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "int_repr is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::int_repr_quant(self);
}
QScheme qscheme(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR(
            "qscheme is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::qscheme_quant(self);
}
Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
    if (self.has_names()) {
        AT_ERROR(
            "set_ is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    // DeviceGuard omitted
    return at::native::set_storage(self, source, storage_offset, size, stride);
}
Tensor & set_quantizer_(Tensor & self, ConstQuantizerPtr quantizer) {
    if (self.has_names()) {
        AT_ERROR(
            "set_quantizer_ is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::set_quantizer_(self, quantizer);
}
Tensor view(const Tensor & self, IntArrayRef size) {
    if (self.has_names()) {
        AT_ERROR(
            "view is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    // DeviceGuard omitted
    return at::native::view(self, size);
}
Tensor & ne_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_out_quantized_cpu(out, self, other);
}
Tensor ne(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_quantized_cpu(self, other);
}
Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_out_quantized_cpu(out, self, other);
}
Tensor ne(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ne_quantized_cpu(self, other);
}
Tensor & eq_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_out_quantized_cpu(out, self, other);
}
Tensor eq(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_quantized_cpu(self, other);
}
Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_out_quantized_cpu(out, self, other);
}
Tensor eq(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::eq_quantized_cpu(self, other);
}
Tensor & ge_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_out_quantized_cpu(out, self, other);
}
Tensor ge(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_quantized_cpu(self, other);
}
Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_out_quantized_cpu(out, self, other);
}
Tensor ge(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::ge_quantized_cpu(self, other);
}
Tensor & le_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_out_quantized_cpu(out, self, other);
}
Tensor le(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_quantized_cpu(self, other);
}
Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_out_quantized_cpu(out, self, other);
}
Tensor le(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::le_quantized_cpu(self, other);
}
Tensor & gt_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_out_quantized_cpu(out, self, other);
}
Tensor gt(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_quantized_cpu(self, other);
}
Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_out_quantized_cpu(out, self, other);
}
Tensor gt(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::gt_quantized_cpu(self, other);
}
Tensor & lt_out(Tensor & out, const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_out_quantized_cpu(out, self, other);
}
Tensor lt(const Tensor & self, Scalar other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_quantized_cpu(self, other);
}
Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::lt_out_quantized_cpu(out, self, other);
}
Tensor lt(const Tensor & self, const Tensor & other) {

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
        AT_ERROR(
            "sort is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::sort_quant(self, dim, descending);
}
std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    if (self.has_names()) {
        AT_ERROR(
            "topk is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
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
        AT_ERROR(
            "_cat is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(tensors));
    return at::native::quantized_cat(tensors, dim);
}
Tensor & _cat_out(Tensor & out, TensorList tensors, int64_t dim) {
    if (out.has_names() || at::has_names(tensors)) {
        AT_ERROR(
            "_cat_out is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(out));
    return at::native::quantized_cat_out(out, tensors, dim);
}
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
    if (self.has_names()) {
        AT_ERROR(
            "_adaptive_avg_pool2d is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_adaptive_avg_pool2d(self, output_size);
}
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    if (self.has_names()) {
        AT_ERROR(
            "avg_pool2d is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    if (self.has_names()) {
        AT_ERROR(
            "upsample_bilinear2d is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_upsample_bilinear2d_cpu(self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    if (self.has_names()) {
        AT_ERROR(
            "upsample_nearest2d is not yet supported with named tensors. Please drop names via "
            "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
            "and set names on the result of the operation.");
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::quantized_upsample_nearest2d_cpu(self, output_size, scales_h, scales_w);
}

}  // namespace QuantizedCPUType

#ifndef USE_STATIC_DISPATCH
namespace {
auto registerer = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>), &QuantizedCPUType::as_strided>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")
    .kernel<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::clamp)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>), &QuantizedCPUType::_empty_affine_quantized>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>), &QuantizedCPUType::_empty_per_channel_affine_quantized>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool), &QuantizedCPUType::quantized_max_pool2d>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, c10::optional<ScalarType>), &QuantizedCPUType::mean>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>), &QuantizedCPUType::mean>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>), &QuantizedCPUType::mean_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::relu(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::relu)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &QuantizedCPUType::relu_>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sigmoid(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::sigmoid)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::tanh(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::tanh)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, c10::optional<MemoryFormat>), &QuantizedCPUType::clone>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::dequantize(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::dequantize)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_scale(Tensor self) -> float")
    .kernel<double (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::q_scale)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_zero_point(Tensor self) -> int")
    .kernel<int64_t (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::q_zero_point)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_per_channel_scales(Tensor self) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &QuantizedCPUType::q_per_channel_scales>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_per_channel_zero_points(Tensor self) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &), &QuantizedCPUType::q_per_channel_zero_points>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::q_per_channel_axis(Tensor self) -> int")
    .impl_unboxedOnlyKernel<int64_t (const Tensor &), &QuantizedCPUType::q_per_channel_axis>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::int_repr(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::int_repr)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::qscheme(Tensor self) -> QScheme")
    .kernel<QScheme (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::qscheme)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef), &QuantizedCPUType::set_>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, ConstQuantizerPtr), &QuantizedCPUType::set_quantizer_>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &QuantizedCPUType::view>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::ne_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ne)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::ne_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ne.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ne)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::eq_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eq.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::eq)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::eq_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::eq)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::ge_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ge.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ge)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::ge_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::ge.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::ge)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::le_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::le.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::le)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::le_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::le.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::le)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::gt_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gt.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::gt)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::gt_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::gt.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::gt)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar), &QuantizedCPUType::lt_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lt.Scalar(Tensor self, Scalar other) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::lt)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), &QuantizedCPUType::lt_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::lt.Tensor(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::lt)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::min(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::min)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::max(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::max)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool), &QuantizedCPUType::sort>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool), &QuantizedCPUType::topk>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::equal(Tensor self, Tensor other) -> bool")
    .kernel<bool (const Tensor &, const Tensor &)>(DispatchKey::QuantizedCPUTensorId, &QuantizedCPUType::equal)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cat(Tensor[] tensors, int dim=0) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (TensorList, int64_t), &QuantizedCPUType::_cat>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, TensorList, int64_t), &QuantizedCPUType::_cat_out>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef), &QuantizedCPUType::_adaptive_avg_pool2d>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), &QuantizedCPUType::avg_pool2d>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>), &QuantizedCPUType::upsample_bilinear2d>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>), &QuantizedCPUType::upsample_nearest2d>(DispatchKey::QuantizedCPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
}
#endif

}
