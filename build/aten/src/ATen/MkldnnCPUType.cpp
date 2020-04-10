// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/MkldnnCPUType.h>

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
Tensor * MkldnnCPUType::add(Tensor & a, Tensor & b) {
  std::cout << "add Tensor with backend MkldnnCPU\n";
  return &a;
}
*/

namespace MkldnnCPUType {

Tensor add_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_add(self, other, alpha);
}
Tensor & add__Tensor(Tensor & self, const Tensor & other, Scalar alpha) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_add_(self, other, alpha);
}
Tensor & add_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_add_out(out, self, other, alpha);
}
Tensor empty_memory_format(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {

    const DeviceGuard device_guard(options.device());
    return at::native::empty_mkldnn(size, options, memory_format);
}
Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
    if (input.has_names() || weight.has_names() || bias.has_names()) {
        AT_ERROR("mkldnn_linear", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::mkldnn_linear(input, weight, bias);
}
Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    if (self.has_names()) {
        AT_ERROR("mkldnn_max_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor mul_Tensor(const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_mul(self, other);
}
Tensor & mul__Tensor(Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_mul_(self, other);
}
Tensor & mul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_mul_out(out, self, other);
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    if (input.has_names() || weight.has_names() || bias.has_names() || running_mean.has_names() || running_var.has_names()) {
        AT_ERROR("native_batch_norm", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::mkldnn_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}
Tensor _mkldnn_reshape(const Tensor & self, IntArrayRef shape) {
    if (self.has_names()) {
        AT_ERROR("_mkldnn_reshape", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::mkldnn_reshape(self, shape);
}
Tensor relu(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_relu(self);
}
Tensor & relu_(Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_relu_(self);
}
Tensor sigmoid(const Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_sigmoid(self);
}
Tensor & sigmoid_(Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_sigmoid_(self);
}
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
    if (self.has_names()) {
        AT_ERROR("_softmax", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_softmax(self, dim, half_to_float);
}
Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
    if (self.has_names()) {
        AT_ERROR("_mkldnn_transpose", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::mkldnn_transpose(self, dim0, dim1);
}
Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
    if (self.has_names()) {
        AT_ERROR("_mkldnn_transpose_", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::mkldnn_transpose_(self, dim0, dim1);
}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_clone(self, memory_format);
}
Tensor & zero_(Tensor & self) {

    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_zero_(self);
}
Tensor to_dense(const Tensor & self) {
    if (self.has_names()) {
        AT_ERROR("to_dense", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_to_dense(self);
}
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
    if (self.has_names()) {
        AT_ERROR("mkldnn_reorder_conv2d_weight", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
}
Tensor view(const Tensor & self, IntArrayRef size) {
    if (self.has_names()) {
        AT_ERROR("view", named_tensors_unsupported_error);
    }
    // DeviceGuard omitted
    return at::native::mkldnn_view(self, size);
}
Tensor & adaptive_avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
    if (out.has_names() || self.has_names()) {
        AT_ERROR("adaptive_avg_pool2d_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_adaptive_avg_pool2d_out(out, self, output_size);
}
Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
    if (self.has_names()) {
        AT_ERROR("mkldnn_adaptive_avg_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_adaptive_avg_pool2d(self, output_size);
}
Tensor & avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    if (out.has_names() || self.has_names()) {
        AT_ERROR("avg_pool2d_out", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_avg_pool2d_out(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    if (self.has_names()) {
        AT_ERROR("avg_pool2d", named_tensors_unsupported_error);
    }
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mkldnn_avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

}  // namespace MkldnnCPUType

namespace {
auto registerer = torch::import()
  .impl("aten::add.Tensor",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::add_Tensor)
  .impl_UNBOXED("aten::add_.Tensor",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::add__Tensor)
  .impl_UNBOXED("aten::add.out",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::add_out_out)
  .impl_UNBOXED("aten::empty.memory_format",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::empty_memory_format)
  .impl_UNBOXED("aten::mkldnn_linear",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mkldnn_linear)
  .impl_UNBOXED("aten::mkldnn_max_pool2d",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mkldnn_max_pool2d)
  .impl("aten::mul.Tensor",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::mul_Tensor)
  .impl_UNBOXED("aten::mul_.Tensor",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mul__Tensor)
  .impl_UNBOXED("aten::mul.out",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mul_out_out)
  .impl_UNBOXED("aten::native_batch_norm",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::native_batch_norm)
  .impl_UNBOXED("aten::_mkldnn_reshape",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::_mkldnn_reshape)
  .impl("aten::relu",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::relu)
  .impl_UNBOXED("aten::relu_",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::relu_)
  .impl("aten::sigmoid",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::sigmoid)
  .impl_UNBOXED("aten::sigmoid_",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::sigmoid_)
  .impl("aten::_softmax",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::_softmax)
  .impl("aten::_mkldnn_transpose",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::_mkldnn_transpose)
  .impl_UNBOXED("aten::_mkldnn_transpose_",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::_mkldnn_transpose_)
  .impl_UNBOXED("aten::clone",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::clone)
  .impl_UNBOXED("aten::zero_",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::zero_)
  .impl("aten::to_dense",
        DispatchKey::MkldnnCPUTensorId, &MkldnnCPUType::to_dense)
  .impl_UNBOXED("aten::mkldnn_reorder_conv2d_weight",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mkldnn_reorder_conv2d_weight)
  .impl_UNBOXED("aten::view",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::view)
  .impl_UNBOXED("aten::adaptive_avg_pool2d.out",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::adaptive_avg_pool2d_out_out)
  .impl_UNBOXED("aten::mkldnn_adaptive_avg_pool2d",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::mkldnn_adaptive_avg_pool2d)
  .impl_UNBOXED("aten::avg_pool2d.out",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::avg_pool2d_out_out)
  .impl_UNBOXED("aten::avg_pool2d",
                DispatchKey::MkldnnCPUTensorId,
                MkldnnCPUType::avg_pool2d);
}

}
