#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>

// @generated from tools/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

namespace VariableType {
// Comment the anonymous namespace so that the generated functions
// can be accessed from outside of the files (register_mobile_autograd.cpp).
// Later when we merge the mobile op registration the anonymous namespace
// will be restored.
// namespace {
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const Tensor & reservedSpace) {
  RECORD_FUNCTION("_batch_norm_impl_index_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, reservedSpace}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_batch_norm_impl_index_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "impl_index", impl_index);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var_transform", save_var_transform);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    jit::tracer::addInputs(node, "reservedSpace", reservedSpace);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1, result2) = TypeDefault::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor _cast_Char(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Char", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Char");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_cast_Char(self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cast_Float(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Float", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Float");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_cast_Float(self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper) {
  RECORD_FUNCTION("_cholesky_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cholesky_solve_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cholesky_solve_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cholesky_solve_helper(self_, A_, upper);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) {
  RECORD_FUNCTION("_convolution_nogroup", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution_nogroup");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) {
  RECORD_FUNCTION("_copy_from", std::vector<c10::IValue>({self, dst}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& dst_ = unpack(dst, "dst", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, dst )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_copy_from"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, dst ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_copy_from");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dst", dst);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> dst__storage_saved =
    dst_.has_storage() ? c10::optional<Storage>(dst_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dst__impl_saved;
  if (dst_.defined()) dst__impl_saved = dst_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_copy_from(self_, dst_, non_blocking);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (dst__storage_saved.has_value())
    AT_ASSERT(dst__storage_saved.value().is_alias_of(dst_.storage()));
  if (dst__impl_saved) AT_ASSERT(dst__impl_saved == dst_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  RECORD_FUNCTION("_ctc_loss_backward", std::vector<c10::IValue>({grad, log_probs, targets, neg_log_likelihood, log_alpha}), Node::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& log_probs_ = unpack(log_probs, "log_probs", 1);
  auto& targets_ = unpack(targets, "targets", 2);
  auto& neg_log_likelihood_ = unpack(neg_log_likelihood, "neg_log_likelihood", 5);
  auto& log_alpha_ = unpack(log_alpha, "log_alpha", 6);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, log_probs, targets, neg_log_likelihood, log_alpha )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_ctc_loss_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, log_probs, targets, neg_log_likelihood, log_alpha ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  c10::optional<Storage> neg_log_likelihood__storage_saved =
    neg_log_likelihood_.has_storage() ? c10::optional<Storage>(neg_log_likelihood_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> neg_log_likelihood__impl_saved;
  if (neg_log_likelihood_.defined()) neg_log_likelihood__impl_saved = neg_log_likelihood_.getIntrusivePtr();
  c10::optional<Storage> log_alpha__storage_saved =
    log_alpha_.has_storage() ? c10::optional<Storage>(log_alpha_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_alpha__impl_saved;
  if (log_alpha_.defined()) log_alpha__impl_saved = log_alpha_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_ctc_loss_backward(grad_, log_probs_, targets_, input_lengths, target_lengths, neg_log_likelihood_, log_alpha_, blank, zero_infinity);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  if (neg_log_likelihood__storage_saved.has_value())
    AT_ASSERT(neg_log_likelihood__storage_saved.value().is_alias_of(neg_log_likelihood_.storage()));
  if (neg_log_likelihood__impl_saved) AT_ASSERT(neg_log_likelihood__impl_saved == neg_log_likelihood_.getIntrusivePtr());
  if (log_alpha__storage_saved.has_value())
    AT_ASSERT(log_alpha__storage_saved.value().is_alias_of(log_alpha_.storage()));
  if (log_alpha__impl_saved) AT_ASSERT(log_alpha__impl_saved == log_alpha_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  RECORD_FUNCTION("_cudnn_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CudnnCtcLossBackward> grad_fn;
  if (compute_requires_grad( log_probs )) {
    grad_fn = std::shared_ptr<CudnnCtcLossBackward>(new CudnnCtcLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
    grad_fn->zero_infinity = zero_infinity;
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cudnn_ctc_loss(log_probs_, targets_, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  RECORD_FUNCTION("_cufft_set_plan_cache_max_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  TypeDefault::_cufft_set_plan_cache_max_size(device_index, max_size);
}
void _cummin_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  RECORD_FUNCTION("_cummin_helper", std::vector<c10::IValue>({self, values, indices}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_cummin_helper(self_, values_, indices_, dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
}
Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  RECORD_FUNCTION("_index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, index, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_index_copy_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, index, source ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_index_copy_(self_, dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _inverse_helper(const Tensor & self) {
  RECORD_FUNCTION("_inverse_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_inverse_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_inverse_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_inverse_helper(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale) {
  RECORD_FUNCTION("_masked_scale", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mask )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_masked_scale"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mask ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_masked_scale");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_masked_scale(self_, mask_, scale);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _min(const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("_min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_min"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_min(self_, dim, keepdim);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
bool _nnpack_available() {
  RECORD_FUNCTION("_nnpack_available", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::_nnpack_available();
  return result;
}
Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
  RECORD_FUNCTION("_pdist_backward", std::vector<c10::IValue>({grad, self, pdist}), Node::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& pdist_ = unpack(pdist, "pdist", 3);
  std::shared_ptr<PdistBackwardBackward> grad_fn;
  if (compute_requires_grad( grad, self, pdist )) {
    grad_fn = std::shared_ptr<PdistBackwardBackward>(new PdistBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self, pdist ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pdist_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "pdist", pdist);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> pdist__storage_saved =
    pdist_.has_storage() ? c10::optional<Storage>(pdist_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> pdist__impl_saved;
  if (pdist_.defined()) pdist__impl_saved = pdist_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_pdist_backward(grad_, self_, p, pdist_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (pdist__storage_saved.has_value())
    AT_ASSERT(pdist__storage_saved.value().is_alias_of(pdist_.storage()));
  if (pdist__impl_saved) AT_ASSERT(pdist__impl_saved == pdist_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("_s_where", std::vector<c10::IValue>({condition, self, other}), Node::peek_at_next_sequence_nr());
  auto& condition_ = unpack(condition, "condition", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<SWhereBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SWhereBackward>(new SWhereBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->condition_ = SavedVariable(condition, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_s_where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> condition__storage_saved =
    condition_.has_storage() ? c10::optional<Storage>(condition_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> condition__impl_saved;
  if (condition_.defined()) condition__impl_saved = condition_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_s_where(condition_, self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (condition__storage_saved.has_value())
    AT_ASSERT(condition__storage_saved.value().is_alias_of(condition_.storage()));
  if (condition__impl_saved) AT_ASSERT(condition__impl_saved == condition_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sample_dirichlet(const Tensor & self, Generator generator) {
  RECORD_FUNCTION("_sample_dirichlet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sample_dirichlet"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sample_dirichlet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sample_dirichlet(self_, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension) {
  RECORD_FUNCTION("_sobol_engine_scramble_", std::vector<c10::IValue>({self, ltm}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_scramble");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_scramble_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ltm", ltm);
    jit::tracer::addInputs(node, "dimension", dimension);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_scramble_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::_sobol_engine_scramble_(self, ltm, dimension);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("_sparse_addmm", std::vector<c10::IValue>({self, sparse, dense, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& sparse_ = unpack(sparse, "sparse", 1);
  auto& dense_ = unpack(dense, "dense", 2);
  std::shared_ptr<SparseAddmmBackward> grad_fn;
  if (compute_requires_grad( self, sparse, dense )) {
    grad_fn = std::shared_ptr<SparseAddmmBackward>(new SparseAddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, sparse, dense ));
    grad_fn->sparse_ = SavedVariable(sparse, false);
    grad_fn->dense_sizes = dense.sizes().vec();
    grad_fn->dense_ = SavedVariable(dense, false);
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> sparse__storage_saved =
    sparse_.has_storage() ? c10::optional<Storage>(sparse_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sparse__impl_saved;
  if (sparse_.defined()) sparse__impl_saved = sparse_.getIntrusivePtr();
  c10::optional<Storage> dense__storage_saved =
    dense_.has_storage() ? c10::optional<Storage>(dense_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dense__impl_saved;
  if (dense_.defined()) dense__impl_saved = dense_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_addmm(self_, sparse_, dense_, beta, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (sparse__storage_saved.has_value())
    AT_ASSERT(sparse__storage_saved.value().is_alias_of(sparse_.storage()));
  if (sparse__impl_saved) AT_ASSERT(sparse__impl_saved == sparse_.getIntrusivePtr());
  if (dense__storage_saved.has_value())
    AT_ASSERT(dense__storage_saved.value().is_alias_of(dense_.storage()));
  if (dense__impl_saved) AT_ASSERT(dense__impl_saved == dense_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("_sparse_coo_tensor_unsafe", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_unsafe");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_sparse_coo_tensor_unsafe(indices, values, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims_and_tensors", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  auto& indices_ = unpack(indices, "indices", 3);
  auto& values_ = unpack(values, "values", 4);
  auto options_ = TensorOptions(options);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward> grad_fn;
  if (compute_requires_grad( values )) {
    grad_fn = std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward>(new SparseCooTensorWithDimsAndTensorsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( values ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->values_sizes = values.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices_, values_, options_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _standard_gamma(const Tensor & self, Generator generator) {
  RECORD_FUNCTION("_standard_gamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<StandardGammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<StandardGammaBackward>(new StandardGammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_standard_gamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_standard_gamma(self_, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper) {
  RECORD_FUNCTION("_symeig_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_symeig_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_symeig_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_symeig_helper(self_, eigenvectors, upper);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) {
  RECORD_FUNCTION("_thnn_fused_lstm_cell", std::vector<c10::IValue>({input_gates, hidden_gates, cx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto input_bias_ = unpack_opt(input_bias, "input_bias", 3);
  auto hidden_bias_ = unpack_opt(hidden_bias, "hidden_bias", 4);
  std::shared_ptr<ThnnFusedLstmCellBackward> grad_fn;
  if (compute_requires_grad( input_gates, hidden_gates, cx, input_bias, hidden_bias )) {
    grad_fn = std::shared_ptr<ThnnFusedLstmCellBackward>(new ThnnFusedLstmCellBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input_gates, hidden_gates, cx, input_bias, hidden_bias ));
    grad_fn->input_gates_ = SavedVariable(input_gates, false);
    grad_fn->hidden_gates_ = SavedVariable(hidden_gates, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->input_bias_ = SavedVariable(input_bias, false);
    grad_fn->hidden_bias_ = SavedVariable(hidden_bias, false);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input_gates__storage_saved =
    input_gates_.has_storage() ? c10::optional<Storage>(input_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input_gates__impl_saved;
  if (input_gates_.defined()) input_gates__impl_saved = input_gates_.getIntrusivePtr();
  c10::optional<Storage> hidden_gates__storage_saved =
    hidden_gates_.has_storage() ? c10::optional<Storage>(hidden_gates_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hidden_gates__impl_saved;
  if (hidden_gates_.defined()) hidden_gates__impl_saved = hidden_gates_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  c10::optional<Storage> input_bias__storage_saved =
    input_bias_.has_storage() ? c10::optional<Storage>(input_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input_bias__impl_saved;
  if (input_bias_.defined()) input_bias__impl_saved = input_bias_.getIntrusivePtr();
  c10::optional<Storage> hidden_bias__storage_saved =
    hidden_bias_.has_storage() ? c10::optional<Storage>(hidden_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hidden_bias__impl_saved;
  if (hidden_bias_.defined()) hidden_bias__impl_saved = hidden_bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_thnn_fused_lstm_cell(input_gates_, hidden_gates_, cx_, input_bias_, hidden_bias_);
  })();
  std::tie(result0, result1, result2) = std::move(tmp);
  #ifndef NDEBUG
  if (input_gates__storage_saved.has_value())
    AT_ASSERT(input_gates__storage_saved.value().is_alias_of(input_gates_.storage()));
  if (input_gates__impl_saved) AT_ASSERT(input_gates__impl_saved == input_gates_.getIntrusivePtr());
  if (hidden_gates__storage_saved.has_value())
    AT_ASSERT(hidden_gates__storage_saved.value().is_alias_of(hidden_gates_.storage()));
  if (hidden_gates__impl_saved) AT_ASSERT(hidden_gates__impl_saved == hidden_gates_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  if (input_bias__storage_saved.has_value())
    AT_ASSERT(input_bias__storage_saved.value().is_alias_of(input_bias_.storage()));
  if (input_bias__impl_saved) AT_ASSERT(input_bias__impl_saved == input_bias_.getIntrusivePtr());
  if (hidden_bias__storage_saved.has_value())
    AT_ASSERT(hidden_bias__storage_saved.value().is_alias_of(hidden_bias_.storage()));
  if (hidden_bias__impl_saved) AT_ASSERT(hidden_bias__impl_saved == hidden_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  RECORD_FUNCTION("_triangular_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_triangular_solve_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_triangular_solve_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_triangular_solve_helper(self_, A_, upper, transpose, unitriangular);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _var(const Tensor & self, bool unbiased) {
  RECORD_FUNCTION("_var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_var"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_var(self_, unbiased);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & abs_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("abs_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("abs");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("abs");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::abs_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool2DBackward>(new AdaptiveMaxPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::adaptive_max_pool2d(self_, output_size);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & adaptive_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  RECORD_FUNCTION("adaptive_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::adaptive_max_pool2d_backward_out(grad_input_, grad_output_, self_, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_max_pool3d_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_max_pool3d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::adaptive_max_pool3d_out(out_, indices_, self_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(out, indices);
}
Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcdiv", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addcdiv(self_, tensor1_, tensor2_, value);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcdiv_", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->tensor1_ = SavedVariable(tensor1, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addcdiv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addcdiv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.addcdiv_(tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & addcmul_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcmul_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcmul");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addcmul");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addcmul_out(out_, self_, tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & addmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmm_out", std::vector<c10::IValue>({out, self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat1_ = unpack(mat1, "mat1", 2);
  auto& mat2_ = unpack(mat2, "mat2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("addmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addmm");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addmm_out(out_, self_, mat1_, mat2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmv", std::vector<c10::IValue>({self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->mat_ = SavedVariable(mat, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addmv(self_, mat_, vec_, beta, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmv_", std::vector<c10::IValue>({self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  check_inplace(self);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->mat_ = SavedVariable(mat, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addmv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addmv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addmv_(self_, mat_, vec_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor alpha_dropout(const Tensor & input, double p, bool train) {
  RECORD_FUNCTION("alpha_dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::alpha_dropout(input, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & alpha_dropout_(Tensor & self, double p, bool train) {
  RECORD_FUNCTION("alpha_dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::alpha_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::alpha_dropout_(self, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor angle(const Tensor & self) {
  RECORD_FUNCTION("angle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AngleBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AngleBackward>(new AngleBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::angle");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::angle(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & atan2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("atan2_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("atan2");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("atan2");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::atan2_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & atan_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("atan_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("atan");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("atan");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::atan_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  RECORD_FUNCTION("avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("avg_pool2d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::avg_pool2d_out(out_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) {
  RECORD_FUNCTION("batch_norm_backward_elemt", std::vector<c10::IValue>({grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu}), Node::peek_at_next_sequence_nr());
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& mean_ = unpack(mean, "mean", 2);
  auto& invstd_ = unpack(invstd, "invstd", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& mean_dy_ = unpack(mean_dy, "mean_dy", 5);
  auto& mean_dy_xmu_ = unpack(mean_dy_xmu, "mean_dy_xmu", 6);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_backward_elemt"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_backward_elemt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "mean_dy", mean_dy);
    jit::tracer::addInputs(node, "mean_dy_xmu", mean_dy_xmu);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_out__storage_saved =
    grad_out_.has_storage() ? c10::optional<Storage>(grad_out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> mean_dy__storage_saved =
    mean_dy_.has_storage() ? c10::optional<Storage>(mean_dy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean_dy__impl_saved;
  if (mean_dy_.defined()) mean_dy__impl_saved = mean_dy_.getIntrusivePtr();
  c10::optional<Storage> mean_dy_xmu__storage_saved =
    mean_dy_xmu_.has_storage() ? c10::optional<Storage>(mean_dy_xmu_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean_dy_xmu__impl_saved;
  if (mean_dy_xmu_.defined()) mean_dy_xmu__impl_saved = mean_dy_xmu_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::batch_norm_backward_elemt(grad_out_, input_, mean_, invstd_, weight_, mean_dy_, mean_dy_xmu_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value())
    AT_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved) AT_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (mean_dy__storage_saved.has_value())
    AT_ASSERT(mean_dy__storage_saved.value().is_alias_of(mean_dy_.storage()));
  if (mean_dy__impl_saved) AT_ASSERT(mean_dy__impl_saved == mean_dy_.getIntrusivePtr());
  if (mean_dy_xmu__storage_saved.has_value())
    AT_ASSERT(mean_dy_xmu__storage_saved.value().is_alias_of(mean_dy_xmu_.storage()));
  if (mean_dy_xmu__impl_saved) AT_ASSERT(mean_dy_xmu__impl_saved == mean_dy_xmu_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, IntArrayRef counts) {
  RECORD_FUNCTION("batch_norm_gather_stats_with_counts", std::vector<c10::IValue>({input, mean, invstd, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& invstd_ = unpack(invstd, "invstd", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, mean, invstd, running_mean, running_var )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_gather_stats_with_counts"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, mean, invstd, running_mean, running_var ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_gather_stats_with_counts");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "counts", counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::batch_norm_gather_stats_with_counts(input_, mean_, invstd_, running_mean_, running_var_, momentum, eps, counts);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("binary_cross_entropy", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<BinaryCrossEntropyBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyBackward>(new BinaryCrossEntropyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::binary_cross_entropy(self_, target_, weight_, reduction);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & binary_cross_entropy_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("binary_cross_entropy_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::binary_cross_entropy_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor bitwise_or_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("bitwise_or", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bitwise_or_Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bitwise_or_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("bitwise_or", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bitwise_or_Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & bitwise_or__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("bitwise_or_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::bitwise_or__Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & bitwise_or__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("bitwise_or_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::bitwise_or__Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor bitwise_xor_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("bitwise_xor", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bitwise_xor_Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bitwise_xor_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("bitwise_xor", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bitwise_xor_Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & bitwise_xor__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("bitwise_xor_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::bitwise_xor__Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & bitwise_xor__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("bitwise_xor_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::bitwise_xor__Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor blackman_window(int64_t window_length, const TensorOptions & options) {
  RECORD_FUNCTION("blackman_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::blackman_window(window_length, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor blackman_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  RECORD_FUNCTION("blackman_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::blackman_window_periodic(window_length, periodic, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  RECORD_FUNCTION("broadcast_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::broadcast_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::broadcast_tensors(tensors);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & cholesky_inverse_out_out(Tensor & out, const Tensor & self, bool upper) {
  RECORD_FUNCTION("cholesky_inverse_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cholesky_inverse");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky_inverse");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_inverse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cholesky_inverse_out(out_, self_, upper);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) {
  RECORD_FUNCTION("cholesky_solve", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<CholeskySolveBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<CholeskySolveBackward>(new CholeskySolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->input2_ = SavedVariable(input2, false);
    grad_fn->upper = upper;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cholesky_solve(self_, input2_, upper);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor clamp_min(const Tensor & self, Scalar min) {
  RECORD_FUNCTION("clamp_min", std::vector<c10::IValue>({self, min}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::clamp_min(self_, min);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & clamp_min_(Tensor & self, Scalar min) {
  RECORD_FUNCTION("clamp_min_", std::vector<c10::IValue>({self, min}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp_min");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_min_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_min_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::clamp_min_(self_, min);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & clamp_out_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  RECORD_FUNCTION("clamp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::clamp_out(out_, self_, min, max);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("clone", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CloneBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CloneBackward>(new CloneBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clone");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::clone(self_, memory_format);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor contiguous(const Tensor & self, MemoryFormat memory_format) {
  RECORD_FUNCTION("contiguous", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::contiguous");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::contiguous(self, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("conv3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv3d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor conv_transpose2d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  RECORD_FUNCTION("conv_transpose2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose2d_input(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  RECORD_FUNCTION("convolution_overrideable", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<ConvolutionOverrideableBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<ConvolutionOverrideableBackward>(new ConvolutionOverrideableBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->transposed = transposed;
    grad_fn->output_padding = output_padding.vec();
    grad_fn->groups = groups;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::convolution_overrideable");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::convolution_overrideable(input_, weight_, bias_, stride, padding, dilation, transposed, output_padding, groups);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & cos_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("cos_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cos");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cos");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cos_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor cosh(const Tensor & self) {
  RECORD_FUNCTION("cosh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cosh(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & cosh_(Tensor & self) {
  RECORD_FUNCTION("cosh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cosh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cosh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::cosh_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  RECORD_FUNCTION("cross", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<CrossBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<CrossBackward>(new CrossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cross(self_, other_, dim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  RECORD_FUNCTION("cudnn_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnBatchNormBackward>(new CudnnBatchNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, exponential_average_factor, epsilon);
  })();
  std::tie(result0, result1, result2, result3) = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  #endif
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
std::tuple<Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  RECORD_FUNCTION("cudnn_convolution_transpose_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackwardBackward>(new CudnnConvolutionTransposeBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_convolution_transpose_backward(self_, grad_output_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("cudnn_convolution_transpose_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_input"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_convolution_transpose_backward_input(grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {
  RECORD_FUNCTION("cudnn_grid_sampler", std::vector<c10::IValue>({self, grid}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<CudnnGridSamplerBackward> grad_fn;
  if (compute_requires_grad( self, grid )) {
    grad_fn = std::shared_ptr<CudnnGridSamplerBackward>(new CudnnGridSamplerBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grid ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grid_ = SavedVariable(grid, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_grid_sampler(self_, grid_);
  })();
  auto output = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  #endif
  return output;
}
bool cudnn_is_acceptable(const Tensor & self) {
  RECORD_FUNCTION("cudnn_is_acceptable", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::cudnn_is_acceptable(self);
  return result;
}
std::tuple<Tensor,Tensor> cummin(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("cummin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumminBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CumminBackward>(new CumminBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cummin(self_, dim);
  })();
  std::tie(values, indices) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> cummin_dimname(const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("cummin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  Tensor values;
  Tensor indices;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(values, indices) = TypeDefault::cummin_dimname(self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  RECORD_FUNCTION("diag_embed", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag_embed");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::diag_embed(self, offset, dim1, dim2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & diag_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("diag_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("diag");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("diag");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diag_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::diag_out(out_, self_, diagonal);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & digamma_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("digamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("digamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("digamma");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::digamma_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
  RECORD_FUNCTION("eig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<EigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<EigBackward>(new EigBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eigenvectors = eigenvectors;
  }
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::eig(self_, eigenvectors);
  })();
  std::tie(eigenvalues, eigenvectors_return) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( eigenvalues, eigenvectors_return ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors_return);
  }
  #endif
  if (grad_fn) {
    grad_fn->eigenvalues_ = SavedVariable(eigenvalues, true);
    grad_fn->eigenvectors_return_ = SavedVariable(eigenvectors_return, true);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
Tensor empty_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("empty", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::empty_names(size, names, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_memory_format(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("empty", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty(size, options_, memory_format);
  })();
  auto result = std::move(tmp);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor eq_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("eq", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::eq(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor eq_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("eq", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::eq(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & eq__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("eq_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<EqBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<EqBackward0>(new EqBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::eq");
    } else {
      op_name = jit::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.eq_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & eq__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("eq_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<EqBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<EqBackward1>(new EqBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::eq");
    } else {
      op_name = jit::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.eq_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & erfinv_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("erfinv_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfinv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("erfinv");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::erfinv_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  RECORD_FUNCTION("fake_quantize_per_tensor_affine", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FakeQuantizePerTensorAffineBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FakeQuantizePerTensorAffineBackward>(new FakeQuantizePerTensorAffineBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->scale = scale;
    grad_fn->zero_point = zero_point;
    grad_fn->quant_min = quant_min;
    grad_fn->quant_max = quant_max;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fake_quantize_per_tensor_affine(self_, scale, zero_point, quant_min, quant_max);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_pack_quantized_matrix(const Tensor & input) {
  RECORD_FUNCTION("fbgemm_pack_quantized_matrix", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::fbgemm_pack_quantized_matrix(input);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_pack_quantized_matrix_KN(const Tensor & input, int64_t K, int64_t N) {
  RECORD_FUNCTION("fbgemm_pack_quantized_matrix", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "K", K);
    jit::tracer::addInputs(node, "N", N);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::fbgemm_pack_quantized_matrix_KN(input, K, N);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor feature_dropout(const Tensor & input, double p, bool train) {
  RECORD_FUNCTION("feature_dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::feature_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::feature_dropout(input, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & feature_dropout_(Tensor & self, double p, bool train) {
  RECORD_FUNCTION("feature_dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::feature_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::feature_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("feature_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::feature_dropout_(self, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap) {
  RECORD_FUNCTION("fill_diagonal_", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fill_diagonal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_diagonal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "wrap", wrap);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_diagonal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::fill_diagonal_(self, fill_value, wrap);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & floor_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("floor_divide_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("floor_divide");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("floor_divide");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::floor_divide_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  RECORD_FUNCTION("fractional_max_pool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<FractionalMaxPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool3DBackwardBackward>(new FractionalMaxPool3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fractional_max_pool3d_backward(grad_output_, self_, kernel_size, output_size, indices_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & full_out_out(Tensor & out, IntArrayRef size, Scalar fill_value) {
  RECORD_FUNCTION("full_out", std::vector<c10::IValue>({out, fill_value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("full_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::full_out_out(out, size, fill_value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & gather_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("gather");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("gather");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gather_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::gather_out(out_, self_, dim, index_, sparse_grad);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & gather_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gather_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::gather_out_dimname_out(out, self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor ge_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("ge", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ge(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor ge_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("ge", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::ge(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & ge__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("ge_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeBackward0>(new GeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ge");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.ge_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & ge__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("ge_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<GeBackward1>(new GeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ge");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.ge_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor gelu(const Tensor & self) {
  RECORD_FUNCTION("gelu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GeluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeluBackward>(new GeluBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gelu(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & geometric_(Tensor & self, double p, Generator generator) {
  RECORD_FUNCTION("geometric_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeometricBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeometricBackward>(new GeometricBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::geometric");
    } else {
      op_name = jit::Symbol::fromQualString("aten::geometric_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("geometric_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.geometric_(p, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("glu_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<GluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<GluBackwardBackward>(new GluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::glu_backward(grad_output_, self_, dim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  RECORD_FUNCTION("grid_sampler_2d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, input, grid )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_2d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, grid ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::grid_sampler_2d_backward(grad_output_, input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  RECORD_FUNCTION("grid_sampler_3d", std::vector<c10::IValue>({input, grid}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<GridSampler3DBackward> grad_fn;
  if (compute_requires_grad( input, grid )) {
    grad_fn = std::shared_ptr<GridSampler3DBackward>(new GridSampler3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
    grad_fn->align_corners = align_corners;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::grid_sampler_3d(input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> gru_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  RECORD_FUNCTION("gru", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::gru_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> gru_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  RECORD_FUNCTION("gru", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::gru_data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor gt_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("gt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gt(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gt_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("gt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gt(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & gt__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("gt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GtBackward0>(new GtBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::gt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.gt_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & gt__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("gt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<GtBackward1>(new GtBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::gt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.gt_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) {
  RECORD_FUNCTION("hardsigmoid_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardsigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::hardsigmoid_backward(grad_output, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self) {
  RECORD_FUNCTION("hardswish_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::hardswish_backward(grad_output, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) {
  RECORD_FUNCTION("hinge_embedding_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hinge_embedding_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::hinge_embedding_loss(self, target, margin, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  RECORD_FUNCTION("histc", std::vector<c10::IValue>({self, min, max}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HistcBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HistcBackward>(new HistcBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::histc(self_, bins, min, max);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hspmm(const Tensor & mat1, const Tensor & mat2) {
  RECORD_FUNCTION("hspmm", std::vector<c10::IValue>({mat1, mat2}), Node::peek_at_next_sequence_nr());
  auto& mat1_ = unpack(mat1, "mat1", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("hspmm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mat1, mat2 ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hspmm(mat1_, mat2_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor imag(const Tensor & self) {
  RECORD_FUNCTION("imag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ImagBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ImagBackward>(new ImagBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::imag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::imag(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  RECORD_FUNCTION("index_copy", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_copy(self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_copy_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  RECORD_FUNCTION("index_copy", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_copy_dimname(self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  RECORD_FUNCTION("index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<IndexCopyBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<IndexCopyBackward>(new IndexCopyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
    if (grad_fn->should_compute_output(1)) {
      grad_fn->source_ = SavedVariable(source, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.index_copy_(dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_copy__dimname(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  RECORD_FUNCTION("index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::index_copy__dimname(self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor index_fill_int_Scalar(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_fill_int_Scalar(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_int_Tensor(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_fill_int_Tensor(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_Dimname_Scalar(const Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_fill_Dimname_Scalar(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_Dimname_Tensor(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::index_fill_Dimname_Tensor(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_fill__int_Scalar(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<IndexFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<IndexFillBackward0>(new IndexFillBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.index_fill_(dim, index_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__int_Tensor(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& value_ = unpack(value, "value", 3);
  check_inplace(self);
  std::shared_ptr<IndexFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<IndexFillBackward1>(new IndexFillBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.index_fill_(dim, index_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__Dimname_Scalar(Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::index_fill__Dimname_Scalar(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__Dimname_Tensor(Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::index_fill__Dimname_Tensor(self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor inverse(const Tensor & self) {
  RECORD_FUNCTION("inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<InverseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<InverseBackward>(new InverseBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::inverse(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) {
  RECORD_FUNCTION("irfft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::irfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "signal_sizes", signal_sizes);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::irfft(self, signal_ndim, normalized, onesided, signal_sizes);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
bool is_nonzero(const Tensor & self) {
  RECORD_FUNCTION("is_nonzero", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::is_nonzero(self);
  return result;
}
bool is_set_to(const Tensor & self, const Tensor & tensor) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor__storage_saved =
    tensor_.has_storage() ? c10::optional<Storage>(tensor_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor__impl_saved;
  if (tensor_.defined()) tensor__impl_saved = tensor_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.is_set_to(tensor_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor__storage_saved.has_value())
    AT_ASSERT(tensor__storage_saved.value().is_alias_of(tensor_.storage()));
  if (tensor__impl_saved) AT_ASSERT(tensor__impl_saved == tensor_.getIntrusivePtr());
  #endif
  return result;
}
bool is_signed(const Tensor & self) {
  auto result = TypeDefault::is_signed(self);
  return result;
}
Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
  RECORD_FUNCTION("isclose", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isclose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "equal_nan", equal_nan);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::isclose(self, other, rtol, atol, equal_nan);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor isfinite(const Tensor & self) {
  RECORD_FUNCTION("isfinite", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isfinite");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::isfinite(self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  RECORD_FUNCTION("kl_div_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<KlDivBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<KlDivBackwardBackward>(new KlDivBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->log_target = log_target;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kl_div_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "log_target", log_target);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::kl_div_backward(grad_output_, self_, target_, reduction, log_target);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor le_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("le", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::le(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor le_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("le", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::le(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & le__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("le_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LeBackward0>(new LeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::le");
    } else {
      op_name = jit::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.le_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & le__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("le_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<LeBackward1>(new LeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::le");
    } else {
      op_name = jit::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.le_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
  RECORD_FUNCTION("leaky_relu", std::vector<c10::IValue>({self, negative_slope}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LeakyReluBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LeakyReluBackward0>(new LeakyReluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::leaky_relu(self_, negative_slope);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
  RECORD_FUNCTION("leaky_relu_", std::vector<c10::IValue>({self, negative_slope}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LeakyReluBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LeakyReluBackward1>(new LeakyReluBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->negative_slope = negative_slope;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::leaky_relu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::leaky_relu_(self_, negative_slope);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
Tensor & lerp_out_Scalar_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight) {
  RECORD_FUNCTION("lerp_out", std::vector<c10::IValue>({out, self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& end_ = unpack(end, "end", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, end )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lerp");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::lerp_out(out_, self_, end_, weight);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & lerp_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight) {
  RECORD_FUNCTION("lerp_out", std::vector<c10::IValue>({out, self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& end_ = unpack(end, "end", 2);
  auto& weight_ = unpack(weight, "weight", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, end, weight )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lerp");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::lerp_out(out_, self_, end_, weight_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor log10(const Tensor & self) {
  RECORD_FUNCTION("log10", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log10Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::log10(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & log10_(Tensor & self) {
  RECORD_FUNCTION("log10_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log10Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log10");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log10_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log10_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log10_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & log_sigmoid_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("log_sigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::log_sigmoid_out_out(out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor logdet(const Tensor & self) {
  RECORD_FUNCTION("logdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogdetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogdetBackward>(new LogdetBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::logdet(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) = TypeDefault::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor lt_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("lt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lt(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor lt_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("lt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lt(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & lt__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("lt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LtBackward0>(new LtBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.lt_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & lt__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("lt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<LtBackward1>(new LtBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.lt_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor masked_select(const Tensor & self, const Tensor & mask) {
  RECORD_FUNCTION("masked_select", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<MaskedSelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaskedSelectBackward>(new MaskedSelectBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mask_sizes = mask.sizes().vec();
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::masked_select(self_, mask_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor matrix_rank_tol(const Tensor & self, double tol, bool symmetric) {
  RECORD_FUNCTION("matrix_rank", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tol", tol);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::matrix_rank_tol(self, tol, symmetric);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor matrix_rank(const Tensor & self, bool symmetric) {
  RECORD_FUNCTION("matrix_rank", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::matrix_rank(self, symmetric);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  RECORD_FUNCTION("max_pool3d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<MaxPool3DWithIndicesBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackwardBackward>(new MaxPool3DWithIndicesBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::max_pool3d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & max_unpool2d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  RECORD_FUNCTION("max_unpool2d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_unpool2d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_unpool2d_out(out_, self_, indices_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> min_dim(const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MinBackward0>(new MinBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::min(self_, dim, keepdim);
  })();
  std::tie(values, indices) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> min_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  Tensor values;
  Tensor indices;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(values, indices) = TypeDefault::min_names_dim(self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor min_other(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MinBackward2> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MinBackward2>(new MinBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::min(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor min(const Tensor & self) {
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MinBackward1>(new MinBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::min(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackward>(new MiopenConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_transpose_backward_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_transpose_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  RECORD_FUNCTION("mkldnn_convolution_backward_weights", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward_weights");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  RECORD_FUNCTION("mkldnn_linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_linear"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_linear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mkldnn_linear(input_, weight_, bias_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("mse_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MseLossBackward>(new MseLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mse_loss(self_, target_, reduction);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & mse_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("mse_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::mse_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor mul_Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("mul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mul(self_, other_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mul_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("mul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mul(self_, other);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & mul__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("mul_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.mul_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & mul__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("mul_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.mul_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("multilabel_margin_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::multilabel_margin_loss(self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & multilabel_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  RECORD_FUNCTION("multilabel_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto& is_target_ = unpack(is_target, "is_target", 5);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target, is_target )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::multilabel_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction, is_target_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & multinomial_out_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, Generator generator) {
  RECORD_FUNCTION("multinomial_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multinomial_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::multinomial_out(out_, self_, num_samples, replacement, generator);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  RECORD_FUNCTION("native_batch_norm_out", std::vector<c10::IValue>({out, save_mean, save_invstd, input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& save_mean_ = unpack(save_mean, "save_mean", 1);
  auto& save_invstd_ = unpack(save_invstd, "save_invstd", 2);
  auto& input_ = unpack(input, "input", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 5);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 6);
  auto running_var_ = unpack_opt(running_var, "running_var", 7);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( input, weight, bias, running_mean, running_var )) {
    throw_error_out_requires_grad("native_batch_norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("native_batch_norm");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_batch_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> save_mean__storage_saved =
    save_mean_.has_storage() ? c10::optional<Storage>(save_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_mean__impl_saved;
  if (save_mean_.defined()) save_mean__impl_saved = save_mean_.getIntrusivePtr();
  c10::optional<Storage> save_invstd__storage_saved =
    save_invstd_.has_storage() ? c10::optional<Storage>(save_invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_invstd__impl_saved;
  if (save_invstd_.defined()) save_invstd__impl_saved = save_invstd_.getIntrusivePtr();
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::native_batch_norm_out(out_, save_mean_, save_invstd_, input_, weight_, bias_, running_mean_, running_var_, training, momentum, eps);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (save_mean__storage_saved.has_value())
    AT_ASSERT(save_mean__storage_saved.value().is_alias_of(save_mean_.storage()));
  if (save_mean__impl_saved) AT_ASSERT(save_mean__impl_saved == save_mean_.getIntrusivePtr());
  if (save_invstd__storage_saved.has_value())
    AT_ASSERT(save_invstd__storage_saved.value().is_alias_of(save_invstd_.storage()));
  if (save_invstd__impl_saved) AT_ASSERT(save_invstd__impl_saved == save_invstd_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
  }
  #endif
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
Tensor neg(const Tensor & self) {
  RECORD_FUNCTION("neg", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::neg(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & neg_(Tensor & self) {
  RECORD_FUNCTION("neg_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::neg");
    } else {
      op_name = jit::Symbol::fromQualString("aten::neg_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::neg_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor new_empty(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("new_empty", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::new_empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::new_empty(self, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & nll_loss2d_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss2d_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::nll_loss2d_out_out(out, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & nll_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::nll_loss_out_out(out, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor nonzero(const Tensor & self) {
  RECORD_FUNCTION("nonzero", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::nonzero(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_ScalarOpt_dtype(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward2>(new NormBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::norm(self_, p, dtype);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor norm_Scalar(const Tensor & self, Scalar p) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self, p}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward0>(new NormBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::norm(self_, p);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor norm_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward3>(new NormBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::norm(self_, p, dim, keepdim, dtype);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor norm_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward1>(new NormBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::norm(self_, p, dim, keepdim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor norm_names_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::norm_names_ScalarOpt_dim_dtype(self, p, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_names_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) {
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::norm_names_ScalarOpt_dim(self, p, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_Tensor_float(const Tensor & mean, double std, Generator generator) {
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({mean}), Node::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  std::shared_ptr<NormalBackward1> grad_fn;
  if (compute_requires_grad( mean )) {
    grad_fn = std::shared_ptr<NormalBackward1>(new NormalBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mean ));
    grad_fn->mean_sizes = mean.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::normal(mean_, std, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_float_Tensor(double mean, const Tensor & std, Generator generator) {
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({std}), Node::peek_at_next_sequence_nr());
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward2> grad_fn;
  if (compute_requires_grad( std )) {
    grad_fn = std::shared_ptr<NormalBackward2>(new NormalBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( std ));
    grad_fn->std_sizes = std.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::normal(mean, std_, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_Tensor_Tensor(const Tensor & mean, const Tensor & std, Generator generator) {
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({mean, std}), Node::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward3> grad_fn;
  if (compute_requires_grad( mean, std )) {
    grad_fn = std::shared_ptr<NormalBackward3>(new NormalBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( mean, std ));
    grad_fn->mean_sizes = mean.sizes().vec();
    grad_fn->std_sizes = std.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::normal(mean_, std_, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_float_float(double mean, double std, IntArrayRef size, Generator generator, const TensorOptions & options) {
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::normal_float_float(mean, std, size, generator, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & normal_(Tensor & self, double mean, double std, Generator generator) {
  RECORD_FUNCTION("normal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NormalBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormalBackward0>(new NormalBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::normal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::normal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.normal_(mean, std, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & nuclear_norm_out_out(Tensor & out, const Tensor & self, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::nuclear_norm_out_out(out, self, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & nuclear_norm_out_dim_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::nuclear_norm_out_dim_out(out, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & orgqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2) {
  RECORD_FUNCTION("orgqr_out", std::vector<c10::IValue>({out, self, input2}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("orgqr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("orgqr");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("orgqr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::orgqr_out(out_, self_, input2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor poisson(const Tensor & self, Generator generator) {
  RECORD_FUNCTION("poisson", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PoissonBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PoissonBackward>(new PoissonBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::poisson");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::poisson(self_, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  RECORD_FUNCTION("poisson_nll_loss", std::vector<c10::IValue>({input, target}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::poisson_nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "log_input", log_input);
    jit::tracer::addInputs(node, "full", full);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::poisson_nll_loss(input, target, log_input, full, eps, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
ScalarType promote_types(ScalarType type1, ScalarType type2) {
  RECORD_FUNCTION("promote_types", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::promote_types(type1, type2);
  return result;
}
Tensor q_per_channel_scales(const Tensor & self) {
  RECORD_FUNCTION("q_per_channel_scales", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("q_per_channel_scales"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::q_per_channel_scales");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::q_per_channel_scales(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  RECORD_FUNCTION("quantized_batch_norm", std::vector<c10::IValue>({input, weight, bias, mean, var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& var_ = unpack(var, "var", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, weight, bias, mean, var )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantized_batch_norm"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, mean, var ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "var", var);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_scale", output_scale);
    jit::tracer::addInputs(node, "output_zero_point", output_zero_point);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> var__storage_saved =
    var_.has_storage() ? c10::optional<Storage>(var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> var__impl_saved;
  if (var_.defined()) var__impl_saved = var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::quantized_batch_norm(input_, weight_, bias_, mean_, var_, eps, output_scale, output_zero_point);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (var__storage_saved.has_value())
    AT_ASSERT(var__storage_saved.value().is_alias_of(var_.storage()));
  if (var__impl_saved) AT_ASSERT(var__impl_saved == var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> quantized_lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, c10::optional<ScalarType> dtype, bool use_dynamic) {
  RECORD_FUNCTION("quantized_lstm", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_lstm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "use_dynamic", use_dynamic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1, result2) = TypeDefault::quantized_lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, dtype, use_dynamic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> quantized_lstm_data(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, c10::optional<ScalarType> dtype, bool use_dynamic) {
  RECORD_FUNCTION("quantized_lstm", std::vector<c10::IValue>({data, batch_sizes}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_lstm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "use_dynamic", use_dynamic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1, result2) = TypeDefault::quantized_lstm_data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("quantized_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantized_max_pool2d"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::quantized_max_pool2d(self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & range_out_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
  RECORD_FUNCTION("range_out", std::vector<c10::IValue>({out, start, end, step}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("range_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::range_out(out_, start, end, step);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & reciprocal_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("reciprocal_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reciprocal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reciprocal");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::reciprocal_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & reflection_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reflection_pad1d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::reflection_pad1d_out(out_, self_, padding);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  RECORD_FUNCTION("renorm", std::vector<c10::IValue>({self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::renorm(self_, p, dim, maxnorm);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  RECORD_FUNCTION("renorm_", std::vector<c10::IValue>({self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::renorm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::renorm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.renorm_(p, dim, maxnorm);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackwardBackward>(new ReplicationPad3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::replication_pad3d_backward(grad_output_, self_, padding);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor reshape(const Tensor & self, IntArrayRef shape) {
  RECORD_FUNCTION("reshape", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reshape");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::reshape(self, shape);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {
  RECORD_FUNCTION("rfft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::rfft(self, signal_ndim, normalized, onesided);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> rnn_relu_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  RECORD_FUNCTION("rnn_relu", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::rnn_relu_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> rnn_relu_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  RECORD_FUNCTION("rnn_relu", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::rnn_relu_data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator generator) {
  RECORD_FUNCTION("rrelu", std::vector<c10::IValue>({self, lower, upper}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::rrelu(self, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator generator) {
  RECORD_FUNCTION("rrelu_", std::vector<c10::IValue>({self, lower, upper}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rrelu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rrelu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::rrelu_(self, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & rrelu_with_noise_out_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator generator) {
  RECORD_FUNCTION("rrelu_with_noise_out", std::vector<c10::IValue>({out, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& noise_ = unpack(noise, "noise", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, noise )) {
    throw_error_out_requires_grad("rrelu_with_noise");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("rrelu_with_noise");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::rrelu_with_noise_out(out_, self_, noise_, lower, upper, training, generator);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor rsub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("rsub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<RsubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<RsubBackward0>(new RsubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rsub(self_, other_, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rsub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("rsub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RsubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RsubBackward1>(new RsubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->alpha = alpha;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rsub(self_, other, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
  RECORD_FUNCTION("sigmoid_backward", std::vector<c10::IValue>({grad_output, output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<SigmoidBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<SigmoidBackwardBackward>(new SigmoidBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sigmoid_backward(grad_output_, output_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sin_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("sin_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sin");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sin");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sin_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor sinh(const Tensor & self) {
  RECORD_FUNCTION("sinh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::sinh(self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sinh_(Tensor & self) {
  RECORD_FUNCTION("sinh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sinh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sinh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sinh_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
std::tuple<Tensor,Tensor> slogdet(const Tensor & self) {
  RECORD_FUNCTION("slogdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SlogdetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SlogdetBackward>(new SlogdetBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor sign;
  Tensor logabsdet;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slogdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slogdet(self_);
  })();
  std::tie(sign, logabsdet) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( logabsdet ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
  }
  #endif
  if (grad_fn) {
    grad_fn->sign_ = SavedVariable(sign, true);
    grad_fn->logabsdet_ = SavedVariable(logabsdet, true);
  }
  return std::make_tuple(std::move(sign), std::move(logabsdet));
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("slow_conv3d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<SlowConv3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<SlowConv3DBackward>(new SlowConv3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slow_conv3d_forward(self_, weight_, kernel_size, bias_, stride, padding);
  })();
  std::tie(output, finput, fgrad_input) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  #endif
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("slow_conv_dilated2d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<SlowConvDilated2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<SlowConvDilated2DBackwardBackward>(new SlowConvDilated2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slow_conv_dilated2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor slow_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  RECORD_FUNCTION("slow_conv_dilated3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<SlowConvDilated3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<SlowConvDilated3DBackward>(new SlowConvDilated3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slow_conv_dilated3d(self_, weight_, kernel_size, bias_, stride, padding, dilation);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("slow_conv_transpose3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 8);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 9);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<SlowConvTranspose3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<SlowConvTranspose3DBackwardBackward>(new SlowConvTranspose3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input_return;
  Tensor grad_weight;
  Tensor grad_bias;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_transpose3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::slow_conv_transpose3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_, output_mask);
  })();
  std::tie(grad_input_return, grad_weight, grad_bias) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input_return, grad_weight, grad_bias ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input_return);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input_return), std::move(grad_weight), std::move(grad_bias));
}
Tensor softshrink(const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("softshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SoftshrinkBackward>(new SoftshrinkBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::softshrink(self_, lambd);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & softshrink_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("softshrink_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::softshrink_backward_out(grad_input_, grad_output_, self_, lambd);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor stack(TensorList tensors, int64_t dim) {
  RECORD_FUNCTION("stack", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<StackBackward> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<StackBackward>(new StackBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::stack(tensors_, dim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) {
  RECORD_FUNCTION("stft", std::vector<c10::IValue>({self, window}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n_fft", n_fft);
    jit::tracer::addInputs(node, "hop_length", hop_length);
    jit::tracer::addInputs(node, "win_length", win_length);
    jit::tracer::addInputs(node, "window", window);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::stft(self, n_fft, hop_length, win_length, window, normalized, onesided);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sub_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("sub_out", std::vector<c10::IValue>({out, self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("sub");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sub");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sub_out(out_, self_, other_, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
  RECORD_FUNCTION("symeig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SymeigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SymeigBackward>(new SymeigBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eigenvectors = eigenvectors;
    grad_fn->upper = upper;
  }
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::symeig(self_, eigenvectors, upper);
  })();
  std::tie(eigenvalues, eigenvectors_return) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( eigenvalues, eigenvectors_return ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors_return);
  }
  #endif
  if (grad_fn) {
    grad_fn->eigenvalues_ = SavedVariable(eigenvalues, true);
    grad_fn->eigenvectors_return_ = SavedVariable(eigenvectors_return, true);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) {
  RECORD_FUNCTION("tensordot", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tensordot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims_self", dims_self);
    jit::tracer::addInputs(node, "dims_other", dims_other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::tensordot(self, other, dims_self, dims_other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("thnn_conv2d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConv2DBackwardBackward>(new ThnnConv2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor grad_input_return;
  Tensor grad_weight;
  Tensor grad_bias;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::thnn_conv2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask);
  })();
  std::tie(grad_input_return, grad_weight, grad_bias) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input_return, grad_weight, grad_bias ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input_return);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input_return), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("thnn_conv2d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::thnn_conv2d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  #endif
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold) {
  RECORD_FUNCTION("threshold_backward", std::vector<c10::IValue>({grad_output, self, threshold}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ThresholdBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ThresholdBackwardBackward>(new ThresholdBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::threshold_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::threshold_backward(grad_output_, self_, threshold);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor to_dense(const Tensor & self) {
  RECORD_FUNCTION("to_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ToDenseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ToDenseBackward>(new ToDenseBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_dense");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.to_dense();
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  RECORD_FUNCTION("triangular_solve", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<TriangularSolveBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<TriangularSolveBackward>(new TriangularSolveBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
    grad_fn->upper = upper;
    grad_fn->transpose = transpose;
    grad_fn->unitriangular = unitriangular;
  }
  Tensor solution;
  Tensor cloned_coefficient;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triangular_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::triangular_solve(self_, A_, upper, transpose, unitriangular);
  })();
  std::tie(solution, cloned_coefficient) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution, cloned_coefficient ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, cloned_coefficient);
  }
  #endif
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(cloned_coefficient));
}
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  RECORD_FUNCTION("triplet_margin_loss", std::vector<c10::IValue>({anchor, positive, negative}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triplet_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "anchor", anchor);
    jit::tracer::addInputs(node, "positive", positive);
    jit::tracer::addInputs(node, "negative", negative);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "swap", swap);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  RECORD_FUNCTION("unique_dim_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("unique_dim_consecutive"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unique_dim_consecutive");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::unique_dim_consecutive(self_, dim, return_inverse, return_counts);
  })();
  std::tie(result0, result1, result2) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_bilinear2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleBilinear2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackward>(new UpsampleBilinear2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_bilinear2d(self_, output_size, align_corners, scales_h, scales_w);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_bilinear2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_bilinear2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bilinear2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_bilinear2d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners, scales_h, scales_w);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_linear1d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_linear1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_linear1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_linear1d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_linear1d_out(out_, self_, output_size, align_corners, scales);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_nearest1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackward>(new UpsampleNearest1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->scales = scales;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_nearest1d(self_, output_size, scales);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_nearest1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_nearest1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales", scales);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_nearest1d_backward_out(grad_input_, grad_output_, output_size, input_size, scales);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_nearest2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_nearest2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest2d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_nearest2d_out(out_, self_, output_size, scales_h, scales_w);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor var(const Tensor & self, bool unbiased) {
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<VarBackward0>(new VarBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::var(self_, unbiased);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor var_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<VarBackward1>(new VarBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::var(self_, dim, unbiased, keepdim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor var_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::var_names_dim(self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor where_self(const Tensor & condition, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("where", std::vector<c10::IValue>({condition, self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::where_self(condition, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> where(const Tensor & condition) {
  RECORD_FUNCTION("where", std::vector<c10::IValue>({condition}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::where(condition);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("zeros_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::zeros_like(self, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
// }
}

namespace {

auto registerer = torch::import()
  .impl_UNBOXED("aten::_batch_norm_impl_index_backward", torch::kAutograd,
          VariableType::_batch_norm_impl_index_backward
        )
  .impl("aten::_cast_Char", torch::kAutograd,
          VariableType::_cast_Char
       )
  .impl("aten::_cast_Float", torch::kAutograd,
          VariableType::_cast_Float
       )
  .impl("aten::_cholesky_solve_helper", torch::kAutograd,
          VariableType::_cholesky_solve_helper
       )
  .impl_UNBOXED("aten::_convolution_nogroup", torch::kAutograd,
          VariableType::_convolution_nogroup
        )
  .impl("aten::_copy_from", torch::kAutograd,
          VariableType::_copy_from
       )
  .impl_UNBOXED("aten::_ctc_loss_backward", torch::kAutograd,
          VariableType::_ctc_loss_backward
        )
  .impl_UNBOXED("aten::_cudnn_ctc_loss", torch::kAutograd,
          VariableType::_cudnn_ctc_loss
        )
  .impl_UNBOXED("aten::_cufft_set_plan_cache_max_size", torch::kAutograd,
          VariableType::_cufft_set_plan_cache_max_size
        )
  .impl_UNBOXED("aten::_cummin_helper", torch::kAutograd,
          VariableType::_cummin_helper
        )
  .impl_UNBOXED("aten::_index_copy_", torch::kAutograd,
          VariableType::_index_copy_
        )
  .impl("aten::_inverse_helper", torch::kAutograd,
          VariableType::_inverse_helper
       )
  .impl("aten::_masked_scale", torch::kAutograd,
          VariableType::_masked_scale
       )
  .impl_UNBOXED("aten::_min", torch::kAutograd,
          VariableType::_min
        )
  .impl("aten::_nnpack_available", torch::kAutograd,
          VariableType::_nnpack_available
       )
  .impl("aten::_pdist_backward", torch::kAutograd,
          VariableType::_pdist_backward
       )
  .impl("aten::_s_where", torch::kAutograd,
          VariableType::_s_where
       )
  .impl_UNBOXED("aten::_sample_dirichlet", torch::kAutograd,
          VariableType::_sample_dirichlet
        )
  .impl_UNBOXED("aten::_sobol_engine_scramble_", torch::kAutograd,
          VariableType::_sobol_engine_scramble_
        )
  .impl("aten::_sparse_addmm", torch::kAutograd,
          VariableType::_sparse_addmm
       )
  .impl_UNBOXED("aten::_sparse_coo_tensor_unsafe", torch::kAutograd,
          VariableType::_sparse_coo_tensor_unsafe
        )
  .impl_UNBOXED("aten::_sparse_coo_tensor_with_dims_and_tensors", torch::kAutograd,
          VariableType::_sparse_coo_tensor_with_dims_and_tensors
        )
  .impl_UNBOXED("aten::_standard_gamma", torch::kAutograd,
          VariableType::_standard_gamma
        )
  .impl_UNBOXED("aten::_symeig_helper", torch::kAutograd,
          VariableType::_symeig_helper
        )
  .impl_UNBOXED("aten::_thnn_fused_lstm_cell", torch::kAutograd,
          VariableType::_thnn_fused_lstm_cell
        )
  .impl_UNBOXED("aten::_triangular_solve_helper", torch::kAutograd,
          VariableType::_triangular_solve_helper
        )
  .impl("aten::_var", torch::kAutograd,
          VariableType::_var
       )
  .impl_UNBOXED("aten::abs.out", torch::kAutograd,
          VariableType::abs_out_out
        )
  .impl_UNBOXED("aten::adaptive_max_pool2d", torch::kAutograd,
          VariableType::adaptive_max_pool2d
        )
  .impl_UNBOXED("aten::adaptive_max_pool2d_backward.grad_input", torch::kAutograd,
          VariableType::adaptive_max_pool2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::adaptive_max_pool3d.out", torch::kAutograd,
          VariableType::adaptive_max_pool3d_out_out
        )
  .impl("aten::addcdiv", torch::kAutograd,
          VariableType::addcdiv
       )
  .impl_UNBOXED("aten::addcdiv_", torch::kAutograd,
          VariableType::addcdiv_
        )
  .impl_UNBOXED("aten::addcmul.out", torch::kAutograd,
          VariableType::addcmul_out_out
        )
  .impl_UNBOXED("aten::addmm.out", torch::kAutograd,
          VariableType::addmm_out_out
        )
  .impl("aten::addmv", torch::kAutograd,
          VariableType::addmv
       )
  .impl_UNBOXED("aten::addmv_", torch::kAutograd,
          VariableType::addmv_
        )
  .impl("aten::alpha_dropout", torch::kAutograd,
          VariableType::alpha_dropout
       )
  .impl_UNBOXED("aten::alpha_dropout_", torch::kAutograd,
          VariableType::alpha_dropout_
        )
  .impl("aten::angle", torch::kAutograd,
          VariableType::angle
       )
  .impl_UNBOXED("aten::atan2.out", torch::kAutograd,
          VariableType::atan2_out_out
        )
  .impl_UNBOXED("aten::atan.out", torch::kAutograd,
          VariableType::atan_out_out
        )
  .impl_UNBOXED("aten::avg_pool1d", torch::kAutograd,
          VariableType::avg_pool1d
        )
  .impl_UNBOXED("aten::avg_pool2d.out", torch::kAutograd,
          VariableType::avg_pool2d_out_out
        )
  .impl_UNBOXED("aten::batch_norm_backward_elemt", torch::kAutograd,
          VariableType::batch_norm_backward_elemt
        )
  .impl_UNBOXED("aten::batch_norm_gather_stats_with_counts", torch::kAutograd,
          VariableType::batch_norm_gather_stats_with_counts
        )
  .impl_UNBOXED("aten::binary_cross_entropy", torch::kAutograd,
          VariableType::binary_cross_entropy
        )
  .impl_UNBOXED("aten::binary_cross_entropy_backward.grad_input", torch::kAutograd,
          VariableType::binary_cross_entropy_backward_out_grad_input
        )
  .impl_UNBOXED("aten::bitwise_or.Scalar", torch::kAutograd,
          VariableType::bitwise_or_Scalar
        )
  .impl_UNBOXED("aten::bitwise_or.Tensor", torch::kAutograd,
          VariableType::bitwise_or_Tensor
        )
  .impl_UNBOXED("aten::bitwise_or_.Scalar", torch::kAutograd,
          VariableType::bitwise_or__Scalar
        )
  .impl_UNBOXED("aten::bitwise_or_.Tensor", torch::kAutograd,
          VariableType::bitwise_or__Tensor
        )
  .impl_UNBOXED("aten::bitwise_xor.Scalar", torch::kAutograd,
          VariableType::bitwise_xor_Scalar
        )
  .impl_UNBOXED("aten::bitwise_xor.Tensor", torch::kAutograd,
          VariableType::bitwise_xor_Tensor
        )
  .impl_UNBOXED("aten::bitwise_xor_.Scalar", torch::kAutograd,
          VariableType::bitwise_xor__Scalar
        )
  .impl_UNBOXED("aten::bitwise_xor_.Tensor", torch::kAutograd,
          VariableType::bitwise_xor__Tensor
        )
  .impl_UNBOXED("aten::blackman_window", torch::kAutograd,
          VariableType::blackman_window
        )
  .impl_UNBOXED("aten::blackman_window.periodic", torch::kAutograd,
          VariableType::blackman_window_periodic
        )
  .impl_UNBOXED("aten::broadcast_tensors", torch::kAutograd,
          VariableType::broadcast_tensors
        )
  .impl_UNBOXED("aten::cholesky_inverse.out", torch::kAutograd,
          VariableType::cholesky_inverse_out_out
        )
  .impl("aten::cholesky_solve", torch::kAutograd,
          VariableType::cholesky_solve
       )
  .impl("aten::clamp_min", torch::kAutograd,
          VariableType::clamp_min
       )
  .impl_UNBOXED("aten::clamp_min_", torch::kAutograd,
          VariableType::clamp_min_
        )
  .impl_UNBOXED("aten::clamp.out", torch::kAutograd,
          VariableType::clamp_out_out
        )
  .impl_UNBOXED("aten::clone", torch::kAutograd,
          VariableType::clone
        )
  .impl_UNBOXED("aten::contiguous", torch::kAutograd,
          VariableType::contiguous
        )
  .impl_UNBOXED("aten::conv3d", torch::kAutograd,
          VariableType::conv3d
        )
  .impl_UNBOXED("aten::conv_transpose2d.input", torch::kAutograd,
          VariableType::conv_transpose2d_input
        )
  .impl_UNBOXED("aten::convolution_overrideable", torch::kAutograd,
          VariableType::convolution_overrideable
        )
  .impl_UNBOXED("aten::cos.out", torch::kAutograd,
          VariableType::cos_out_out
        )
  .impl("aten::cosh", torch::kAutograd,
          VariableType::cosh
       )
  .impl_UNBOXED("aten::cosh_", torch::kAutograd,
          VariableType::cosh_
        )
  .impl("aten::cross", torch::kAutograd,
          VariableType::cross
       )
  .impl_UNBOXED("aten::cudnn_batch_norm", torch::kAutograd,
          VariableType::cudnn_batch_norm
        )
  .impl_UNBOXED("aten::cudnn_convolution_transpose_backward", torch::kAutograd,
          VariableType::cudnn_convolution_transpose_backward
        )
  .impl_UNBOXED("aten::cudnn_convolution_transpose_backward_input", torch::kAutograd,
          VariableType::cudnn_convolution_transpose_backward_input
        )
  .impl("aten::cudnn_grid_sampler", torch::kAutograd,
          VariableType::cudnn_grid_sampler
       )
  .impl("aten::cudnn_is_acceptable", torch::kAutograd,
          VariableType::cudnn_is_acceptable
       )
  .impl_UNBOXED("aten::cummin", torch::kAutograd,
          VariableType::cummin
        )
  .impl_UNBOXED("aten::cummin.dimname", torch::kAutograd,
          VariableType::cummin_dimname
        )
  .impl("aten::diag_embed", torch::kAutograd,
          VariableType::diag_embed
       )
  .impl_UNBOXED("aten::diag.out", torch::kAutograd,
          VariableType::diag_out_out
        )
  .impl_UNBOXED("aten::digamma.out", torch::kAutograd,
          VariableType::digamma_out_out
        )
  .impl_UNBOXED("aten::eig", torch::kAutograd,
          VariableType::eig
        )
  .impl_UNBOXED("aten::empty.names", torch::kAutograd,
          VariableType::empty_names
        )
  .impl_UNBOXED("aten::empty.memory_format", torch::kAutograd,
          VariableType::empty_memory_format
        )
  .impl("aten::eq.Scalar", torch::kAutograd,
          VariableType::eq_Scalar
       )
  .impl("aten::eq.Tensor", torch::kAutograd,
          VariableType::eq_Tensor
       )
  .impl_UNBOXED("aten::eq_.Scalar", torch::kAutograd,
          VariableType::eq__Scalar
        )
  .impl_UNBOXED("aten::eq_.Tensor", torch::kAutograd,
          VariableType::eq__Tensor
        )
  .impl_UNBOXED("aten::erfinv.out", torch::kAutograd,
          VariableType::erfinv_out_out
        )
  .impl("aten::fake_quantize_per_tensor_affine", torch::kAutograd,
          VariableType::fake_quantize_per_tensor_affine
       )
  .impl("aten::fbgemm_pack_quantized_matrix", torch::kAutograd,
          VariableType::fbgemm_pack_quantized_matrix
       )
  .impl("aten::fbgemm_pack_quantized_matrix.KN", torch::kAutograd,
          VariableType::fbgemm_pack_quantized_matrix_KN
       )
  .impl("aten::feature_dropout", torch::kAutograd,
          VariableType::feature_dropout
       )
  .impl_UNBOXED("aten::feature_dropout_", torch::kAutograd,
          VariableType::feature_dropout_
        )
  .impl_UNBOXED("aten::fill_diagonal_", torch::kAutograd,
          VariableType::fill_diagonal_
        )
  .impl_UNBOXED("aten::floor_divide.out", torch::kAutograd,
          VariableType::floor_divide_out_out
        )
  .impl_UNBOXED("aten::fractional_max_pool3d_backward", torch::kAutograd,
          VariableType::fractional_max_pool3d_backward
        )
  .impl_UNBOXED("aten::full.out", torch::kAutograd,
          VariableType::full_out_out
        )
  .impl_UNBOXED("aten::gather.out", torch::kAutograd,
          VariableType::gather_out_out
        )
  .impl_UNBOXED("aten::gather.dimname_out", torch::kAutograd,
          VariableType::gather_out_dimname_out
        )
  .impl("aten::ge.Scalar", torch::kAutograd,
          VariableType::ge_Scalar
       )
  .impl("aten::ge.Tensor", torch::kAutograd,
          VariableType::ge_Tensor
       )
  .impl_UNBOXED("aten::ge_.Scalar", torch::kAutograd,
          VariableType::ge__Scalar
        )
  .impl_UNBOXED("aten::ge_.Tensor", torch::kAutograd,
          VariableType::ge__Tensor
        )
  .impl("aten::gelu", torch::kAutograd,
          VariableType::gelu
       )
  .impl_UNBOXED("aten::geometric_", torch::kAutograd,
          VariableType::geometric_
        )
  .impl("aten::glu_backward", torch::kAutograd,
          VariableType::glu_backward
       )
  .impl_UNBOXED("aten::grid_sampler_2d_backward", torch::kAutograd,
          VariableType::grid_sampler_2d_backward
        )
  .impl("aten::grid_sampler_3d", torch::kAutograd,
          VariableType::grid_sampler_3d
       )
  .impl_UNBOXED("aten::gru.input", torch::kAutograd,
          VariableType::gru_input
        )
  .impl_UNBOXED("aten::gru.data", torch::kAutograd,
          VariableType::gru_data
        )
  .impl("aten::gt.Scalar", torch::kAutograd,
          VariableType::gt_Scalar
       )
  .impl("aten::gt.Tensor", torch::kAutograd,
          VariableType::gt_Tensor
       )
  .impl_UNBOXED("aten::gt_.Scalar", torch::kAutograd,
          VariableType::gt__Scalar
        )
  .impl_UNBOXED("aten::gt_.Tensor", torch::kAutograd,
          VariableType::gt__Tensor
        )
  .impl("aten::hardsigmoid_backward", torch::kAutograd,
          VariableType::hardsigmoid_backward
       )
  .impl("aten::hardswish_backward", torch::kAutograd,
          VariableType::hardswish_backward
       )
  .impl("aten::hinge_embedding_loss", torch::kAutograd,
          VariableType::hinge_embedding_loss
       )
  .impl("aten::histc", torch::kAutograd,
          VariableType::histc
       )
  .impl("aten::hspmm", torch::kAutograd,
          VariableType::hspmm
       )
  .impl("aten::imag", torch::kAutograd,
          VariableType::imag
       )
  .impl("aten::index_copy", torch::kAutograd,
          VariableType::index_copy
       )
  .impl_UNBOXED("aten::index_copy.dimname", torch::kAutograd,
          VariableType::index_copy_dimname
        )
  .impl_UNBOXED("aten::index_copy_", torch::kAutograd,
          VariableType::index_copy_
        )
  .impl_UNBOXED("aten::index_copy_.dimname", torch::kAutograd,
          VariableType::index_copy__dimname
        )
  .impl("aten::index_fill.int_Scalar", torch::kAutograd,
          VariableType::index_fill_int_Scalar
       )
  .impl("aten::index_fill.int_Tensor", torch::kAutograd,
          VariableType::index_fill_int_Tensor
       )
  .impl_UNBOXED("aten::index_fill.Dimname_Scalar", torch::kAutograd,
          VariableType::index_fill_Dimname_Scalar
        )
  .impl_UNBOXED("aten::index_fill.Dimname_Tensor", torch::kAutograd,
          VariableType::index_fill_Dimname_Tensor
        )
  .impl_UNBOXED("aten::index_fill_.int_Scalar", torch::kAutograd,
          VariableType::index_fill__int_Scalar
        )
  .impl_UNBOXED("aten::index_fill_.int_Tensor", torch::kAutograd,
          VariableType::index_fill__int_Tensor
        )
  .impl_UNBOXED("aten::index_fill_.Dimname_Scalar", torch::kAutograd,
          VariableType::index_fill__Dimname_Scalar
        )
  .impl_UNBOXED("aten::index_fill_.Dimname_Tensor", torch::kAutograd,
          VariableType::index_fill__Dimname_Tensor
        )
  .impl("aten::inverse", torch::kAutograd,
          VariableType::inverse
       )
  .impl_UNBOXED("aten::irfft", torch::kAutograd,
          VariableType::irfft
        )
  .impl("aten::is_nonzero", torch::kAutograd,
          VariableType::is_nonzero
       )
  .impl("aten::is_set_to", torch::kAutograd,
          VariableType::is_set_to
       )
  .impl("aten::is_signed", torch::kAutograd,
          VariableType::is_signed
       )
  .impl("aten::isclose", torch::kAutograd,
          VariableType::isclose
       )
  .impl("aten::isfinite", torch::kAutograd,
          VariableType::isfinite
       )
  .impl("aten::kl_div_backward", torch::kAutograd,
          VariableType::kl_div_backward
       )
  .impl("aten::le.Scalar", torch::kAutograd,
          VariableType::le_Scalar
       )
  .impl("aten::le.Tensor", torch::kAutograd,
          VariableType::le_Tensor
       )
  .impl_UNBOXED("aten::le_.Scalar", torch::kAutograd,
          VariableType::le__Scalar
        )
  .impl_UNBOXED("aten::le_.Tensor", torch::kAutograd,
          VariableType::le__Tensor
        )
  .impl("aten::leaky_relu", torch::kAutograd,
          VariableType::leaky_relu
       )
  .impl_UNBOXED("aten::leaky_relu_", torch::kAutograd,
          VariableType::leaky_relu_
        )
  .impl_UNBOXED("aten::lerp.Scalar_out", torch::kAutograd,
          VariableType::lerp_out_Scalar_out
        )
  .impl_UNBOXED("aten::lerp.Tensor_out", torch::kAutograd,
          VariableType::lerp_out_Tensor_out
        )
  .impl("aten::log10", torch::kAutograd,
          VariableType::log10
       )
  .impl_UNBOXED("aten::log10_", torch::kAutograd,
          VariableType::log10_
        )
  .impl_UNBOXED("aten::log_sigmoid.out", torch::kAutograd,
          VariableType::log_sigmoid_out_out
        )
  .impl("aten::logdet", torch::kAutograd,
          VariableType::logdet
       )
  .impl_UNBOXED("aten::lstm_cell", torch::kAutograd,
          VariableType::lstm_cell
        )
  .impl("aten::lt.Scalar", torch::kAutograd,
          VariableType::lt_Scalar
       )
  .impl("aten::lt.Tensor", torch::kAutograd,
          VariableType::lt_Tensor
       )
  .impl_UNBOXED("aten::lt_.Scalar", torch::kAutograd,
          VariableType::lt__Scalar
        )
  .impl_UNBOXED("aten::lt_.Tensor", torch::kAutograd,
          VariableType::lt__Tensor
        )
  .impl("aten::masked_select", torch::kAutograd,
          VariableType::masked_select
       )
  .impl("aten::matrix_rank.tol", torch::kAutograd,
          VariableType::matrix_rank_tol
       )
  .impl("aten::matrix_rank", torch::kAutograd,
          VariableType::matrix_rank
       )
  .impl_UNBOXED("aten::max_pool3d", torch::kAutograd,
          VariableType::max_pool3d
        )
  .impl_UNBOXED("aten::max_pool3d_with_indices_backward", torch::kAutograd,
          VariableType::max_pool3d_with_indices_backward
        )
  .impl_UNBOXED("aten::max_unpool2d.out", torch::kAutograd,
          VariableType::max_unpool2d_out_out
        )
  .impl_UNBOXED("aten::min.dim", torch::kAutograd,
          VariableType::min_dim
        )
  .impl_UNBOXED("aten::min.names_dim", torch::kAutograd,
          VariableType::min_names_dim
        )
  .impl("aten::min.other", torch::kAutograd,
          VariableType::min_other
       )
  .impl("aten::min", torch::kAutograd,
          VariableType::min
       )
  .impl_UNBOXED("aten::miopen_convolution", torch::kAutograd,
          VariableType::miopen_convolution
        )
  .impl_UNBOXED("aten::miopen_convolution_transpose_backward_weight", torch::kAutograd,
          VariableType::miopen_convolution_transpose_backward_weight
        )
  .impl_UNBOXED("aten::mkldnn_convolution_backward_weights", torch::kAutograd,
          VariableType::mkldnn_convolution_backward_weights
        )
  .impl_UNBOXED("aten::mkldnn_linear", torch::kAutograd,
          VariableType::mkldnn_linear
        )
  .impl("aten::mse_loss", torch::kAutograd,
          VariableType::mse_loss
       )
  .impl_UNBOXED("aten::mse_loss_backward.grad_input", torch::kAutograd,
          VariableType::mse_loss_backward_out_grad_input
        )
  .impl("aten::mul.Tensor", torch::kAutograd,
          VariableType::mul_Tensor
       )
  .impl("aten::mul.Scalar", torch::kAutograd,
          VariableType::mul_Scalar
       )
  .impl_UNBOXED("aten::mul_.Tensor", torch::kAutograd,
          VariableType::mul__Tensor
        )
  .impl_UNBOXED("aten::mul_.Scalar", torch::kAutograd,
          VariableType::mul__Scalar
        )
  .impl("aten::multilabel_margin_loss", torch::kAutograd,
          VariableType::multilabel_margin_loss
       )
  .impl_UNBOXED("aten::multilabel_margin_loss_backward.grad_input", torch::kAutograd,
          VariableType::multilabel_margin_loss_backward_out_grad_input
        )
  .impl_UNBOXED("aten::multinomial.out", torch::kAutograd,
          VariableType::multinomial_out_out
        )
  .impl_UNBOXED("aten::native_batch_norm.out", torch::kAutograd,
          VariableType::native_batch_norm_out_out
        )
  .impl("aten::neg", torch::kAutograd,
          VariableType::neg
       )
  .impl_UNBOXED("aten::neg_", torch::kAutograd,
          VariableType::neg_
        )
  .impl_UNBOXED("aten::new_empty", torch::kAutograd,
          VariableType::new_empty
        )
  .impl_UNBOXED("aten::nll_loss2d.out", torch::kAutograd,
          VariableType::nll_loss2d_out_out
        )
  .impl_UNBOXED("aten::nll_loss.out", torch::kAutograd,
          VariableType::nll_loss_out_out
        )
  .impl("aten::nonzero", torch::kAutograd,
          VariableType::nonzero
       )
  .impl_UNBOXED("aten::norm.ScalarOpt_dtype", torch::kAutograd,
          VariableType::norm_ScalarOpt_dtype
        )
  .impl("aten::norm.Scalar", torch::kAutograd,
          VariableType::norm_Scalar
       )
  .impl_UNBOXED("aten::norm.ScalarOpt_dim_dtype", torch::kAutograd,
          VariableType::norm_ScalarOpt_dim_dtype
        )
  .impl_UNBOXED("aten::norm.ScalarOpt_dim", torch::kAutograd,
          VariableType::norm_ScalarOpt_dim
        )
  .impl_UNBOXED("aten::norm.names_ScalarOpt_dim_dtype", torch::kAutograd,
          VariableType::norm_names_ScalarOpt_dim_dtype
        )
  .impl_UNBOXED("aten::norm.names_ScalarOpt_dim", torch::kAutograd,
          VariableType::norm_names_ScalarOpt_dim
        )
  .impl_UNBOXED("aten::normal.Tensor_float", torch::kAutograd,
          VariableType::normal_Tensor_float
        )
  .impl_UNBOXED("aten::normal.float_Tensor", torch::kAutograd,
          VariableType::normal_float_Tensor
        )
  .impl_UNBOXED("aten::normal.Tensor_Tensor", torch::kAutograd,
          VariableType::normal_Tensor_Tensor
        )
  .impl_UNBOXED("aten::normal.float_float", torch::kAutograd,
          VariableType::normal_float_float
        )
  .impl_UNBOXED("aten::normal_", torch::kAutograd,
          VariableType::normal_
        )
  .impl_UNBOXED("aten::nuclear_norm.out", torch::kAutograd,
          VariableType::nuclear_norm_out_out
        )
  .impl_UNBOXED("aten::nuclear_norm.dim_out", torch::kAutograd,
          VariableType::nuclear_norm_out_dim_out
        )
  .impl_UNBOXED("aten::orgqr.out", torch::kAutograd,
          VariableType::orgqr_out_out
        )
  .impl_UNBOXED("aten::poisson", torch::kAutograd,
          VariableType::poisson
        )
  .impl("aten::poisson_nll_loss", torch::kAutograd,
          VariableType::poisson_nll_loss
       )
  .impl_UNBOXED("aten::promote_types", torch::kAutograd,
          VariableType::promote_types
        )
  .impl_UNBOXED("aten::q_per_channel_scales", torch::kAutograd,
          VariableType::q_per_channel_scales
        )
  .impl_UNBOXED("aten::quantized_batch_norm", torch::kAutograd,
          VariableType::quantized_batch_norm
        )
  .impl_UNBOXED("aten::quantized_lstm", torch::kAutograd,
          VariableType::quantized_lstm
        )
  .impl_UNBOXED("aten::quantized_lstm.data", torch::kAutograd,
          VariableType::quantized_lstm_data
        )
  .impl_UNBOXED("aten::quantized_max_pool2d", torch::kAutograd,
          VariableType::quantized_max_pool2d
        )
  .impl_UNBOXED("aten::range.out", torch::kAutograd,
          VariableType::range_out_out
        )
  .impl_UNBOXED("aten::reciprocal.out", torch::kAutograd,
          VariableType::reciprocal_out_out
        )
  .impl_UNBOXED("aten::reflection_pad1d.out", torch::kAutograd,
          VariableType::reflection_pad1d_out_out
        )
  .impl("aten::renorm", torch::kAutograd,
          VariableType::renorm
       )
  .impl_UNBOXED("aten::renorm_", torch::kAutograd,
          VariableType::renorm_
        )
  .impl_UNBOXED("aten::replication_pad3d_backward", torch::kAutograd,
          VariableType::replication_pad3d_backward
        )
  .impl_UNBOXED("aten::reshape", torch::kAutograd,
          VariableType::reshape
        )
  .impl("aten::rfft", torch::kAutograd,
          VariableType::rfft
       )
  .impl_UNBOXED("aten::rnn_relu.input", torch::kAutograd,
          VariableType::rnn_relu_input
        )
  .impl_UNBOXED("aten::rnn_relu.data", torch::kAutograd,
          VariableType::rnn_relu_data
        )
  .impl_UNBOXED("aten::rrelu", torch::kAutograd,
          VariableType::rrelu
        )
  .impl_UNBOXED("aten::rrelu_", torch::kAutograd,
          VariableType::rrelu_
        )
  .impl_UNBOXED("aten::rrelu_with_noise.out", torch::kAutograd,
          VariableType::rrelu_with_noise_out_out
        )
  .impl("aten::rsub.Tensor", torch::kAutograd,
          VariableType::rsub_Tensor
       )
  .impl("aten::rsub.Scalar", torch::kAutograd,
          VariableType::rsub_Scalar
       )
  .impl("aten::sigmoid_backward", torch::kAutograd,
          VariableType::sigmoid_backward
       )
  .impl_UNBOXED("aten::sin.out", torch::kAutograd,
          VariableType::sin_out_out
        )
  .impl("aten::sinh", torch::kAutograd,
          VariableType::sinh
       )
  .impl_UNBOXED("aten::sinh_", torch::kAutograd,
          VariableType::sinh_
        )
  .impl_UNBOXED("aten::slogdet", torch::kAutograd,
          VariableType::slogdet
        )
  .impl_UNBOXED("aten::slow_conv3d_forward", torch::kAutograd,
          VariableType::slow_conv3d_forward
        )
  .impl_UNBOXED("aten::slow_conv_dilated2d_backward", torch::kAutograd,
          VariableType::slow_conv_dilated2d_backward
        )
  .impl_UNBOXED("aten::slow_conv_dilated3d", torch::kAutograd,
          VariableType::slow_conv_dilated3d
        )
  .impl_UNBOXED("aten::slow_conv_transpose3d_backward.output_mask", torch::kAutograd,
          VariableType::slow_conv_transpose3d_backward_output_mask
        )
  .impl("aten::softshrink", torch::kAutograd,
          VariableType::softshrink
       )
  .impl_UNBOXED("aten::softshrink_backward.grad_input", torch::kAutograd,
          VariableType::softshrink_backward_out_grad_input
        )
  .impl_UNBOXED("aten::stack", torch::kAutograd,
          VariableType::stack
        )
  .impl_UNBOXED("aten::stft", torch::kAutograd,
          VariableType::stft
        )
  .impl_UNBOXED("aten::sub.out", torch::kAutograd,
          VariableType::sub_out_out
        )
  .impl_UNBOXED("aten::symeig", torch::kAutograd,
          VariableType::symeig
        )
  .impl_UNBOXED("aten::tensordot", torch::kAutograd,
          VariableType::tensordot
        )
  .impl_UNBOXED("aten::thnn_conv2d_backward.output_mask", torch::kAutograd,
          VariableType::thnn_conv2d_backward_output_mask
        )
  .impl_UNBOXED("aten::thnn_conv2d_forward.output", torch::kAutograd,
          VariableType::thnn_conv2d_forward_out_output
        )
  .impl("aten::threshold_backward", torch::kAutograd,
          VariableType::threshold_backward
       )
  .impl("aten::to_dense", torch::kAutograd,
          VariableType::to_dense
       )
  .impl_UNBOXED("aten::triangular_solve", torch::kAutograd,
          VariableType::triangular_solve
        )
  .impl("aten::triplet_margin_loss", torch::kAutograd,
          VariableType::triplet_margin_loss
       )
  .impl_UNBOXED("aten::unique_dim_consecutive", torch::kAutograd,
          VariableType::unique_dim_consecutive
        )
  .impl_UNBOXED("aten::upsample_bilinear2d", torch::kAutograd,
          VariableType::upsample_bilinear2d
        )
  .impl_UNBOXED("aten::upsample_bilinear2d_backward.grad_input", torch::kAutograd,
          VariableType::upsample_bilinear2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::upsample_linear1d.out", torch::kAutograd,
          VariableType::upsample_linear1d_out_out
        )
  .impl_UNBOXED("aten::upsample_nearest1d", torch::kAutograd,
          VariableType::upsample_nearest1d
        )
  .impl_UNBOXED("aten::upsample_nearest1d_backward.grad_input", torch::kAutograd,
          VariableType::upsample_nearest1d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::upsample_nearest2d.out", torch::kAutograd,
          VariableType::upsample_nearest2d_out_out
        )
  .impl("aten::var", torch::kAutograd,
          VariableType::var
       )
  .impl_UNBOXED("aten::var.dim", torch::kAutograd,
          VariableType::var_dim
        )
  .impl_UNBOXED("aten::var.names_dim", torch::kAutograd,
          VariableType::var_names_dim
        )
  .impl("aten::where.self", torch::kAutograd,
          VariableType::where_self
       )
  .impl_UNBOXED("aten::where", torch::kAutograd,
          VariableType::where
        )
  .impl_UNBOXED("aten::zeros_like", torch::kAutograd,
          VariableType::zeros_like
        );

}

}} // namespace torch::autograd
