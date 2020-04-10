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
Tensor & __ilshift___Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
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
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ilshift__(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __ilshift___Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
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
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__ilshift__(other_);
  }
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
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __ior___Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::__ior___Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __ior___Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::__ior___Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __ixor___Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::__ixor___Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __ixor___Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::__ixor___Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor __lshift___Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
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
    return at::__lshift__(self_, other);
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
Tensor __lshift___Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
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
    return at::__lshift__(self_, other_);
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
Tensor __or___Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::__or___Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor __or___Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::__or___Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor __xor___Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::__xor___Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor __xor___Tensor(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::__xor___Tensor(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("_addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("_addr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_addr");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_addr_out", out);
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
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_addr_out(out_, self_, vec1_, vec2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
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
Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("_baddbmm_mkl_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_baddbmm_mkl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _cast_Double(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Double", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Double");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_cast_Double(self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cast_Short(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Short", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Short");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_cast_Short(self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  RECORD_FUNCTION("_cudnn_rnn", std::vector<c10::IValue>({input, weight_buf, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto weight_buf_ = unpack_opt(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto cx_ = unpack_opt(cx, "cx", 5);
  auto dropout_state_ = unpack_opt(dropout_state, "dropout_state", 14);
  check_no_requires_grad(weight_buf, "weight_buf");
  std::shared_ptr<CudnnRnnBackward> grad_fn;
  if (compute_requires_grad( input, weight, hx, cx )) {
    grad_fn = std::shared_ptr<CudnnRnnBackward>(new CudnnRnnBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, hx, cx ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = make_saved_variable_list(weight);
    grad_fn->weight_stride0 = weight_stride0;
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->mode = mode;
    grad_fn->hidden_size = hidden_size;
    grad_fn->num_layers = num_layers;
    grad_fn->batch_first = batch_first;
    grad_fn->dropout = dropout;
    grad_fn->train = train;
    grad_fn->bidirectional = bidirectional;
    grad_fn->batch_sizes = batch_sizes.vec();
    grad_fn->dropout_state_ = SavedVariable(dropout_state, false);
    grad_fn->weight_size_ = weight.size();
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> weight__storage_saved(weight_.size());
  for (const Tensor& tensor : weight_)
    weight__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> weight__impl_saved(weight_.size());
  for (size_t i=0; i<weight_.size(); i++)
    if (weight_[i].defined()) weight__impl_saved[i] = weight_[i].getIntrusivePtr();
  c10::optional<Storage> weight_buf__storage_saved =
    weight_buf_.has_storage() ? c10::optional<Storage>(weight_buf_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight_buf__impl_saved;
  if (weight_buf_.defined()) weight_buf__impl_saved = weight_buf_.getIntrusivePtr();
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  c10::optional<Storage> dropout_state__storage_saved =
    dropout_state_.has_storage() ? c10::optional<Storage>(dropout_state_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> dropout_state__impl_saved;
  if (dropout_state_.defined()) dropout_state__impl_saved = dropout_state_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cudnn_rnn(input_, weight_, weight_stride0, weight_buf_, hx_, cx_, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state_);
  })();
  std::tie(result0, result1, result2, result3, result4) = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__storage_saved[i].has_value())
      AT_ASSERT(weight__storage_saved[i].value().is_alias_of(weight_[i].storage()));
  }
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__impl_saved[i])
      AT_ASSERT(weight__impl_saved[i] == weight_[i].getIntrusivePtr());
  }
  if (weight_buf__storage_saved.has_value())
    AT_ASSERT(weight_buf__storage_saved.value().is_alias_of(weight_buf_.storage()));
  if (weight_buf__impl_saved) AT_ASSERT(weight_buf__impl_saved == weight_buf_.getIntrusivePtr());
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  if (dropout_state__storage_saved.has_value())
    AT_ASSERT(dropout_state__storage_saved.value().is_alias_of(dropout_state_.storage()));
  if (dropout_state__impl_saved) AT_ASSERT(dropout_state__impl_saved == dropout_state_.getIntrusivePtr());
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
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  #endif
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
Tensor & _cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("_cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_cumsum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_cumsum");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cumsum_out", out);
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
    at::_cumsum_out(out_, self_, dim);
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
int64_t _dimV(const Tensor & self) {
  RECORD_FUNCTION("_dimV", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_._dimV();
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  RECORD_FUNCTION("_embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto per_sample_weights_ = unpack_opt(per_sample_weights, "per_sample_weights", 6);
  std::shared_ptr<EmbeddingBagBackward> grad_fn;
  if (compute_requires_grad( weight, per_sample_weights )) {
    grad_fn = std::shared_ptr<EmbeddingBagBackward>(new EmbeddingBagBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( weight, per_sample_weights ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->weight_ = SavedVariable(weight, false);
    }
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->offsets_ = SavedVariable(offsets, false);
    grad_fn->mode = mode;
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
    if (grad_fn->should_compute_output(0)) {
      grad_fn->per_sample_weights_ = SavedVariable(per_sample_weights, false);
    }
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
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> per_sample_weights__storage_saved =
    per_sample_weights_.has_storage() ? c10::optional<Storage>(per_sample_weights_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> per_sample_weights__impl_saved;
  if (per_sample_weights_.defined()) per_sample_weights__impl_saved = per_sample_weights_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_embedding_bag(weight_, indices_, offsets_, scale_grad_by_freq, mode, sparse, per_sample_weights_, include_last_offset);
  })();
  std::tie(result0, result1, result2, result3) = std::move(tmp);
  #ifndef NDEBUG
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (per_sample_weights__storage_saved.has_value())
    AT_ASSERT(per_sample_weights__storage_saved.value().is_alias_of(per_sample_weights_.storage()));
  if (per_sample_weights__impl_saved) AT_ASSERT(per_sample_weights__impl_saved == per_sample_weights_.getIntrusivePtr());
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
Tensor _embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  RECORD_FUNCTION("_embedding_bag_sparse_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, per_sample_weights}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
  RECORD_FUNCTION("_gather_sparse_backward", std::vector<c10::IValue>({self, index, grad}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_gather_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_gather_sparse_backward(self, dim, index, grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe) {
  RECORD_FUNCTION("_index_put_impl_", std::vector<c10::IValue>({self, values}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto indices_ = unpack(indices, "indices", 1);
  auto& values_ = unpack(values, "values", 2);
  check_inplace(self);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<IndexPutImplBackward> grad_fn;
  if (compute_requires_grad( self, values )) {
    grad_fn = std::shared_ptr<IndexPutImplBackward>(new IndexPutImplBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, values ));
    grad_fn->indices_ = make_saved_variable_list(indices);
    grad_fn->values_info = values;
    grad_fn->accumulate = accumulate;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    jit::tracer::addInputs(node, "unsafe", unsafe);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_index_put_impl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> indices__storage_saved(indices_.size());
  for (const Tensor& tensor : indices_)
    indices__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> indices__impl_saved(indices_.size());
  for (size_t i=0; i<indices_.size(); i++)
    if (indices_[i].defined()) indices__impl_saved[i] = indices_[i].getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_index_put_impl_(self_, indices_, values_, accumulate, unsafe);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  for (size_t i=0; i<indices_.size(); i++) {
    if (indices__storage_saved[i].has_value())
      AT_ASSERT(indices__storage_saved[i].value().is_alias_of(indices_[i].storage()));
  }
  for (size_t i=0; i<indices_.size(); i++) {
    if (indices__impl_saved[i])
      AT_ASSERT(indices__impl_saved[i] == indices_[i].getIntrusivePtr());
  }
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
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
Tensor _indices(const Tensor & self) {
  RECORD_FUNCTION("_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_indices");
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
    return self_._indices();
  })();
  auto result = as_view(self, tmp, /* is_differentiable */ false);
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
Scalar _local_scalar_dense(const Tensor & self) {
  RECORD_FUNCTION("_local_scalar_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_local_scalar_dense(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> _max_out_max(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("_max_out", std::vector<c10::IValue>({max, max_indices, self}), Node::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_indices_ = unpack(max_indices, "max_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_max");
  }
  if (compute_requires_grad( max, max_indices )) {
    throw_error_out_requires_grad("_max");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_indices", max_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> max__storage_saved =
    max_.has_storage() ? c10::optional<Storage>(max_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max__impl_saved;
  if (max_.defined()) max__impl_saved = max_.getIntrusivePtr();
  c10::optional<Storage> max_indices__storage_saved =
    max_indices_.has_storage() ? c10::optional<Storage>(max_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max_indices__impl_saved;
  if (max_indices_.defined()) max_indices__impl_saved = max_indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_max_out(max_, max_indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (max__storage_saved.has_value())
    AT_ASSERT(max__storage_saved.value().is_alias_of(max_.storage()));
  if (max__impl_saved) AT_ASSERT(max__impl_saved == max_.getIntrusivePtr());
  if (max_indices__storage_saved.has_value())
    AT_ASSERT(max_indices__storage_saved.value().is_alias_of(max_indices_.storage()));
  if (max_indices__impl_saved) AT_ASSERT(max_indices__impl_saved == max_indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(max);
  increment_version(max_indices);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( max, max_indices ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_indices);
  }
  #endif
  return std::forward_as_tuple(max, max_indices);
}
Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  RECORD_FUNCTION("_mkldnn_transpose", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_mkldnn_transpose"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
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
    return at::_mkldnn_transpose(self_, dim0, dim1);
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
Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  RECORD_FUNCTION("_mkldnn_transpose_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_mkldnn_transpose_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_", self);
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
    at::_mkldnn_transpose_(self_, dim0, dim1);
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
std::tuple<Tensor &,Tensor &> _mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("_mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_mode");
  }
  if (compute_requires_grad( values, indices )) {
    throw_error_out_requires_grad("_mode");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
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
    at::_mode_out(values_, indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  increment_version(indices);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values, indices ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor _nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) {
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_weight", std::vector<c10::IValue>({input, grad_output}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weightsize", weightsize);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _pdist_forward(const Tensor & self, double p) {
  RECORD_FUNCTION("_pdist_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PdistBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PdistBackward>(new PdistBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::_pdist_forward");
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
    return at::_pdist_forward(self_, p);
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
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  RECORD_FUNCTION("_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SoftmaxBackward>(new SoftmaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
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
    return at::_softmax(self_, dim, half_to_float);
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
Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, options_);
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
Tensor _sparse_sum(const Tensor & self) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_sparse_sum(self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_sum_dtype(const Tensor & self, ScalarType dtype) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_sparse_sum_dtype(self, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_sum_dim(const Tensor & self, IntArrayRef dim) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SparseSumBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SparseSumBackward>(new SparseSumBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim.vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
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
    return at::_sparse_sum(self_, dim);
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
Tensor _sparse_sum_dim_dtype(const Tensor & self, IntArrayRef dim, ScalarType dtype) {
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::_sparse_sum_dim_dtype(self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & input_bias, const Tensor & hidden_bias, const Tensor & cx, const Tensor & cy) {
  RECORD_FUNCTION("_thnn_differentiable_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_differentiable_lstm_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1, result2, result3, result4) = TypeDefault::_thnn_differentiable_lstm_cell_backward(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  RECORD_FUNCTION("_thnn_fused_gru_cell", std::vector<c10::IValue>({input_gates, hidden_gates, hx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& hx_ = unpack(hx, "hx", 2);
  auto input_bias_ = unpack_opt(input_bias, "input_bias", 3);
  auto hidden_bias_ = unpack_opt(hidden_bias, "hidden_bias", 4);
  std::shared_ptr<ThnnFusedGruCellBackward> grad_fn;
  if (compute_requires_grad( input_gates, hidden_gates, hx, input_bias, hidden_bias )) {
    grad_fn = std::shared_ptr<ThnnFusedGruCellBackward>(new ThnnFusedGruCellBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input_gates, hidden_gates, hx, input_bias, hidden_bias ));
    grad_fn->input_gates_ = SavedVariable(input_gates, false);
    grad_fn->hidden_gates_ = SavedVariable(hidden_gates, false);
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->input_bias_ = SavedVariable(input_bias, false);
    grad_fn->hidden_bias_ = SavedVariable(hidden_bias, false);
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_gru_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "hx", hx);
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
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
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
    return at::_thnn_fused_gru_cell(input_gates_, hidden_gates_, hx_, input_bias_, hidden_bias_);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (input_gates__storage_saved.has_value())
    AT_ASSERT(input_gates__storage_saved.value().is_alias_of(input_gates_.storage()));
  if (input_gates__impl_saved) AT_ASSERT(input_gates__impl_saved == input_gates_.getIntrusivePtr());
  if (hidden_gates__storage_saved.has_value())
    AT_ASSERT(hidden_gates__storage_saved.value().is_alias_of(hidden_gates_.storage()));
  if (hidden_gates__impl_saved) AT_ASSERT(hidden_gates__impl_saved == hidden_gates_.getIntrusivePtr());
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (input_bias__storage_saved.has_value())
    AT_ASSERT(input_bias__storage_saved.value().is_alias_of(input_bias_.storage()));
  if (input_bias__impl_saved) AT_ASSERT(input_bias__impl_saved == input_bias_.getIntrusivePtr());
  if (hidden_bias__storage_saved.has_value())
    AT_ASSERT(hidden_bias__storage_saved.value().is_alias_of(hidden_bias_.storage()));
  if (hidden_bias__impl_saved) AT_ASSERT(hidden_bias__impl_saved == hidden_bias_.getIntrusivePtr());
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
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
  RECORD_FUNCTION("_thnn_fused_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, cx, cy, workspace}), Node::peek_at_next_sequence_nr());
  auto grad_hy_ = unpack_opt(grad_hy, "grad_hy", 0);
  auto grad_cy_ = unpack_opt(grad_cy, "grad_cy", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto& cy_ = unpack(cy, "cy", 3);
  auto& workspace_ = unpack(workspace, "workspace", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_hy, grad_cy, cx, cy, workspace )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_fused_lstm_cell_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_hy, grad_cy, cx, cy, workspace ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_hy__storage_saved =
    grad_hy_.has_storage() ? c10::optional<Storage>(grad_hy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_hy__impl_saved;
  if (grad_hy_.defined()) grad_hy__impl_saved = grad_hy_.getIntrusivePtr();
  c10::optional<Storage> grad_cy__storage_saved =
    grad_cy_.has_storage() ? c10::optional<Storage>(grad_cy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_cy__impl_saved;
  if (grad_cy_.defined()) grad_cy__impl_saved = grad_cy_.getIntrusivePtr();
  c10::optional<Storage> cx__storage_saved =
    cx_.has_storage() ? c10::optional<Storage>(cx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cx__impl_saved;
  if (cx_.defined()) cx__impl_saved = cx_.getIntrusivePtr();
  c10::optional<Storage> cy__storage_saved =
    cy_.has_storage() ? c10::optional<Storage>(cy_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> cy__impl_saved;
  if (cy_.defined()) cy__impl_saved = cy_.getIntrusivePtr();
  c10::optional<Storage> workspace__storage_saved =
    workspace_.has_storage() ? c10::optional<Storage>(workspace_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> workspace__impl_saved;
  if (workspace_.defined()) workspace__impl_saved = workspace_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_thnn_fused_lstm_cell_backward(grad_hy_, grad_cy_, cx_, cy_, workspace_, has_bias);
  })();
  std::tie(result0, result1, result2, result3, result4) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_hy__storage_saved.has_value())
    AT_ASSERT(grad_hy__storage_saved.value().is_alias_of(grad_hy_.storage()));
  if (grad_hy__impl_saved) AT_ASSERT(grad_hy__impl_saved == grad_hy_.getIntrusivePtr());
  if (grad_cy__storage_saved.has_value())
    AT_ASSERT(grad_cy__storage_saved.value().is_alias_of(grad_cy_.storage()));
  if (grad_cy__impl_saved) AT_ASSERT(grad_cy__impl_saved == grad_cy_.getIntrusivePtr());
  if (cx__storage_saved.has_value())
    AT_ASSERT(cx__storage_saved.value().is_alias_of(cx_.storage()));
  if (cx__impl_saved) AT_ASSERT(cx__impl_saved == cx_.getIntrusivePtr());
  if (cy__storage_saved.has_value())
    AT_ASSERT(cy__storage_saved.value().is_alias_of(cy_.storage()));
  if (cy__impl_saved) AT_ASSERT(cy__impl_saved == cy_.getIntrusivePtr());
  if (workspace__storage_saved.has_value())
    AT_ASSERT(workspace__storage_saved.value().is_alias_of(workspace_.storage()));
  if (workspace__impl_saved) AT_ASSERT(workspace__impl_saved == workspace_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3, result4 ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
  RECORD_FUNCTION("_trilinear", std::vector<c10::IValue>({i1, i2, i3}), Node::peek_at_next_sequence_nr());
  auto& i1_ = unpack(i1, "i1", 0);
  auto& i2_ = unpack(i2, "i2", 1);
  auto& i3_ = unpack(i3, "i3", 2);
  std::shared_ptr<TrilinearBackward> grad_fn;
  if (compute_requires_grad( i1, i2, i3 )) {
    grad_fn = std::shared_ptr<TrilinearBackward>(new TrilinearBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( i1, i2, i3 ));
    grad_fn->i1_ = SavedVariable(i1, false);
    grad_fn->i2_ = SavedVariable(i2, false);
    grad_fn->i3_ = SavedVariable(i3, false);
    grad_fn->expand1 = expand1.vec();
    grad_fn->expand2 = expand2.vec();
    grad_fn->expand3 = expand3.vec();
    grad_fn->sumdim = sumdim.vec();
    grad_fn->unroll_dim = unroll_dim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_trilinear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "i1", i1);
    jit::tracer::addInputs(node, "i2", i2);
    jit::tracer::addInputs(node, "i3", i3);
    jit::tracer::addInputs(node, "expand1", expand1);
    jit::tracer::addInputs(node, "expand2", expand2);
    jit::tracer::addInputs(node, "expand3", expand3);
    jit::tracer::addInputs(node, "sumdim", sumdim);
    jit::tracer::addInputs(node, "unroll_dim", unroll_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> i1__storage_saved =
    i1_.has_storage() ? c10::optional<Storage>(i1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i1__impl_saved;
  if (i1_.defined()) i1__impl_saved = i1_.getIntrusivePtr();
  c10::optional<Storage> i2__storage_saved =
    i2_.has_storage() ? c10::optional<Storage>(i2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i2__impl_saved;
  if (i2_.defined()) i2__impl_saved = i2_.getIntrusivePtr();
  c10::optional<Storage> i3__storage_saved =
    i3_.has_storage() ? c10::optional<Storage>(i3_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> i3__impl_saved;
  if (i3_.defined()) i3__impl_saved = i3_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_trilinear(i1_, i2_, i3_, expand1, expand2, expand3, sumdim, unroll_dim);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (i1__storage_saved.has_value())
    AT_ASSERT(i1__storage_saved.value().is_alias_of(i1_.storage()));
  if (i1__impl_saved) AT_ASSERT(i1__impl_saved == i1_.getIntrusivePtr());
  if (i2__storage_saved.has_value())
    AT_ASSERT(i2__storage_saved.value().is_alias_of(i2_.storage()));
  if (i2__impl_saved) AT_ASSERT(i2__impl_saved == i2_.getIntrusivePtr());
  if (i3__storage_saved.has_value())
    AT_ASSERT(i3__storage_saved.value().is_alias_of(i3_.storage()));
  if (i3__impl_saved) AT_ASSERT(i3__impl_saved == i3_.getIntrusivePtr());
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
Tensor _unsafe_view(const Tensor & self, IntArrayRef size) {
  RECORD_FUNCTION("_unsafe_view", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsafeViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsafeViewBackward>(new UnsafeViewBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unsafe_view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
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
    return at::_unsafe_view(self_, size);
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
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {
  RECORD_FUNCTION("_weight_norm_cuda_interface", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  auto& v_ = unpack(v, "v", 0);
  auto& g_ = unpack(g, "g", 1);
  std::shared_ptr<WeightNormCudaInterfaceBackward> grad_fn;
  if (compute_requires_grad( v, g )) {
    grad_fn = std::shared_ptr<WeightNormCudaInterfaceBackward>(new WeightNormCudaInterfaceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( v, g ));
    grad_fn->v_ = SavedVariable(v, false);
    grad_fn->g_ = SavedVariable(g, false);
    grad_fn->dim = dim;
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_cuda_interface");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> v__storage_saved =
    v_.has_storage() ? c10::optional<Storage>(v_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> v__impl_saved;
  if (v_.defined()) v__impl_saved = v_.getIntrusivePtr();
  c10::optional<Storage> g__storage_saved =
    g_.has_storage() ? c10::optional<Storage>(g_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> g__impl_saved;
  if (g_.defined()) g__impl_saved = g_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_weight_norm_cuda_interface(v_, g_, dim);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (v__storage_saved.has_value())
    AT_ASSERT(v__storage_saved.value().is_alias_of(v_.storage()));
  if (v__impl_saved) AT_ASSERT(v__impl_saved == v_.getIntrusivePtr());
  if (g__storage_saved.has_value())
    AT_ASSERT(g__storage_saved.value().is_alias_of(g_.storage()));
  if (g__impl_saved) AT_ASSERT(g__impl_saved == g_.getIntrusivePtr());
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
Tensor abs(const Tensor & self) {
  RECORD_FUNCTION("abs", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::abs(self_);
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
Tensor & abs_(Tensor & self) {
  RECORD_FUNCTION("abs_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AbsBackward>(new AbsBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::abs");
    } else {
      op_name = jit::Symbol::fromQualString("aten::abs_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
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
    at::abs_(self_);
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
Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::adaptive_avg_pool1d(self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & adaptive_avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool2d_out", out);
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
    at::adaptive_avg_pool2d_out(out_, self_, output_size);
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
Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  RECORD_FUNCTION("adaptive_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<AdaptiveMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool2DBackwardBackward>(new AdaptiveMaxPool2DBackwardBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
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
    return at::adaptive_max_pool2d_backward(grad_output_, self_, indices_);
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
std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveMaxPool3DBackward>(new AdaptiveMaxPool3DBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
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
    return at::adaptive_max_pool3d(self_, output_size);
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
Tensor & adaptive_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  RECORD_FUNCTION("adaptive_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
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
    at::adaptive_max_pool3d_backward_out(grad_input_, grad_output_, self_, indices_);
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
Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcmul", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->tensor2_ = SavedVariable(tensor2, false);
    }
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
    op_name = jit::Symbol::fromQualString("aten::addcmul");
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
    return at::addcmul(self_, tensor1_, tensor2_, value);
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
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  RECORD_FUNCTION("addcmul_", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcmulBackward>(new AddcmulBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->tensor2_ = SavedVariable(tensor2, false);
    }
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
      op_name = jit::Symbol::fromQualString("aten::addcmul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addcmul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_", self);
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
    self_.addcmul_(tensor1_, tensor2_, value);
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
Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->beta = beta;
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::addmm(self_, mat1_, mat2_, beta, alpha);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
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
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addmm_", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<AddmmBackward>(new AddmmBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->beta = beta;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
    self_.addmm_(mat1_, mat2_, beta, alpha);
  }
  #ifndef NDEBUG
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
Tensor & addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("addr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addr");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", out);
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
  c10::optional<Storage> vec1__storage_saved =
    vec1_.has_storage() ? c10::optional<Storage>(vec1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec1__impl_saved;
  if (vec1_.defined()) vec1__impl_saved = vec1_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addr_out(out_, self_, vec1_, vec2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec1__storage_saved.has_value())
    AT_ASSERT(vec1__storage_saved.value().is_alias_of(vec1_.storage()));
  if (vec1__impl_saved) AT_ASSERT(vec1__impl_saved == vec1_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
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
Tensor align_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("align_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::align_as(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> align_tensors(TensorList tensors) {
  RECORD_FUNCTION("align_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::align_tensors(tensors);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor align_to(const Tensor & self, DimnameList names) {
  RECORD_FUNCTION("align_to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::align_to(self, names);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor align_to_ellipsis_idx(const Tensor & self, DimnameList order, int64_t ellipsis_idx) {
  RECORD_FUNCTION("align_to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "order", order);
    jit::tracer::addInputs(node, "ellipsis_idx", ellipsis_idx);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::align_to_ellipsis_idx(self, order, ellipsis_idx);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  RECORD_FUNCTION("argmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("argmax"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmax");
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
    return at::argmax(self_, dim, keepdim);
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
Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::argsort(self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor argsort_dimname(const Tensor & self, Dimname dim, bool descending) {
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::argsort_dimname(self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor atan(const Tensor & self) {
  RECORD_FUNCTION("atan", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::atan(self_);
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
Tensor atan2(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("atan2", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::atan2");
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
    return at::atan2(self_, other_);
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
Tensor & atan2_(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("atan2_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<Atan2Backward>(new Atan2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atan2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_", self);
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
    self_.atan2_(other_);
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
Tensor & atan_(Tensor & self) {
  RECORD_FUNCTION("atan_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AtanBackward>(new AtanBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::atan");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
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
    at::atan_(self_);
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
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AvgPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AvgPool2DBackward>(new AvgPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->divisor_override = divisor_override;
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
    return at::avg_pool2d(self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
Tensor & avg_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_backward_out", grad_input);
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
    at::avg_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
Tensor & avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  RECORD_FUNCTION("avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("avg_pool3d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_out", out);
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
    at::avg_pool3d_out(out_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
Tensor & baddbmm_out_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("baddbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("baddbmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("baddbmm");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", out);
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
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::baddbmm_out(out_, self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
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
Tensor bartlett_window(int64_t window_length, const TensorOptions & options) {
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bartlett_window(window_length, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bartlett_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::bartlett_window_periodic(window_length, periodic, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) {
  RECORD_FUNCTION("batch_norm_update_stats", std::vector<c10::IValue>({input, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 1);
  auto running_var_ = unpack_opt(running_var, "running_var", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, running_mean, running_var )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_update_stats"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, running_mean, running_var ));
  }
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_update_stats");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
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
    return at::batch_norm_update_stats(input_, running_mean_, running_var_, momentum);
  })();
  std::tie(result0, result1) = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
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
Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("binary_cross_entropy_backward", std::vector<c10::IValue>({grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<BinaryCrossEntropyBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyBackwardBackward>(new BinaryCrossEntropyBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
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
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::binary_cross_entropy_backward(grad_output_, self_, target_, weight_, reduction);
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
Tensor & bitwise_and_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("bitwise_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("bitwise_and");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_and");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
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
    at::bitwise_and_out(out_, self_, other_);
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
Tensor & bitwise_and_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("bitwise_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_and");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_and");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
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
    at::bitwise_and_out(out_, self_, other);
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
Tensor & bitwise_not_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("bitwise_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_not");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_not");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_out", out);
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
    at::bitwise_not_out(out_, self_);
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
Tensor & bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  RECORD_FUNCTION("bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("bmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bmm");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bmm_out", out);
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
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::bmm_out(out_, self_, mat2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
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
Tensor cdist(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  RECORD_FUNCTION("cdist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cdist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::cdist(x1, x2, p, compute_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor celu(const Tensor & self, Scalar alpha) {
  RECORD_FUNCTION("celu", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::celu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::celu(self, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & celu_(Tensor & self, Scalar alpha) {
  RECORD_FUNCTION("celu_", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::celu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::celu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("celu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::celu_(self, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor cholesky_inverse(const Tensor & self, bool upper) {
  RECORD_FUNCTION("cholesky_inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CholeskyInverseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CholeskyInverseBackward>(new CholeskyInverseBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->upper = upper;
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
    return at::cholesky_inverse(self_, upper);
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
Tensor & cholesky_out_out(Tensor & out, const Tensor & self, bool upper) {
  RECORD_FUNCTION("cholesky_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cholesky");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_out", out);
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
    at::cholesky_out(out_, self_, upper);
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
Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  RECORD_FUNCTION("clamp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
    grad_fn->max = max;
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
    return at::clamp(self_, min, max);
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
Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  RECORD_FUNCTION("clamp_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampBackward>(new ClampBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
    grad_fn->max = max;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
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
    at::clamp_(self_, min, max);
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
Tensor & clamp_max_out_out(Tensor & out, const Tensor & self, Scalar max) {
  RECORD_FUNCTION("clamp_max_out", std::vector<c10::IValue>({out, self, max}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_max");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp_max");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", out);
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
    at::clamp_max_out(out_, self_, max);
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
Tensor & conj_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("conj_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("conj");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("conj");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conj");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("conj_out", out);
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
    at::conj_out(out_, self_);
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
Tensor conv_transpose3d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  RECORD_FUNCTION("conv_transpose3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose3d_input(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("convolution_backward_overrideable", std::vector<c10::IValue>({grad_output, input, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ConvolutionBackwardOverrideableBackward> grad_fn;
  if (compute_requires_grad( grad_output, input, weight )) {
    grad_fn = std::shared_ptr<ConvolutionBackwardOverrideableBackward>(new ConvolutionBackwardOverrideableBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->groups = groups;
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
    op_name = jit::Symbol::fromQualString("aten::convolution_backward_overrideable");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
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
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::convolution_backward_overrideable(grad_output_, input_, weight_, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
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
Tensor cos(const Tensor & self) {
  RECORD_FUNCTION("cos", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::cos(self_);
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
Tensor & cos_(Tensor & self) {
  RECORD_FUNCTION("cos_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CosBackward>(new CosBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::cos");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cos_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_", self);
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
    at::cos_(self_);
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
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon, const Tensor & reserveSpace) {
  RECORD_FUNCTION("cudnn_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var, reserveSpace}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 5);
  auto save_var_ = unpack_opt(save_var, "save_var", 6);
  auto& reserveSpace_ = unpack(reserveSpace, "reserveSpace", 8);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( input, grad_output, weight, save_mean, save_var, reserveSpace )) {
    grad_fn = std::shared_ptr<CudnnBatchNormBackwardBackward>(new CudnnBatchNormBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grad_output, weight, save_mean, save_var, reserveSpace ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_var_ = SavedVariable(save_var, false);
    grad_fn->epsilon = epsilon;
    grad_fn->reserveSpace_ = SavedVariable(reserveSpace, false);
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
    op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    jit::tracer::addInputs(node, "reserveSpace", reserveSpace);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  c10::optional<Storage> save_mean__storage_saved =
    save_mean_.has_storage() ? c10::optional<Storage>(save_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_mean__impl_saved;
  if (save_mean_.defined()) save_mean__impl_saved = save_mean_.getIntrusivePtr();
  c10::optional<Storage> save_var__storage_saved =
    save_var_.has_storage() ? c10::optional<Storage>(save_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_var__impl_saved;
  if (save_var_.defined()) save_var__impl_saved = save_var_.getIntrusivePtr();
  c10::optional<Storage> reserveSpace__storage_saved =
    reserveSpace_.has_storage() ? c10::optional<Storage>(reserveSpace_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> reserveSpace__impl_saved;
  if (reserveSpace_.defined()) reserveSpace__impl_saved = reserveSpace_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon, reserveSpace_);
  })();
  std::tie(result0, result1, result2) = std::move(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  if (save_mean__storage_saved.has_value())
    AT_ASSERT(save_mean__storage_saved.value().is_alias_of(save_mean_.storage()));
  if (save_mean__impl_saved) AT_ASSERT(save_mean__impl_saved == save_mean_.getIntrusivePtr());
  if (save_var__storage_saved.has_value())
    AT_ASSERT(save_var__storage_saved.value().is_alias_of(save_var_.storage()));
  if (save_var__impl_saved) AT_ASSERT(save_var__impl_saved == save_var_.getIntrusivePtr());
  if (reserveSpace__storage_saved.has_value())
    AT_ASSERT(reserveSpace__storage_saved.value().is_alias_of(reserveSpace_.storage()));
  if (reserveSpace__impl_saved) AT_ASSERT(reserveSpace__impl_saved == reserveSpace_.getIntrusivePtr());
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
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
  RECORD_FUNCTION("cudnn_grid_sampler_backward", std::vector<c10::IValue>({self, grid, grad_output}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto& grad_output_ = unpack(grad_output, "grad_output", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, grid, grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_grid_sampler_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grid, grad_output ));
  }
  Tensor grad_self;
  Tensor grad_grid;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "grad_output", grad_output);
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
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_grid_sampler_backward(self_, grid_, grad_output_);
  })();
  std::tie(grad_self, grad_grid) = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_self, grad_grid ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_self);
    jit::tracer::addOutput(node, grad_grid);
  }
  #endif
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
std::tuple<Tensor &,Tensor &> cummax_out_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("cummax_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cummax");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("cummax");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
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
    at::cummax_out(values_, indices_, self_, dim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> cummax_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("cummax_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::cummax_out_dimname_out(values, indices, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor & cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumsum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cumsum");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
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
    at::cumsum_out(out_, self_, dim, dtype);
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
Tensor & cumsum_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::cumsum_out_dimname_out(out, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor diag(const Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("diag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DiagBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DiagBackward>(new DiagBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->diagonal = diagonal;
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
    return at::diag(self_, diagonal);
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
Tensor digamma(const Tensor & self) {
  RECORD_FUNCTION("digamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::digamma(self_);
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
Tensor & digamma_(Tensor & self) {
  RECORD_FUNCTION("digamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DigammaBackward>(new DigammaBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::digamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::digamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_", self);
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
    self_.digamma_();
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
Tensor & elu_out_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  RECORD_FUNCTION("elu_out", std::vector<c10::IValue>({out, self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("elu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("elu");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_out", out);
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
    at::elu_out(out_, self_, alpha, scale, input_scale);
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
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  RECORD_FUNCTION("embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
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
    op_name = jit::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1, result2, result3) = TypeDefault::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  RECORD_FUNCTION("embedding_dense_backward", std::vector<c10::IValue>({grad_output, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<EmbeddingDenseBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<EmbeddingDenseBackwardBackward>(new EmbeddingDenseBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->indices_ = SavedVariable(indices, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::embedding_dense_backward(grad_output_, indices_, num_weights, padding_idx, scale_grad_by_freq);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
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
Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("empty_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::empty_like(self, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
  RECORD_FUNCTION("empty_strided", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::empty_strided(size, stride, options_);
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
Tensor & erfc_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("erfc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("erfc");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_out", out);
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
    at::erfc_out(out_, self_);
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
Tensor erfinv(const Tensor & self) {
  RECORD_FUNCTION("erfinv", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::erfinv(self_);
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
Tensor & erfinv_(Tensor & self) {
  RECORD_FUNCTION("erfinv_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfinvBackward>(new ErfinvBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::erfinv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erfinv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_", self);
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
    self_.erfinv_();
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
Tensor expand(const Tensor & self, IntArrayRef size, bool implicit) {
  RECORD_FUNCTION("expand", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpandBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpandBackward>(new ExpandBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "implicit", implicit);
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
    return self_.expand(size, implicit);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE);
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
Tensor & expm1_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("expm1_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("expm1");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("expm1");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expm1");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_out", out);
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
    at::expm1_out(out_, self_);
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
Tensor fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  RECORD_FUNCTION("fake_quantize_per_tensor_affine_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::fake_quantize_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  RECORD_FUNCTION("fft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::fft(self, signal_ndim, normalized);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_using_ints(const Tensor & self, int64_t start_dim, int64_t end_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::flatten_using_ints(self, start_dim, end_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_named_out_dim(const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::flatten_named_out_dim(self, start_dim, end_dim, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_using_names(const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::flatten_using_names(self, start_dim, end_dim, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_DimnameList(const Tensor & self, DimnameList dims, Dimname out_dim) {
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::flatten_DimnameList(self, dims, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor floor_divide(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("floor_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("floor_divide"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
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
    return at::floor_divide(self_, other_);
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
Tensor floor_divide_Scalar(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("floor_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::floor_divide_Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & floor_divide__Tensor(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("floor_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("floor_divide_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
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
    self_.floor_divide_(other_);
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
Tensor & floor_divide__Scalar(Tensor & self, Scalar other) {
  RECORD_FUNCTION("floor_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::floor_divide__Scalar(self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & floor_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("floor_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("floor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("floor");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_out", out);
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
    at::floor_out(out_, self_);
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
Tensor full_names(IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) {
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::full_names(size, fill_value, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::full(size, fill_value, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<GatherBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GatherBackward>(new GatherBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->sparse_grad = sparse_grad;
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
    tracer_state->graph->insertNode(node);
  
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
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gather(self_, dim, index_, sparse_grad);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
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
Tensor gather_dimname(const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::gather_dimname(self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gelu_backward(const Tensor & grad, const Tensor & self) {
  RECORD_FUNCTION("gelu_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("gelu_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad, self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gelu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
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
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gelu_backward(grad_, self_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
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
std::tuple<Tensor,Tensor> grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  RECORD_FUNCTION("grid_sampler_3d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, input, grid )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_3d_backward"), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d_backward");
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
    return at::grid_sampler_3d_backward(grad_output_, input_, grid_, interpolation_mode, padding_mode, align_corners);
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
Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor hann_window(int64_t window_length, const TensorOptions & options) {
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::hann_window(window_length, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hann_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::hann_window_periodic(window_length, periodic, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hardshrink(const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("hardshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardshrinkBackward>(new HardshrinkBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::hardshrink");
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
    return at::hardshrink(self_, lambd);
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
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  RECORD_FUNCTION("ifft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ifft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::ifft(self, signal_ndim, normalized);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_select_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) {
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("index_select");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("index_select");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
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
    at::index_select_out(out_, self_, dim, index_);
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
Tensor & index_select_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index) {
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::index_select_out_dimname_out(out, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor indices(const Tensor & self) {
  RECORD_FUNCTION("indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::indices");
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
    return self_.indices();
  })();
  auto result = as_view(self, tmp, /* is_differentiable */ false);
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
bool is_complex(const Tensor & self) {
  RECORD_FUNCTION("is_complex", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::is_complex(self);
  return result;
}
bool is_same_size(const Tensor & self, const Tensor & other) {
  auto result = TypeDefault::is_same_size(self, other);
  return result;
}
Tensor & l1_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("l1_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("l1_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("l1_loss");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_out", out);
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
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::l1_loss_out(out_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
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
Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {
  RECORD_FUNCTION("layer_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::layer_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enable", cudnn_enable);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope, bool self_is_result) {
  RECORD_FUNCTION("leaky_relu_backward", std::vector<c10::IValue>({grad_output, self, negative_slope}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<LeakyReluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<LeakyReluBackwardBackward>(new LeakyReluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
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
    return at::leaky_relu_backward(grad_output_, self_, negative_slope, self_is_result);
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
Tensor lerp_Scalar(const Tensor & self, const Tensor & end, Scalar weight) {
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  std::shared_ptr<LerpBackward0> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward0>(new LerpBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> end__storage_saved =
    end_.has_storage() ? c10::optional<Storage>(end_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> end__impl_saved;
  if (end_.defined()) end__impl_saved = end_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lerp(self_, end_, weight);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
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
Tensor lerp_Tensor(const Tensor & self, const Tensor & end, const Tensor & weight) {
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<LerpBackward1> grad_fn;
  if (compute_requires_grad( self, end, weight )) {
    grad_fn = std::shared_ptr<LerpBackward1>(new LerpBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end, weight ));
    grad_fn->weight_ = SavedVariable(weight, false);
    if (grad_fn->should_compute_output(2)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    if (grad_fn->should_compute_output(2)) {
      grad_fn->end_ = SavedVariable(end, false);
    }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::lerp(self_, end_, weight_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
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
Tensor & lerp__Scalar(Tensor & self, const Tensor & end, Scalar weight) {
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  check_inplace(self);
  std::shared_ptr<LerpBackward0> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::shared_ptr<LerpBackward0>(new LerpBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
    self_.lerp_(end_, weight);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (end__storage_saved.has_value())
    AT_ASSERT(end__storage_saved.value().is_alias_of(end_.storage()));
  if (end__impl_saved) AT_ASSERT(end__impl_saved == end_.getIntrusivePtr());
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
Tensor & lerp__Tensor(Tensor & self, const Tensor & end, const Tensor & weight) {
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  check_inplace(self);
  std::shared_ptr<LerpBackward1> grad_fn;
  if (compute_requires_grad( self, end, weight )) {
    grad_fn = std::shared_ptr<LerpBackward1>(new LerpBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, end, weight ));
    grad_fn->weight_ = SavedVariable(weight, false);
    if (grad_fn->should_compute_output(2)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    if (grad_fn->should_compute_output(2)) {
      grad_fn->end_ = SavedVariable(end, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
    self_.lerp_(end_, weight_);
  }
  #ifndef NDEBUG
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
Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  RECORD_FUNCTION("linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::linear(input, weight, bias);
  return result;
}
Tensor log_sigmoid(const Tensor & self) {
  RECORD_FUNCTION("log_sigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::log_sigmoid(self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & log_sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  RECORD_FUNCTION("log_sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& buffer_ = unpack(buffer, "buffer", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, buffer )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
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
  c10::optional<Storage> buffer__storage_saved =
    buffer_.has_storage() ? c10::optional<Storage>(buffer_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> buffer__impl_saved;
  if (buffer_.defined()) buffer__impl_saved = buffer_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log_sigmoid_backward_out(grad_input_, grad_output_, self_, buffer_);
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
  if (buffer__storage_saved.has_value())
    AT_ASSERT(buffer__storage_saved.value().is_alias_of(buffer_.storage()));
  if (buffer__impl_saved) AT_ASSERT(buffer__impl_saved == buffer_.getIntrusivePtr());
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
Tensor & logical_or_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("logical_or_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logical_or");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logical_or");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_out", out);
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
    at::logical_or_out(out_, self_, other_);
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
Tensor & logical_xor_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("logical_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logical_xor");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logical_xor");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_out", out);
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
    at::logical_xor_out(out_, self_, other_);
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
Tensor & logspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) {
  RECORD_FUNCTION("logspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
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
    at::logspace_out(out_, start, end, steps, base);
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
Tensor & logsumexp_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("logsumexp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logsumexp");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
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
    at::logsumexp_out(out_, self_, dim, keepdim);
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
Tensor & logsumexp_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim) {
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::logsumexp_out_names_out(out, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & matmul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("matmul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::matmul_out_out(out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor &,Tensor &> max_out_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  auto& max_ = unpack(max, "max", 0);
  auto& max_values_ = unpack(max_values, "max_values", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( max )) {
    throw_error_out_requires_grad("max");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> max__storage_saved =
    max_.has_storage() ? c10::optional<Storage>(max_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max__impl_saved;
  if (max_.defined()) max__impl_saved = max_.getIntrusivePtr();
  c10::optional<Storage> max_values__storage_saved =
    max_values_.has_storage() ? c10::optional<Storage>(max_values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> max_values__impl_saved;
  if (max_values_.defined()) max_values__impl_saved = max_values_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_out(max_, max_values_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (max__storage_saved.has_value())
    AT_ASSERT(max__storage_saved.value().is_alias_of(max_.storage()));
  if (max__impl_saved) AT_ASSERT(max__impl_saved == max_.getIntrusivePtr());
  if (max_values__storage_saved.has_value())
    AT_ASSERT(max_values__storage_saved.value().is_alias_of(max_values_.storage()));
  if (max_values__impl_saved) AT_ASSERT(max_values__impl_saved == max_values_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(max);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( max ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  #endif
  return std::forward_as_tuple(max, max_values);
}
std::tuple<Tensor &,Tensor &> max_out_names_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::max_out_names_dim_max(max, max_values, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  #endif
  return std::forward_as_tuple(max, max_values);
}
Tensor & max_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", out);
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
    at::max_out(out_, self_, other_);
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
Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  RECORD_FUNCTION("max_unpool2d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<MaxUnpool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxUnpool2DBackward>(new MaxUnpool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size.vec();
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
    return at::max_unpool2d(self_, indices_, output_size);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
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
Tensor & max_unpool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  RECORD_FUNCTION("max_unpool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_backward_out", grad_input);
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
    at::max_unpool2d_backward_out(grad_input_, grad_output_, self_, indices_, output_size);
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
Tensor & max_unpool3d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("max_unpool3d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_unpool3d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_out", out);
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
    at::max_unpool3d_out(out_, self_, indices_, output_size, stride, padding);
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
Tensor min_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::min_values(self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor min_values_names(const Tensor & self, DimnameList dim, bool keepdim) {
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::min_values_names(self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("miopen_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<MiopenConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackwardBackward>(new MiopenConvolutionBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
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
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
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
    return at::miopen_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1, result2) = std::move(tmp);
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
Tensor miopen_convolution_backward_bias(const Tensor & grad_output) {
  RECORD_FUNCTION("miopen_convolution_backward_bias", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_bias"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_convolution_backward_bias(grad_output_);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
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
Tensor miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_input"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
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
    return at::miopen_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenConvolutionTransposeBackward>(new MiopenConvolutionTransposeBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
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
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
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
    return at::miopen_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("mkldnn_adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_adaptive_avg_pool2d"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d");
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
    return at::mkldnn_adaptive_avg_pool2d(self_, output_size);
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
Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("mkldnn_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MkldnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MkldnnConvolutionBackward>(new MkldnnConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
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
    return at::mkldnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups);
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
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("mkldnn_reorder_conv2d_weight", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_reorder_conv2d_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_reorder_conv2d_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
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
    return at::mkldnn_reorder_conv2d_weight(self_, padding, stride, dilation, groups);
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
std::tuple<Tensor &,Tensor &> mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mode");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("mode");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
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
    at::mode_out(values_, indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> mode_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::mode_out_dimname_out(values, indices, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("mse_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MseLossBackwardBackward>(new MseLossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
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
    op_name = jit::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
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
    return at::mse_loss_backward(grad_output_, self_, target_, reduction);
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
Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  RECORD_FUNCTION("multilabel_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& is_target_ = unpack(is_target, "is_target", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target, is_target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("multilabel_margin_loss_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, is_target ));
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
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::multilabel_margin_loss_backward(grad_output_, self_, target_, reduction, is_target_);
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
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
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
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out_output(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("multilabel_margin_loss_forward_out", std::vector<c10::IValue>({output, is_target, self, target}), Node::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& is_target_ = unpack(is_target, "is_target", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "is_target", is_target);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> is_target__storage_saved =
    is_target_.has_storage() ? c10::optional<Storage>(is_target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> is_target__impl_saved;
  if (is_target_.defined()) is_target__impl_saved = is_target_.getIntrusivePtr();
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
    at::multilabel_margin_loss_forward_out(output_, is_target_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (is_target__storage_saved.has_value())
    AT_ASSERT(is_target__storage_saved.value().is_alias_of(is_target_.storage()));
  if (is_target__impl_saved) AT_ASSERT(is_target__impl_saved == is_target_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  #endif
  return std::forward_as_tuple(output, is_target);
}
Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator generator) {
  RECORD_FUNCTION("multinomial", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
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
    return at::multinomial(self_, num_samples, replacement, generator);
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
Tensor mvlgamma(const Tensor & self, int64_t p) {
  RECORD_FUNCTION("mvlgamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::mvlgamma");
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
    return at::mvlgamma(self_, p);
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
Tensor & mvlgamma_(Tensor & self, int64_t p) {
  RECORD_FUNCTION("mvlgamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MvlgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MvlgammaBackward>(new MvlgammaBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mvlgamma_", self);
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
    self_.mvlgamma_(p);
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
Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  RECORD_FUNCTION("narrow", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::narrow(self, dim, start, length);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor narrow_Tensor(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) {
  RECORD_FUNCTION("narrow", std::vector<c10::IValue>({self, start}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::narrow_Tensor(self, dim, start, length);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  RECORD_FUNCTION("native_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<NativeBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<NativeBatchNormBackward>(new NativeBatchNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->eps = eps;
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
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
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
    return at::native_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, momentum, eps);
  })();
  std::tie(result0, result1, result2) = std::move(tmp);
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
  }
  #endif
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & ne_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
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
    at::ne_out(out_, self_, other);
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
Tensor & ne_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
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
    at::ne_out(out_, self_, other_);
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
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor new_full(const Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  RECORD_FUNCTION("new_full", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::new_full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::new_full(self, size, fill_value, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::nll_loss(self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss2d", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::nll_loss2d(self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & nll_loss2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  RECORD_FUNCTION("nll_loss2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_backward_out", grad_input);
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
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::nll_loss2d_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
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
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
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
Tensor & nll_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  RECORD_FUNCTION("nll_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_backward_out", grad_input);
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
  c10::optional<Storage> total_weight__storage_saved =
    total_weight_.has_storage() ? c10::optional<Storage>(total_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> total_weight__impl_saved;
  if (total_weight_.defined()) total_weight__impl_saved = total_weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::nll_loss_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction, ignore_index, total_weight_);
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
  if (total_weight__storage_saved.has_value())
    AT_ASSERT(total_weight__storage_saved.value().is_alias_of(total_weight_.storage()));
  if (total_weight__impl_saved) AT_ASSERT(total_weight__impl_saved == total_weight_.getIntrusivePtr());
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
Tensor nuclear_norm(const Tensor & self, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::nuclear_norm(self, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nuclear_norm_dim(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::nuclear_norm_dim(self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor orgqr(const Tensor & self, const Tensor & input2) {
  RECORD_FUNCTION("orgqr", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<OrgqrBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::shared_ptr<OrgqrBackward>(new OrgqrBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
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
    return at::orgqr(self_, input2_);
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
  return result;
}
Tensor & ormqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  RECORD_FUNCTION("ormqr_out", std::vector<c10::IValue>({out, self, input2, input3}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  auto& input3_ = unpack(input3, "input3", 3);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    throw_error_out_requires_grad("ormqr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("ormqr");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ormqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "input3", input3);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "transpose", transpose);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ormqr_out", out);
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
  c10::optional<Storage> input3__storage_saved =
    input3_.has_storage() ? c10::optional<Storage>(input3_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input3__impl_saved;
  if (input3_.defined()) input3__impl_saved = input3_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ormqr_out(out_, self_, input2_, input3_, left, transpose);
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
  if (input3__storage_saved.has_value())
    AT_ASSERT(input3__storage_saved.value().is_alias_of(input3_.storage()));
  if (input3__impl_saved) AT_ASSERT(input3__impl_saved == input3_.getIntrusivePtr());
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
Tensor permute(const Tensor & self, IntArrayRef dims) {
  RECORD_FUNCTION("permute", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PermuteBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PermuteBackward>(new PermuteBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims.vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::permute");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
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
    return self_.permute(dims);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE);
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
Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor) {
  RECORD_FUNCTION("pixel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pixel_shuffle");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upscale_factor", upscale_factor);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::pixel_shuffle(self, upscale_factor);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  RECORD_FUNCTION("put_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<PutBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<PutBackward>(new PutBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->index_ = SavedVariable(index, false);
    if (grad_fn->should_compute_output(0)) {
      grad_fn->source_ = SavedVariable(source, false);
    }
    grad_fn->accumulate = accumulate;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_", self);
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
    self_.put_(index_, source_, accumulate);
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
int64_t q_zero_point(const Tensor & self) {
  RECORD_FUNCTION("q_zero_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::q_zero_point(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
  RECORD_FUNCTION("quantize_per_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
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
    return at::quantize_per_tensor(self_, scale, zero_point, dtype);
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
std::vector<Tensor> quantize_per_tensor_tensors(TensorList tensors, const Tensor & scales, const Tensor & zero_points, ScalarType dtype) {
  RECORD_FUNCTION("quantize_per_tensor", std::vector<c10::IValue>({scales, zero_points}), Node::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto& scales_ = unpack(scales, "scales", 1);
  auto& zero_points_ = unpack(zero_points, "zero_points", 2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "dtype", dtype);
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
  c10::optional<Storage> scales__storage_saved =
    scales_.has_storage() ? c10::optional<Storage>(scales_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> scales__impl_saved;
  if (scales_.defined()) scales__impl_saved = scales_.getIntrusivePtr();
  c10::optional<Storage> zero_points__storage_saved =
    zero_points_.has_storage() ? c10::optional<Storage>(zero_points_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> zero_points__impl_saved;
  if (zero_points_.defined()) zero_points__impl_saved = zero_points_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::quantize_per_tensor(tensors_, scales_, zero_points_, dtype);
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
  if (scales__storage_saved.has_value())
    AT_ASSERT(scales__storage_saved.value().is_alias_of(scales_.storage()));
  if (scales__impl_saved) AT_ASSERT(scales__impl_saved == scales_.getIntrusivePtr());
  if (zero_points__storage_saved.has_value())
    AT_ASSERT(zero_points__storage_saved.value().is_alias_of(zero_points_.storage()));
  if (zero_points__impl_saved) AT_ASSERT(zero_points__impl_saved == zero_points_.getIntrusivePtr());
  #endif
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> quantized_gru_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  RECORD_FUNCTION("quantized_gru", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru");
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
  std::tie(result0, result1) = TypeDefault::quantized_gru_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> quantized_gru_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  RECORD_FUNCTION("quantized_gru", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru");
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
  std::tie(result0, result1) = TypeDefault::quantized_gru_data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  RECORD_FUNCTION("quantized_lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_lstm_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  std::tie(result0, result1) = TypeDefault::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & rand_out_out(Tensor & out, IntArrayRef size) {
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::rand_out_out(out, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & rand_out_generator_out(Tensor & out, IntArrayRef size, Generator generator) {
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::rand_out_generator_out(out, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randn_out_out(Tensor & out, IntArrayRef size) {
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::randn_out_out(out, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randn_out_generator_out(Tensor & out, IntArrayRef size, Generator generator) {
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::randn_out_generator_out(out, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor range_step(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end, step}), Node::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::range_step(start, end, step, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor range(Scalar start, Scalar end, const TensorOptions & options) {
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::range(start, end, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor real(const Tensor & self) {
  RECORD_FUNCTION("real", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RealBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RealBackward>(new RealBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::real");
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
    return at::real(self_);
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
Tensor reciprocal(const Tensor & self) {
  RECORD_FUNCTION("reciprocal", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
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
    return at::reciprocal(self_);
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
Tensor & reciprocal_(Tensor & self) {
  RECORD_FUNCTION("reciprocal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReciprocalBackward>(new ReciprocalBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::reciprocal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::reciprocal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_", self);
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
    at::reciprocal_(self_);
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
Tensor refine_names(const Tensor & self, DimnameList names) {
  RECORD_FUNCTION("refine_names", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::refine_names");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::refine_names(self, names);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor reflection_pad1d(const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReflectionPad1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReflectionPad1DBackward>(new ReflectionPad1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
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
    return at::reflection_pad1d(self_, padding);
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
Tensor & reflection_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_backward_out", grad_input);
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
    at::reflection_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
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
Tensor & reflection_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("reflection_pad2d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_out", out);
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
    at::reflection_pad2d_out(out_, self_, padding);
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
Tensor relu(const Tensor & self) {
  RECORD_FUNCTION("relu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReluBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward0>(new ReluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::relu");
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
    return at::relu(self_);
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
Tensor & relu_(Tensor & self) {
  RECORD_FUNCTION("relu_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReluBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReluBackward1>(new ReluBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::relu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::relu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_", self);
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
    at::relu_(self_);
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
Tensor & remainder_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("remainder");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
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
    at::remainder_out(out_, self_, other);
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
Tensor & remainder_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("remainder");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
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
    at::remainder_out(out_, self_, other_);
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
Tensor & replication_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("replication_pad1d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_out", out);
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
    at::replication_pad1d_out(out_, self_, padding);
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
Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  auto result = TypeDefault::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator generator) {
  RECORD_FUNCTION("rrelu_with_noise", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward0>(new RreluWithNoiseBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> noise__storage_saved =
    noise_.has_storage() ? c10::optional<Storage>(noise_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> noise__impl_saved;
  if (noise_.defined()) noise__impl_saved = noise_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rrelu_with_noise(self_, noise_, lower, upper, training, generator);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
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
Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator generator) {
  RECORD_FUNCTION("rrelu_with_noise_", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_inplace(self);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RreluWithNoiseBackward1>(new RreluWithNoiseBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
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
    at::rrelu_with_noise_(self_, noise_, lower, upper, training, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (noise__storage_saved.has_value())
    AT_ASSERT(noise__storage_saved.value().is_alias_of(noise_.storage()));
  if (noise__impl_saved) AT_ASSERT(noise__impl_saved == noise_.getIntrusivePtr());
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
Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::scatter_add(self, dim, index, src);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor scatter_add_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::scatter_add_dimname(self, dim, index, src);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  RECORD_FUNCTION("scatter_add_", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<ScatterAddBackward> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<ScatterAddBackward>(new ScatterAddBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->index_ = SavedVariable(index, false);
    }
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_", self);
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
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.scatter_add_(dim, index_, src_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
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
Tensor select_Dimname(const Tensor & self, Dimname dim, int64_t index) {
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::select_Dimname(self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor select_int(const Tensor & self, int64_t dim, int64_t index) {
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SelectBackward>(new SelectBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->index = index;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
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
    return at::select(self_, dim, index);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE);
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
Tensor sin(const Tensor & self) {
  RECORD_FUNCTION("sin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
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
    return at::sin(self_);
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
Tensor & sin_(Tensor & self) {
  RECORD_FUNCTION("sin_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinBackward>(new SinBackward(), deleteNode);
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
      op_name = jit::Symbol::fromQualString("aten::sin");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sin_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_", self);
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
    at::sin_(self_);
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
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("slow_conv_dilated3d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<SlowConvDilated3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<SlowConvDilated3DBackwardBackward>(new SlowConvDilated3DBackwardBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated3d_backward");
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
    return at::slow_conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
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
Tensor & soft_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("soft_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("soft_margin_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("soft_margin_loss");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", out);
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
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::soft_margin_loss_out(out_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
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
Tensor softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::softmax_int(self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::softmax_Dimname(self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & softplus_out_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) {
  RECORD_FUNCTION("softplus_out", std::vector<c10::IValue>({out, self, beta, threshold}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softplus");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("softplus");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_out", out);
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
    at::softplus_out(out_, self_, beta, threshold);
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
Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  RECORD_FUNCTION("softshrink_backward", std::vector<c10::IValue>({grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<SoftshrinkBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftshrinkBackwardBackward>(new SoftshrinkBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
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
    return at::softshrink_backward(grad_output_, self_, lambd);
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
std::tuple<Tensor &,Tensor &> sort_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sort");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("sort");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
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
    at::sort_out(values_, indices_, self_, dim, descending);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> sort_out_dimname_values(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool descending) {
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::sort_out_dimname_values(values, indices, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor squeeze(const Tensor & self) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward0>(new SqueezeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
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
    return at::squeeze(self_);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE);
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
Tensor squeeze_dim(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward1>(new SqueezeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
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
    return at::squeeze(self_, dim);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE);
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
Tensor squeeze_dimname(const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::squeeze_dimname(self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & squeeze_(Tensor & self) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward2>(new SqueezeBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
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
    self_.squeeze_();
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & squeeze__dim(Tensor & self, int64_t dim) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward3> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqueezeBackward3>(new SqueezeBackward3(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
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
    self_.squeeze_(dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & squeeze__dimname(Tensor & self, Dimname dim) {
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::squeeze__dimname(self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & std_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("std");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("std");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
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
    at::std_out(out_, self_, dim, unbiased, keepdim);
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
Tensor & std_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::std_out_names_out(out, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor sub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
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
    return at::sub(self_, other_, alpha);
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
Tensor sub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
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
    return at::sub(self_, other, alpha);
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
Tensor & sub__Tensor(Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SubBackward0>(new SubBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
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
    self_.sub_(other_, alpha);
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
Tensor & sub__Scalar(Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SubBackward1>(new SubBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
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
    self_.sub_(other, alpha);
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
Tensor & sum_out_IntList_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sum");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
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
    at::sum_out(out_, self_, dim, keepdim, dtype);
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
Tensor & sum_out_DimnameList_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::sum_out_DimnameList_out(out, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & take_out_out(Tensor & out, const Tensor & self, const Tensor & index) {
  RECORD_FUNCTION("take_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("take");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("take");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::take");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("take_out", out);
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
    at::take_out(out_, self_, index_);
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
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("thnn_conv2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConv2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConv2DBackward>(new ThnnConv2DBackward(), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
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
    return at::thnn_conv2d_forward(self_, weight_, kernel_size, bias_, stride, padding);
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
Tensor & thnn_conv_depthwise2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  RECORD_FUNCTION("thnn_conv_depthwise2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  TypeDefault::thnn_conv_depthwise2d_out_out(out, self, weight, kernel_size, bias, stride, padding, dilation);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor to_dense_backward(const Tensor & grad, const Tensor & input) {
  RECORD_FUNCTION("to_dense_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::to_dense_backward(grad, input);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> topk_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  RECORD_FUNCTION("topk_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("topk");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("topk");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::topk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
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
    at::topk_out(values_, indices_, self_, k, dim, largest, sorted);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor & trunc_out_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("trunc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("trunc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("trunc");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trunc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_out", out);
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
    at::trunc_out(out_, self_);
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
std::vector<Tensor> unbind_int(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnbindBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnbindBackward>(new UnbindBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
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
    return at::unbind(self_, dim);
  })();
  auto result = as_view(/* base */ self, /* output */ tmp, /* is_differentiable */ true, /* creation_meta */ CreationMeta::MULTI_OUTPUT_NODE);
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
std::vector<Tensor> unbind_Dimname(const Tensor & self, Dimname dim) {
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::unbind_Dimname(self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  RECORD_FUNCTION("unique_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("unique_consecutive"), deleteNode);
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
    op_name = jit::Symbol::fromQualString("aten::unique_consecutive");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
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
    return at::unique_consecutive(self_, return_inverse, return_counts, dim);
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
Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_bilinear2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBilinear2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleBilinear2DBackwardBackward>(new UpsampleBilinear2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
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
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_bilinear2d_backward(grad_output_, output_size, input_size, align_corners, scales_h, scales_w);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
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
Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_linear1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleLinear1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleLinear1DBackward>(new UpsampleLinear1DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales = scales;
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
    return at::upsample_linear1d(self_, output_size, align_corners, scales);
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
Tensor & upsample_linear1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_linear1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_backward_out", grad_input);
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
    at::upsample_linear1d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners, scales);
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
Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  RECORD_FUNCTION("upsample_nearest1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleNearest1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleNearest1DBackwardBackward>(new UpsampleNearest1DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->scales = scales;
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_nearest1d_backward(grad_output_, output_size, input_size, scales);
  })();
  auto result = std::move(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
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
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_nearest2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleNearest2DBackward>(new UpsampleNearest2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
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
    return at::upsample_nearest2d(self_, output_size, scales_h, scales_w);
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
Tensor & upsample_nearest2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_nearest2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_backward_out", grad_input);
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
    at::upsample_nearest2d_backward_out(grad_input_, grad_output_, output_size, input_size, scales_h, scales_w);
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
Tensor & upsample_nearest3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  RECORD_FUNCTION("upsample_nearest3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest3d");
  }
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_out", out);
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
    at::upsample_nearest3d_out(out_, self_, output_size, scales_d, scales_h, scales_w);
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
Tensor view_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("view_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  auto result = TypeDefault::view_as(self, other);
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
  .impl_UNBOXED("aten::__ilshift__.Scalar", torch::kAutograd,
          VariableType::__ilshift___Scalar
        )
  .impl_UNBOXED("aten::__ilshift__.Tensor", torch::kAutograd,
          VariableType::__ilshift___Tensor
        )
  .impl_UNBOXED("aten::__ior__.Scalar", torch::kAutograd,
          VariableType::__ior___Scalar
        )
  .impl_UNBOXED("aten::__ior__.Tensor", torch::kAutograd,
          VariableType::__ior___Tensor
        )
  .impl_UNBOXED("aten::__ixor__.Scalar", torch::kAutograd,
          VariableType::__ixor___Scalar
        )
  .impl_UNBOXED("aten::__ixor__.Tensor", torch::kAutograd,
          VariableType::__ixor___Tensor
        )
  .impl("aten::__lshift__.Scalar", torch::kAutograd,
          VariableType::__lshift___Scalar
       )
  .impl("aten::__lshift__.Tensor", torch::kAutograd,
          VariableType::__lshift___Tensor
       )
  .impl("aten::__or__.Scalar", torch::kAutograd,
          VariableType::__or___Scalar
       )
  .impl("aten::__or__.Tensor", torch::kAutograd,
          VariableType::__or___Tensor
       )
  .impl("aten::__xor__.Scalar", torch::kAutograd,
          VariableType::__xor___Scalar
       )
  .impl("aten::__xor__.Tensor", torch::kAutograd,
          VariableType::__xor___Tensor
       )
  .impl_UNBOXED("aten::_addr.out", torch::kAutograd,
          VariableType::_addr_out_out
        )
  .impl_UNBOXED("aten::_baddbmm_mkl_", torch::kAutograd,
          VariableType::_baddbmm_mkl_
        )
  .impl("aten::_cast_Double", torch::kAutograd,
          VariableType::_cast_Double
       )
  .impl("aten::_cast_Short", torch::kAutograd,
          VariableType::_cast_Short
       )
  .impl_UNBOXED("aten::_cudnn_rnn", torch::kAutograd,
          VariableType::_cudnn_rnn
        )
  .impl_UNBOXED("aten::_cumsum.out", torch::kAutograd,
          VariableType::_cumsum_out_out
        )
  .impl("aten::_dimV", torch::kAutograd,
          VariableType::_dimV
       )
  .impl_UNBOXED("aten::_embedding_bag", torch::kAutograd,
          VariableType::_embedding_bag
        )
  .impl_UNBOXED("aten::_embedding_bag_sparse_backward", torch::kAutograd,
          VariableType::_embedding_bag_sparse_backward
        )
  .impl("aten::_gather_sparse_backward", torch::kAutograd,
          VariableType::_gather_sparse_backward
       )
  .impl_UNBOXED("aten::_index_put_impl_", torch::kAutograd,
          VariableType::_index_put_impl_
        )
  .impl_UNBOXED("aten::_indices", torch::kAutograd,
          VariableType::_indices
        )
  .impl("aten::_local_scalar_dense", torch::kAutograd,
          VariableType::_local_scalar_dense
       )
  .impl_UNBOXED("aten::_max.max", torch::kAutograd,
          VariableType::_max_out_max
        )
  .impl("aten::_mkldnn_transpose", torch::kAutograd,
          VariableType::_mkldnn_transpose
       )
  .impl_UNBOXED("aten::_mkldnn_transpose_", torch::kAutograd,
          VariableType::_mkldnn_transpose_
        )
  .impl_UNBOXED("aten::_mode.values", torch::kAutograd,
          VariableType::_mode_out_values
        )
  .impl_UNBOXED("aten::_nnpack_spatial_convolution_backward_weight", torch::kAutograd,
          VariableType::_nnpack_spatial_convolution_backward_weight
        )
  .impl("aten::_pdist_forward", torch::kAutograd,
          VariableType::_pdist_forward
       )
  .impl("aten::_softmax", torch::kAutograd,
          VariableType::_softmax
       )
  .impl_UNBOXED("aten::_sparse_coo_tensor_with_dims", torch::kAutograd,
          VariableType::_sparse_coo_tensor_with_dims
        )
  .impl("aten::_sparse_sum", torch::kAutograd,
          VariableType::_sparse_sum
       )
  .impl_UNBOXED("aten::_sparse_sum.dtype", torch::kAutograd,
          VariableType::_sparse_sum_dtype
        )
  .impl_UNBOXED("aten::_sparse_sum.dim", torch::kAutograd,
          VariableType::_sparse_sum_dim
        )
  .impl_UNBOXED("aten::_sparse_sum.dim_dtype", torch::kAutograd,
          VariableType::_sparse_sum_dim_dtype
        )
  .impl_UNBOXED("aten::_thnn_differentiable_lstm_cell_backward", torch::kAutograd,
          VariableType::_thnn_differentiable_lstm_cell_backward
        )
  .impl_UNBOXED("aten::_thnn_fused_gru_cell", torch::kAutograd,
          VariableType::_thnn_fused_gru_cell
        )
  .impl_UNBOXED("aten::_thnn_fused_lstm_cell_backward", torch::kAutograd,
          VariableType::_thnn_fused_lstm_cell_backward
        )
  .impl_UNBOXED("aten::_trilinear", torch::kAutograd,
          VariableType::_trilinear
        )
  .impl_UNBOXED("aten::_unsafe_view", torch::kAutograd,
          VariableType::_unsafe_view
        )
  .impl_UNBOXED("aten::_weight_norm_cuda_interface", torch::kAutograd,
          VariableType::_weight_norm_cuda_interface
        )
  .impl("aten::abs", torch::kAutograd,
          VariableType::abs
       )
  .impl_UNBOXED("aten::abs_", torch::kAutograd,
          VariableType::abs_
        )
  .impl_UNBOXED("aten::adaptive_avg_pool1d", torch::kAutograd,
          VariableType::adaptive_avg_pool1d
        )
  .impl_UNBOXED("aten::adaptive_avg_pool2d.out", torch::kAutograd,
          VariableType::adaptive_avg_pool2d_out_out
        )
  .impl("aten::adaptive_max_pool2d_backward", torch::kAutograd,
          VariableType::adaptive_max_pool2d_backward
       )
  .impl_UNBOXED("aten::adaptive_max_pool3d", torch::kAutograd,
          VariableType::adaptive_max_pool3d
        )
  .impl_UNBOXED("aten::adaptive_max_pool3d_backward.grad_input", torch::kAutograd,
          VariableType::adaptive_max_pool3d_backward_out_grad_input
        )
  .impl("aten::addcmul", torch::kAutograd,
          VariableType::addcmul
       )
  .impl_UNBOXED("aten::addcmul_", torch::kAutograd,
          VariableType::addcmul_
        )
  .impl("aten::addmm", torch::kAutograd,
          VariableType::addmm
       )
  .impl_UNBOXED("aten::addmm_", torch::kAutograd,
          VariableType::addmm_
        )
  .impl_UNBOXED("aten::addr.out", torch::kAutograd,
          VariableType::addr_out_out
        )
  .impl_UNBOXED("aten::align_as", torch::kAutograd,
          VariableType::align_as
        )
  .impl_UNBOXED("aten::align_tensors", torch::kAutograd,
          VariableType::align_tensors
        )
  .impl_UNBOXED("aten::align_to", torch::kAutograd,
          VariableType::align_to
        )
  .impl_UNBOXED("aten::align_to.ellipsis_idx", torch::kAutograd,
          VariableType::align_to_ellipsis_idx
        )
  .impl_UNBOXED("aten::argmax", torch::kAutograd,
          VariableType::argmax
        )
  .impl("aten::argsort", torch::kAutograd,
          VariableType::argsort
       )
  .impl_UNBOXED("aten::argsort.dimname", torch::kAutograd,
          VariableType::argsort_dimname
        )
  .impl("aten::atan", torch::kAutograd,
          VariableType::atan
       )
  .impl("aten::atan2", torch::kAutograd,
          VariableType::atan2
       )
  .impl_UNBOXED("aten::atan2_", torch::kAutograd,
          VariableType::atan2_
        )
  .impl_UNBOXED("aten::atan_", torch::kAutograd,
          VariableType::atan_
        )
  .impl_UNBOXED("aten::avg_pool2d", torch::kAutograd,
          VariableType::avg_pool2d
        )
  .impl_UNBOXED("aten::avg_pool2d_backward.grad_input", torch::kAutograd,
          VariableType::avg_pool2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::avg_pool3d.out", torch::kAutograd,
          VariableType::avg_pool3d_out_out
        )
  .impl_UNBOXED("aten::baddbmm.out", torch::kAutograd,
          VariableType::baddbmm_out_out
        )
  .impl_UNBOXED("aten::bartlett_window", torch::kAutograd,
          VariableType::bartlett_window
        )
  .impl_UNBOXED("aten::bartlett_window.periodic", torch::kAutograd,
          VariableType::bartlett_window_periodic
        )
  .impl_UNBOXED("aten::batch_norm_update_stats", torch::kAutograd,
          VariableType::batch_norm_update_stats
        )
  .impl_UNBOXED("aten::binary_cross_entropy_backward", torch::kAutograd,
          VariableType::binary_cross_entropy_backward
        )
  .impl_UNBOXED("aten::bitwise_and.Tensor_out", torch::kAutograd,
          VariableType::bitwise_and_out_Tensor_out
        )
  .impl_UNBOXED("aten::bitwise_and.Scalar_out", torch::kAutograd,
          VariableType::bitwise_and_out_Scalar_out
        )
  .impl_UNBOXED("aten::bitwise_not.out", torch::kAutograd,
          VariableType::bitwise_not_out_out
        )
  .impl_UNBOXED("aten::bmm.out", torch::kAutograd,
          VariableType::bmm_out_out
        )
  .impl_UNBOXED("aten::cdist", torch::kAutograd,
          VariableType::cdist
        )
  .impl("aten::celu", torch::kAutograd,
          VariableType::celu
       )
  .impl_UNBOXED("aten::celu_", torch::kAutograd,
          VariableType::celu_
        )
  .impl("aten::cholesky_inverse", torch::kAutograd,
          VariableType::cholesky_inverse
       )
  .impl_UNBOXED("aten::cholesky.out", torch::kAutograd,
          VariableType::cholesky_out_out
        )
  .impl("aten::clamp", torch::kAutograd,
          VariableType::clamp
       )
  .impl_UNBOXED("aten::clamp_", torch::kAutograd,
          VariableType::clamp_
        )
  .impl_UNBOXED("aten::clamp_max.out", torch::kAutograd,
          VariableType::clamp_max_out_out
        )
  .impl_UNBOXED("aten::conj.out", torch::kAutograd,
          VariableType::conj_out_out
        )
  .impl_UNBOXED("aten::conv_transpose3d.input", torch::kAutograd,
          VariableType::conv_transpose3d_input
        )
  .impl_UNBOXED("aten::convolution_backward_overrideable", torch::kAutograd,
          VariableType::convolution_backward_overrideable
        )
  .impl("aten::cos", torch::kAutograd,
          VariableType::cos
       )
  .impl_UNBOXED("aten::cos_", torch::kAutograd,
          VariableType::cos_
        )
  .impl_UNBOXED("aten::cudnn_batch_norm_backward", torch::kAutograd,
          VariableType::cudnn_batch_norm_backward
        )
  .impl_UNBOXED("aten::cudnn_grid_sampler_backward", torch::kAutograd,
          VariableType::cudnn_grid_sampler_backward
        )
  .impl_UNBOXED("aten::cummax.out", torch::kAutograd,
          VariableType::cummax_out_out
        )
  .impl_UNBOXED("aten::cummax.dimname_out", torch::kAutograd,
          VariableType::cummax_out_dimname_out
        )
  .impl_UNBOXED("aten::cumsum.out", torch::kAutograd,
          VariableType::cumsum_out_out
        )
  .impl_UNBOXED("aten::cumsum.dimname_out", torch::kAutograd,
          VariableType::cumsum_out_dimname_out
        )
  .impl("aten::diag", torch::kAutograd,
          VariableType::diag
       )
  .impl("aten::digamma", torch::kAutograd,
          VariableType::digamma
       )
  .impl_UNBOXED("aten::digamma_", torch::kAutograd,
          VariableType::digamma_
        )
  .impl_UNBOXED("aten::elu.out", torch::kAutograd,
          VariableType::elu_out_out
        )
  .impl_UNBOXED("aten::embedding_bag", torch::kAutograd,
          VariableType::embedding_bag
        )
  .impl("aten::embedding_dense_backward", torch::kAutograd,
          VariableType::embedding_dense_backward
       )
  .impl_UNBOXED("aten::empty_like", torch::kAutograd,
          VariableType::empty_like
        )
  .impl_UNBOXED("aten::empty_strided", torch::kAutograd,
          VariableType::empty_strided
        )
  .impl_UNBOXED("aten::erfc.out", torch::kAutograd,
          VariableType::erfc_out_out
        )
  .impl("aten::erfinv", torch::kAutograd,
          VariableType::erfinv
       )
  .impl_UNBOXED("aten::erfinv_", torch::kAutograd,
          VariableType::erfinv_
        )
  .impl_UNBOXED("aten::expand", torch::kAutograd,
          VariableType::expand
        )
  .impl_UNBOXED("aten::expm1.out", torch::kAutograd,
          VariableType::expm1_out_out
        )
  .impl("aten::fake_quantize_per_tensor_affine_backward", torch::kAutograd,
          VariableType::fake_quantize_per_tensor_affine_backward
       )
  .impl("aten::fft", torch::kAutograd,
          VariableType::fft
       )
  .impl("aten::flatten.using_ints", torch::kAutograd,
          VariableType::flatten_using_ints
       )
  .impl_UNBOXED("aten::flatten.named_out_dim", torch::kAutograd,
          VariableType::flatten_named_out_dim
        )
  .impl_UNBOXED("aten::flatten.using_names", torch::kAutograd,
          VariableType::flatten_using_names
        )
  .impl_UNBOXED("aten::flatten.DimnameList", torch::kAutograd,
          VariableType::flatten_DimnameList
        )
  .impl_UNBOXED("aten::floor_divide", torch::kAutograd,
          VariableType::floor_divide
        )
  .impl_UNBOXED("aten::floor_divide.Scalar", torch::kAutograd,
          VariableType::floor_divide_Scalar
        )
  .impl_UNBOXED("aten::floor_divide_.Tensor", torch::kAutograd,
          VariableType::floor_divide__Tensor
        )
  .impl_UNBOXED("aten::floor_divide_.Scalar", torch::kAutograd,
          VariableType::floor_divide__Scalar
        )
  .impl_UNBOXED("aten::floor.out", torch::kAutograd,
          VariableType::floor_out_out
        )
  .impl_UNBOXED("aten::full.names", torch::kAutograd,
          VariableType::full_names
        )
  .impl_UNBOXED("aten::full", torch::kAutograd,
          VariableType::full
        )
  .impl("aten::gather", torch::kAutograd,
          VariableType::gather
       )
  .impl_UNBOXED("aten::gather.dimname", torch::kAutograd,
          VariableType::gather_dimname
        )
  .impl("aten::gelu_backward", torch::kAutograd,
          VariableType::gelu_backward
       )
  .impl_UNBOXED("aten::grid_sampler_3d_backward", torch::kAutograd,
          VariableType::grid_sampler_3d_backward
        )
  .impl_UNBOXED("aten::gru_cell", torch::kAutograd,
          VariableType::gru_cell
        )
  .impl_UNBOXED("aten::hann_window", torch::kAutograd,
          VariableType::hann_window
        )
  .impl_UNBOXED("aten::hann_window.periodic", torch::kAutograd,
          VariableType::hann_window_periodic
        )
  .impl("aten::hardshrink", torch::kAutograd,
          VariableType::hardshrink
       )
  .impl("aten::ifft", torch::kAutograd,
          VariableType::ifft
       )
  .impl_UNBOXED("aten::index_select.out", torch::kAutograd,
          VariableType::index_select_out_out
        )
  .impl_UNBOXED("aten::index_select.dimname_out", torch::kAutograd,
          VariableType::index_select_out_dimname_out
        )
  .impl_UNBOXED("aten::indices", torch::kAutograd,
          VariableType::indices
        )
  .impl("aten::is_complex", torch::kAutograd,
          VariableType::is_complex
       )
  .impl("aten::is_same_size", torch::kAutograd,
          VariableType::is_same_size
       )
  .impl_UNBOXED("aten::l1_loss.out", torch::kAutograd,
          VariableType::l1_loss_out_out
        )
  .impl_UNBOXED("aten::layer_norm", torch::kAutograd,
          VariableType::layer_norm
        )
  .impl("aten::leaky_relu_backward", torch::kAutograd,
          VariableType::leaky_relu_backward
       )
  .impl("aten::lerp.Scalar", torch::kAutograd,
          VariableType::lerp_Scalar
       )
  .impl("aten::lerp.Tensor", torch::kAutograd,
          VariableType::lerp_Tensor
       )
  .impl_UNBOXED("aten::lerp_.Scalar", torch::kAutograd,
          VariableType::lerp__Scalar
        )
  .impl_UNBOXED("aten::lerp_.Tensor", torch::kAutograd,
          VariableType::lerp__Tensor
        )
  .impl_UNBOXED("aten::linear", torch::kAutograd,
          VariableType::linear
        )
  .impl("aten::log_sigmoid", torch::kAutograd,
          VariableType::log_sigmoid
       )
  .impl_UNBOXED("aten::log_sigmoid_backward.grad_input", torch::kAutograd,
          VariableType::log_sigmoid_backward_out_grad_input
        )
  .impl_UNBOXED("aten::logical_or.out", torch::kAutograd,
          VariableType::logical_or_out_out
        )
  .impl_UNBOXED("aten::logical_xor.out", torch::kAutograd,
          VariableType::logical_xor_out_out
        )
  .impl_UNBOXED("aten::logspace.out", torch::kAutograd,
          VariableType::logspace_out_out
        )
  .impl_UNBOXED("aten::logsumexp.out", torch::kAutograd,
          VariableType::logsumexp_out_out
        )
  .impl_UNBOXED("aten::logsumexp.names_out", torch::kAutograd,
          VariableType::logsumexp_out_names_out
        )
  .impl_UNBOXED("aten::matmul.out", torch::kAutograd,
          VariableType::matmul_out_out
        )
  .impl_UNBOXED("aten::max.dim_max", torch::kAutograd,
          VariableType::max_out_dim_max
        )
  .impl_UNBOXED("aten::max.names_dim_max", torch::kAutograd,
          VariableType::max_out_names_dim_max
        )
  .impl_UNBOXED("aten::max.out", torch::kAutograd,
          VariableType::max_out_out
        )
  .impl_UNBOXED("aten::max_unpool2d", torch::kAutograd,
          VariableType::max_unpool2d
        )
  .impl_UNBOXED("aten::max_unpool2d_backward.grad_input", torch::kAutograd,
          VariableType::max_unpool2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::max_unpool3d.out", torch::kAutograd,
          VariableType::max_unpool3d_out_out
        )
  .impl_UNBOXED("aten::min_values", torch::kAutograd,
          VariableType::min_values
        )
  .impl_UNBOXED("aten::min_values.names", torch::kAutograd,
          VariableType::min_values_names
        )
  .impl_UNBOXED("aten::miopen_convolution_backward", torch::kAutograd,
          VariableType::miopen_convolution_backward
        )
  .impl("aten::miopen_convolution_backward_bias", torch::kAutograd,
          VariableType::miopen_convolution_backward_bias
       )
  .impl_UNBOXED("aten::miopen_convolution_backward_input", torch::kAutograd,
          VariableType::miopen_convolution_backward_input
        )
  .impl_UNBOXED("aten::miopen_convolution_transpose", torch::kAutograd,
          VariableType::miopen_convolution_transpose
        )
  .impl_UNBOXED("aten::mkldnn_adaptive_avg_pool2d", torch::kAutograd,
          VariableType::mkldnn_adaptive_avg_pool2d
        )
  .impl_UNBOXED("aten::mkldnn_convolution", torch::kAutograd,
          VariableType::mkldnn_convolution
        )
  .impl_UNBOXED("aten::mkldnn_reorder_conv2d_weight", torch::kAutograd,
          VariableType::mkldnn_reorder_conv2d_weight
        )
  .impl_UNBOXED("aten::mode.values", torch::kAutograd,
          VariableType::mode_out_values
        )
  .impl_UNBOXED("aten::mode.dimname_out", torch::kAutograd,
          VariableType::mode_out_dimname_out
        )
  .impl("aten::mse_loss_backward", torch::kAutograd,
          VariableType::mse_loss_backward
       )
  .impl("aten::multilabel_margin_loss_backward", torch::kAutograd,
          VariableType::multilabel_margin_loss_backward
       )
  .impl_UNBOXED("aten::multilabel_margin_loss_forward.output", torch::kAutograd,
          VariableType::multilabel_margin_loss_forward_out_output
        )
  .impl_UNBOXED("aten::multinomial", torch::kAutograd,
          VariableType::multinomial
        )
  .impl("aten::mvlgamma", torch::kAutograd,
          VariableType::mvlgamma
       )
  .impl_UNBOXED("aten::mvlgamma_", torch::kAutograd,
          VariableType::mvlgamma_
        )
  .impl_UNBOXED("aten::narrow", torch::kAutograd,
          VariableType::narrow
        )
  .impl_UNBOXED("aten::narrow.Tensor", torch::kAutograd,
          VariableType::narrow_Tensor
        )
  .impl_UNBOXED("aten::native_batch_norm", torch::kAutograd,
          VariableType::native_batch_norm
        )
  .impl_UNBOXED("aten::ne.Scalar_out", torch::kAutograd,
          VariableType::ne_out_Scalar_out
        )
  .impl_UNBOXED("aten::ne.Tensor_out", torch::kAutograd,
          VariableType::ne_out_Tensor_out
        )
  .impl_UNBOXED("aten::new_full", torch::kAutograd,
          VariableType::new_full
        )
  .impl_UNBOXED("aten::nll_loss", torch::kAutograd,
          VariableType::nll_loss
        )
  .impl_UNBOXED("aten::nll_loss2d", torch::kAutograd,
          VariableType::nll_loss2d
        )
  .impl_UNBOXED("aten::nll_loss2d_backward.grad_input", torch::kAutograd,
          VariableType::nll_loss2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::nll_loss_backward.grad_input", torch::kAutograd,
          VariableType::nll_loss_backward_out_grad_input
        )
  .impl("aten::nuclear_norm", torch::kAutograd,
          VariableType::nuclear_norm
       )
  .impl_UNBOXED("aten::nuclear_norm.dim", torch::kAutograd,
          VariableType::nuclear_norm_dim
        )
  .impl("aten::orgqr", torch::kAutograd,
          VariableType::orgqr
       )
  .impl_UNBOXED("aten::ormqr.out", torch::kAutograd,
          VariableType::ormqr_out_out
        )
  .impl_UNBOXED("aten::permute", torch::kAutograd,
          VariableType::permute
        )
  .impl("aten::pixel_shuffle", torch::kAutograd,
          VariableType::pixel_shuffle
       )
  .impl_UNBOXED("aten::put_", torch::kAutograd,
          VariableType::put_
        )
  .impl("aten::q_zero_point", torch::kAutograd,
          VariableType::q_zero_point
       )
  .impl_UNBOXED("aten::quantize_per_tensor", torch::kAutograd,
          VariableType::quantize_per_tensor
        )
  .impl_UNBOXED("aten::quantize_per_tensor.tensors", torch::kAutograd,
          VariableType::quantize_per_tensor_tensors
        )
  .impl_UNBOXED("aten::quantized_gru.input", torch::kAutograd,
          VariableType::quantized_gru_input
        )
  .impl_UNBOXED("aten::quantized_gru.data", torch::kAutograd,
          VariableType::quantized_gru_data
        )
  .impl_UNBOXED("aten::quantized_lstm_cell", torch::kAutograd,
          VariableType::quantized_lstm_cell
        )
  .impl_UNBOXED("aten::rand.out", torch::kAutograd,
          VariableType::rand_out_out
        )
  .impl_UNBOXED("aten::rand.generator_out", torch::kAutograd,
          VariableType::rand_out_generator_out
        )
  .impl_UNBOXED("aten::randn.out", torch::kAutograd,
          VariableType::randn_out_out
        )
  .impl_UNBOXED("aten::randn.generator_out", torch::kAutograd,
          VariableType::randn_out_generator_out
        )
  .impl_UNBOXED("aten::range.step", torch::kAutograd,
          VariableType::range_step
        )
  .impl_UNBOXED("aten::range", torch::kAutograd,
          VariableType::range
        )
  .impl("aten::real", torch::kAutograd,
          VariableType::real
       )
  .impl("aten::reciprocal", torch::kAutograd,
          VariableType::reciprocal
       )
  .impl_UNBOXED("aten::reciprocal_", torch::kAutograd,
          VariableType::reciprocal_
        )
  .impl_UNBOXED("aten::refine_names", torch::kAutograd,
          VariableType::refine_names
        )
  .impl_UNBOXED("aten::reflection_pad1d", torch::kAutograd,
          VariableType::reflection_pad1d
        )
  .impl_UNBOXED("aten::reflection_pad1d_backward.grad_input", torch::kAutograd,
          VariableType::reflection_pad1d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::reflection_pad2d.out", torch::kAutograd,
          VariableType::reflection_pad2d_out_out
        )
  .impl("aten::relu", torch::kAutograd,
          VariableType::relu
       )
  .impl_UNBOXED("aten::relu_", torch::kAutograd,
          VariableType::relu_
        )
  .impl_UNBOXED("aten::remainder.Scalar_out", torch::kAutograd,
          VariableType::remainder_out_Scalar_out
        )
  .impl_UNBOXED("aten::remainder.Tensor_out", torch::kAutograd,
          VariableType::remainder_out_Tensor_out
        )
  .impl_UNBOXED("aten::replication_pad1d.out", torch::kAutograd,
          VariableType::replication_pad1d_out_out
        )
  .impl_UNBOXED("aten::rnn_relu_cell", torch::kAutograd,
          VariableType::rnn_relu_cell
        )
  .impl_UNBOXED("aten::rrelu_with_noise", torch::kAutograd,
          VariableType::rrelu_with_noise
        )
  .impl_UNBOXED("aten::rrelu_with_noise_", torch::kAutograd,
          VariableType::rrelu_with_noise_
        )
  .impl("aten::scatter_add", torch::kAutograd,
          VariableType::scatter_add
       )
  .impl_UNBOXED("aten::scatter_add.dimname", torch::kAutograd,
          VariableType::scatter_add_dimname
        )
  .impl_UNBOXED("aten::scatter_add_", torch::kAutograd,
          VariableType::scatter_add_
        )
  .impl_UNBOXED("aten::select.Dimname", torch::kAutograd,
          VariableType::select_Dimname
        )
  .impl_UNBOXED("aten::select.int", torch::kAutograd,
          VariableType::select_int
        )
  .impl("aten::sin", torch::kAutograd,
          VariableType::sin
       )
  .impl_UNBOXED("aten::sin_", torch::kAutograd,
          VariableType::sin_
        )
  .impl_UNBOXED("aten::slow_conv_dilated3d_backward", torch::kAutograd,
          VariableType::slow_conv_dilated3d_backward
        )
  .impl_UNBOXED("aten::soft_margin_loss.out", torch::kAutograd,
          VariableType::soft_margin_loss_out_out
        )
  .impl_UNBOXED("aten::softmax.int", torch::kAutograd,
          VariableType::softmax_int
        )
  .impl_UNBOXED("aten::softmax.Dimname", torch::kAutograd,
          VariableType::softmax_Dimname
        )
  .impl_UNBOXED("aten::softplus.out", torch::kAutograd,
          VariableType::softplus_out_out
        )
  .impl("aten::softshrink_backward", torch::kAutograd,
          VariableType::softshrink_backward
       )
  .impl_UNBOXED("aten::sort.values", torch::kAutograd,
          VariableType::sort_out_values
        )
  .impl_UNBOXED("aten::sort.dimname_values", torch::kAutograd,
          VariableType::sort_out_dimname_values
        )
  .impl_UNBOXED("aten::squeeze", torch::kAutograd,
          VariableType::squeeze
        )
  .impl_UNBOXED("aten::squeeze.dim", torch::kAutograd,
          VariableType::squeeze_dim
        )
  .impl_UNBOXED("aten::squeeze.dimname", torch::kAutograd,
          VariableType::squeeze_dimname
        )
  .impl_UNBOXED("aten::squeeze_", torch::kAutograd,
          VariableType::squeeze_
        )
  .impl_UNBOXED("aten::squeeze_.dim", torch::kAutograd,
          VariableType::squeeze__dim
        )
  .impl_UNBOXED("aten::squeeze_.dimname", torch::kAutograd,
          VariableType::squeeze__dimname
        )
  .impl_UNBOXED("aten::std.out", torch::kAutograd,
          VariableType::std_out_out
        )
  .impl_UNBOXED("aten::std.names_out", torch::kAutograd,
          VariableType::std_out_names_out
        )
  .impl("aten::sub.Tensor", torch::kAutograd,
          VariableType::sub_Tensor
       )
  .impl("aten::sub.Scalar", torch::kAutograd,
          VariableType::sub_Scalar
       )
  .impl_UNBOXED("aten::sub_.Tensor", torch::kAutograd,
          VariableType::sub__Tensor
        )
  .impl_UNBOXED("aten::sub_.Scalar", torch::kAutograd,
          VariableType::sub__Scalar
        )
  .impl_UNBOXED("aten::sum.IntList_out", torch::kAutograd,
          VariableType::sum_out_IntList_out
        )
  .impl_UNBOXED("aten::sum.DimnameList_out", torch::kAutograd,
          VariableType::sum_out_DimnameList_out
        )
  .impl_UNBOXED("aten::take.out", torch::kAutograd,
          VariableType::take_out_out
        )
  .impl_UNBOXED("aten::thnn_conv2d_forward", torch::kAutograd,
          VariableType::thnn_conv2d_forward
        )
  .impl_UNBOXED("aten::thnn_conv_depthwise2d.out", torch::kAutograd,
          VariableType::thnn_conv_depthwise2d_out_out
        )
  .impl("aten::to_dense_backward", torch::kAutograd,
          VariableType::to_dense_backward
       )
  .impl_UNBOXED("aten::topk.values", torch::kAutograd,
          VariableType::topk_out_values
        )
  .impl_UNBOXED("aten::trunc.out", torch::kAutograd,
          VariableType::trunc_out_out
        )
  .impl_UNBOXED("aten::unbind.int", torch::kAutograd,
          VariableType::unbind_int
        )
  .impl_UNBOXED("aten::unbind.Dimname", torch::kAutograd,
          VariableType::unbind_Dimname
        )
  .impl_UNBOXED("aten::unique_consecutive", torch::kAutograd,
          VariableType::unique_consecutive
        )
  .impl_UNBOXED("aten::upsample_bilinear2d_backward", torch::kAutograd,
          VariableType::upsample_bilinear2d_backward
        )
  .impl_UNBOXED("aten::upsample_linear1d", torch::kAutograd,
          VariableType::upsample_linear1d
        )
  .impl_UNBOXED("aten::upsample_linear1d_backward.grad_input", torch::kAutograd,
          VariableType::upsample_linear1d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::upsample_nearest1d_backward", torch::kAutograd,
          VariableType::upsample_nearest1d_backward
        )
  .impl_UNBOXED("aten::upsample_nearest2d", torch::kAutograd,
          VariableType::upsample_nearest2d
        )
  .impl_UNBOXED("aten::upsample_nearest2d_backward.grad_input", torch::kAutograd,
          VariableType::upsample_nearest2d_backward_out_grad_input
        )
  .impl_UNBOXED("aten::upsample_nearest3d.out", torch::kAutograd,
          VariableType::upsample_nearest3d_out_out
        )
  .impl("aten::view_as", torch::kAutograd,
          VariableType::view_as
       );

}

}} // namespace torch::autograd
