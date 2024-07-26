#include "tensor.h"

Tensor *tensor_create(Buffer *buf) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  t->buf = buf;
  t->grad = NULL;
  t->ctx = NULL;
  return t;
}

void tensor_destroy(Tensor *t) {
  if (t == NULL) {
    return; // Nothing to destroy if the tensor is NULL
  }

  if (t->buf) {
    buffer_destroy(t->buf);
    t->buf = NULL;
  }

  if (t->grad) {
    buffer_destroy(t->grad);
    t->grad = NULL;
  }

  if (t->ctx) {
    free(t->ctx);
    t->ctx = NULL;
  }
}

bool validate_op(OpType op) {
  switch (op) {
  case OP_MUL:
    return true; // Multiplication
  case OP_RELU:
    return true; // Rectified Linear Unit
  case OP_DOT:
    return true; // Dot Product
  case OP_SUM:
    return true; // Summation
  case OP_LOGSOFTMAX:
    return true; // Log Softmax
  case OP_NLL:
    return true; // Negative Log Likelihood
  default:
    return false; // Invalid operation type
  }
}

Tensor *context_forward(OpType op, Tensor **inputs, int num_inputs) {
  switch (op) {
  case OP_MUL: {
    Tensor *result = tensor_create(mul(inputs[0]->buf, inputs[1]->buf));
    return result;
  }
  case OP_RELU: {
    Tensor *result = tensor_create(relu(inputs[0]->buf));
    return result;
  }
  case OP_DOT: {
    Tensor *result = tensor_create(dot(inputs[0]->buf, inputs[1]->buf));
    return result;
  }
  case OP_SUM: {
    Tensor *result = tensor_create(sum(inputs[0]->buf));
    return result;
  }
  case OP_LOGSOFTMAX: {
    Buffer *out = log_softmax(inputs[0]->buf);
    Tensor *result = tensor_create(out);
    return result;
  }
  case OP_NLL: {
    Tensor *result = tensor_create(nll(inputs[0]->buf, inputs[1]->buf));
    return result;
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }
}

Buffer **context_backward(Context *ctx, Buffer *grad_output) {
  Buffer **grads = (Buffer **)malloc(ctx->num_inputs * sizeof(Buffer *));

  switch (ctx->op) {
  case OP_MUL: {
    grads[0] = mul(grad_output, ctx->inputs[1]->buf);
    grads[1] = mul(grad_output, ctx->inputs[0]->buf);
    break;
  }
  case OP_RELU: {
    grads[0] = buffer_copy(grad_output);
    for (int i = 0; i < grads[0]->st->numel; i++) {
      int grad_ix = view_index(grads[0]->st, i);
      int input_ix = view_index(ctx->inputs[0]->buf->st, i);
      if (ctx->inputs[0]->buf->data[input_ix] <= 0) {
        grads[0]->data[grad_ix] = 0.0f;
      }
    }
    break;
  }
  case OP_DOT: {
    grads[0] =
        dot_backward(grad_output, ctx->inputs[0]->buf, ctx->inputs[1]->buf, 0);
    grads[1] =
        dot_backward(grad_output, ctx->inputs[0]->buf, ctx->inputs[1]->buf, 1);
    break;
  }
  case OP_SUM: {
    grads[0] = full_like(ctx->inputs[0]->buf, grad_output->data[0]);
    break;
  }
  case OP_LOGSOFTMAX: {
    grads[0] = log_softmax_backward(grad_output, ctx->saved_buffer);
    break;
  }
  case OP_NLL: {
    grads[0] = nll_backward(grad_output, ctx->inputs[1]->buf);
    grads[1] = NULL;
    break;
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }

  return grads;
}

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {

  for (int i = 0; i < num_inputs; i++) {
    if (!inputs[i]) {
      fprintf(stderr, "Input tensor is NULL\n");
      return NULL;
    }
  }

  Context *ctx = (Context *)malloc(sizeof(Context));
  if (!ctx) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  ctx->op = op;
  ctx->num_inputs = num_inputs;
  ctx->inputs = inputs;

  Tensor *result = context_forward(op, inputs, num_inputs);
  if (op == OP_LOGSOFTMAX) {
    ctx->saved_buffer = result->buf;
  }
  if (!result) {
    fprintf(stderr, "Error: context_forward returned NULL\n");
    return NULL;
  }

  result->ctx = ctx;

  return result;
}

void backward(Tensor *t, bool implicit) {
  if (!t->ctx)
    return;

  if (implicit) {
    assert(t->buf->st->numel == 1 && "Can only backprop scalar");
    float one[1] = {1.0f};
    int shape[1] = {1};
    int ndim = 1;
    int size = 1;
    t->grad = buffer_data_create(one, size, shape, ndim, true);
  }

  assert(t->grad != NULL);

  Buffer **grads = context_backward(t->ctx, t->grad);
  for (int i = 0; i < t->ctx->num_inputs; i++) {
    if (grads[i] != NULL) {
      assert(grads[i]->st->numel == t->ctx->inputs[i]->buf->st->numel &&
             "grad shape != tensor shape");
      t->ctx->inputs[i]->grad = grads[i];
      backward(t->ctx->inputs[i], 0);
    }
  }
  free(grads);

  return;
}
