#include "tensor.h"
#include "buffer.h"
#include "ops.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create(Buffer *buf) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  t->buf = buf;
  t->grad = NULL;
  t->ctx = NULL;
  return t;
}

Buffer *context_forward(Context *self, Tensor **inputs, int num_inputs) {
  switch (self->op) {
  case OP_MUL: {
    self->saved_buffers = (Buffer **)malloc(2 * sizeof(Buffer *));
    self->saved_buffers[0] = buffer_copy(inputs[0]->buf);
    self->saved_buffers[1] = buffer_copy(inputs[1]->buf);
    self->num_saved_buffers = 2;
    return mul(inputs[0]->buf, inputs[1]->buf);
  }
  case OP_RELU: {
    self->saved_buffers = (Buffer **)malloc(sizeof(Buffer *));
    self->saved_buffers[0] = buffer_copy(inputs[0]->buf);
    self->num_saved_buffers = 1;
    return relu(inputs[0]->buf);
  }
  case OP_DOT: {
    self->saved_buffers = (Buffer **)malloc(2 * sizeof(Buffer *));
    self->saved_buffers[0] = buffer_copy(inputs[0]->buf);
    self->saved_buffers[1] = buffer_copy(inputs[1]->buf);
    self->num_saved_buffers = 2;
    return dot(inputs[0]->buf, inputs[1]->buf);
    break;
  }
  case OP_SUM: {
    self->saved_buffers = (Buffer **)malloc(sizeof(Buffer *));
    self->saved_buffers[0] = buffer_copy(inputs[0]->buf);
    self->num_saved_buffers = 1;
    return sum(inputs[0]->buf);
  }
  case OP_LOGSOFTMAX: {
    self->saved_buffers = (Buffer **)malloc(sizeof(Buffer *));
    Buffer *result = log_softmax(inputs[0]->buf);
    self->saved_buffers[0] = buffer_copy(result);
    self->num_saved_buffers = 1;
    return result;
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }
}

Buffer **context_backward(Context *self, Buffer *grad_output) {
  Buffer **grads = (Buffer **)malloc(self->num_parents * sizeof(Buffer *));

  for (int i = 0; i < self->num_parents; i++) {
    grads[i] = NULL;
  }

  if (!grads) {
    fprintf(stderr, "Error: malloc failed for grads\n");
    return NULL;
  }
  switch (self->op) {
  case OP_MUL: {
    grads[0] = mul(grad_output, self->saved_buffers[1]);
    grads[1] = mul(grad_output, self->saved_buffers[0]);
    break;
  }
  case OP_RELU: {
    grads[0] = buffer_copy(grad_output);
    for (int i = 0; i < grads[0]->st->numel; i++) {
      int ix = view_index(grads[0]->st, i);
      if (self->saved_buffers[0]->data[ix] < 0) {
        grads[0]->data[ix] = 0.0f;
      }
    }
    break;
  }
  case OP_DOT: {
    grads[0] = dot_backward(grad_output, self->saved_buffers[0],
                            self->saved_buffers[1], 0);
    grads[1] = dot_backward(grad_output, self->saved_buffers[0],
                            self->saved_buffers[1], 1);
    break;
  }
  case OP_SUM: {
    grads[0] = full_like(self->saved_buffers[0], grad_output->data[0]);
    break;
  }
  case OP_LOGSOFTMAX: {
    grads[0] = log_softmax_backward(grad_output, self->saved_buffers[0]);
    break;
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }

  for (int i = 0; i < self->num_parents; i++) {
    if (grads[i] == NULL) {
      fprintf(stderr, "Error: grad[%d] is NULL for op %d\n", i, self->op);
      // Free all previously allocated grads
      for (int j = 0; j < i; j++) {
        if (grads[j] != NULL) {
          buffer_destroy(grads[j]);
        }
      }
      free(grads);
      return NULL;
    }
  }

  return grads;
}

// When an op is run, it creates a new result Tensor that holds a context
// The context contains the ops that created the tensor (no copy), information
// about the operation to be used in backward pass, and any tensors that need to
// be saved for backwards (copy)

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {
  Context *f = (Context *)malloc(sizeof(Context));
  if (!f) {
    fprintf(stderr, "Error: malloc failed for Context\n");
    return NULL;
  }
  f->op = op;
  f->parents = inputs; // Directly set parents to the input tensor pointers
  f->num_parents = num_inputs;
  f->saved_buffers = NULL;
  f->num_saved_buffers = 0;

  Buffer *result = context_forward(f, inputs, num_inputs);

  if (!result) {
    fprintf(stderr, "Error: context_forward returned NULL\n");
    free(f);
    return NULL;
  }

  Tensor *ret = tensor_create(result);
  if (!ret) {
    buffer_destroy(result);
    free(f);
    return NULL;
  }
  ret->ctx = f;
  return ret;
}

void graph_destroy(Tensor *tensor) {
  if (!tensor) {
    printf("final is NULL\n");
    return;
  }

  if (tensor->ctx) {
    for (int i = 0; i < tensor->ctx->num_saved_buffers; i++) {
      buffer_destroy(tensor->ctx->saved_buffers[i]);
    }
    for (int i = 0; i < tensor->ctx->num_parents; i++) {
      graph_destroy(tensor->ctx->parents[i]);
    }
    free(tensor->ctx);
    free(tensor->buf);
    if (tensor->grad) {
      free(tensor->grad);
    }
  }
  free(tensor);
}

void tensor_backward(Tensor *t, int implicit) {
  if (!t->ctx)
    return;

  if (implicit) {
    assert(t->buf->st->numel == 1 && "Can only backprop scalar");
    float one = 1.0f;
    t->grad =
        buffer_data_create(&one, 1, t->buf->st->shape, t->buf->st->ndim, true);
  }

  assert(t->grad != NULL);
  Buffer **grads = context_backward(t->ctx, t->grad);
  for (int i = 0; i < t->ctx->num_parents; i++) {
    Tensor *parent = t->ctx->parents[i];
    assert(grads[i]->st->numel == parent->buf->st->numel &&
           "grad shape != tensor shape");
    parent->grad = grads[i];
    tensor_backward(parent, 0);
  }
  free(grads);
}
