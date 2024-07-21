#include "buffer.h"
#include "ops.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  OP_MUL,
  OP_RELU,
  OP_DOT,
  OP_SUM,
  OP_LOGSOFTMAX,
} OpType;

typedef struct Context context;

typedef struct {
  Buffer *buf;
  Buffer *grad;
  Context *ctx;
} Tensor;

typedef struct {
  OpType op;
  Tensor **parents;
  int num_parents;
  Buffer **saved_tensors;
  int num_saved_tensors;
} Context;

Tensor *tensor_create(Buffer *buf) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  t->buf = buf;
  t->grad = NULL;
  t->ctx = NULL;
  return t;
}

void tensor_destroy(Tensor *t) {
  if (t) {
    buffer_destroy(t->buf);
    buffer_destroy(t->grad);
    free(t);
  }
}

Buffer *context_forward(Context *self, Buffer **inputs, int num_inputs) {
  switch (self->op) {
  case OP_MUL: {
    self->saved_tensors[0] = inputs[0];
    self->saved_tensors[1] = inputs[1];
    self->num_saved_tensors = 2;
    return mul(inputs[0], inputs[1]);
  }
  case OP_RELU: {
    self->saved_tensors[0] = inputs[0];
    self->num_saved_tensors = 1;
    return relu(inputs[0]);
  }
  case OP_DOT: {
    self->saved_tensors[0] = inputs[0];
    self->saved_tensors[1] = inputs[1];
    self->num_saved_tensors = 2;
    return matrix_vector_dot(inputs[0], inputs[1]);
  }
  case OP_SUM: {
    self->saved_tensors[0] = inputs[0];
    self->num_saved_tensors = 1;
    return sum(inputs[0]);
  }
  case OP_LOGSOFTMAX: {
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }
}

Buffer **context_backward(Context *self, Buffer *grad_output) { return NULL; }

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {
  Context *f = malloc(sizeof(Context));
  f->op = op;
  f->parents = malloc(num_inputs * sizeof(Tensor *));
  memcpy(f->parents, inputs, num_inputs * sizeof(Tensor *));
  f->num_parents = num_inputs;

  Buffer **input_buffers = malloc(num_inputs * sizeof(Buffer *));
  for (int i = 0; i < num_inputs; i++) {
    input_buffers[i] = inputs[i]->buf;
  }

  Buffer *result = context_forward(f, input_buffers, num_inputs);
  free(input_buffers);

  Tensor *ret = tensor_create(result);
  ret->ctx = f;
  return ret;
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
