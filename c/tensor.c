#include "buffer.h"
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

typedef struct Function Function;

typedef struct {
  Buffer *buf;
  Buffer *grad;
  Function *ctx;
} Tensor;

struct Function {
  OpType op;
  Tensor **parents;
  int num_parents;
  Buffer **saved_tensors;
  int num_saved_tensors;
};

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

Buffer *function_forward(Function *self, Buffer **inputs, int num_inputs) {
  return NULL;
}

Buffer **function_backward(Function *self, Buffer *grad_output) { return NULL; }

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {
  Function *f = malloc(sizeof(Function));
  f->op = op;
  f->parents = malloc(num_inputs * sizeof(Tensor *));
  memcpy(f->parents, inputs, num_inputs * sizeof(Tensor *));
  f->num_parents = num_inputs;

  Buffer **input_buffers = malloc(num_inputs * sizeof(Buffer *));
  for (int i = 0; i < num_inputs; i++) {
    input_buffers[i] = inputs[i]->buf;
  }

  Buffer *result = function_forward(f, input_buffers, num_inputs);
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

  Buffer **grads = function_backward(t->ctx, t->grad);

  for (int i = 0; i < t->ctx->num_parents; i++) {
    Tensor *parent = t->ctx->parents[i];
    assert(grads[i]->st->numel == parent->buf->st->numel &&
           "grad shape != tensor shape");
    parent->grad = grads[i];
    tensor_backward(parent, 0);
  }

  free(grads);
}
