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
    self->saved_tensors = malloc(2 * sizeof(Buffer *));
    self->saved_tensors[0] = buffer_create(
        inputs[0]->buf->data, inputs[0]->buf->size, inputs[0]->buf->st, true);
    self->saved_tensors[1] = buffer_create(
        inputs[1]->buf->data, inputs[1]->buf->size, inputs[1]->buf->st, true);
    self->num_saved_tensors = 2;
    return mul(inputs[0]->buf, inputs[1]->buf);
  }
  case OP_RELU: {
    self->saved_tensors = malloc(sizeof(Buffer *));
    self->saved_tensors[0] = buffer_create(
        inputs[0]->buf->data, inputs[0]->buf->size, inputs[0]->buf->st, true);
    self->num_saved_tensors = 1;
    return relu(inputs[0]->buf);
  }
  case OP_MVDOT: {
    self->saved_tensors = malloc(2 * sizeof(Buffer *));
    self->saved_tensors[0] = buffer_create(
        inputs[0]->buf->data, inputs[0]->buf->size, inputs[0]->buf->st, true);
    self->saved_tensors[1] = buffer_create(
        inputs[1]->buf->data, inputs[1]->buf->size, inputs[1]->buf->st, true);
    self->num_saved_tensors = 2;
    return matrix_vector_dot(inputs[0]->buf, inputs[1]->buf);
  }
  case OP_VMDOT: {
    self->saved_tensors = malloc(2 * sizeof(Buffer *));
    self->saved_tensors[0] = buffer_create(
        inputs[0]->buf->data, inputs[0]->buf->size, inputs[0]->buf->st, true);
    self->saved_tensors[1] = buffer_create(
        inputs[1]->buf->data, inputs[1]->buf->size, inputs[1]->buf->st, true);
    self->num_saved_tensors = 2;
    return vector_matrix_dot(inputs[0]->buf, inputs[1]->buf);
  }
  case OP_SUM: {
    self->saved_tensors = malloc(sizeof(Buffer *));
    self->saved_tensors[0] = buffer_create(
        inputs[0]->buf->data, inputs[0]->buf->size, inputs[0]->buf->st, true);
    self->num_saved_tensors = 1;
    return sum(inputs[0]->buf);
  }
  case OP_LOGSOFTMAX: {
    self->saved_tensors = malloc(sizeof(Buffer *));
    Buffer *result = log_softmax(inputs[0]->buf);
    self->saved_tensors[0] =
        buffer_create(result->data, result->size, result->st, true);
    self->num_saved_tensors = 1;
    return result;
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }
}

Buffer **context_backward(Context *self, Tensor *grad_output) { return NULL; }

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {
  Context *f = (Context *)malloc(sizeof(Context));
  if (!f) {
    fprintf(stderr, "Error: malloc failed for Context\n");
    return NULL;
  }
  f->op = op;
  f->parents = inputs; // Directly set parents to the input tensor pointers
  f->num_parents = num_inputs;
  f->saved_tensors = NULL;
  f->num_saved_tensors = 0;

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

/* void tensor_backward(Tensor *t, int implicit) { */
/*   if (!t->ctx) */
/*     return; */
/**/
/*   if (implicit) { */
/*     assert(t->buf->st->numel == 1 && "Can only backprop scalar"); */
/*     float one = 1.0f; */
/*     t->grad = */
/*         buffer_data_create(&one, 1, t->buf->st->shape, t->buf->st->ndim,
 * true); */
/*   } */
/**/
/*   assert(t->grad != NULL); */
/**/
/*   Buffer **grads = context_backward(t->ctx, t->grad); */
/**/
/*   for (int i = 0; i < t->ctx->num_parents; i++) { */
/*     Tensor *parent = t->ctx->parents[i]; */
/*     assert(grads[i]->st->numel == parent->buf->st->numel && */
/*            "grad shape != tensor shape"); */
/*     parent->grad = grads[i]; */
/*     tensor_backward(parent, 0); */
/*   } */
/**/
/*   free(grads); */
/* } */
