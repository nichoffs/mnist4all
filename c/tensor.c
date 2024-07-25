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

Buffer *context_forward(OpType op, Buffer **inputs, int num_inputs) {
  switch (op) {
  case OP_MUL: {
    return mul(inputs[0], inputs[1]);
  }
  case OP_RELU: {
    return relu(inputs[0]);
  }
  case OP_DOT: {
    return dot(inputs[0], inputs[1]);
  }
  case OP_SUM: {
    return sum(inputs[0]);
  }
  case OP_LOGSOFTMAX: {
    return log_softmax(inputs[0]);
  }
  case OP_NLL: {
    return nll(inputs[0], inputs[1]);
  }
  default: {
    fprintf(stderr, "Invalid op type\n");
    return NULL;
  }
  }
}

Tensor *apply_op(OpType op, Tensor **inputs, int num_inputs) {

  Buffer **input_buffers = (Buffer **)malloc(num_inputs * sizeof(Buffer *));
  for (int i = 0; i < num_inputs; i++) {
    input_buffers[i] = inputs[i]->buf;
  }

  Buffer *result = context_forward(op, input_buffers, num_inputs);
  free(input_buffers);

  if (!result) {
    fprintf(stderr, "Error: context_forward returned NULL\n");
    return NULL;
  }

  Tensor *ret = tensor_create(result);
  if (!ret) {
    buffer_destroy(result);
    return NULL;
  }

  return ret;
}
