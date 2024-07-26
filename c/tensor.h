#ifndef TENOSR_H
#define TENOSR_H

#include "buffer.h"
#include "ops.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Operation types
typedef enum {
    OP_MUL,
    OP_RELU,
    OP_DOT,
    OP_SUM,
    OP_LOGSOFTMAX,
    OP_NLL,
} OpType;

// Forward declarations
typedef struct Context Context;
typedef struct Tensor Tensor;

struct Tensor {
  Buffer *buf;
  Buffer *grad;
  Context *ctx;
};

struct Context {
    OpType op;

    Tensor** inputs;
    int num_inputs;

    Buffer* saved_buffer;
};

// Tensor operations
Tensor* tensor_create(Buffer* buf);
void tensor_destroy(Tensor* t);
void backward(Tensor* t, bool implicit);

// Context operations
Tensor* context_forward(OpType op, Tensor** inputs, int num_inputs);

// Operation application
Tensor* apply_op(OpType op, Tensor** inputs, int num_inputs);

#endif
