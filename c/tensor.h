#ifndef TENOSR_H
#define TENOSR_H


#include "buffer.h"

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

// Tensor operations
Tensor* tensor_create(Buffer* buf);
void tensor_destroy(Tensor* t);
void tensor_backward(Tensor* t, int implicit);

// Context operations
Buffer* context_forward(OpType op, Buffer** inputs, int num_inputs);

// Operation application
Tensor* apply_op(OpType op, Tensor** inputs, int num_inputs);

#endif
