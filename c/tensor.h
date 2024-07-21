#ifndef TENOSR_H
#define TENOSR_H


#include "buffer.h"

// Operation types
typedef enum {
    OP_MUL,
    OP_RELU,
    OP_MVDOT,
    OP_VMDOT,
    OP_SUM,
    OP_LOGSOFTMAX,
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
  Tensor **parents;
  int num_parents;
  Buffer **saved_tensors;
  int num_saved_tensors;
};

// Tensor operations
Tensor* tensor_create(Buffer* buf);
void tensor_destroy(Tensor* t);
void tensor_backward(Tensor* t, int implicit);

// Graph operations
void graph_destroy(Tensor *ret);

// Context operations
Buffer* context_forward(Context* self, Tensor** inputs, int num_inputs);
Buffer** context_backward(Context* self, Tensor* grad_output);

// Operation application
Tensor* apply_op(OpType op, Tensor** inputs, int num_inputs);

#endif
