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
} OpType;

// Forward declarations
typedef struct Function Function;
typedef struct Tensor Tensor;

// Tensor structure
struct Tensor {
    Buffer* buf;
    Buffer* grad;
    Function* ctx;
};

// Function structure
struct Function {
    OpType op;
    Tensor** parents;
    int num_parents;
    Buffer** saved_tensors;
    int num_saved_tensors;
};

// Tensor operations
Tensor* tensor_create(Buffer* buf);
void tensor_destroy(Tensor* t);
void tensor_backward(Tensor* t, int implicit);

// Function operations
Buffer* function_forward(Function* self, Buffer** inputs, int num_inputs);
Buffer** function_backward(Function* self, Buffer* grad_output);

// Operation application
Tensor* apply_op(OpType op, Tensor** inputs, int num_inputs);

#endif
