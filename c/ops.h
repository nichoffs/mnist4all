#ifndef OPS_H
#define OPS_H

#include "buffer.h"
#include "shapetracker.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Type definitions
typedef float (*UnaryOpFunc)(float);
typedef float (*BinaryOpFunc)(float, float);

// Unary Operations
Buffer* unary_op(Buffer* buf, UnaryOpFunc op);
Buffer* square_root(Buffer* buf);
Buffer* logarithm(Buffer* buf);
Buffer* exponent(Buffer* buf);
Buffer *logsumexp(Buffer *x);
Buffer *log_softmax(Buffer *buf); 
Buffer *log_softmax_backward(Buffer *output, Buffer *grad_output);
Buffer* relu(Buffer* buf);

// Binary Operations
Buffer* binary_op(Buffer* buf1, Buffer* buf2, BinaryOpFunc op);
Buffer* add(Buffer* buf1, Buffer* buf2);
Buffer* sub(Buffer* buf1, Buffer* buf2);
Buffer* mul(Buffer* buf1, Buffer* buf2);
Buffer* divide(Buffer* buf1, Buffer* buf2);

// Matrix Operations

Buffer *dot(Buffer *batch_vectors, Buffer *matrix);
Buffer *dot_backward(Buffer *grad_output, Buffer *input1, Buffer *input2,
                     int input_index);

// Reduce Operations
Buffer* sum(Buffer* buf);
Buffer* sumAxis(Buffer* buf, int axis);
Buffer* maxAxis(Buffer* buf, int axis);
Buffer* nll(Buffer* pred, Buffer* target);
Buffer *nll_backward(Buffer *grad_output, Buffer *target);

// Movement Operations
Buffer* T(Buffer* buf);
Buffer* slice(Buffer* buf, int* start, int* end);
Buffer* flatten(Buffer* buf);
Buffer *flattenAxes(Buffer *buf, int ax1, int ax2);
Buffer* unsqueeze(Buffer* buf, int axis);
Buffer* expand(Buffer* buf, int axis, int new_size);

#endif // OPS_H
