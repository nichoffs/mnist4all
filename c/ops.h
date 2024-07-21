#ifndef OPS_H
#define OPS_H

#include "buffer.h"
#include "shapetracker.h"

// Type definitions
typedef float (*UnaryOpFunc)(float);
typedef float (*BinaryOpFunc)(float, float);

// Unary Operations
Buffer* unary_op(Buffer* buf, UnaryOpFunc op);
Buffer* square_root(Buffer* buf);
Buffer* logarithm(Buffer* buf);
Buffer* exponent(Buffer* buf);
Buffer* log_softmax(Buffer* buf);
Buffer* relu(Buffer* buf);

// Binary Operations
Buffer* binary_op(Buffer* buf1, Buffer* buf2, BinaryOpFunc op);
Buffer* add(Buffer* buf1, Buffer* buf2);
Buffer* sub(Buffer* buf1, Buffer* buf2);
Buffer* mul(Buffer* buf1, Buffer* buf2);
Buffer* divide(Buffer* buf1, Buffer* buf2);

// Matrix Operations
Buffer* matrix_vector_dot(Buffer* matrix, Buffer* vector);

// Reduce Operations
Buffer* sum(Buffer* buf);
Buffer* sumAxis(Buffer* buf, int axis);
Buffer* maxAxis(Buffer* buf, int axis);

// Movement Operations
Buffer* T(Buffer* buf);
Buffer* slice(Buffer* buf, int* start, int* end);
float indexBuffer(Buffer* buf, int* indices);
Buffer* flatten(Buffer* buf);
Buffer* unsqueeze(Buffer* buf, int axis);
Buffer* expand(Buffer* buf, int axis, int new_size);

#endif // OPS_H
