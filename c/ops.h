#ifndef OPS_H
#define OPS_H

#include "buffer.h"
#include "shapetracker.h"

// Unary Ops

typedef float (*UnaryOpFunc)(float);

Buffer *unary_op(Buffer *buf, UnaryOpFunc);

Buffer *square_root(Buffer *buf);
Buffer *logarithm(Buffer *buf);
Buffer *exponent(Buffer *buf);

// Binary Ops

typedef float (*BinaryOpFunc)(float, float);

Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc);

Buffer *add(Buffer *buf1, Buffer *buf2);
Buffer *sub(Buffer *buf1, Buffer *buf2);
Buffer *mul(Buffer *buf1, Buffer *buf2);
Buffer *divide(Buffer *buf1, Buffer *buf2);

Buffer *matrix_vector_dot(Buffer *matrix, Buffer *vector);

// Reduce Ops

Buffer *sum(Buffer *buf);
Buffer *sumAxis(Buffer *buf, int axis);

Buffer *maxAxis(Buffer *buf, int axis);

// Movement Ops

Buffer *T(Buffer *buf);
Buffer* slice(Buffer *buf, int *start, int *end);
float indexBuffer(Buffer *buf, int *indices);
Buffer* reshape(Buffer* buf, int *shape);

#endif
