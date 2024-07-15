#include "buffer.h"
#include "shapetracker.h"

// Unary Ops

Buffer *square_root(Buffer *buf);
Buffer *logarithm(Buffer *buf);
Buffer *exponent(Buffer *buf);

// Binary Ops

Buffer *add(Buffer *buf1, Buffer *buf2);
Buffer *sub(Buffer *buf1, Buffer *buf2);
Buffer *mul(Buffer *buf1, Buffer *buf2);
Buffer *divide(Buffer *buf1, Buffer *buf2);

// Reduce Ops

// Movement Ops

int calculateIndex(ShapeTracker *st, int *indices);
float indexBuffer(Buffer *buf, int *indices);

