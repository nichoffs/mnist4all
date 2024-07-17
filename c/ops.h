#include "buffer.h"
#include "shapetracker.h"

// Load Ops

Buffer *full_like(Buffer *buf, float value);
Buffer *randint(int *shape, int size, int min, int max);

// Unary Ops

typedef float (*UnaryOpFunc)(float);
typedef float (*BinaryOpFunc)(float, float);

Buffer *square_root(Buffer *buf);
Buffer *logarithm(Buffer *buf);
Buffer *exponent(Buffer *buf);

// Binary Ops

Buffer *add(Buffer *buf1, Buffer *buf2);
Buffer *sub(Buffer *buf1, Buffer *buf2);
Buffer *mul(Buffer *buf1, Buffer *buf2);
Buffer *divide(Buffer *buf1, Buffer *buf2);

// Reduce Ops

Buffer *sum(Buffer *buf);
Buffer *sumAxis(Buffer *buf, int axis);

// Movement Ops

int calculateIndex(ShapeTracker *st, int *indices);
float indexBuffer(Buffer *buf, int *indices);
Buffer* reshape(Buffer* buf, int *shape);

