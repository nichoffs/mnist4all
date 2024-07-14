#include "buffer.h"
#include "shapetracker.h"

// Unary Ops

Buffer *sqrt(Buffer *buf);
Buffer *log(Buffer *buf);
Buffer *exp(Buffer *buf);

// Binary Ops

// Reduce Ops

// Movement Ops

int calculateIndex(ShapeTracker *st, int *indices);
float index(Buffer *buf, int *indices);

