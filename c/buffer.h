#ifndef BUFFER_H
#define BUFFER_H

#include <stdbool.h>
#include "shapetracker.h"

typedef struct Buffer {
  float *data;
  ShapeTracker *shapeTracker;
} Buffer;

Buffer *initBuffer(float *data, int *shape, int size, bool copy);
Buffer *copyBuffer(Buffer* buf);
void freeBuffer(Buffer *buffer);

#endif // BUFFER_H

