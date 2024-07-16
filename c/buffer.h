#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"

typedef struct Buffer {
  float *data;
  ShapeTracker *shapeTracker;
} Buffer;

Buffer *full_like(Buffer *buf, float value);
Buffer *randint(int *shape, int size, int min, int max);
Buffer *createBuffer(float *data, int *shape, int size);
Buffer *copyBuffer(Buffer *buf);
void printBuffer(Buffer *buf);
void freeBuffer(Buffer *buffer);

#endif // BUFFER_H

