#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"

typedef struct Buffer {
  float *data;
  ShapeTracker *shapeTracker;
} Buffer;

Buffer *full_like(Buffer *buf, float value);
Buffer *uniformBuffer(int *shape, int size, int min, int max);
Buffer *createBuffer(float *data, int *shape, int size);
float index(Buffer *buf, int *indices);
void freeBuffer(Buffer *buffer);

#endif // BUFFER_H

