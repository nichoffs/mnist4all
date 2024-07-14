#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"

typedef struct Buffer {
  float *data;
  ShapeTracker *shapeTracker;
} Buffer;

Buffer *createBuffer(float *data, int *shape);
void freeBuffer(Buffer *buffer);

#endif // BUFFER_H

