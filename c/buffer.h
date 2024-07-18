#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"
#include <stdbool.h>

typedef struct {
  float *data;
  int size; // number of elements in data
  ShapeTracker *st; // view of data
  bool copy;
} Buffer;

Buffer *buffer_create(float *data, int size, ShapeTracker *st, bool copy);
Buffer *buffer_data_create(float *data, int size, int *shape, int ndim, bool copy);
void buffer_destroy(Buffer* buf);


#endif
