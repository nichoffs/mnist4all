#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Unary Ops

Buffer *square_root(Buffer *buf) {
  int size = buf->shapeTracker->size;

  float *data = (float *)malloc(size * sizeof(float));
  int *shape = (int *)malloc((buf->shapeTracker->ndim + 1) * sizeof(int));

  // Copy data and shape correctly
  memcpy(data, buf->data, size * sizeof(float));
  memcpy(shape, buf->shapeTracker->shape,
         (buf->shapeTracker->ndim + 1) * sizeof(int));

  Buffer *new_buf = createBuffer(data, shape, size);
  if (!new_buf) {
    free(data);
    free(shape);
    return NULL;
  }

  // Calculate the square root for the new buffer's data
  for (int i = 0; i < size; i++) {
    new_buf->data[i] = sqrt(buf->data[i]);
  }

  return new_buf;
}

// Movement Ops

int calculateIndex(ShapeTracker *st, int *indices) {
  int index = 0;
  for (int i = 0; i < st->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= st->shape[i]) {
      fprintf(stderr, "Index %d out of bounds for dimension %d.\n", indices[i],
              i);
      return -1;
    }
    index += indices[i] * st->strides[i];
  }
  return index;
}

float indexBuffer(Buffer *buf, int *indices) {
  int idx = calculateIndex(buf->shapeTracker, indices);
  if (idx == -1) {
    fprintf(stderr, "Invalid index.\n");
    return 0; // Handle error appropriately
  }
  return buf->data[idx];
}
