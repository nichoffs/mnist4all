#include "ops.h"
#include <stdio.h>

// Unary Ops

/* Buffer *sqrt(Buffer *buf) { */
/*   for (int i = 0; i < buf->shapeTracker->size; i++) { */
/*     buf->data */
/*   } */
/* } */

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

float index(Buffer *buf, int *indices) {
  int idx = calculateIndex(buf->shapeTracker, indices);
  if (idx == -1) {
    fprintf(stderr, "Invalid index.\n");
    return 0; // Handle error appropriately
  }
  return buf->data[idx];
}
