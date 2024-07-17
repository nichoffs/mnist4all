#include "buffer.h"
#include <stdio.h>

void reportMemoryError(const char *type) {
  fprintf(stderr, "Failed to allocate memory for %s\n", type);
}

static void _printData(Buffer *buf, int dim, int offset) {
  int ndim = buf->shapeTracker->ndim;
  int *shape = buf->shapeTracker->shape;
  float *data = buf->data;

  if (dim == ndim - 1) {
    printf("[");
    for (int i = 0; i < shape[dim]; i++) {
      printf("%f", data[offset + i]);
      if (i < shape[dim] - 1) {
        printf(", ");
      }
    }
    printf("]");
  } else {
    printf("[");
    int stride = buf->shapeTracker->strides[dim];
    for (int i = 0; i < shape[dim]; i++) {
      _printData(buf, dim + 1, offset + i * stride);
      if (i < shape[dim] - 1) {
        printf(",\n");
        for (int j = 0; j <= dim; j++) {
          printf(" ");
        }
      }
    }
    printf("]");
  }
}

void printBuffer(Buffer *buf) {
  if (!buf || !buf->data || !buf->shapeTracker) {
    fprintf(stderr, "Buffer or its data/shapeTracker is NULL\n");
    return;
  }

  int ndim = buf->shapeTracker->ndim;
  int *shape = buf->shapeTracker->shape;

  if (ndim == 0) {
    printf("[]\n");
    return;
  }

  _printData(buf, 0, 0);
  printf("\n");
}
