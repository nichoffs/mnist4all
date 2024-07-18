#include "utils.h"

static void data_print(Buffer *buf, int dim, int offset) {
  int ndim = buf->st->ndim;
  int *shape = buf->st->shape;
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
    int stride = buf->st->stride[dim];
    for (int i = 0; i < shape[dim]; i++) {
      data_print(buf, dim + 1, offset + i * stride);
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

void buffer_print(Buffer *buf) {
  if (!buf || !buf->data || !buf->st) {
    fprintf(stderr, "Buffer or its data/ShapeTracker is NULL\n");
    return;
  }

  int ndim = buf->st->ndim;

  if (ndim == 0) {
    printf("[]\n");
    return;
  }

  data_print(buf, 0, buf->st->offset);
  printf("\n");
}

void shapetracker_print(Buffer *buf) {
  if (!buf || !buf->data || !buf->st) {
    fprintf(stderr, "Buffer or its data/ShapeTracker is NULL\n");
    return;
  }

  int ndim = buf->st->ndim;
  if (ndim == 0) {
    printf("[]\n");
    return;
  }

  data_print(buf, 0, buf->st->offset);
  printf("\n");
}
