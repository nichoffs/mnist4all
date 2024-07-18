#include "buffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO: HOW TO HANDLE FREEING MEMORY WHERE COPY IS FALSE

Buffer *buffer_create(float *data, int size, ShapeTracker *st, bool copy) {
  Buffer *buf = (Buffer *)malloc(sizeof(Buffer));
  if (!buf) {
    fprintf(stderr, "Cannot allocate memory for Buffer\n");
    return NULL;
  }

  if (!data || !st) {
    fprintf(stderr, "ShapeTracker or input data is NULL\n");
    return NULL;
  }

  buf->st = st;

  if (copy) {
    buf->data = (float *)malloc(sizeof(float) * size);
    if (!buf->data) {
      fprintf(stderr, "Cannot allocate memory for Buffer data\n");
      return NULL;
    }
    memcpy(buf->data, data, sizeof(float) * size);
  } else {
    buf->data = data;
  }

  buf->copy = copy;

  return buf;
}

Buffer *buffer_data_create(float *data, int size, int *shape, int ndim,
                           bool copy) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }

  int *stride = (int *)malloc(sizeof(int) * ndim);
  if (!stride) {
    fprintf(stderr, "Cannot allocate memory for stride\n");
    return NULL;
  }

  stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }

  ShapeTracker *st = shapetracker_create(shape, stride, 0, ndim);
  Buffer *buf = buffer_create(data, size, st, copy);
  return buf;
}

void buffer_destroy(Buffer *buf) {
  if (!buf)
    return;
  if (buf->copy) {
    free(buf->data);
  }
  shapetracker_destroy(buf->st);
  free(buf);
}
