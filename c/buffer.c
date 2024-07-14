#include "buffer.h"
#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Buffer *createBuffer(float *data, int *shape, int size) {
  Buffer *x = (Buffer *)malloc(sizeof(Buffer));
  if (!x) {
    fprintf(stderr, "Failed to allocate memory for Buffer\n");
    return NULL;
  }

  x->data = (float *)malloc(size * sizeof(float));
  if (!x->data) {
    fprintf(stderr, "Failed to allocate memory for Buffer data\n");
    free(x);
    return NULL;
  }

  if (data != NULL) {
    memcpy(x->data, data, size * sizeof(float));
  }

  int ndim = 0;
  while (shape[ndim] != 0) {
    ndim++;
  }

  int *shape_copy = (int *)malloc((ndim + 1) * sizeof(int));
  if (!shape_copy) {
    fprintf(stderr, "Failed to allocate memory for shape\n");
    free(x->data);
    free(x);
    return NULL;
  }
  memcpy(shape_copy, shape, (ndim + 1) * sizeof(int));

  x->shapeTracker = createShapeTracker(shape_copy, size);
  if (!x->shapeTracker) {
    free(x->data);
    free(shape_copy);
    free(x);
    return NULL;
  }

  return x;
}

Buffer *full_like(Buffer *buf, float value) {
  float *data = (float *)malloc(buf->shapeTracker->size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for data\n");
    return NULL;
  }

  for (int i = 0; i < buf->shapeTracker->size; i++) {
    data[i] = value;
  }

  Buffer *new_buf =
      createBuffer(data, buf->shapeTracker->shape, buf->shapeTracker->size);
  if (!new_buf) {
    free(data); // Free data if buffer creation failed
  }
  return new_buf;
}

Buffer *uniformBuffer(int *shape, int size, int min, int max) {
  float *data = (float *)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for data\n");
    return NULL;
  }
  for (int i = 0; i < size; i++) {
    data[i] = rand() % (max - min + 1) + min;
  }

  Buffer *new_buf = createBuffer(data, shape, size);
  if (!new_buf) {
    free(data); // Free data if buffer creation failed
  }
  return new_buf;
}

void freeBuffer(Buffer *buffer) {
  if (buffer) {
    if (buffer->data) {
      free(buffer->data);
    }
    if (buffer->shapeTracker) {
      if (buffer->shapeTracker->shape) {
        free(buffer->shapeTracker->shape);
      }
      freeShapeTracker(buffer->shapeTracker);
    }
    free(buffer);
  }
}

