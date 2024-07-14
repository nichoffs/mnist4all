#include "buffer.h"
#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>

Buffer *full_like(Buffer *buf, float value) {
  float *data = (float *)malloc(buf->shapeTracker->size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for data\n");
    return NULL;
  }

  for (int i = 0; i < buf->shapeTracker->size; i++) {
    data[i] = value;
  }

  return createBuffer(data, buf->shapeTracker->shape, buf->shapeTracker->size);
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
  return createBuffer(data, shape, size);
}

Buffer *createBuffer(float *data, int *shape, int size) {
  Buffer *x = (Buffer *)malloc(sizeof(Buffer));
  if (!x) {
    fprintf(stderr, "Failed to allocate memory for Buffer\n");
    exit(EXIT_FAILURE);
  }

  x->data = data;
  x->shapeTracker = createShapeTracker(shape, size);

  if (!x->shapeTracker) {
    free(x);
    return NULL;
  }

  return x;
}

int calculateIndex(ShapeTracker *st, int *indices) {
  int index = 0;
  for (int i = 0; i < st->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= st->shape[i]) {
      fprintf(stderr, "Index %d out of bounds for dimension %d.\n", indices[i],
              i);
      return -1; // Return -1 or another error code to indicate an out-of-bounds
                 // index
    }
    index += indices[i] * st->strides[i];
  }
  return index;
}

// Function to access data in a Buffer based on an array of indices
float index(Buffer *buf, int *indices) {
  int idx = calculateIndex(buf->shapeTracker, indices);
  if (idx == -1) {
    fprintf(stderr, "Invalid index.\n");
    return 0; // Handle error appropriately
  }
  return buf->data[idx];
}

void freeBuffer(Buffer *buffer) {
  freeShapeTracker(buffer->shapeTracker);
  free(buffer);
}
