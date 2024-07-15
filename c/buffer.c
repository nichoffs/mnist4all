#include "buffer.h"
#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO: Eliminate data memcpy and just insist that any data passed in is
// mallocd so it can be cleared but we still don't need to copy

Buffer *createBuffer(float *data, int *shape, int size) {
  Buffer *x = (Buffer *)malloc(sizeof(Buffer));
  if (!x) {
    fprintf(stderr, "Failed to allocate memory for Buffer\n");
    return NULL;
  }

  if (data == NULL) {
    fprintf(stderr, "Input data was not allocated properly!\n");
    free(x);
    return NULL;
  }

  x->data = data;

  int ndim = 0;
  while (shape[ndim] != 0) {
    ndim++;
  }

  if (shape == NULL) {
    fprintf(stderr, "Shape was not allocated properly!");
    free(x->data);
    free(x);
    return NULL;
  }

  x->shapeTracker = createShapeTracker(shape, size);
  if (!x->shapeTracker) {
    free(x->data);
    free(x);
    return NULL;
  }
  return x;
}

Buffer *full_like(Buffer *buf, float value) {
  if (!buf) {
    fprintf(stderr, "Buffer is NULL");
    return NULL;
  }

  float *data = (float *)malloc(buf->shapeTracker->size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for data\n");
    return NULL;
  }
  for (int i = 0; i < buf->shapeTracker->size; i++) {
    data[i] = value;
  }

  // + 1 for the sentinel value
  int *shape = (int *)malloc((buf->shapeTracker->ndim + 1) * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Failed to allocate memory for shape\n");
    return NULL;
  }
  for (int i = 0; i < buf->shapeTracker->ndim; i++) {
    shape[i] = buf->shapeTracker->shape[i];
  }

  Buffer *new_buf = createBuffer(data, shape, buf->shapeTracker->size);

  if (!new_buf) {
    free(data); // Free data if buffer creation failed
  }

  return new_buf;
}

// TODO: THIS IS CURRENTLY GENERATING ONLY INTS IN THE RANGE AND INCLUDES TOP -
// FIX assumes shape was mallocd
Buffer *uniform(int *shape, int size, int min, int max) {
  float *data = (float *)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for data\n");
    return NULL;
  }

  if (!shape) {
    fprintf(stderr, "Shape data was null!\n");
    return NULL;
  }

  for (int i = 0; i < size; i++) {
    data[i] = rand() % (max - min + 1) + min;
  }

  Buffer *new_buf = createBuffer(data, shape, size);
  if (!new_buf) {
    free(data); // Free data if buffer creation failed
    free(shape);
    return NULL;
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
