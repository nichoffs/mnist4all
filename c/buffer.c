#include "buffer.h"
#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// TODO: HOW TO HANDLE FREEING MEMORY WHERE COPY IS FALSE

static Buffer *allocate_buffer() {
  Buffer *buf = (Buffer *)malloc(sizeof(Buffer));
  if (!buf) {
    fprintf(stderr, "Cannot allocate memory for Buffer\n");
  }
  return buf;
}

static float *allocate_data(int size) {
  float *data = (float *)malloc(sizeof(float) * size);
  if (!data) {
    fprintf(stderr, "Cannot allocate memory for Buffer data\n");
  }
  return data;
}

static int *calculate_strides(const int *shape, int ndim) {
  int *stride = (int *)malloc(sizeof(int) * ndim);
  if (!stride) {
    fprintf(stderr, "Cannot allocate memory for stride\n");
    return NULL;
  }

  stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

Buffer *buffer_create(float *data, int size, ShapeTracker *st, bool copy) {
  if (!data || !st) {
    fprintf(stderr, "ShapeTracker or input data is NULL\n");
    return NULL;
  }

  Buffer *buf = allocate_buffer();
  if (!buf)
    return NULL;

  buf->st = st;
  buf->size = size;
  buf->copy = copy;

  if (copy) {
    buf->data = allocate_data(size);
    if (!buf->data) {
      free(buf);
      return NULL;
    }
    memcpy(buf->data, data, sizeof(float) * size);
  } else {
    buf->data = data;
  }

  return buf;
}

Buffer *buffer_data_create(float *data, int size, int *shape, int ndim,
                           bool copy) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }

  int *stride = calculate_strides(shape, ndim);
  if (!stride)
    return NULL;

  ShapeTracker *st = shapetracker_create(shape, stride, 0, ndim);
  free(stride); // shapetracker_create makes a copy, so we can free this

  Buffer *buf = buffer_create(data, size, st, copy);
  if (!buf)
    shapetracker_destroy(st);

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

Buffer *zeros(int *shape, int ndim) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }

  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }

  if (!size) {
    fprintf(stderr, "Size is 0\n");
    return NULL;
  }

  float *data = (float *)calloc(size, sizeof(float));

  Buffer *buf = buffer_data_create(data, size, shape, ndim, false);

  return buf;
}

Buffer *randint(int *shape, int ndim, int low, int high) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }
  if (low >= high) {
    fprintf(stderr, "Invalid range: low must be less than high\n");
    return NULL;
  }

  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }

  if (!size) {
    fprintf(stderr, "Size is 0\n");
    return NULL;
  }

  float *data = (float *)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for randint data\n");
    return NULL;
  }

  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  int range = high - low;
  for (int i = 0; i < size; i++) {
    data[i] = (float)(low + (rand() % range));
  }

  Buffer *buf = buffer_data_create(data, size, shape, ndim, false);
  if (!buf) {
    free(data);
  }
  return buf;
}

Buffer *uniform(int *shape, int ndim, float low, float high) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }
  if (low >= high) {
    fprintf(stderr, "Invalid range: low must be less than high\n");
    return NULL;
  }

  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  if (!size) {
    fprintf(stderr, "Size is 0\n");
    return NULL;
  }

  float *data = (float *)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Failed to allocate memory for uniform data\n");
    return NULL;
  }

  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  float range = high - low;
  for (int i = 0; i < size; i++) {
    float random = (float)rand() / RAND_MAX;
    data[i] = low + random * range;
  }

  Buffer *buf = buffer_data_create(data, size, shape, ndim, false);
  if (!buf) {
    free(data);
  }
  return buf;
}
