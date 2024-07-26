#include "buffer.h"

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

int *calculate_strides(const int *shape, int ndim) {
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

Buffer *buffer_copy(Buffer *buf) {
  if (!buf) {
    fprintf(stderr, "Cannot copy NULL buffer\n");
    return NULL;
  }

  ShapeTracker *st = shapetracker_create(buf->st->shape, buf->st->stride,
                                         buf->st->offset, buf->st->ndim);
  return buffer_create(buf->data, buf->size, st, true);
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
  free(stride);

  Buffer *buf = buffer_create(data, size, st, copy);
  if (!buf)
    shapetracker_destroy(st);

  return buf;
}

void buffer_destroy(Buffer *buf) {
  if (!buf)
    return;
  if (buf->copy)
    free(buf->data);
  shapetracker_destroy(buf->st);
  free(buf);
}

static int calculate_size(int *shape, int ndim) {
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

Buffer *zeros(int *shape, int ndim) {
  if (!shape) {
    fprintf(stderr, "Shape is NULL\n");
    return NULL;
  }

  int size = calculate_size(shape, ndim);
  if (size == 0) {
    fprintf(stderr, "Size is 0\n");
    return NULL;
  }

  float *data = (float *)calloc(size, sizeof(float));
  return buffer_data_create(data, size, shape, ndim, false);
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

  int size = calculate_size(shape, ndim);
  if (size == 0) {
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
    data[i] = low + ((float)rand() / RAND_MAX) * range;
  }

  Buffer *buf = buffer_data_create(data, size, shape, ndim, false);
  if (!buf)
    free(data);
  return buf;
}

Buffer *full_like(Buffer *buf, float fill_value) {
  if (!buf) {
    fprintf(stderr, "Buffer is NULL\n");
    return NULL;
  }

  Buffer *ret = buffer_copy(buf);
  if (!ret) {
    fprintf(stderr, "Buffer copy failed\n");
    return NULL;
  }

  for (int i = 0; i < ret->st->numel; i++) {
    int ix = view_index(ret->st, i);
    ret->data[ix] = fill_value;
  }

  return ret;
}
