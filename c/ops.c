#include "ops.h"
#include "shapetracker.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// unary op
Buffer *unary_op(Buffer *buf, UnaryOpFunc uop) {
  if (!buf) {
    fprintf(stderr, "Input to unaryop is none\n");
    return NULL;
  }

  float *data = (float *)malloc(sizeof(float) * buf->size);
  for (int i = 0; i < buf->st->numel; i++) {
    int ix = view_index(buf->st, i);
    data[i] = uop(buf->data[ix]);
  }

  Buffer *ret =
      buffer_data_create(data, buf->size, buf->st->shape, buf->st->ndim, false);

  return ret;
}

Buffer *square_root(Buffer *buf) { return unary_op(buf, sqrtf); }
Buffer *logarithm(Buffer *buf) { return unary_op(buf, logf); }
Buffer *exponent(Buffer *buf) { return unary_op(buf, expf); }

// binary op

Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc op_func) {
  if (!buf1 || !buf2) {
    fprintf(stderr, "One of the inputs to to binaryop is none\n");
    return NULL;
  }

  float *data = (float *)malloc(sizeof(float) * buf1->size);

  for (int i = 0; i < buf1->st->numel; i++) {
    int ix1 = view_index(buf1->st, i);
    int ix2 = view_index(buf2->st, i);
    data[i] = op_func(buf1->data[ix1], buf2->data[ix2]);
  }

  Buffer *ret = buffer_data_create(data, buf1->size, buf1->st->shape,
                                   buf1->st->ndim, true);

  return ret;
}

float add_func(float a, float b) { return a + b; }
float sub_func(float a, float b) { return a - b; }
float mul_func(float a, float b) { return a * b; }
float div_func(float a, float b) { return a / b; }

Buffer *add(Buffer *buf1, Buffer *buf2) {
  return binary_op(buf1, buf2, add_func);
}
Buffer *sub(Buffer *buf1, Buffer *buf2) {
  return binary_op(buf1, buf2, sub_func);
}
Buffer *mul(Buffer *buf1, Buffer *buf2) {
  return binary_op(buf1, buf2, mul_func);
}
Buffer *divide(Buffer *buf1, Buffer *buf2) {
  return binary_op(buf1, buf2, div_func);
}

#include "ops.h"
#include <stdio.h>
#include <stdlib.h>

Buffer *sumAxis(Buffer *buf, int axis) {
  if (!buf || axis < 0 || axis >= buf->st->ndim) {
    fprintf(stderr, "Invalid input or axis for sumAxis\n");
    return NULL;
  }

  // Calculate the shape of the result
  int *new_shape = (int *)malloc(sizeof(int) * (buf->st->ndim - 1));
  int new_size = 1;
  int j = 0;
  for (int i = 0; i < buf->st->ndim; i++) {
    if (i != axis) {
      new_shape[j] = buf->st->shape[i];
      new_size *= new_shape[j];
      j++;
    }
  }

  // Allocate memory for the result
  float *result_data = (float *)calloc(new_size, sizeof(float));
  if (!result_data) {
    fprintf(stderr, "Memory allocation failed in sumAxis\n");
    free(new_shape);
    return NULL;
  }

  // Create start and end arrays for slicing
  int *start = (int *)calloc(buf->st->ndim, sizeof(int));
  int *end = (int *)malloc(buf->st->ndim * sizeof(int));
  for (int i = 0; i < buf->st->ndim; i++) {
    end[i] = buf->st->shape[i];
  }

  for (int i = 0; i < buf->st->shape[axis]; i++) {
    start[axis] = i;
    end[axis] = i + 1;

    // Slice the buffer
    Buffer *slice_buf = slice(buf, start, end);
    if (!slice_buf) {
      fprintf(stderr, "Slicing failed in sumAxis\n");
      free(new_shape);
      free(result_data);
      free(start);
      free(end);
      return NULL;
    }

    for (int j = 0; j < new_size; j++) {
      result_data[j] += slice_buf->data[view_index(slice_buf->st, j)];
    }

    buffer_destroy(slice_buf);
  }

  Buffer *result = buffer_data_create(result_data, new_size, new_shape,
                                      buf->st->ndim - 1, false);

  free(new_shape);
  free(start);
  free(end);
  return result;
}

// movement op

Buffer *slice(Buffer *buf, int *start, int *end) {
  int *newShape = (int *)malloc(sizeof(int) * buf->st->ndim);
  int offset = 0;
  for (int i = 0; i < buf->st->ndim; i++) {
    if (start[i] < 0 || end[i] > buf->st->shape[i] || start[i] >= end[i]) {
      fprintf(stderr, "Invalid slice\n");
      return NULL;
    }
    newShape[i] = end[i] - start[i];
    offset += start[i] * buf->st->stride[i];
  }
  ShapeTracker *st =
      shapetracker_create(newShape, buf->st->stride, offset, buf->st->ndim);
  Buffer *new_buf = buffer_create(buf->data, buf->size, st, false);
  free(newShape);
  return new_buf;
}
