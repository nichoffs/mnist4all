#include "ops.h"
#include "shapetracker.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper functions

static Buffer *allocate_result_buffer(int size, int *shape, int ndim) {
  float *data = (float *)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }
  return buffer_data_create(data, size, shape, ndim, false);
}

static int *calculate_new_shape(Buffer *buf, int axis, int *new_size) {
  int *new_shape = (int *)malloc(sizeof(int) * (buf->st->ndim - 1));
  if (!new_shape) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  *new_size = 1;
  int j = 0;
  for (int i = 0; i < buf->st->ndim; i++) {
    if (i != axis) {
      new_shape[j] = buf->st->shape[i];
      *new_size *= new_shape[j];
      j++;
    }
  }
  return new_shape;
}

// Unary operations

Buffer *unary_op(Buffer *buf, UnaryOpFunc uop) {
  if (!buf) {
    fprintf(stderr, "Input to unary op is NULL\n");
    return NULL;
  }

  Buffer *ret =
      allocate_result_buffer(buf->size, buf->st->shape, buf->st->ndim);
  if (!ret)
    return NULL;

  for (int i = 0; i < buf->st->numel; i++) {
    int ix = view_index(buf->st, i);
    ret->data[i] = uop(buf->data[ix]);
  }

  return ret;
}

Buffer *square_root(Buffer *buf) { return unary_op(buf, sqrtf); }
Buffer *logarithm(Buffer *buf) { return unary_op(buf, logf); }
Buffer *exponent(Buffer *buf) { return unary_op(buf, expf); }

// Binary operations

Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc op_func) {
  if (!buf1 || !buf2) {
    fprintf(stderr, "One of the inputs to binary op is NULL\n");
    return NULL;
  }

  Buffer *ret =
      allocate_result_buffer(buf1->size, buf1->st->shape, buf1->st->ndim);
  if (!ret)
    return NULL;

  for (int i = 0; i < buf1->st->numel; i++) {
    int ix1 = view_index(buf1->st, i);
    int ix2 = view_index(buf2->st, i);
    ret->data[i] = op_func(buf1->data[ix1], buf2->data[ix2]);
  }

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

// Matrix operations

Buffer *matrix_vector_dot(Buffer *matrix, Buffer *vector) {
  if (!matrix || !vector || matrix->st->ndim != 2 || vector->st->ndim != 1) {
    fprintf(stderr, "Invalid inputs for matrix_vector_dot\n");
    return NULL;
  }

  if (matrix->st->shape[1] != vector->st->shape[0]) {
    fprintf(stderr, "Matrix columns must match vector length\n");
    return NULL;
  }

  int rows = matrix->st->shape[0];
  int cols = matrix->st->shape[1];
  int result_shape[1] = {rows};

  Buffer *result = allocate_result_buffer(rows, result_shape, 1);
  if (!result)
    return NULL;

  for (int i = 0; i < rows; i++) {
    result->data[i] = 0;
    for (int j = 0; j < cols; j++) {
      int matrix_index = view_index(matrix->st, i * cols + j);
      int vector_index = view_index(vector->st, j);
      result->data[i] +=
          matrix->data[matrix_index] * vector->data[vector_index];
    }
  }

  return result;
}

// Reduce operations

Buffer *sum(Buffer *buf) {
  if (!buf) {
    fprintf(stderr, "Buffer to sum is NULL\n");
    return NULL;
  }

  float *sum = (float *)malloc(sizeof(float));
  if (!sum) {
    fprintf(stderr, "Memory allocation failed in sum\n");
    return NULL;
  }
  *sum = 0;

  for (int i = 0; i < buf->st->numel; i++) {
    int ix = view_index(buf->st, i);
    *sum += buf->data[ix];
  }

  int shape[1] = {1};
  Buffer *ret = buffer_data_create(sum, 1, shape, 1, false);
  if (!ret)
    free(sum);
  return ret;
}

Buffer *sumAxis(Buffer *buf, int axis) {
  if (!buf || axis < 0 || axis >= buf->st->ndim) {
    fprintf(stderr, "Invalid input or axis for sumAxis\n");
    return NULL;
  }

  int new_size;
  int *new_shape = calculate_new_shape(buf, axis, &new_size);
  if (!new_shape)
    return NULL;

  Buffer *result =
      allocate_result_buffer(new_size, new_shape, buf->st->ndim - 1);
  if (!result) {
    free(new_shape);
    return NULL;
  }

  int *start = (int *)calloc(buf->st->ndim, sizeof(int));
  int *end = (int *)malloc(buf->st->ndim * sizeof(int));
  if (!start || !end) {
    fprintf(stderr, "Memory allocation failed in sumAxis\n");
    free(new_shape);
    free(start);
    free(end);
    buffer_destroy(result);
    return NULL;
  }

  for (int i = 0; i < buf->st->ndim; i++) {
    end[i] = buf->st->shape[i];
  }

  for (int i = 0; i < buf->st->shape[axis]; i++) {
    start[axis] = i;
    end[axis] = i + 1;

    Buffer *slice_buf = slice(buf, start, end);
    if (!slice_buf) {
      fprintf(stderr, "Slicing failed in sumAxis\n");
      free(new_shape);
      free(start);
      free(end);
      buffer_destroy(result);
      return NULL;
    }

    for (int j = 0; j < new_size; j++) {
      result->data[j] += slice_buf->data[view_index(slice_buf->st, j)];
    }

    buffer_destroy(slice_buf);
  }

  free(new_shape);
  free(start);
  free(end);
  return result;
}

Buffer *maxAxis(Buffer *buf, int axis) {
  if (!buf || axis < 0 || axis >= buf->st->ndim) {
    fprintf(stderr, "Invalid input or axis for maxAxis\n");
    return NULL;
  }

  int new_size;
  int *new_shape = calculate_new_shape(buf, axis, &new_size);
  if (!new_shape)
    return NULL;

  Buffer *result =
      allocate_result_buffer(new_size, new_shape, buf->st->ndim - 1);
  if (!result) {
    free(new_shape);
    return NULL;
  }

  for (int i = 0; i < new_size; i++) {
    result->data[i] = -FLT_MAX;
  }

  int *start = (int *)calloc(buf->st->ndim, sizeof(int));
  int *end = (int *)malloc(buf->st->ndim * sizeof(int));
  if (!start || !end) {
    fprintf(stderr, "Memory allocation failed in maxAxis\n");
    free(new_shape);
    free(start);
    free(end);
    buffer_destroy(result);
    return NULL;
  }

  for (int i = 0; i < buf->st->ndim; i++) {
    end[i] = buf->st->shape[i];
  }

  for (int i = 0; i < buf->st->shape[axis]; i++) {
    start[axis] = i;
    end[axis] = i + 1;

    Buffer *slice_buf = slice(buf, start, end);
    if (!slice_buf) {
      fprintf(stderr, "Slicing failed in maxAxis\n");
      free(new_shape);
      free(start);
      free(end);
      buffer_destroy(result);
      return NULL;
    }

    for (int j = 0; j < new_size; j++) {
      float slice_value = slice_buf->data[view_index(slice_buf->st, j)];
      if (slice_value > result->data[j]) {
        result->data[j] = slice_value;
      }
    }

    buffer_destroy(slice_buf);
  }

  free(new_shape);
  free(start);
  free(end);
  return result;
}

// Movement operations

Buffer *slice(Buffer *buf, int *start, int *end) {
  int *newShape = (int *)malloc(sizeof(int) * buf->st->ndim);
  if (!newShape) {
    fprintf(stderr, "Memory allocation failed in slice\n");
    return NULL;
  }

  int offset = 0;
  for (int i = 0; i < buf->st->ndim; i++) {
    if (start[i] < 0 || end[i] > buf->st->shape[i] || start[i] >= end[i]) {
      fprintf(stderr, "Invalid slice\n");
      free(newShape);
      return NULL;
    }
    newShape[i] = end[i] - start[i];
    offset += start[i] * buf->st->stride[i];
  }

  ShapeTracker *st =
      shapetracker_create(newShape, buf->st->stride, offset, buf->st->ndim);
  if (!st) {
    fprintf(stderr, "Failed to create ShapeTracker in slice\n");
    free(newShape);
    return NULL;
  }

  Buffer *new_buf = buffer_create(buf->data, buf->size, st, false);
  if (!new_buf) {
    fprintf(stderr, "Failed to create Buffer in slice\n");
    shapetracker_destroy(st);
  }

  free(newShape);
  return new_buf;
}

Buffer *T(Buffer *buf) {
  if (!buf || buf->st->ndim != 2) {
    fprintf(stderr, "Invalid input for T\n");
    return NULL;
  }

  int new_shape[2] = {buf->st->shape[1], buf->st->shape[0]};
  int new_stride[2] = {buf->st->stride[1], buf->st->stride[0]};

  ShapeTracker *st =
      shapetracker_create(new_shape, new_stride, buf->st->offset, 2);
  if (!st) {
    fprintf(stderr, "Failed to create ShapeTracker in T\n");
    return NULL;
  }

  Buffer *new_buf = buffer_create(buf->data, buf->size, st, false);
  if (!new_buf) {
    fprintf(stderr, "Failed to create Buffer in T\n");
    shapetracker_destroy(st);
    return NULL;
  }

  return new_buf;
}
