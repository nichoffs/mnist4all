#include "ops.h"
#include "shapetracker.h"
#include "utils.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper functions

static int validate_op_input(Buffer *buf) {
  if (!buf) {
    fprintf(stderr, "Input buffer is NULL\n");
    return 0;
  }
  if (!buf->data) {
    fprintf(stderr, "Input buffer data is NULL\n");
    return 0;
  }
  if (!buf->st) {
    fprintf(stderr, "Input buffer ShapeTracker is NULL\n");
    return 0;
  }
  if (!buf->st->shape) {
    fprintf(stderr, "Input buffer shape is NULL\n");
    return 0;
  }
  if (buf->st->numel == 0 || buf->size == 0) {
    fprintf(stderr, "Input buffer has zero elements\n");
    return 0;
  }
  return 1;
}

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
  if (!validate_op_input(buf)) {
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

float relu_func(float a) { return a > 0 ? a : 0; }

Buffer *square_root(Buffer *buf) { return unary_op(buf, sqrtf); }
Buffer *logarithm(Buffer *buf) { return unary_op(buf, logf); }
Buffer *exponent(Buffer *buf) { return unary_op(buf, expf); }
Buffer *relu(Buffer *buf) { return unary_op(buf, relu_func); }

Buffer *logsumexp(Buffer *x) {
  Buffer *c = maxAxis(x, 1);

  // Unsqueeze c to allow broadcasting
  Buffer *c_unsqueezed = unsqueeze(c, 1);

  // Broadcast c to match x's shape
  Buffer *c_broadcasted = expand(c_unsqueezed, 1, x->st->shape[1]);

  // x - c_broadcasted
  Buffer *x_minus_c = sub(x, c_broadcasted);

  // exp(x - c_broadcasted)
  Buffer *exp_x_minus_c = exponent(x_minus_c);

  // sum(exp(x - c_broadcasted), axis=1)
  Buffer *sum_exp = sumAxis(exp_x_minus_c, 1);

  // log(sum_exp)
  Buffer *log_sum_exp = logarithm(sum_exp);

  // c + log(sum_exp)
  Buffer *result = add(c, log_sum_exp);

  // Clean up temporary buffers
  buffer_destroy(c_unsqueezed);
  buffer_destroy(c_broadcasted);
  buffer_destroy(x_minus_c);
  buffer_destroy(exp_x_minus_c);
  buffer_destroy(sum_exp);
  buffer_destroy(log_sum_exp);

  return result;
}

Buffer *log_softmax(Buffer *buf) {
  Buffer *lse = logsumexp(buf);

  // Unsqueeze lse to allow broadcasting
  Buffer *lse_unsqueezed = unsqueeze(lse, 1);

  // Broadcast lse to match buf's shape
  Buffer *lse_broadcasted = expand(lse_unsqueezed, 1, buf->st->shape[1]);

  // buf - lse_broadcasted
  Buffer *result = sub(buf, lse_broadcasted);

  // Clean up temporary buffers
  buffer_destroy(lse);
  buffer_destroy(lse_unsqueezed);
  buffer_destroy(lse_broadcasted);

  return result;
}

// Binary operations

Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc op_func) {
  if (!validate_op_input(buf1) || !validate_op_input(buf2)) {
    return NULL;
  }

  if (buf1->st->ndim != buf2->st->ndim) {
    fprintf(stderr,
            "Binary Op Shape Mismatch: Different number of dimensions\n");
    return NULL;
  }

  for (int i = 0; i < buf1->st->ndim; i++) {
    if (buf1->st->shape[i] != buf2->st->shape[i]) {
      fprintf(stderr, "Binary Op Shape Mismatch: Dimension %d differs\n", i);
      return NULL;
    }
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

// (1,N) @ (N,M) = (1,M)
Buffer *vector_matrix_dot(Buffer *vector, Buffer *matrix) {
  if (!validate_op_input(vector) || !validate_op_input(matrix)) {
    return NULL;
  }
  if (vector->st->ndim != 2 || matrix->st->ndim != 2) {
    fprintf(stderr, "Invalid dimensions for vector_matrix_dot\n");
    return NULL;
  }
  if (vector->st->shape[0] != 1 ||
      vector->st->shape[1] != matrix->st->shape[0]) {
    fprintf(stderr, "Vector shape must be (1,N) and match matrix rows (N,M)\n");
    return NULL;
  }
  int vector_length = vector->st->shape[1];
  int cols = matrix->st->shape[1];
  int result_shape[] = {1, cols};
  Buffer *result = allocate_result_buffer(cols, result_shape, 2);
  if (!result)
    return NULL;
  for (int j = 0; j < cols; j++) {
    result->data[j] = 0;
    for (int i = 0; i < vector_length; i++) {
      int vector_index = view_index(vector->st, i);
      int matrix_index = view_index(matrix->st, i * cols + j);
      result->data[j] +=
          vector->data[vector_index] * matrix->data[matrix_index];
    }
  }
  return result;
}
Buffer *matrix_vector_dot(Buffer *matrix, Buffer *vector) {
  if (!validate_op_input(matrix) || !validate_op_input(vector)) {
    return NULL;
  }

  if (matrix->st->ndim != 2 || vector->st->ndim != 1) {
    fprintf(stderr, "Invalid dimensions for matrix_vector_dot\n");
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

Buffer *flatten(Buffer *buf) {
  if (!buf || buf->st->numel == 0) {
    fprintf(stderr, "Invalid input for flatten\n");
    return NULL;
  }

  int shape[1] = {buf->st->numel};
  int stride[1] = {1};

  ShapeTracker *st = shapetracker_create(shape, stride, 0, 1);

  Buffer *ret = buffer_create(buf->data, buf->size, st, false);

  return ret;
}

Buffer *unsqueeze(Buffer *buf, int axis) {
  if (!buf || buf->st->numel == 0) {
    fprintf(stderr, "Invalid input for flatten\n");
    return NULL;
  }

  int ndim = buf->st->ndim + 1;
  int shape[ndim];
  int stride[ndim];
  int j = 0;

  for (int i = 0; i < ndim; i++) {
    if (axis == i) {
      shape[i] = 1;
      stride[i] = (i > 0) ? stride[i - 1] : buf->st->stride[0];
    } else {
      shape[i] = buf->st->shape[j];
      stride[i] = buf->st->stride[j];
      j++;
    }
  }

  ShapeTracker *st = shapetracker_create(shape, stride, 0, ndim);
  if (!st) {
    fprintf(stderr, "Failed to create ShapeTracker in unsqueeze\n");
    return NULL;
  }

  Buffer *ret = buffer_create(buf->data, buf->size, st, false);
  if (!ret) {
    fprintf(stderr, "Failed to create Buffer in unsqueeze\n");
    shapetracker_destroy(st);
  }

  return ret;
}

Buffer *expand(Buffer *buf, int axis, int new_size) {
  if (!buf || buf->st->numel == 0) {
    fprintf(stderr, "Invalid input for expand\n");
    return NULL;
  }
  if (axis < 0 || axis >= buf->st->ndim) {
    fprintf(stderr, "Invalid axis for expand\n");
    return NULL;
  }
  if (buf->st->shape[axis] != 1) {
    fprintf(stderr, "Invalid axis shape for expand -- must be one!\n");
    return NULL;
  }
  if (new_size < 1) {
    fprintf(stderr, "Invalid new size for expand -- must be positive!\n");
    return NULL;
  }

  int *new_shape = (int *)malloc(buf->st->ndim * sizeof(int));
  int *new_stride = (int *)malloc(buf->st->ndim * sizeof(int));
  if (!new_shape || !new_stride) {
    fprintf(stderr, "Memory allocation failed in expand\n");
    free(new_shape);
    free(new_stride);
    return NULL;
  }

  for (int i = 0; i < buf->st->ndim; i++) {
    new_shape[i] = buf->st->shape[i];
    new_stride[i] = buf->st->stride[i];
    if (i == axis) {
      new_shape[i] = new_size;
      new_stride[i] = 0;
    }
  }

  ShapeTracker *st = shapetracker_create(new_shape, new_stride, buf->st->offset,
                                         buf->st->ndim);
  if (!st) {
    fprintf(stderr, "Failed to create ShapeTracker in expand\n");
    free(new_shape);
    free(new_stride);
    return NULL;
  }

  Buffer *new_buf = buffer_create(buf->data, buf->size, st, false);
  if (!new_buf) {
    fprintf(stderr, "Failed to create Buffer in expand\n");
    shapetracker_destroy(st);
    free(new_shape);
    free(new_stride);
    return NULL;
  }

  free(new_shape);
  free(new_stride);
  return new_buf;
}
