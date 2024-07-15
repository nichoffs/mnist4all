#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Unary Ops

// Define the type for a unary operation function
typedef float (*UnaryOpFunc)(float);
typedef float (*BinaryOpFunc)(float, float);

// Unary operation function
Buffer *unary_op(Buffer *buf, UnaryOpFunc op_func) {
  int size = buf->shapeTracker->size;

  float *data = (float *)malloc(size * sizeof(float));
  int *shape = (int *)malloc((buf->shapeTracker->ndim + 1) * sizeof(int));

  // Copy data and shape correctly
  memcpy(data, buf->data, size * sizeof(float));
  memcpy(shape, buf->shapeTracker->shape,
         (buf->shapeTracker->ndim + 1) * sizeof(int));

  Buffer *new_buf = createBuffer(data, shape, size);
  if (!new_buf) {
    free(data);
    free(shape);
    return NULL;
  }

  // Apply the unary operation
  for (int i = 0; i < size; i++) {
    new_buf->data[i] = op_func(buf->data[i]);
  }

  return new_buf;
}

Buffer *square_root(Buffer *buf) { return unary_op(buf, sqrtf); }

Buffer *logarithm(Buffer *buf) { return unary_op(buf, logf); }

Buffer *exponent(Buffer *buf) { return unary_op(buf, expf); }

// Binary Op

// TODO: CHECK SHAPES
Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc op_func) {
  int size = buf1->shapeTracker->size;

  float *data = (float *)calloc(size, sizeof(float));
  if (!data) {
    fprintf(stderr, "Error: Failed to allocate memory for data.\n");
    return NULL;
  }
  int *shape = (int *)malloc((buf1->shapeTracker->ndim + 1) * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Error: Failed to allocate memory for shape.\n");
    free(data);
    return NULL;
  }
  for (int i = 0; i < buf1->shapeTracker->ndim; i++) {
    shape[i] = buf1->shapeTracker->shape[i];
  }

  Buffer *new_buf = createBuffer(data, shape, size);
  if (!new_buf) {
    free(data);
    free(shape);
    return NULL;
  }

  // Apply the unary operation
  for (int i = 0; i < size; i++) {
    new_buf->data[i] = op_func(buf1->data[i], buf2->data[i]);
  }

  return new_buf;
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

// Movement Ops

int calculateIndex(ShapeTracker *st, int *indices) {
  int index = 0;
  for (int i = 0; i < st->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= st->shape[i]) {
      fprintf(stderr, "Index %d out of bounds for dimension %d.\n", indices[i],
              i);
      return -1;
    }
    index += indices[i] * st->strides[i];
  }
  return index;
}

float indexBuffer(Buffer *buf, int *indices) {
  int idx = calculateIndex(buf->shapeTracker, indices);
  if (idx == -1) {
    fprintf(stderr, "Invalid index.\n");
    return 0; // Handle error appropriately
  }
  return buf->data[idx];
}
