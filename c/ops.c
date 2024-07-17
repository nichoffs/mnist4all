#include "ops.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load Ops

// Unary Ops
Buffer *unary_op(Buffer *buf, UnaryOpFunc op_func) {
  Buffer *new_buf = copyBuffer(buf);
  if (!new_buf) {
    return NULL;
  }

  // Apply the unary operation
  for (int i = 0; i < new_buf->shapeTracker->size; i++) {
    new_buf->data[i] = op_func(buf->data[i]);
  }

  return new_buf;
}

Buffer *square_root(Buffer *buf) { return unary_op(buf, sqrtf); }
Buffer *logarithm(Buffer *buf) { return unary_op(buf, logf); }
Buffer *exponent(Buffer *buf) { return unary_op(buf, expf); }

// Binary Op
Buffer *binary_op(Buffer *buf1, Buffer *buf2, BinaryOpFunc op_func) {
  Buffer *new_buf = copyBuffer(buf1);

  if (new_buf == NULL) {
    return NULL;
  }

  for (int i = 0; i < buf1->shapeTracker->ndim + 1; i++) {
    if (buf1->shapeTracker->ndim != buf2->shapeTracker->ndim) {
      printf("Number of dimensions for binary op don't match\n");
      return NULL;
    }
    if (buf1->shapeTracker->shape[i] != buf2->shapeTracker->shape[i]) {
      printf("Shapes for binary op don't match\n");
      return NULL;
    }
  }

  for (int i = 0; i < new_buf->shapeTracker->size; i++) {
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

// Reduce Ops

Buffer *sum(Buffer *buf) {
  // init shape on stack because it's copied anyway
  int newShape[2] = {1, 0};
  float *data = malloc(sizeof(float));
  if (!data) {
    reportMemoryError("sum data or shape");
    return NULL;
  }
  newShape[0] = 1;
  newShape[1] = 0;
  Buffer *sum = initBuffer(data, newShape, 1, false);
  for (int i = 0; i < buf->shapeTracker->size; i++) {
    sum->data[0] += buf->data[i];
  }
  return sum;
}

// Movement Ops

// assumes input with slice along each axis
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

// TODO: Figure out error handling
int *validateSliceIndices(ShapeTracker *st, int *slices) {
  for (int i = 0; i < st->ndim; i++) {
    int start = slices[i][0];
    int end = slices[i][1];
    if (start < 0 || end > st->shape[i] || start >= end) {
      fprintf(stderr, "Invalid slice indices [%d, %d] for dimension %d.\n",
              start, end, i);
      return 0;
    }
  }
  return 1;
}

Buffer *sliceBuffer(Buffer *buf, int *slices) {
  if (!buf || !slices) {
    reportMemoryError("buffer or indices");
    return NULL;
  }
  if (!validateSliceIndices(buf->shapeTracker, slices)) {
    printf("Invalid slice indices\n");
    return NULL;
  }
}

/* Buffer *indexAxisBuffer(Buffer *buf, int axis, int index) {} */
