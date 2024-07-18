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
  Buffer *ret = buffer_create(buf->data, buf->size, buf->st, true);
  for (int i = 0; i < buf->st->numel; i++) {
    ret->data[i] = uop(buf->data[i]);
  }
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

  Buffer *ret = buffer_create(buf1->data, buf1->size, buf1->st, true);

  for (int i = 0; i < ret->size; i++) {
    ret->data[i] = op_func(buf1->data[i], buf2->data[i]);
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
