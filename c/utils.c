#include "utils.h"
#include "tensor.h"
#include <stdio.h>

static void print_indent(int depth) {
  for (int i = 0; i < depth; i++) {
    printf(" ");
  }
}

static void data_print(Buffer *buf, int dim, int offset) {
  int ndim = buf->st->ndim;
  int *shape = buf->st->shape;
  int *stride = buf->st->stride;
  float *data = buf->data;

  printf("[");

  if (dim == ndim - 1) {
    for (int i = 0; i < shape[dim]; i++) {
      printf("%f", data[offset + i * stride[dim]]);
      if (i < shape[dim] - 1) {
        printf(", ");
      }
    }
  } else {
    for (int i = 0; i < shape[dim]; i++) {
      if (i > 0) {
        printf(",\n");
        print_indent(dim + 1);
      }
      data_print(buf, dim + 1, offset + i * stride[dim]);
    }
  }

  printf("]");
}

static void print_buffer_content(Buffer *buf) {
  if (buf->st->ndim == 0) {
    printf("[]\n");
  } else {
    data_print(buf, 0, buf->st->offset);
    printf("\n");
  }
}

static bool is_buffer_valid(Buffer *buf) {
  if (!buf || !buf->data || !buf->st) {
    fprintf(stderr, "Buffer or its data/ShapeTracker is NULL\n");
    return false;
  }
  return true;
}

void buffer_print(Buffer *buf) {
  if (!is_buffer_valid(buf))
    return;
  print_buffer_content(buf);
}

void shapetracker_print(Buffer *buf) {
  if (!is_buffer_valid(buf))
    return;
  print_buffer_content(buf);
}

void op_print(OpType op) {
  printf("Operation: ");
  switch (op) {
  case OP_MUL:
    printf("Multiplication\n");
    break;
  case OP_RELU:
    printf("ReLU\n");
    break;
  case OP_DOT:
    printf("Dot Product\n");
    break;
  case OP_SUM:
    printf("Sum\n");
    break;
  case OP_LOGSOFTMAX:
    printf("Log Softmax\n");
    break;
  case OP_NLL:
    printf("NLL\n");
    break;
  default:
    printf("Unknown\n");
  }
}

void shape_print(Buffer *buf) {
  if (!buf) {
    printf("Shape can't be printed -- buffer null\n");
  }
  printf("(");
  for (int i = 0; i < buf->st->ndim; i++) {
    printf("%d", buf->st->shape[i]);
    if (i < buf->st->ndim - 1) {
      printf(", ");
    }
  }
  printf(")\n");
}

void context_print(Context *ctx) {
  if (!ctx) {
    fprintf(stderr, "Context is NULL\n");
    return;
  }

  printf("Context:\n");
  op_print(ctx->op);

  printf("  Number of parents: %d\n", ctx->num_parents);
  printf("  Number of saved tensors: %d\n", ctx->num_saved_buffers);

  if (ctx->saved_buffers && ctx->num_saved_buffers > 0) {
    printf("  Saved tensors:\n");
    for (int i = 0; i < ctx->num_saved_buffers; i++) {
      printf("    Tensor %d:\n", i);
      shape_print(ctx->saved_buffers[i]);
      /* shapetracker_print(ctx->saved_buffers[i]); */
    }
  }
}
