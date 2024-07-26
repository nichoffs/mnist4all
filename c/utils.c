#include "utils.h"

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
  fprintf(stderr, "Operation: ");
  switch (op) {
  case OP_MUL:
    fprintf(stderr, "Multiplication\n");
    break;
  case OP_RELU:
    fprintf(stderr, "ReLU\n");
    break;
  case OP_DOT:
    fprintf(stderr, "Dot Product\n");
    break;
  case OP_SUM:
    fprintf(stderr, "Sum\n");
    break;
  case OP_LOGSOFTMAX:
    fprintf(stderr, "Log Softmax\n");
    break;
  case OP_NLL:
    fprintf(stderr, "NLL\n");
    break;
  default:
    fprintf(stderr, "Unknown (%d)\n", op);
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
  printf("Context:\n");
  op_print(ctx->op);
  printf("Inputs:\n");
  for (int i = 0; i < ctx->num_inputs; i++) {
    printf("Input %d:\n", i);
    shape_print(ctx->inputs[i]->buf);
  }
}
