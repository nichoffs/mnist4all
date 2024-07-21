#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper functions

static int calculate_numel(int *shape, int ndim) {
  int numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= shape[i];
  }
  return numel;
}

static int *allocate_and_copy(int *source, int size) {
  int *dest = (int *)malloc(size * sizeof(int));
  if (!dest) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }
  memcpy(dest, source, size * sizeof(int));
  return dest;
}

// ShapeTracker functions

ShapeTracker *shapetracker_create(int *shape, int *stride, int offset,
                                  int ndim) {
  if (!shape || !stride) {
    fprintf(stderr, "Shape or stride is NULL\n");
    return NULL;
  }

  ShapeTracker *st = (ShapeTracker *)malloc(sizeof(ShapeTracker));
  if (!st) {
    fprintf(stderr, "Cannot allocate memory for ShapeTracker\n");
    return NULL;
  }

  st->shape = allocate_and_copy(shape, ndim);
  st->stride = allocate_and_copy(stride, ndim);

  if (!st->shape || !st->stride) {
    shapetracker_destroy(st);
    return NULL;
  }

  st->offset = offset;
  st->ndim = ndim;
  st->numel = calculate_numel(shape, ndim);

  return st;
}

void shapetracker_destroy(ShapeTracker *st) {
  if (st) {
    free(st->shape);
    free(st->stride);
    free(st);
  }
}

int view_index(ShapeTracker *st, int numel) {
  if (!st || numel < 0 || numel >= st->numel) {
    return -1;
  }

  int physical_index = st->offset;
  int remaining = numel;

  for (int i = 0; i < st->ndim; i++) {
    int prod = 1;
    for (int j = i + 1; j < st->ndim; j++) {
      prod *= st->shape[j];
    }
    int coord = remaining / prod;
    remaining %= prod;
    physical_index += coord * st->stride[i];
  }

  return physical_index;
}
