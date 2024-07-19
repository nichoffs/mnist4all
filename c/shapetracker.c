#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ShapeTracker *shapetracker_create(int *shape, int *stride, int offset,
                                  int ndim) {
  ShapeTracker *st = (ShapeTracker *)malloc(sizeof(ShapeTracker));

  if (!st) {
    fprintf(stderr, "Cannot allocate memory for shapetracker\n");
    return NULL;
  }
  if (!shape || !stride) {
    fprintf(stderr, "Shape or stride is NULL\n");
    return NULL;
  }

  st->shape = (int *)malloc(ndim * sizeof(int));
  memcpy(st->shape, shape, ndim * sizeof(int));
  st->stride = (int *)malloc(ndim * sizeof(int));
  memcpy(st->stride, stride, ndim * sizeof(int));

  st->offset = offset;
  st->ndim = ndim;
  st->numel = _numel(shape, ndim);

  return st;
}

ShapeTracker *shapetracker_destroy(ShapeTracker *st) {
  if (!st) {
    fprintf(stderr, "ShapeTracker is NULL\n");
    return NULL;
  }
  free(st->shape);
  free(st->stride);
  free(st);
  return NULL;
}

// to convert a logical index to a physical index, use a two step process
// 1. convert logical index to coordinate system w.r.t view
// 2. perform usual indexing operation to retrieve physical index
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

int _numel(int *shape, int ndim) {
  int numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= shape[i];
  }
  return numel;
}
