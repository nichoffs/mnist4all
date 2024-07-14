#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>

int calculate_ndim(int *shape) {
  int ndim = 0;
  while (shape[ndim] != 0) {
    ndim++;
  }
  return ndim;
}

int calculate_numel(int *shape, int ndim) {
  int total = 1;
  for (int i = 0; i < ndim; i++) {
    total *= shape[i];
  }
  return total;
}

int *calculate_strides(int *shape, int ndim) {
  int *strides = (int *)malloc(ndim * sizeof(int));
  for (int i = ndim - 1; i >= 0; i--) {
    if (i == ndim - 1) {
      strides[i] = 1;
    } else {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }
  return strides;
}

ShapeTracker *createShapeTracker(int *shape) {
  ShapeTracker *st = (ShapeTracker *)malloc(sizeof(ShapeTracker));
  if (!st) {
    fprintf(stderr, "Failed to allocate memory for ShapeTracker\n");
    exit(EXIT_FAILURE);
  }

  st->shape = shape;
  st->ndim = calculate_ndim(st->shape);
  st->numel = calculate_numel(st->shape, st->ndim);
  st->strides = calculate_strides(st->shape, st->ndim);

  printf("ShapeTracker created with %d dimensions and %d elements.\n", st->ndim,
         st->numel);

  for (int i = 0; i < st->ndim; i++) {
    printf("Shape[%d] = %d, Strides[%d] = %d\n", i, st->shape[i], i,
           st->strides[i]);
  }

  return st;
}

void freeShapeTracker(ShapeTracker *st) { free(st); }
