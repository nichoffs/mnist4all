#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ShapeTracker *copyShapeTracker(ShapeTracker *st) {
  int *newShape = (int *)malloc((st->ndim + 1) * sizeof(int));
  memcpy(newShape, st->shape, (st->ndim + 1) * sizeof(int));
  ShapeTracker *newShapeTracker = createShapeTracker(newShape, st->size);
  return newShapeTracker;
}

// strides are reverse cumprod of shapes
int *calculate_strides(int *shape, int ndim) {
  int *strides = (int *)malloc(ndim * sizeof(int));
  if (!strides) {
    return NULL; // Return NULL if memory allocation fails
  }
  for (int i = ndim - 1; i >= 0; i--) {
    if (i == ndim - 1) {
      strides[i] = 1;
    } else {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }
  return strides;
}

int calculate_ndim(int *shape) {
  int ndim = 0;
  while (shape[ndim] != 0) {
    ndim++;
  }
  return ndim;
}

int verifyShape(int *shape, int ndim, int size) {
  int numEl = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "The shape must have positive values.\n");
      return 0; // Return 0 to indicate error
    }
    numEl *= shape[i];
  }
  if (numEl != size) {
    fprintf(stderr, "The number of elements in the shape does not match the "
                    "size of the buffer.\n");
    return 0; // Return 0 to indicate error
  }
  return 1; // Return 1 on success
}

ShapeTracker *createShapeTracker(int *shape, int size) {
  ShapeTracker *st = (ShapeTracker *)malloc(sizeof(ShapeTracker));
  if (!st) {
    fprintf(stderr, "Failed to allocate memory for ShapeTracker\n");
    return NULL; // Return NULL on allocation failure
  }

  st->ndim = calculate_ndim(shape);
  st->shape = shape;
  st->size = size;
  st->strides = calculate_strides(shape, st->ndim);
  if (!st->strides) {
    free(st->shape);
    free(st);
    return NULL;
  }

  if (!verifyShape(shape, st->ndim, size)) {
    freeShapeTracker(st);
    return NULL;
  }

  return st;
}

void freeShapeTracker(ShapeTracker *st) {
  if (st) {
    free(st->shape);
    free(st->strides); // Ensure to free strides
    free(st);          // Then free the ShapeTracker itself
  }
}
