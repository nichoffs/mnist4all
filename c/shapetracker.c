#include "shapetracker.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// strides are reverse cumprod of shapes
int *_default_strides(int *shape, int ndim) {
  int *strides = (int *)malloc(ndim * sizeof(int));
  if (!strides) {
    return NULL; // Return NULL if memory allocation fails
  }
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

int _calculate_ndim(int *shape) {
  int ndim = 0;
  while (shape[ndim] != 0) {
    ndim++;
  }
  return ndim;
}

// Shape attribute data is always copied
ShapeTracker *initShapeTracker(int *shape, int size) {

  ShapeTracker *st = malloc(sizeof(ShapeTracker));
  if (st == NULL) {
    reportMemoryError("ShapeTracker");
    return NULL;
  }

  st->ndim = _calculate_ndim(shape);
  st->shape = malloc((st->ndim + 1) * sizeof(int));
  if (st->shape == NULL) {
    free(st);
    reportMemoryError("shape");
    return NULL;
  }
  memcpy(st->shape, shape, (st->ndim + 1) * sizeof(int));

  st->strides = _default_strides(shape, st->ndim);
  if (st->strides == NULL) {
    free(st->shape);
    free(st);
    reportMemoryError("strides");
    return NULL;
  }

  if (!verifyShape(shape, st->ndim, size)) {
    freeShapeTracker(st);
    return NULL;
  }

  st->size = size;
  return st;
}

void freeShapeTracker(ShapeTracker *st) {
  if (st) {
    free(st->shape);
    free(st->strides); // Ensure to free strides
    free(st);          // Then free the ShapeTracker itself
  }
}
