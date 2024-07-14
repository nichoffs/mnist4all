#include "buffer.h"
#include "shapetracker.h"
#include <stdio.h>
#include <stdlib.h>

Buffer *createBuffer(float *data, int *shape) {
  Buffer *x = (Buffer *)malloc(sizeof(Buffer));
  if (!x) {
    fprintf(stderr, "Failed to allocate memory for Buffer\n");
    exit(EXIT_FAILURE);
  }

  x->data = data;
  x->shapeTracker = createShapeTracker(shape);

  return x;
}

void freeBuffer(Buffer *buffer) {
  freeShapeTracker(buffer->shapeTracker);
  free(buffer);
}

int main() {
  int shape[] = {2, 2, 3, 0}; // Example shape with 0 as the terminator
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Example data

  Buffer *buffer = createBuffer(data, shape);

  if (buffer) {
    printf("Buffer created successfully.\n");
  } else {
    printf("Failed to create buffer.\n");
  }

  // Remember to free the allocated memory
  freeBuffer(buffer);

  return 0;
}
