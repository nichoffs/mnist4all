#include "../buffer.h"
#include "../ops.h" // Ensure you include ops.h for the index function
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  // VALIDATE SQUARE ROOT

  int size = 10;
  float *data = (float *)malloc(size * sizeof(float));
  int *shape = (int *)malloc(3 * sizeof(int));
  for (int i = 0; i < size; i++) {
    data[i] = (float)i;
  }
  shape[0] = 5;
  shape[1] = 2;
  shape[2] = 0;

  Buffer *buf = createBuffer(data, shape, size);
  Buffer *res = square_root(buf);

  for (int i = 0; i < size; i++) {
    printf("Original: %f, Square Root: %f\n", buf->data[i], res->data[i]);
  }

  return 0;
}
