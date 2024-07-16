#include "../buffer.h"
#include "../ops.h" // Ensure you include ops.h for the index function
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  // VALIDATE ADDITION

  // INITIALIZE ADDING ARRAYS
  float *data1 = (float *)malloc(size * sizeof(float));
  float *data2 = (float *)malloc(size * sizeof(float));
  int *shape1 = (int *)malloc(3 * sizeof(int));
  int *shape2 = (int *)malloc(3 * sizeof(int));
  memcpy(shape1, shape, 3 * sizeof(int));
  memcpy(shape2, shape, 3 * sizeof(int));
  Buffer *buf1 = createBuffer(data1, shape1, size);
  Buffer *buf2 = createBuffer(data2, shape2, size);

  // PERFORM ADDITION - ADDS SHOULD EQUAL 10

  Buffer *add_res = add(buf1, buf2);
  for (int i = 0; i < size; i++) {
    printf("Buf1: %f, Buf2: %f, Sum: %f\n", buf1->data[i], buf2->data[i],
           add_res->data[i]);
  }

  freeBuffer(buf);
  freeBuffer(res);
  freeBuffer(buf1);
  freeBuffer(buf2);
  freeBuffer(add_res);

  return 0;
}
