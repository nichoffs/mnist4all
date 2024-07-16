#include "../buffer.h"
#include "../ops.h" // Ensure you include ops.h for the index function
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void perform_unary_test(Buffer *(*unary_op)(Buffer *), float *expected_data) {
  int size = 10;
  float *data = (float *)malloc(10 * sizeof(float));
  for (int i = 0; i < size; i++) {
    data[i] = i;
  }
  int *shape = (int *)malloc(3 * sizeof(int));
  shape[0] = 5.0;
  shape[1] = 2.0;
  shape[2] = 0.0;

  Buffer *buf = createBuffer(data, shape, size);
  if (!buf) {
    fprintf(stderr, "Failed to create buffer\n");
    return;
  }

  Buffer *res = unary_op(buf);
  if (!res) {
    fprintf(stderr, "Failed to perform unary operation\n");
    freeBuffer(buf);
    return;
  }

  for (int i = 0; i < size; i++) {
    assert(res->data[i] == expected_data[i]);
  }

  freeBuffer(buf);
  freeBuffer(res);
}

void perform_binary_test(Buffer *(*binary_op)(Buffer *, Buffer *),
                         float *expected_data) {
  int size = 10;
  float *data1 = malloc(size * sizeof(float));
  float *data2 = malloc(size * sizeof(float));

  for (int i = 0; i < 10; i++) {
    data1[i] = ((float)i);
    data2[i] = (10.0 - (float)i);
  }
  int *shape1 = (int *)malloc(3 * sizeof(int));
  int *shape2 = (int *)malloc(3 * sizeof(int));
  shape1[0] = 5.0;
  shape1[1] = 2.0;
  shape1[2] = 0.0;
  memcpy(shape2, shape1, 3 * sizeof(int));

  Buffer *buf1 = createBuffer(data1, shape1, size);
  Buffer *buf2 = createBuffer(data2, shape2, size);

  if (!buf1 || !buf2) {
    fprintf(stderr, "Failed to create buffers\n");
    freeBuffer(buf1);
    freeBuffer(buf2);
    return;
  }

  Buffer *res = binary_op(buf1, buf2);
  if (!res) {
    fprintf(stderr, "Failed to perform binary operation\n");
    freeBuffer(buf1);
    freeBuffer(buf2);
    return;
  }

  for (int i = 0; i < size; i++) {
    assert(res->data[i] == expected_data[i]);
  }
  freeBuffer(buf1);
  freeBuffer(buf2);
  freeBuffer(res);
}

int main() {

  // unary op tests
  float expected_data1[10];
  for (int i = 0; i < 10; i++) {
    expected_data1[i] = sqrtf((float)i);
  }
  perform_unary_test(square_root, expected_data1);
  for (int i = 0; i < 10; i++) {
    expected_data1[i] = expf((float)i);
  }
  perform_unary_test(exponent, expected_data1);
  for (int i = 0; i < 10; i++) {
    expected_data1[i] = logf((float)i);
  }
  perform_unary_test(logarithm, expected_data1);

  // binary op tests

  float expected_data2[10];
  for (int i = 0; i < 10; i++) {
    expected_data2[i] = (float)i + (10.0 - (float)i);
  }
  perform_binary_test(add, expected_data2);
  for (int i = 0; i < 10; i++) {
    expected_data2[i] = (float)i - (10.0 - (float)i);
  }
  perform_binary_test(sub, expected_data2);
  for (int i = 0; i < 10; i++) {
    expected_data2[i] = (float)i * (10.0 - (float)i);
  }
  perform_binary_test(mul, expected_data2);
  for (int i = 0; i < 10; i++) {
    expected_data2[i] = (float)i / (10.0 - (float)i);
  }
  perform_binary_test(divide, expected_data2);

  return 0;
}
