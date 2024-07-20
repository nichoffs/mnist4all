#include "../buffer.h"
#include "../ops.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

int compare_buffers(Buffer *buf1, Buffer *buf2) {
  if (buf1->size != buf2->size)
    return 0;
  for (int i = 0; i < buf1->size; i++) {
    if (fabs(buf1->data[i] - buf2->data[i]) > EPSILON)
      return 0;
  }
  return 1;
}

Buffer *create_test_buffer(float *data, int size, int *shape, int ndim) {
  return buffer_data_create(data, size, shape, ndim, true);
}

void test_square_root() {
  printf("Testing square_root operation...\n");

  // Test case 1: Basic square root
  float data1[] = {0.0f, 1.0f, 4.0f, 9.0f};
  float expected1[] = {0.0f, 1.0f, 2.0f, 3.0f};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *result1 = square_root(buf1);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Square root of zero
  float data2[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result2 = square_root(buf2);
  assert(compare_buffers(result2, buf2));

  // Test case 3: Square root of very small number
  float data3[] = {1e-10f, 1e-10f, 1e-10f, 1e-10f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result3 = square_root(buf3);
  for (int i = 0; i < 4; i++) {
    assert(fabs(result3->data[i] - sqrtf(1e-10f)) < EPSILON);
  }

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);

  printf("Square root operation tests passed!\n");
}

void test_logarithm() {
  printf("Testing logarithm operation...\n");

  // Test case 1: Basic logarithm
  float data1[] = {1.0f, 2.0f, exp(1.0f), 10.0f};
  float expected1[] = {0.0f, logf(2.0f), 1.0f, logf(10.0f)};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *result1 = logarithm(buf1);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Logarithm of 1
  float data2[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float expected2[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result2 = logarithm(buf2);
  Buffer *expected_buf2 = create_test_buffer(expected2, 4, shape1, 2);
  assert(compare_buffers(result2, expected_buf2));

  // Test case 3: Logarithm of very small number
  float data3[] = {1e-10f, 1e-10f, 1e-10f, 1e-10f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result3 = logarithm(buf3);
  for (int i = 0; i < 4; i++) {
    assert(fabs(result3->data[i] - logf(1e-10f)) < EPSILON);
  }

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);
  buffer_destroy(expected_buf2);

  printf("Logarithm operation tests passed!\n");
}

void test_exponent() {
  printf("Testing exponent operation...\n");

  // Test case 1: Basic exponent
  float data1[] = {0.0f, 1.0f, 2.0f, -1.0f};
  float expected1[] = {1.0f, exp(1.0f), exp(2.0f), exp(-1.0f)};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *result1 = exponent(buf1);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Exponent of 0
  float data2[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float expected2[] = {1.0f, 1.0f, 1.0f, 1.0f};
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result2 = exponent(buf2);
  Buffer *expected_buf2 = create_test_buffer(expected2, 4, shape1, 2);
  assert(compare_buffers(result2, expected_buf2));

  // Test case 3: Exponent of large number
  float data3[] = {10.0f, 20.0f, 30.0f, 40.0f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result3 = exponent(buf3);
  for (int i = 0; i < 4; i++) {
    assert(fabs(result3->data[i] - expf(data3[i])) < EPSILON * expf(data3[i]));
  }

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);
  buffer_destroy(expected_buf2);

  printf("Exponent operation tests passed!\n");
}

void test_unary_op_errors() {
  printf("Testing unary op error handling...\n");

  // Test case 1: NULL input
  Buffer *result = square_root(NULL);
  assert(result == NULL);

  // Test case 2: Empty buffer
  int shape[] = {0};
  Buffer *empty_buf = create_test_buffer(NULL, 0, shape, 1);
  result = square_root(empty_buf);
  assert(result == NULL);

  // Clean up
  buffer_destroy(empty_buf);
  buffer_destroy(result);

  printf("Unary op error handling tests passed!\n");
}

int main() {
  test_square_root();
  test_logarithm();
  test_exponent();
  test_unary_op_errors();
  printf("All unary operation tests passed successfully!\n");
  return 0;
}
