#include "../buffer.h"
#include "../ops.h"
#include "../shapetracker.h"
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

void test_add() {
  printf("Testing add operation...\n");

  // Test case 1: Basic addition
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float expected1[] = {6.0f, 8.0f, 10.0f, 12.0f};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result1 = add(buf1, buf2);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Addition with zero
  float data3[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result2 = add(buf1, buf3);
  assert(compare_buffers(result2, buf1));

  // Test case 3: Addition with negative numbers
  float data4[] = {-1.0f, -2.0f, -3.0f, -4.0f};
  float expected3[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf4 = create_test_buffer(data4, 4, shape1, 2);
  Buffer *result3 = add(buf1, buf4);
  Buffer *expected_buf3 = create_test_buffer(expected3, 4, shape1, 2);
  assert(compare_buffers(result3, expected_buf3));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(buf4);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);
  buffer_destroy(expected_buf3);

  printf("Add operation tests passed!\n");
}

void test_sub() {
  printf("Testing subtract operation...\n");

  // Test case 1: Basic subtraction
  float data1[] = {10.0f, 8.0f, 6.0f, 4.0f};
  float data2[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float expected1[] = {9.0f, 6.0f, 3.0f, 0.0f};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result1 = sub(buf1, buf2);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Subtraction with zero
  float data3[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result2 = sub(buf1, buf3);
  assert(compare_buffers(result2, buf1));

  // Test case 3: Subtraction resulting in negative numbers
  float expected3[] = {-9.0f, -6.0f, -3.0f, 0.0f};
  Buffer *result3 = sub(buf2, buf1);
  Buffer *expected_buf3 = create_test_buffer(expected3, 4, shape1, 2);
  assert(compare_buffers(result3, expected_buf3));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);
  buffer_destroy(expected_buf3);

  printf("Subtract operation tests passed!\n");
}

void test_mul() {
  printf("Testing multiply operation...\n");

  // Test case 1: Basic multiplication
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float expected1[] = {5.0f, 12.0f, 21.0f, 32.0f};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result1 = mul(buf1, buf2);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Multiplication by zero
  float data3[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float expected2[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result2 = mul(buf1, buf3);
  Buffer *expected_buf2 = create_test_buffer(expected2, 4, shape1, 2);
  assert(compare_buffers(result2, expected_buf2));

  // Test case 3: Multiplication by one
  float data4[] = {1.0f, 1.0f, 1.0f, 1.0f};
  Buffer *buf4 = create_test_buffer(data4, 4, shape1, 2);
  Buffer *result3 = mul(buf1, buf4);
  assert(compare_buffers(result3, buf1));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(buf4);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);
  buffer_destroy(expected_buf2);

  printf("Multiply operation tests passed!\n");
}

void test_divide() {
  printf("Testing divide operation...\n");

  // Test case 1: Basic division
  float data1[] = {10.0f, 8.0f, 6.0f, 4.0f};
  float data2[] = {2.0f, 2.0f, 2.0f, 2.0f};
  float expected1[] = {5.0f, 4.0f, 3.0f, 2.0f};
  int shape1[] = {2, 2};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);
  Buffer *buf2 = create_test_buffer(data2, 4, shape1, 2);
  Buffer *result1 = divide(buf1, buf2);
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, shape1, 2);
  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: Division by one
  float data3[] = {1.0f, 1.0f, 1.0f, 1.0f};
  Buffer *buf3 = create_test_buffer(data3, 4, shape1, 2);
  Buffer *result2 = divide(buf1, buf3);
  assert(compare_buffers(result2, buf1));

  // Test case 3: Division by zero (check for infinity)
  float data4[] = {0.0f, 0.0f, 0.0f, 0.0f};
  Buffer *buf4 = create_test_buffer(data4, 4, shape1, 2);
  Buffer *result3 = divide(buf1, buf4);
  for (int i = 0; i < 4; i++) {
    assert(isinf(result3->data[i]));
  }

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
  buffer_destroy(buf4);
  buffer_destroy(result1);
  buffer_destroy(result2);
  buffer_destroy(result3);
  buffer_destroy(expected_buf1);

  printf("Divide operation tests passed!\n");
}

void test_binary_op_errors() {
  printf("Testing binary op error handling...\n");

  // Test case 1: NULL input
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int shape[] = {2, 2};
  Buffer *buf = create_test_buffer(data, 4, shape, 2);
  Buffer *result = add(NULL, buf);
  assert(result == NULL);
  result = add(buf, NULL);
  assert(result == NULL);

  // Test case 2: Mismatched shapes
  float data2[] = {1.0f, 2.0f};
  int shape2[] = {2, 1};
  Buffer *buf2 = create_test_buffer(data2, 2, shape2, 2);
  result = add(buf, buf2);
  assert(result == NULL);

  // Clean up
  buffer_destroy(buf);
  buffer_destroy(buf2);

  printf("Binary op error handling tests passed!\n");
}

int main() {
  test_add();
  test_sub();
  test_mul();
  test_divide();
  test_binary_op_errors();
  printf("All binary operation tests passed successfully!\n");
  return 0;
}
