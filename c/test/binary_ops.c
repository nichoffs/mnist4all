#include "test.h"

void test_add() {

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
}

void test_sub() {

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
}

void test_mul() {

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
}

void test_divide() {

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
}

void test_binary_op_errors() {

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
}
