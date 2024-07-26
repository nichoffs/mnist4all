#include "movement.h"

void test_slice() {

  // Test case 1: 2D slice
  float data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int shape1[] = {3, 3};
  Buffer *buf1 = create_test_buffer(data1, 9, shape1, 2);

  int start1[] = {1, 1};
  int end1[] = {3, 3};
  Buffer *result1 = slice(buf1, start1, end1);

  float expected1[] = {5, 6, 8, 9};
  int expected_shape1[] = {2, 2};
  Buffer *expected_buf1 = create_test_buffer(expected1, 4, expected_shape1, 2);

  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: 1D slice
  float data2[] = {1, 2, 3, 4, 5};
  int shape2[] = {5};
  Buffer *buf2 = create_test_buffer(data2, 5, shape2, 1);

  int start2[] = {1};
  int end2[] = {4};
  Buffer *result2 = slice(buf2, start2, end2);

  float expected2[] = {2, 3, 4};
  int expected_shape2[] = {3};
  Buffer *expected_buf2 = create_test_buffer(expected2, 3, expected_shape2, 1);

  assert(compare_buffers(result2, expected_buf2));

  // Test case 3: 3D slice
  float data3[27];
  for (int i = 0; i < 27; i++)
    data3[i] = i + 1;
  int shape3[] = {3, 3, 3};
  Buffer *buf3 = create_test_buffer(data3, 27, shape3, 3);

  int start3[] = {1, 0, 1};
  int end3[] = {3, 2, 3};
  Buffer *result3 = slice(buf3, start3, end3);

  float expected3[] = {11, 12, 14, 15, 20, 21, 23, 24};
  int expected_shape3[] = {2, 2, 2};
  Buffer *expected_buf3 = create_test_buffer(expected3, 8, expected_shape3, 3);

  assert(compare_buffers(result3, expected_buf3));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(result1);
  buffer_destroy(expected_buf1);
  buffer_destroy(buf2);
  buffer_destroy(result2);
  buffer_destroy(expected_buf2);
  buffer_destroy(buf3);
  buffer_destroy(result3);
  buffer_destroy(expected_buf3);
}

void test_transpose() {

  // Test case 1: 2x3 matrix
  float data1[] = {1, 2, 3, 4, 5, 6};
  int shape1[] = {2, 3};
  Buffer *buf1 = create_test_buffer(data1, 6, shape1, 2);

  Buffer *result1 = T(buf1);

  float expected1[] = {1, 4, 2, 5, 3, 6};
  int expected_shape1[] = {3, 2};
  Buffer *expected_buf1 = create_test_buffer(expected1, 6, expected_shape1, 2);

  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: 3x2 matrix
  float data2[] = {1, 2, 3, 4, 5, 6};
  int shape2[] = {3, 2};
  Buffer *buf2 = create_test_buffer(data2, 6, shape2, 2);

  Buffer *result2 = T(buf2);

  float expected2[] = {1, 3, 5, 2, 4, 6};
  int expected_shape2[] = {2, 3};
  Buffer *expected_buf2 = create_test_buffer(expected2, 6, expected_shape2, 2);

  assert(compare_buffers(result2, expected_buf2));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(result1);
  buffer_destroy(expected_buf1);
  buffer_destroy(buf2);
  buffer_destroy(result2);
  buffer_destroy(expected_buf2);
}

void test_flatten() {

  // Test case 1: 2D array
  float data1[] = {1, 2, 3, 4, 5, 6};
  int shape1[] = {2, 3};
  Buffer *buf1 = create_test_buffer(data1, 6, shape1, 2);

  Buffer *result1 = flatten(buf1);

  float expected1[] = {1, 2, 3, 4, 5, 6};
  int expected_shape1[] = {6};
  Buffer *expected_buf1 = create_test_buffer(expected1, 6, expected_shape1, 1);

  assert(compare_buffers(result1, expected_buf1));
  assert(result1->st->ndim == 1);
  assert(result1->st->shape[0] == 6);

  // Test case 2: 3D array
  float data2[24];
  for (int i = 0; i < 24; i++)
    data2[i] = i + 1;
  int shape2[] = {2, 3, 4};
  Buffer *buf2 = create_test_buffer(data2, 24, shape2, 3);

  Buffer *result2 = flatten(buf2);

  float expected2[24];
  for (int i = 0; i < 24; i++)
    expected2[i] = i + 1;
  int expected_shape2[] = {24};
  Buffer *expected_buf2 = create_test_buffer(expected2, 24, expected_shape2, 1);

  assert(compare_buffers(result2, expected_buf2));
  assert(result2->st->ndim == 1);
  assert(result2->st->shape[0] == 24);

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(result1);
  buffer_destroy(expected_buf1);
  buffer_destroy(buf2);
  buffer_destroy(result2);
  buffer_destroy(expected_buf2);
}

void test_unsqueeze() {

  // Test case 1: 1D array, unsqueeze at beginning
  float data1[] = {1, 2, 3, 4};
  int shape1[] = {4};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 1);

  Buffer *result1 = unsqueeze(buf1, 0);

  assert(result1->st->ndim == 2);
  assert(result1->st->shape[0] == 1);
  assert(result1->st->shape[1] == 4);
  assert(compare_buffers(result1, buf1));

  // Test case 2: 2D array, unsqueeze in middle
  float data2[] = {1, 2, 3, 4, 5, 6};
  int shape2[] = {2, 3};
  Buffer *buf2 = create_test_buffer(data2, 6, shape2, 2);

  Buffer *result2 = unsqueeze(buf2, 1);

  assert(result2->st->ndim == 3);
  assert(result2->st->shape[0] == 2);
  assert(result2->st->shape[1] == 1);
  assert(result2->st->shape[2] == 3);
  assert(compare_buffers(result2, buf2));

  // Test case 3: 3D array, unsqueeze at end
  float data3[24];
  for (int i = 0; i < 24; i++)
    data3[i] = i + 1;
  int shape3[] = {2, 3, 4};
  Buffer *buf3 = create_test_buffer(data3, 24, shape3, 3);

  Buffer *result3 = unsqueeze(buf3, 3);

  assert(result3->st->ndim == 4);
  assert(result3->st->shape[0] == 2);
  assert(result3->st->shape[1] == 3);
  assert(result3->st->shape[2] == 4);
  assert(result3->st->shape[3] == 1);
  assert(compare_buffers(result3, buf3));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(result1);
  buffer_destroy(buf2);
  buffer_destroy(result2);
  buffer_destroy(buf3);
  buffer_destroy(result3);
}

void test_expand() {

  // Test case 1: 2D array, expand second dimension
  float data1[] = {1, 2, 3, 4};
  int shape1[] = {4, 1};
  Buffer *buf1 = create_test_buffer(data1, 4, shape1, 2);

  Buffer *result1 = expand(buf1, 1, 2);

  assert(result1 != NULL);
  assert(result1->st->ndim == 2);
  assert(result1->st->shape[0] == 4);
  assert(result1->st->shape[1] == 2);
  assert(result1->st->stride[1] == 0);

  float expected1[] = {1, 1, 2, 2, 3, 3, 4, 4};
  int expected_shape1[] = {4, 2};
  Buffer *expected_buf1 = create_test_buffer(expected1, 8, expected_shape1, 2);

  assert(compare_buffers(result1, expected_buf1));

  // Test case 2: 3D array, expand middle dimension
  float data2[] = {1, 2, 3, 4, 5, 6};
  int shape2[] = {2, 1, 3};
  Buffer *buf2 = create_test_buffer(data2, 6, shape2, 3);

  Buffer *result2 = expand(buf2, 1, 4);

  assert(result2 != NULL);
  assert(result2->st->ndim == 3);
  assert(result2->st->shape[0] == 2);
  assert(result2->st->shape[1] == 4);
  assert(result2->st->shape[2] == 3);
  assert(result2->st->stride[1] == 0);

  float expected2[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                       4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6};
  int expected_shape2[] = {2, 4, 3};
  Buffer *expected_buf2 = create_test_buffer(expected2, 24, expected_shape2, 3);

  assert(compare_buffers(result2, expected_buf2));

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(result1);
  buffer_destroy(expected_buf1);
  buffer_destroy(buf2);
  buffer_destroy(result2);
  buffer_destroy(expected_buf2);
}
