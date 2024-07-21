#include "../buffer.h"
#include "../ops.h"
#include "../utils.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-6

// Helper function to compare buffers
int compare_buffers(Buffer *buf1, Buffer *buf2) {
  if (buf1->st->numel != buf2->st->numel) {
    printf("Number of elements mismatch: %d vs %d\n", buf1->st->numel,
           buf2->st->numel);
    return 0;
  }
  for (int i = 0; i < buf1->st->numel; i++) {
    int ix1 = view_index(buf1->st, i);
    int ix2 = view_index(buf2->st, i);
    if (fabs(buf1->data[ix1] - buf2->data[ix2]) > EPSILON) {
      printf("Value mismatch at index %d: %f vs %f\n", i, buf1->data[ix1],
             buf2->data[ix2]);
      return 0;
    }
  }
  return 1;
}

// Helper function to create a test buffer
Buffer *create_test_buffer(float *data, int size, int *shape, int ndim) {
  return buffer_data_create(data, size, shape, ndim, true);
}

void test_matrix_vector_dot() {
  printf("Testing matrix_vector_dot operation...\n");

  // Test case 1: Basic matrix-vector multiplication
  float matrix_data1[] = {1, 2, 3, 4, 5, 6};
  int matrix_shape1[] = {2, 3};
  Buffer *matrix1 = create_test_buffer(matrix_data1, 6, matrix_shape1, 2);

  float vector_data1[] = {1, 2, 3};
  int vector_shape1[] = {3};
  Buffer *vector1 = create_test_buffer(vector_data1, 3, vector_shape1, 1);

  Buffer *result1 = matrix_vector_dot(matrix1, vector1);

  float expected_data1[] = {14, 32};
  int expected_shape1[] = {2};
  Buffer *expected1 = create_test_buffer(expected_data1, 2, expected_shape1, 1);

  assert(compare_buffers(result1, expected1));

  // Test case 2: Matrix-vector multiplication with larger dimensions
  float matrix_data2[20];
  for (int i = 0; i < 20; i++)
    matrix_data2[i] = i + 1;
  int matrix_shape2[] = {4, 5};
  Buffer *matrix2 = create_test_buffer(matrix_data2, 20, matrix_shape2, 2);

  float vector_data2[] = {1, 2, 3, 4, 5};
  int vector_shape2[] = {5};
  Buffer *vector2 = create_test_buffer(vector_data2, 5, vector_shape2, 1);

  Buffer *result2 = matrix_vector_dot(matrix2, vector2);

  float expected_data2[] = {55, 130, 205, 280};
  int expected_shape2[] = {4};
  Buffer *expected2 = create_test_buffer(expected_data2, 4, expected_shape2, 1);

  assert(compare_buffers(result2, expected2));

  // Test case 3: Error handling - incompatible dimensions
  float vector_data3[] = {1, 2};
  int vector_shape3[] = {2};
  Buffer *vector3 = create_test_buffer(vector_data3, 2, vector_shape3, 1);

  Buffer *result3 = matrix_vector_dot(matrix2, vector3);
  assert(result3 == NULL);

  // Clean up
  buffer_destroy(matrix1);
  buffer_destroy(vector1);
  buffer_destroy(result1);
  buffer_destroy(expected1);
  buffer_destroy(matrix2);
  buffer_destroy(vector2);
  buffer_destroy(result2);
  buffer_destroy(expected2);
  buffer_destroy(vector3);

  printf("matrix_vector_dot tests passed!\n");
}

void test_vector_matrix_dot() {
  printf("Testing vector_matrix_dot operation...\n");

  // Test case 1: Basic vector-matrix multiplication
  float vector_data1[] = {1, 2};
  int vector_shape1[] = {1, 2};
  Buffer *vector1 = create_test_buffer(vector_data1, 2, vector_shape1, 2);

  float matrix_data1[] = {1, 2, 3, 4, 5, 6};
  int matrix_shape1[] = {2, 3};
  Buffer *matrix1 = create_test_buffer(matrix_data1, 6, matrix_shape1, 2);

  Buffer *result1 = vector_matrix_dot(vector1, matrix1);

  float expected_data1[] = {9, 12, 15};
  int expected_shape1[] = {1, 3};
  Buffer *expected1 = create_test_buffer(expected_data1, 3, expected_shape1, 2);

  assert(compare_buffers(result1, expected1));

  // Test case 2: Vector-matrix multiplication with larger dimensions
  float vector_data2[] = {1, 2, 3, 4};
  int vector_shape2[] = {1, 4};
  Buffer *vector2 = create_test_buffer(vector_data2, 4, vector_shape2, 2);

  float matrix_data2[20];
  for (int i = 0; i < 20; i++)
    matrix_data2[i] = i + 1;
  int matrix_shape2[] = {4, 5};
  Buffer *matrix2 = create_test_buffer(matrix_data2, 20, matrix_shape2, 2);

  Buffer *result2 = vector_matrix_dot(vector2, matrix2);

  float expected_data2[] = {110, 120, 130, 140, 150};
  int expected_shape2[] = {1, 5};
  Buffer *expected2 = create_test_buffer(expected_data2, 5, expected_shape2, 2);

  assert(compare_buffers(result2, expected2));

  // Test case 3: Error handling - incompatible dimensions
  float vector_data3[] = {1, 2, 3};
  int vector_shape3[] = {1, 3};
  Buffer *vector3 = create_test_buffer(vector_data3, 3, vector_shape3, 2);

  Buffer *result3 = vector_matrix_dot(vector3, matrix2);
  assert(result3 == NULL);

  // Test case 4: Error handling - incorrect vector shape
  float vector_data4[] = {1, 2};
  int vector_shape4[] = {2, 1};
  Buffer *vector4 = create_test_buffer(vector_data4, 2, vector_shape4, 2);

  Buffer *result4 = vector_matrix_dot(vector4, matrix1);
  assert(result4 == NULL);

  // Clean up
  buffer_destroy(vector1);
  buffer_destroy(matrix1);
  buffer_destroy(result1);
  buffer_destroy(expected1);
  buffer_destroy(vector2);
  buffer_destroy(matrix2);
  buffer_destroy(result2);
  buffer_destroy(expected2);
  buffer_destroy(vector3);
  buffer_destroy(vector4);

  printf("vector_matrix_dot tests passed!\n");
}

int main() {
  test_matrix_vector_dot();
  test_vector_matrix_dot();
  printf("All matrix and vector dot product tests passed successfully!\n");
  return 0;
}
