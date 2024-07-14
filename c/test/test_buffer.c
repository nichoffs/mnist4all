#include "../buffer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_buffer_creation() {
  printf("Testing buffer creation...\n");

  // Test case 1: Create a 2D buffer
  float data_2d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  int shape_2d[] = {2, 3, 0}; // 2x3 matrix, 0 terminates the shape
  Buffer *buf_2d = createBuffer(data_2d, shape_2d, 6);

  assert(buf_2d != NULL);
  assert(buf_2d->shapeTracker != NULL);
  assert(buf_2d->shapeTracker->ndim == 2);
  assert(buf_2d->shapeTracker->shape[0] == 2);
  assert(buf_2d->shapeTracker->shape[1] == 3);
  assert(buf_2d->shapeTracker->strides[0] == 3);
  assert(buf_2d->shapeTracker->strides[1] == 1);

  // Test case 2: Create a 3D buffer
  float data_3d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  int shape_3d[] = {2, 2, 2, 0}; // 2x2x2 cube
  Buffer *buf_3d = createBuffer(data_3d, shape_3d, 8);

  assert(buf_3d != NULL);
  assert(buf_3d->shapeTracker != NULL);
  assert(buf_3d->shapeTracker->ndim == 3);
  assert(buf_3d->shapeTracker->shape[0] == 2);
  assert(buf_3d->shapeTracker->shape[1] == 2);
  assert(buf_3d->shapeTracker->shape[2] == 2);
  assert(buf_3d->shapeTracker->strides[0] == 4);
  assert(buf_3d->shapeTracker->strides[1] == 2);
  assert(buf_3d->shapeTracker->strides[2] == 1);

  // Test case 3: Attempt to create a buffer with invalid shape
  float data_invalid[] = {1.0, 2.0, 3.0};
  int shape_invalid[] = {2, 2, 0}; // Shape doesn't match data size
  Buffer *buf_invalid = createBuffer(data_invalid, shape_invalid, 3);

  assert(buf_invalid == NULL); // Should fail to create

  freeBuffer(buf_2d);
  freeBuffer(buf_3d);

  printf("Buffer creation tests passed!\n");
}

void test_buffer_indexing() {
  printf("Testing buffer indexing...\n");

  // Create a 3D buffer for testing
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  int shape[] = {2, 2, 2, 0};
  Buffer *buf = createBuffer(data, shape, 8);

  // Test case 1: Valid indexing
  int indices1[] = {0, 0, 0};
  assert(index(buf, indices1) == 1.0);

  int indices2[] = {1, 1, 1};
  assert(index(buf, indices2) == 8.0);

  int indices3[] = {0, 1, 1};
  assert(index(buf, indices3) == 4.0);

  // Test case 2: Index out of bounds (should print error message)
  int indices_out_of_bounds[] = {2, 0, 0};
  float result = index(buf, indices_out_of_bounds);
  assert(result == 0.0); // Assuming 0.0 is returned for invalid index

  freeBuffer(buf);

  printf("Buffer indexing tests passed!\n");
}

void test_buffer_memory_management() {
  printf("Testing buffer memory management...\n");

  // Allocate a large number of buffers to stress test memory management
  const int num_buffers = 1000;
  Buffer *buffers[num_buffers];

  for (int i = 0; i < num_buffers; i++) {
    float *data = (float *)malloc(sizeof(float) * 100);
    for (int j = 0; j < 100; j++) {
      data[j] = (float)j;
    }
    int shape[] = {10, 10, 0};
    buffers[i] = createBuffer(data, shape, 100);
    assert(buffers[i] != NULL);
  }

  // Free all allocated buffers
  for (int i = 0; i < num_buffers; i++) {
    freeBuffer(buffers[i]);
  }

  printf("Buffer memory management tests passed!\n");
}

void test_full_like() {
  printf("Testing full_like function...\n");

  // Create an initial buffer
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  int shape[] = {2, 3, 0}; // 2x3 matrix
  Buffer *buf = createBuffer(data, shape, 6);

  // Test full_like function
  float fill_value = 7.5;
  Buffer *new_buf = full_like(buf, fill_value);

  // Verify the new buffer
  assert(new_buf != NULL);
  assert(new_buf->shapeTracker != NULL);
  assert(new_buf->shapeTracker->ndim == buf->shapeTracker->ndim);
  assert(new_buf->shapeTracker->shape[0] == buf->shapeTracker->shape[0]);
  assert(new_buf->shapeTracker->shape[1] == buf->shapeTracker->shape[1]);
  assert(new_buf->shapeTracker->strides[0] == buf->shapeTracker->strides[0]);
  assert(new_buf->shapeTracker->strides[1] == buf->shapeTracker->strides[1]);

  for (int i = 0; i < 6; i++) {
    assert(new_buf->data[i] == fill_value);
  }

  freeBuffer(buf);
  freeBuffer(new_buf);

  printf("full_like tests passed!\n");
}

int main() {
  test_buffer_creation();
  test_buffer_indexing();
  test_buffer_memory_management();
  test_full_like();

  printf("All buffer tests passed successfully!\n");
  return 0;
}
