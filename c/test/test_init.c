#include "../buffer.h"
#include "../shapetracker.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void test_buffer_create() {
  printf("Testing buffer_create...\n");

  // Test case 1: Basic creation with copy = true
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int shape1[] = {2, 2};
  int stride1[] = {2, 1};
  ShapeTracker *st1 = shapetracker_create(shape1, stride1, 0, 2);
  Buffer *buf1 = buffer_create(data1, 4, st1, true);

  assert(buf1 != NULL);
  assert(buf1->size == 4);
  assert(buf1->copy == true);
  assert(buf1->data != data1); // Should be a copy
  assert(memcmp(buf1->data, data1, 4 * sizeof(float)) == 0);
  assert(buf1->st == st1);

  // Test case 2: Creation with copy = false
  float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  ShapeTracker *st2 = shapetracker_create(shape1, stride1, 0, 2);
  Buffer *buf2 = buffer_create(data2, 4, st2, false);

  assert(buf2 != NULL);
  assert(buf2->size == 4);
  assert(buf2->copy == false);
  assert(buf2->data == data2); // Should be the same pointer
  assert(buf2->st == st2);

  // Test case 3: Creation with NULL data
  ShapeTracker *st3 = shapetracker_create(shape1, stride1, 0, 2);
  Buffer *buf3 = buffer_create(NULL, 4, st3, true);
  assert(buf3 == NULL);

  // Test case 4: Creation with NULL ShapeTracker
  Buffer *buf4 = buffer_create(data1, 4, NULL, true);
  assert(buf4 == NULL);

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  shapetracker_destroy(st3);

  printf("All buffer_create tests passed!\n");
}

void test_buffer_data_create() {
  printf("Testing buffer_data_create...\n");

  // Test case 1: Basic creation with copy = true
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int shape1[] = {2, 3};
  Buffer *buf1 = buffer_data_create(data1, 6, shape1, 2, true);

  assert(buf1 != NULL);
  assert(buf1->size == 6);
  assert(buf1->copy == true);
  assert(buf1->data != data1); // Should be a copy
  assert(memcmp(buf1->data, data1, 6 * sizeof(float)) == 0);
  assert(buf1->st != NULL);
  assert(buf1->st->ndim == 2);
  assert(buf1->st->shape[0] == 2 && buf1->st->shape[1] == 3);
  assert(buf1->st->stride[0] == 3 && buf1->st->stride[1] == 1);

  // Test case 2: Creation with copy = false
  float data2[] = {7.0f, 8.0f, 9.0f, 10.0f};
  int shape2[] = {2, 2};
  Buffer *buf2 = buffer_data_create(data2, 4, shape2, 2, false);

  assert(buf2 != NULL);
  assert(buf2->size == 4);
  assert(buf2->copy == false);
  assert(buf2->data == data2); // Should be the same pointer
  assert(buf2->st != NULL);
  assert(buf2->st->ndim == 2);
  assert(buf2->st->shape[0] == 2 && buf2->st->shape[1] == 2);
  assert(buf2->st->stride[0] == 2 && buf2->st->stride[1] == 1);

  // Test case 3: Creation with NULL data
  Buffer *buf3 = buffer_data_create(NULL, 4, shape2, 2, true);
  assert(buf3 == NULL);

  // Test case 4: Creation with NULL shape
  Buffer *buf4 = buffer_data_create(data2, 4, NULL, 2, true);
  assert(buf4 == NULL);

  // Test case 5: Creation with 1D shape
  float data5[] = {1.0f, 2.0f, 3.0f};
  int shape5[] = {3};
  Buffer *buf5 = buffer_data_create(data5, 3, shape5, 1, true);

  assert(buf5 != NULL);
  assert(buf5->size == 3);
  assert(buf5->st->ndim == 1);
  assert(buf5->st->shape[0] == 3);
  assert(buf5->st->stride[0] == 1);

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf5);

  printf("All buffer_data_create tests passed!\n");
}

int main() {
  test_buffer_create();
  test_buffer_data_create();
  printf("All buffer initialization tests passed successfully!\n");
  return 0;
}
