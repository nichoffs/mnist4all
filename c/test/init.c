#include "init.h"

void test_buffer_create() {

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
}

void test_buffer_data_create() {

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
}

void test_zeros() {

  // Test case 1: 1D zeros
  int shape1[] = {5};
  Buffer *buf1 = zeros(shape1, 1);
  assert(buf1 != NULL);
  assert(buf1->size == 5);
  assert(buf1->st->ndim == 1);
  assert(buf1->st->shape[0] == 5);
  assert(buf1->st->stride[0] == 1);
  for (int i = 0; i < 5; i++) {
    assert(buf1->data[i] == 0.0f);
  }

  // Test case 2: 2D zeros
  int shape2[] = {3, 4};
  Buffer *buf2 = zeros(shape2, 2);
  assert(buf2 != NULL);
  assert(buf2->size == 12);
  assert(buf2->st->ndim == 2);
  assert(buf2->st->shape[0] == 3 && buf2->st->shape[1] == 4);
  assert(buf2->st->stride[0] == 4 && buf2->st->stride[1] == 1);
  for (int i = 0; i < 12; i++) {
    assert(buf2->data[i] == 0.0f);
  }

  // Test case 3: 3D zeros
  int shape3[] = {2, 3, 2};
  Buffer *buf3 = zeros(shape3, 3);
  assert(buf3 != NULL);
  assert(buf3->size == 12);
  assert(buf3->st->ndim == 3);
  assert(buf3->st->shape[0] == 2 && buf3->st->shape[1] == 3 &&
         buf3->st->shape[2] == 2);
  assert(buf3->st->stride[0] == 6 && buf3->st->stride[1] == 2 &&
         buf3->st->stride[2] == 1);
  for (int i = 0; i < 12; i++) {
    assert(buf3->data[i] == 0.0f);
  }

  // Test case 4: Empty shape (should return NULL)
  int shape4[] = {0};
  Buffer *buf4 = zeros(shape4, 1);
  assert(buf4 == NULL);

  // Test case 5: NULL shape (should return NULL)
  Buffer *buf5 = zeros(NULL, 2);
  assert(buf5 == NULL);

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
}

void test_uniform() {

  // Test case 1: 1D uniform with range [0.0, 1.0)
  int shape1[] = {100};
  Buffer *buf1 = uniform(shape1, 1, 0.0f, 1.0f);
  assert(buf1 != NULL);
  assert(buf1->size == 100);
  assert(buf1->st->ndim == 1);
  assert(buf1->st->shape[0] == 100);
  assert(buf1->st->stride[0] == 1);
  for (int i = 0; i < 100; i++) {
    assert(buf1->data[i] >= 0.0f && buf1->data[i] < 1.0f);
  }

  // Test case 2: 2D uniform with range [-5.0, 5.0)
  int shape2[] = {10, 10};
  Buffer *buf2 = uniform(shape2, 2, -5.0f, 5.0f);
  assert(buf2 != NULL);
  assert(buf2->size == 100);
  assert(buf2->st->ndim == 2);
  assert(buf2->st->shape[0] == 10 && buf2->st->shape[1] == 10);
  assert(buf2->st->stride[0] == 10 && buf2->st->stride[1] == 1);
  for (int i = 0; i < 100; i++) {
    assert(buf2->data[i] >= -5.0f && buf2->data[i] < 5.0f);
  }

  // Test case 3: 3D uniform with range [100.0, 200.0)
  int shape3[] = {5, 5, 5};
  Buffer *buf3 = uniform(shape3, 3, 100.0f, 200.0f);
  assert(buf3 != NULL);
  assert(buf3->size == 125);
  assert(buf3->st->ndim == 3);
  assert(buf3->st->shape[0] == 5 && buf3->st->shape[1] == 5 &&
         buf3->st->shape[2] == 5);
  assert(buf3->st->stride[0] == 25 && buf3->st->stride[1] == 5 &&
         buf3->st->stride[2] == 1);
  for (int i = 0; i < 125; i++) {
    assert(buf3->data[i] >= 100.0f && buf3->data[i] < 200.0f);
  }

  // Test case 4: Empty shape (should return NULL)
  int shape4[] = {0};
  Buffer *buf4 = uniform(shape4, 1, 0.0f, 1.0f);
  assert(buf4 == NULL);

  // Test case 5: NULL shape (should return NULL)
  Buffer *buf5 = uniform(NULL, 2, 0.0f, 1.0f);
  assert(buf5 == NULL);

  // Test case 6: Invalid range (low >= high)
  int shape6[] = {5};
  Buffer *buf6 = uniform(shape6, 1, 1.0f, 1.0f);
  assert(buf6 == NULL);

  // Clean up
  buffer_destroy(buf1);
  buffer_destroy(buf2);
  buffer_destroy(buf3);
}
