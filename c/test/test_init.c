#include "../buffer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

ShapeTracker *mock_shapetracker_create(int *shape, int *stride, int offset,
                                       int ndim) {
  ShapeTracker *st = (ShapeTracker *)malloc(sizeof(ShapeTracker));
  st->shape = shape;
  st->stride = stride;
  st->offset = offset;
  st->ndim = ndim;
  return st;
}

void test_buffer_create() {
  printf("Testing buffer_create...\n");

  float data[] = {1.0, 2.0, 3.0, 4.0};
  int size = 4;
  int shape[] = {2, 2};
  int stride[] = {2, 1};
  ShapeTracker *st = mock_shapetracker_create(shape, stride, 0, 2);

  Buffer *buf1 = buffer_create(data, size, st, true);
  assert(buf1 != NULL);
  assert(buf1->data != data);
  assert(buf1->st == st);
  for (int i = 0; i < size; i++) {
    assert(buf1->data[i] == data[i]);
  }

  Buffer *buf2 = buffer_create(data, size, st, false);
  assert(buf2 != NULL);
  assert(buf2->data == data);
  assert(buf2->st == st);

  Buffer *buf3 = buffer_create(NULL, size, st, true);
  assert(buf3 == NULL);

  Buffer *buf4 = buffer_create(data, size, NULL, true);
  assert(buf4 == NULL);

  printf("buffer_create tests passed.\n");

  free(buf1->data);
  free(buf1);
  free(buf2);
  free(st);
}

void test_buffer_data_create() {
  printf("Testing buffer_data_create...\n");

  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  int size = 6;
  int shape[] = {2, 3};
  int ndim = 2;

  Buffer *buf1 = buffer_data_create(data, size, shape, ndim, true);
  assert(buf1 != NULL);
  assert(buf1->data != data);
  for (int i = 0; i < size; i++) {
    assert(buf1->data[i] == data[i]);
  }
  assert(buf1->st != NULL);
  assert(buf1->st->ndim == ndim);
  for (int i = 0; i < ndim; i++) {
    assert(buf1->st->shape[i] == shape[i]);
  }

  Buffer *buf2 = buffer_data_create(data, size, shape, ndim, false);
  assert(buf2 != NULL);
  assert(buf2->data == data);
  assert(buf2->st != NULL);
  assert(buf2->st->ndim == ndim);

  Buffer *buf3 = buffer_data_create(data, size, NULL, ndim, true);
  assert(buf3 == NULL);

  printf("buffer_data_create tests passed.\n");

  buffer_destroy(buf1);
  buffer_destroy(buf2);
  /* free(buf2); */
}

int main() {
  test_buffer_create();
  test_buffer_data_create();
  printf("All tests passed!\n");
  return 0;
}
