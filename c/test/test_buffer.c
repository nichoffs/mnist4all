#include "../buffer.h"
#include "../ops.h" // Ensure you include ops.h for the index function
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

  // VALIDATE CREATION

  int size_1 = 4;
  float *data_1 = (float *)malloc(size_1 * sizeof(float));
  for (int i = 0; i < size_1; i++) {
    data_1[i] = i + 1.0;
  }

  int *shape_1 = (int *)malloc(3 * sizeof(int));
  shape_1[0] = 2;
  shape_1[1] = 2;
  shape_1[2] = 0;

  Buffer *buf_1 = createBuffer(data_1, shape_1, size_1);

  assert(buf_1 != NULL);
  assert(buf_1->shapeTracker != NULL);
  assert(buf_1->shapeTracker->ndim == 2);
  assert(buf_1->shapeTracker->shape[0] == 2);
  assert(buf_1->shapeTracker->shape[1] == 2);
  assert(buf_1->shapeTracker->strides[0] == 2);
  assert(buf_1->shapeTracker->strides[1] == 1);
  assert(buf_1->data[0] == 1.0);
  assert(buf_1->data[1] == 2.0);
  assert(buf_1->data[2] == 3.0);
  assert(buf_1->data[3] == 4.0);
  assert(buf_1->shapeTracker->size == 4);

  // INVALID SHAPE CREATION

  int size_2 = 4;
  float *data_2 = (float *)malloc(size_2 * sizeof(float));
  for (int i = 0; i < size_2; i++) {
    data_2[i] = i + 1.0;
  }

  int *shape_2 = (int *)malloc(3 * sizeof(int));
  shape_2[0] = 2;
  shape_2[1] = 1;
  shape_2[2] = 0;

  Buffer *buf_2 = createBuffer(data_2, shape_2, size_2);

  assert(buf_2 == NULL);

  // VALIDATE FULL_LIKE

  int size_3 = 6;
  int *shape_3 = (int *)malloc(3 * sizeof(int));
  shape_3[0] = 2;
  shape_3[1] = 3;
  shape_3[2] = 0;
  float *data_3 = (float *)malloc(size_3 * sizeof(float));
  Buffer *buf_3_orig = createBuffer(data_3, shape_3, size_3);

  Buffer *buf_3_like = full_like(buf_3_orig, 5);

  assert(buf_3_like != NULL);
  assert(buf_3_like->shapeTracker != NULL);
  assert(buf_3_like->shapeTracker->ndim == 2);
  assert(buf_3_like->shapeTracker->shape[0] == 2);
  assert(buf_3_like->shapeTracker->shape[1] == 3);
  assert(buf_3_like->shapeTracker->strides[0] == 3);
  assert(buf_3_like->shapeTracker->strides[1] == 1);
  assert(buf_3_like->data[0] == 5.0);
  assert(buf_3_like->data[1] == 5.0);
  assert(buf_3_like->data[2] == 5.0);
  assert(buf_3_like->data[3] == 5.0);
  assert(buf_3_like->shapeTracker->size == 6);

  // VALIDATE RANDINT

  int size_4 = 6;
  int *shape_4 = (int *)malloc(3 * sizeof(int));
  shape_4[0] = 2;
  shape_4[1] = 3;
  shape_4[2] = 0;
  Buffer *buf_4 = randint(shape_4, size_4, 3, 5);

  assert(buf_4 != NULL);
  assert(buf_4->shapeTracker != NULL);
  assert(buf_4->shapeTracker->ndim == 2);
  assert(buf_4->shapeTracker->shape[0] == 2);
  assert(buf_4->shapeTracker->shape[1] == 3);
  assert(buf_4->shapeTracker->strides[0] == 3);
  assert(buf_4->shapeTracker->strides[1] == 1);
  assert(buf_4->shapeTracker->size == 6);
  for (int i = 0; i < size_4; i++) {
    printf("%f\n", buf_4->data[i]);
  }

  return 0;
}
