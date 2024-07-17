#include "../buffer.h"
#include "../utils.h"
#include <assert.h>
#include <stdlib.h>

int main() {

  // init
  int size = 10;
  int shape[3] = {5, 2, 0};

  // stack allocated memory
  float data1[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  // heap allocated memory
  float *data2 = (float *)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    data2[i] = i;
  }

  // initialize buffers - stack allocated memory must have copy true
  Buffer *buf1 = initBuffer(data1, shape, size, true);
  Buffer *buf2 = initBuffer(data2, shape, size, true);
  Buffer *buf3 = initBuffer(data2, shape, size, false);

  // validate data insertion
  for (int i = 0; i < size; i++) {
    assert(buf1->data[i] == data1[i]);
    assert(buf2->data[i] == data1[i]);
    assert(buf3->data[i] == data1[i]);
  }

  // validate shape
  for (int i = 0; i < buf1->shapeTracker->ndim + 1; i++) {
    assert(buf1->shapeTracker->shape[i] == shape[i]);
    assert(buf2->shapeTracker->shape[i] == shape[i]);
    assert(buf3->shapeTracker->shape[i] == shape[i]);
  }

  // if copy were false, freeing would throw error because no malloc
  freeBuffer(buf1);
  freeBuffer(buf2);
  freeBuffer(buf3);

  return 0;
}
