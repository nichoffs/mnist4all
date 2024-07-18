#include "../buffer.h"
#include "../ops.h"
#include "../utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  float data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  int size = 12;
  int shape[] = {3, 4, 0};

  int start[2] = {1, 1};
  int end[2] = {3, 3};

  Buffer *buf1 = buffer_data_create(data, size, shape, 2, true);
  buffer_print(buf1);

  Buffer *slicedBuf = slice(buf1, start, end);
  shapetracker_print(buf1);

  return 0;
}
