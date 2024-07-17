#include "../buffer.h"
#include "../ops.h"
#include "../utils.h"
#include <assert.h>
#include <math.h>

int main() {

  int size = 10;
  int shape[3] = {5, 2, 0};
  float data[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  Buffer *buf = initBuffer(data, shape, size, true);
  Buffer *sum_buf = sum(buf);

  assert(sum_buf->data[0] == 45.0);

  return 0;
}
