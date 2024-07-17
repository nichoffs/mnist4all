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
  Buffer *sqrt_buf = square_root(buf);
  Buffer *log_buf = logarithm(buf);
  Buffer *exp_buf = exponent(buf);

  for (int i = 0; i < size; i++) {
    assert(sqrt_buf->data[i] == sqrtf(data[i]));
    assert(log_buf->data[i] == logf(data[i]));
    assert(exp_buf->data[i] == expf(data[i]));
  }

  return 0;
}
