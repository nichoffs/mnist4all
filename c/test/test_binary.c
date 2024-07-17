#include "../buffer.h"
#include "../ops.h"
#include "../utils.h"
#include <assert.h>
#include <stdio.h>

int main() {

  int size = 10;
  int shape[3] = {5, 2, 0};
  float data1[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  float data2[10] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

  Buffer *buf1 = initBuffer(data1, shape, size, true);
  Buffer *buf2 = initBuffer(data2, shape, size, true);

  Buffer *add_buf = add(buf1, buf2);
  Buffer *sub_buf = sub(buf1, buf2);
  Buffer *mul_buf = mul(buf1, buf2);
  Buffer *div_buf = divide(buf1, buf2);

  for (int i = 0; i < size; i++) {
    assert(add_buf->data[i] == data1[i] + data2[i]);
    assert(sub_buf->data[i] == data1[i] - data2[i]);
    assert(mul_buf->data[i] == data1[i] * data2[i]);
    assert(div_buf->data[i] == data1[i] / data2[i]);
  }

  int newShape[3] = {10, 0};
  float data3[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  Buffer *buf3 = initBuffer(data3, newShape, size, true);
  Buffer *add_buf_error = add(buf1, buf3);
  assert(add_buf_error == NULL);

  return 0;
}
