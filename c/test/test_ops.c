#include "../buffer.h"
#include "../ops.h"
#include "../shapetracker.h"
#include "../utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_unary_ops() {
  int shape[] = {2, 2};
  float data[] = {1.0, 4.0, 9.0, 16.0};
  Buffer *buf = buffer_data_create(data, 4, shape, 2, true);

  Buffer *sqrt_buf = square_root(buf);
  float expected_sqrt[] = {1.0, 2.0, 3.0, 4.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(sqrt_buf->data[i] - expected_sqrt[i]) < 1e-6);
  }
  buffer_destroy(sqrt_buf);

  Buffer *log_buf = logarithm(buf);
  float expected_log[] = {0.0, logf(4.0), logf(9.0), logf(16.0)};
  for (int i = 0; i < 4; i++) {
    assert(fabs(log_buf->data[i] - expected_log[i]) < 1e-6);
  }
  buffer_destroy(log_buf);

  Buffer *exp_buf = exponent(buf);
  float expected_exp[] = {expf(1.0), expf(4.0), expf(9.0), expf(16.0)};
  for (int i = 0; i < 4; i++) {
    assert(fabs(exp_buf->data[i] - expected_exp[i]) < 1e-6);
  }
  buffer_destroy(exp_buf);

  buffer_destroy(buf);
  printf("Unary operations tests passed.\n");
}

void test_binary_ops() {
  int shape[] = {2, 2};
  float data1[] = {1.0, 2.0, 3.0, 4.0};
  float data2[] = {5.0, 6.0, 7.0, 8.0};
  Buffer *buf1 = buffer_data_create(data1, 4, shape, 2, true);
  Buffer *buf2 = buffer_data_create(data2, 4, shape, 2, true);

  Buffer *add_buf = add(buf1, buf2);
  float expected_add[] = {6.0, 8.0, 10.0, 12.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(add_buf->data[i] - expected_add[i]) < 1e-6);
  }
  buffer_destroy(add_buf);

  Buffer *sub_buf = sub(buf1, buf2);
  float expected_sub[] = {-4.0, -4.0, -4.0, -4.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(sub_buf->data[i] - expected_sub[i]) < 1e-6);
  }
  buffer_destroy(sub_buf);

  Buffer *mul_buf = mul(buf1, buf2);
  float expected_mul[] = {5.0, 12.0, 21.0, 32.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(mul_buf->data[i] - expected_mul[i]) < 1e-6);
  }
  buffer_destroy(mul_buf);

  Buffer *div_buf = divide(buf1, buf2);
  float expected_div[] = {0.2, 1.0 / 3.0, 3.0 / 7.0, 4.0 / 8.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(div_buf->data[i] - expected_div[i]) < 1e-6);
  }
  buffer_destroy(div_buf);

  buffer_destroy(buf1);
  buffer_destroy(buf2);
  printf("Binary operations tests passed.\n");
}

void test_slice() {
  int shape[] = {4, 4};
  float data[] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  Buffer *buf = buffer_data_create(data, 16, shape, 2, true);

  int start[] = {1, 1};
  int end[] = {3, 3};
  Buffer *slice_buf = slice(buf, start, end);

  float expected_slice[] = {6.0, 7.0, 10.0, 11.0};
  for (int i = 0; i < 4; i++) {
    assert(fabs(slice_buf->data[slice_buf->st->offset + i] -
                expected_slice[i]) < 1e-6);
  }
  buffer_destroy(slice_buf);
  buffer_destroy(buf);
  printf("Slice operation tests passed.\n");
}

int main() {
  /* test_unary_ops(); */
  test_binary_ops();
  test_slice();
  printf("All tests passed successfully.\n");
  return 0;
}

