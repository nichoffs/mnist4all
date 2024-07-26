#include "test.h"
#include "binary_ops.h"
#include "dot.h"
#include "movement.h"
#include "tensor_backward.h"
#include "tensor_forward.h"
#include "unary_ops.h"

#define EPSILON 1e-6

int compare_buffers(Buffer *buf1, Buffer *buf2) {
  if (buf1->st->numel != buf2->st->numel)
    return 0;
  for (int i = 0; i < buf1->st->numel; i++) {
    int ix1 = view_index(buf1->st, i);
    int ix2 = view_index(buf2->st, i);
    if (fabs(buf1->data[ix1] - buf2->data[ix2]) > EPSILON)
      return 0;
  }
  return 1;
}

Buffer *create_test_buffer(float *data, int size, int *shape, int ndim) {
  return buffer_data_create(data, size, shape, ndim, true);
}

int main() {
  // Binary ops tests
  test_add();
  test_sub();
  test_mul();
  test_divide();
  test_binary_op_errors();

  printf("Binary ops tests passed!\n");

  // Unary ops tests
  test_square_root();
  test_logarithm();
  test_exponent();
  test_unary_op_errors();
  test_log_softmax();

  printf("Unary ops tests passed!\n");

  // Movement tests
  test_slice();
  test_transpose();
  test_flatten();
  test_unsqueeze();
  test_expand();

  printf("Movement ops tests passed!\n");

  // Tensor tests
  test_forward_pass();
  test_backward_pass();

  printf("Tensor tests passed!\n");

  test_dot();

  printf("Dot tests passed!\n");

  printf("All tests passed!\n");
  return 0;
}
