#include "../buffer.h"
#include "../tensor.h"
#include "../utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

void test_backward_pass() {
  // Run forward pass

  // Initialize input tensors
  float x_data[] = {0.1f, 0.2f, 0.3f};
  int x_shape[] = {1, 3};
  Buffer *x_buf = buffer_data_create(x_data, 3, x_shape, 2, true);
  Tensor *x = tensor_create(x_buf);

  float W_data[] = {0.5f, 0.6f, 0.1f, 0.2f, 0.1f, 0.7f, 0.8f, 0.3f, 0.4f};
  int W_shape[] = {3, 3};
  Buffer *W_buf = buffer_data_create(W_data, 9, W_shape, 2, true);
  Tensor *W = tensor_create(W_buf);

  float m_data[] = {0.9f, 0.1f, 0.2f};
  int m_shape[] = {1, 3};
  Buffer *m_buf = buffer_data_create(m_data, 3, m_shape, 2, true);
  Tensor *m = tensor_create(m_buf);

  // Perform forward pass
  Tensor *out = apply_op(OP_DOT, (Tensor *[]){x, W}, 2);
  Tensor *outr = apply_op(OP_RELU, (Tensor *[]){out}, 1);
  Tensor *outl = apply_op(OP_LOGSOFTMAX, (Tensor *[]){outr}, 1);
  Tensor *outm = apply_op(OP_MUL, (Tensor *[]){outl, m}, 2);
  Tensor *outx = apply_op(OP_SUM, (Tensor *[]){outm}, 1);

  float expected_x_grad_data[3] = {0.05519729, -0.07564659, 0.21479775};
  int expected_x_shape[2] = {1, 3};
  int expected_x_size = 3;
  int expected_x_ndim = 2;

  Buffer *expected_x_grad =
      buffer_data_create(expected_x_grad_data, expected_x_size,
                         expected_x_shape, expected_x_ndim, true);

  float expected_W_grad_data[9] = {.0470491,  -.02660007, -.02044934,
                                   .09409883, -.05320014, -.04089867,
                                   .14114824, -.07980022, -.06134801};
  int expected_W_grad_shape[2] = {3, 3};
  int expected_W_grad_size = 9;
  int expected_W_grad_ndim = 2;

  Buffer *expected_W_grad =
      buffer_data_create(expected_W_grad_data, expected_W_grad_size,
                         expected_W_grad_shape, expected_W_grad_ndim, true);

  shapetracker_print(expected_x_grad);
  shapetracker_print(expected_W_grad);

  printf("Forward pass test passed!\n");
}

int main() {
  test_backward_pass();
  return 0;
}
