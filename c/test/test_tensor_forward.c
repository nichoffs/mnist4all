#include "../buffer.h"
#include "../tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

void test_forward_pass() {
  printf("Testing forward pass...\n");

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

  // Print result
  printf("Result: ");
  for (int i = 0; i < outx->buf->size; i++) {
    printf("%f ", outx->buf->data[i]);
  }
  printf("\n");

  // Expected result (calculated using numpy)
  float expected_result = -1.2609298;

  // Compare result
  assert(fabs(outx->buf->data[0] - expected_result) < EPSILON);

  printf("Forward pass test passed!\n");
}

int main() {
  test_forward_pass();
  return 0;
}
