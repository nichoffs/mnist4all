#include "../buffer.h"
#include "../tensor.h"
#include "../utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

void test_log_softmax_backward() {
  printf("Testing forward pass...\n");

  // Initialize input tensors
  float x_data[] = {1, 2, 3};
  int x_shape[] = {1, 3};
  Buffer *x_buf = buffer_data_create(x_data, 3, x_shape, 2, true);
  Tensor *x = tensor_create(x_buf);

  float y_data[] = {1, 2, 3};
  int y_shape[] = {1, 3};
  Buffer *y_buf = buffer_data_create(y_data, 3, y_shape, 2, true);
  Tensor *y = tensor_create(y_buf);

  // Perform forward pass
  Tensor *outl = apply_op(OP_LOGSOFTMAX, (Tensor *[]){x}, 1);
  Tensor *outx = apply_op(OP_NLL, (Tensor *[]){outl, y}, 2);

  shapetracker_print(outl->buf);
  shapetracker_print(outx->buf);
  backward(outx, 1);
  shapetracker_print(outl->grad);
  shapetracker_print(outx->grad);

  printf("Forward pass test passed!\n");
}

int main() {
  test_log_softmax_backward();
  return 0;
}
