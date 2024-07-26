#include "dot.h"

#define EPSILON 1e-6

void test_dot() {
  printf("Testing dot function...\n");

  // Test case 1: Valid input (provided example)
  {
    float data1[6] = {1, 2, 3, 4, 5, 6};
    int shape1[2] = {2, 3};
    int size1 = 6;
    int ndim1 = 2;
    float weight_data1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int weight_shape1[2] = {3, 3};
    int weight_size1 = 9;
    int weight_ndim1 = 2;
    Buffer *buf1 = buffer_data_create(data1, size1, shape1, ndim1, true);
    Buffer *weight1 = buffer_data_create(weight_data1, weight_size1,
                                         weight_shape1, weight_ndim1, true);
    Buffer *out1 = dot(buf1, weight1);

    assert(out1 != NULL);
    assert(out1->st->ndim == 2);
    assert(out1->st->shape[0] == 2);
    assert(out1->st->shape[1] == 3);

    float expected1[6] = {30, 36, 42, 66, 81, 96};
    for (int i = 0; i < 6; i++) {
      assert(fabs(out1->data[i] - expected1[i]) < EPSILON);
    }

    buffer_destroy(buf1);
    buffer_destroy(weight1);
    buffer_destroy(out1);
  }

  // Test case 2: Invalid input - mismatched dimensions
  {
    float data2[6] = {1, 2, 3, 4, 5, 6};
    int shape2[2] = {2, 3};
    float weight_data2[6] = {1, 2, 3, 4, 5, 6};
    int weight_shape2[2] = {2, 3}; // Should be {3, something}
    Buffer *buf2 = buffer_data_create(data2, 6, shape2, 2, true);
    Buffer *weight2 =
        buffer_data_create(weight_data2, 6, weight_shape2, 2, true);
    Buffer *out2 = dot(buf2, weight2);

    assert(out2 == NULL);

    buffer_destroy(buf2);
    buffer_destroy(weight2);
  }

  // Test case 3: Invalid input - incorrect number of dimensions
  {
    float data3[3] = {1, 2, 3};
    int shape3[1] = {3};
    float weight_data3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int weight_shape3[2] = {3, 3};
    Buffer *buf3 = buffer_data_create(data3, 3, shape3, 1, true);
    Buffer *weight3 =
        buffer_data_create(weight_data3, 9, weight_shape3, 2, true);
    Buffer *out3 = dot(buf3, weight3);

    assert(out3 == NULL);

    buffer_destroy(buf3);
    buffer_destroy(weight3);
  }

  // Test case 4: Edge case - 1x1 matrices
  {
    float data4[1] = {2};
    int shape4[2] = {1, 1};
    float weight_data4[1] = {3};
    int weight_shape4[2] = {1, 1};
    Buffer *buf4 = buffer_data_create(data4, 1, shape4, 2, true);
    Buffer *weight4 =
        buffer_data_create(weight_data4, 1, weight_shape4, 2, true);
    Buffer *out4 = dot(buf4, weight4);

    assert(out4 != NULL);
    assert(out4->st->ndim == 2);
    assert(out4->st->shape[0] == 1);
    assert(out4->st->shape[1] == 1);
    assert(fabs(out4->data[0] - 6) < EPSILON);

    buffer_destroy(buf4);
    buffer_destroy(weight4);
    buffer_destroy(out4);
  }

  // Test case 5: Large batch size
  {
    int batch_size = 1000;
    int input_dim = 10;
    int output_dim = 5;
    float *data5 = malloc(batch_size * input_dim * sizeof(float));
    float *weight_data5 = malloc(input_dim * output_dim * sizeof(float));
    for (int i = 0; i < batch_size * input_dim; i++)
      data5[i] = 1.0f;
    for (int i = 0; i < input_dim * output_dim; i++)
      weight_data5[i] = 1.0f;

    int shape5[2] = {batch_size, input_dim};
    int weight_shape5[2] = {input_dim, output_dim};
    Buffer *buf5 =
        buffer_data_create(data5, batch_size * input_dim, shape5, 2, true);
    Buffer *weight5 = buffer_data_create(weight_data5, input_dim * output_dim,
                                         weight_shape5, 2, true);
    Buffer *out5 = dot(buf5, weight5);

    assert(out5 != NULL);
    assert(out5->st->ndim == 2);
    assert(out5->st->shape[0] == batch_size);
    assert(out5->st->shape[1] == output_dim);
    for (int i = 0; i < batch_size * output_dim; i++) {
      assert(fabs(out5->data[i] - input_dim) < EPSILON);
    }

    buffer_destroy(buf5);
    buffer_destroy(weight5);
    buffer_destroy(out5);
    free(data5);
    free(weight_data5);
  }
}
