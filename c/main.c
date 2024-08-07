#include "buffer.h"
#include "dataloader.h"
#include "tensor.h"
#include <math.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BS 128
#define LR .01
#define EPOCHS 1000
#define SAMPLE_SIZE 784
#define NUM_CLASSES 10
#define NUM_SAMPLES 60000
#define PROGRESS_BAR_WIDTH 50

static Buffer *getImages(Buffer *images, int *indices, int num_indices,
                         int sample_size) {
  // Validate input
  if (!images || !indices) {
    fprintf(stderr, "Invalid input: images or indices is NULL\n");
    return NULL;
  }

  if (images->st->ndim != 2) {
    fprintf(stderr, "Invalid images buffer: must be 2-dimensional\n");
    return NULL;
  }

  if (images->st->shape[1] != sample_size) {
    fprintf(stderr, "Invalid sample size: doesn't match images buffer\n");
    return NULL;
  }

  for (int i = 0; i < num_indices; i++) {
    if (indices[i] < 0 || indices[i] >= images->st->shape[0]) {
      fprintf(stderr, "Invalid index %d: out of range\n", indices[i]);
      return NULL;
    }
  }

  int new_shape[2] = {num_indices, sample_size};

  int new_size = num_indices * sample_size;
  float *new_data = malloc(new_size * sizeof(float));
  if (!new_data) {
    fprintf(stderr, "Memory allocation failed for new data\n");
    return NULL;
  }

  for (int i = 0; i < num_indices; i++) {
    int images_offset = indices[i] * sample_size;
    int dest_offset = i * sample_size;
    for (int j = 0; j < sample_size; j++) {
      int images_index = view_index(images->st, images_offset + j);
      new_data[dest_offset + j] = images->data[images_index];
    }
  }

  Buffer *result = buffer_data_create(new_data, new_size, new_shape, 2, false);
  if (!result) {
    fprintf(stderr, "Failed to create result buffer\n");
    free(new_data);
    return NULL;
  }

  return result;
}

static Buffer *getLabels(Buffer *labels, int *indices, int num_indices,
                         int num_classes) {
  if (!labels || !indices) {
    fprintf(stderr, "Invalid input: labels or indices is NULL\n");
    return NULL;
  }
  if (labels->st->ndim != 1) {
    fprintf(stderr, "Invalid labels buffer: must be 1-dimensional\n");
    return NULL;
  }

  for (int i = 0; i < num_indices; i++) {
    if (indices[i] < 0 || indices[i] >= labels->st->shape[0]) {
      fprintf(stderr, "Invalid index %d: out of range\n", indices[i]);
      return NULL;
    }
  }

  int new_shape[2] = {num_indices, num_classes};

  int new_size = num_indices * num_classes;
  float *new_data = calloc(new_size, sizeof(float));
  if (!new_data) {
    fprintf(stderr, "Memory allocation failed for new data\n");
    return NULL;
  }

  for (int i = 0; i < num_indices; i++) {
    int label = (int)labels->data[indices[i]];
    if (label < 0 || label >= num_classes) {
      fprintf(stderr, "Invalid label %d: out of range\n", label);
      free(new_data);
      return NULL;
    }
    new_data[i * num_classes + label] = -1.0f;
  }

  Buffer *result = buffer_data_create(new_data, new_size, new_shape, 2, false);
  if (!result) {
    fprintf(stderr, "Failed to create result buffer\n");
    free(new_data);
    return NULL;
  }

  return result;
}

static int *rand_int(int low, int high, int size, int *arr) {
  if (low >= high || size <= 0) {
    fprintf(stderr, "Invalid range or size for rand_int\n");
    return NULL;
  }

  if (!arr) {
    fprintf(stderr, "Memory allocation failed in rand_int\n");
    return NULL;
  }

  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  int range = high - low;
  for (int i = 0; i < size; i++) {
    arr[i] = low + (rand() % range);
  }

  return arr;
}

static Buffer *layer_init(int m, int h) {
  int shape[2] = {m, h};
  Buffer *ret = uniform(shape, 2, -1, 1);
  int div = sqrtf(m * h);
  for (int i = 0; i < ret->size; i++) {
    ret->data[i] /= div;
  }
  if (!ret) {
    printf("layer init failed!");
    exit(1);
  }
  return ret;
}

typedef struct {
  Tensor *l1;
  Tensor *l2;
} MNISTClassifier;

static MNISTClassifier *create_mnist_classifier() {
  MNISTClassifier *model = malloc(sizeof(MNISTClassifier));
  model->l1 = tensor_create(layer_init(784, 128));
  model->l2 = tensor_create(layer_init(128, 10));
  return model;
}

void sgd_step(Tensor **params, int num_params, float lr) {
  for (int i = 0; i < num_params; i++) {
    for (int j = 0; j < params[i]->buf->st->numel; j++) {
      params[i]->buf->data[j] -= lr * params[i]->grad->data[j];
    }
    buffer_destroy(params[i]->grad);
    params[i]->grad = NULL;
  }
}

void print_progress_bar(int epoch, int total_epochs, float loss) {
  float progress = (float)(epoch + 1) / total_epochs;
  int bar_width = (int)(progress * PROGRESS_BAR_WIDTH);

  printf("\033[K"); // Clear the line
  printf("Epoch %4d/%d [", epoch + 1, total_epochs);

  for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
    if (i < bar_width)
      printf("=");
    else if (i == bar_width)
      printf(">");
    else
      printf(" ");
  }

  printf("] %3d%% | Loss: %.4f\n", (int)(progress * 100.0), loss);
}

int main() {
  Buffer *train_images, *train_labels, *test_images, *test_labels;

  load_mnist_datasets(&train_images, &train_labels, &test_images, &test_labels);

  if (!train_images || !train_labels || !test_images || !test_labels) {
    fprintf(stderr, "Failed to load one or more datasets.\n");
    return 1;
  }

  int *indices = (int *)malloc(BS * sizeof(int));

  MNISTClassifier *model = create_mnist_classifier();
  Tensor *h1;
  Tensor *h1_relu;
  Tensor *logits;
  Tensor *probs;
  Tensor *loss;

  Buffer *images;
  Buffer *labels;

  Tensor *x;
  Tensor *y;

  for (int epoch = 0; epoch < EPOCHS; epoch++) {

    rand_int(0, NUM_SAMPLES, BS, indices);
    images = getImages(train_images, indices, BS, SAMPLE_SIZE);
    labels = getLabels(train_labels, indices, BS, NUM_CLASSES);

    x = tensor_create(images);
    y = tensor_create(labels);

    // Forward pass
    h1 = apply_op(OP_DOT, (Tensor *[]){x, model->l1}, 2);
    h1_relu = apply_op(OP_RELU, (Tensor *[]){h1}, 1);
    logits = apply_op(OP_DOT, (Tensor *[]){h1_relu, model->l2}, 2);
    probs = apply_op(OP_LOGSOFTMAX, (Tensor *[]){logits}, 1);
    loss = apply_op(OP_NLL, (Tensor *[]){probs, y}, 2);

    print_progress_bar(epoch, EPOCHS, loss->buf->data[0]);
    if (epoch < EPOCHS - 1) {
      printf("\033[1A");
    }

    // Backward pass
    backward(loss, 1);

    tensor_destroy(h1);
    tensor_destroy(h1_relu);
    tensor_destroy(logits);
    tensor_destroy(probs);
    tensor_destroy(loss);
    tensor_destroy(x);
    tensor_destroy(y);

    // Update parameters
    sgd_step((Tensor *[]){model->l1, model->l2}, 2, LR);

    // Clean up
  }

  return 0;
}
