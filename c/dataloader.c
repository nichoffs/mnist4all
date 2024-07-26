#include "dataloader.h"

#define MNIST_IMAGE_MAGIC 2051
#define MNIST_LABEL_MAGIC 2049

Buffer *load_mnist_gzip(const char *filename, int is_images) {
  char filepath[256];
  snprintf(filepath, sizeof(filepath), "../data/%s", filename);

  gzFile file = gzopen(filepath, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s\n", filepath);
    return NULL;
  }

  int magic;
  gzread(file, &magic, sizeof(int));
  magic = __builtin_bswap32(magic);

  if (magic != (is_images ? MNIST_IMAGE_MAGIC : MNIST_LABEL_MAGIC)) {
    fprintf(stderr, "Error: Invalid magic number in file %s\n", filepath);
    gzclose(file);
    return NULL;
  }

  int num_items;
  gzread(file, &num_items, sizeof(int));
  num_items = __builtin_bswap32(num_items);

  int num_rows = 0, num_cols = 0;
  if (is_images) {
    gzread(file, &num_rows, sizeof(int));
    gzread(file, &num_cols, sizeof(int));
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
  }

  int total_size = num_items * (is_images ? (num_rows * num_cols) : 1);
  int ndim = is_images ? 2 : 1;
  int shape[2] = {num_items, num_rows * num_cols};
  if (!is_images) {
    shape[0] = num_items;
  }

  float *data = (float *)malloc(total_size * sizeof(float));
  if (!data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    gzclose(file);
    return NULL;
  }

  unsigned char *temp_buffer = (unsigned char *)malloc(total_size);
  if (!temp_buffer) {
    fprintf(stderr, "Error: Memory allocation failed for temporary buffer\n");
    free(data);
    gzclose(file);
    return NULL;
  }

  int bytes_read = gzread(file, temp_buffer, total_size);
  if (bytes_read != total_size) {
    fprintf(stderr, "Error: Unexpected end of file %s\n", filepath);
    free(data);
    free(temp_buffer);
    gzclose(file);
    return NULL;
  }

  for (int i = 0; i < total_size; i++) {
    data[i] = (float)temp_buffer[i];
  }

  free(temp_buffer);
  gzclose(file);

  Buffer *buf = buffer_data_create(data, total_size, shape, ndim, false);
  if (!buf) {
    fprintf(stderr, "Error: Failed to create buffer\n");
    free(data);
    return NULL;
  }

  return buf;
}

void load_mnist_datasets(Buffer **train_images, Buffer **train_labels,
                         Buffer **test_images, Buffer **test_labels) {
  *train_images = load_mnist_gzip("train-images-idx3-ubyte.gz", 1);
  *train_labels = load_mnist_gzip("train-labels-idx1-ubyte.gz", 0);
  *test_images = load_mnist_gzip("t10k-images-idx3-ubyte.gz", 1);
  *test_labels = load_mnist_gzip("t10k-labels-idx1-ubyte.gz", 0);

  if (!*train_images || !*train_labels || !*test_images || !*test_labels) {
    fprintf(stderr, "Error: Failed to load one or more MNIST datasets\n");
    // Clean up any successfully loaded datasets
    if (*train_images)
      buffer_destroy(*train_images);
    if (*train_labels)
      buffer_destroy(*train_labels);
    if (*test_images)
      buffer_destroy(*test_images);
    if (*test_labels)
      buffer_destroy(*test_labels);
    *train_images = *train_labels = *test_images = *test_labels = NULL;
  }
}
