#ifndef DATALOADER_H
#define DATALOADER_H

#include "buffer.h"

void load_mnist_datasets(Buffer **train_images, Buffer **train_labels,
                         Buffer **test_images, Buffer **test_labels);

Buffer *load_mnist_gzip(const char *filename, int is_images);

#endif
