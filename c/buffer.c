#include "buffer.h"
#include "shapetracker.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Initialization

/* To initialize a buffer, you need to pass in data, shape, size (# elements),
 * and a copy flag. */
/* If copy is true, data will be remalloc'd and copied into the buffer. */
/* If copy is false, the input data pointer will be used. */
/* Keep in mind, if you want to initialize an array on the stack, */
/* copy must be true so that the buffer can be correctly freed. */

Buffer *initBuffer(float *data, int *shape, int size, bool copy) {

  if (data == NULL || shape == NULL) {
    reportMemoryError("data or shape input");
    return NULL;
  }

  Buffer *x = (Buffer *)malloc(sizeof(Buffer));
  if (!x) {
    reportMemoryError("buffer");
    return NULL;
  }

  if (copy) {
    float *dataCopy = (float *)malloc(size * sizeof(float));
    if (!dataCopy) {
      reportMemoryError("data copy");
      free(x);
    }
    memcpy(dataCopy, data, size * sizeof(float));
    x->data = dataCopy;
  } else {
    x->data = data;
  }

  x->shapeTracker = initShapeTracker(shape, size);
  if (!x->shapeTracker) {
    free(x->data);
    free(x);
    return NULL;
  }
  return x;
}

Buffer *copyBuffer(Buffer *buf) {
  if (buf == NULL) {
    reportMemoryError("copybuffer input");
    return NULL;
  }

  return initBuffer(buf->data, buf->shapeTracker->shape,
                    buf->shapeTracker->size, true);
}

// Helpers

void freeBuffer(Buffer *buffer) {
  if (buffer) {
    if (buffer->data) {
      free(buffer->data);
    }
    if (buffer->shapeTracker) {
      freeShapeTracker(buffer->shapeTracker);
    }
    free(buffer);
  }
}
