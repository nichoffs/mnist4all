#ifndef TEST_H
#define TEST_H

#include "../buffer.h"
#include "../ops.h"
#include "../tensor.h"
#include "../shapetracker.h"
#include "../utils.h"
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Utility function
int compare_buffers(Buffer *buf1, Buffer *buf2);
Buffer *create_test_buffer(float *data, int size, int *shape, int ndim);

#endif
