#ifndef UTILS_H
#define UTILS_H

#include "buffer.h"
#include "ops.h"
#include <stdio.h>

static void data_print(Buffer *buf, int dim, int offset);
void buffer_print(Buffer *buf);

void shapetracker_print(Buffer *buf); 

#endif
