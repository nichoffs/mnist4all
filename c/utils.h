#ifndef UTILS_H
#define UTILS_H

#include "buffer.h"
#include <stdio.h>

/**
 * @brief Prints the contents of a Buffer to stdout.
 * @param buf Pointer to the Buffer to print
 */
void buffer_print(Buffer *buf);

/**
 * @brief Prints the contents of a Buffer's ShapeTracker to stdout.
 * @param buf Pointer to the Buffer whose ShapeTracker to print
 */
void shapetracker_print(Buffer *buf);

#endif // UTILS_H
