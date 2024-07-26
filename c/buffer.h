#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * @struct Buffer
 * @brief Represents a tensor buffer with shape information.
 *
 * @param data Pointer to the float array holding the tensor data.
 * @param size Number of elements in the data array.
 * @param st Pointer to the ShapeTracker containing shape and stride information.
 * @param copy Boolean flag indicating whether the data is owned by this buffer.
 */
typedef struct {
    float* data;
    int size;
    ShapeTracker* st;
    bool copy;
} Buffer;


int *calculate_strides(const int *shape, int ndim);

/**
 * @brief Creates a new Buffer with existing data and shape information.
 *
 * @param data Pointer to the float array of data.
 * @param size Number of elements in the data array.
 * @param st Pointer to a ShapeTracker with shape and stride information for the VIEW.
 * @param copy If true, data is copied; if false, the pointer is used directly.
 * @return Pointer to the newly created Buffer, or NULL if creation failed.
 */
Buffer* buffer_create(float* data, int size, ShapeTracker* st, bool copy);

/*
 * @brief Copies the content of a buffer into a new one
 *
 * @param Buffer* Pointer to the Buffer to be copied
 * @return Pointer to the newly created Buffer, or NULL if creation failed.
 */
Buffer* buffer_copy(Buffer* buf);


/*
 * @brief Creates a new Buffer with data and shape information.
 *
 * This function creates a new ShapeTracker internally.
 *
 * @param data Pointer to the float array of data.
 * @param size Number of elements in the data array.
 * @param shape Array of integers representing the shape of the data - row major.
 * @param ndim Number of dimensions in the shape array.
 * @param copy If true, data is copied; if false, the pointer is used directly.
 * @return Pointer to the newly created Buffer, or NULL if creation failed.
 */
Buffer* buffer_data_create(float* data, int size, int* shape, int ndim, bool copy);

/*
 * @brief Destroys a Buffer and frees associated memory.
 *
 * If the buffer's copy flag is true, this function also frees the data array.
 *
 * @param buf Pointer to the Buffer to be destroyed.
 */
void buffer_destroy(Buffer* buf);

/*
 * @brief Initializes a buffer according to shape with zeros
 *
 * @param int* The shape of the buffer
 * @param int The number of dimensions in the shape
 * @return Pointer to the newly created Buffer, or NULL if creation failed.
 */
Buffer* zeros(int* shape, int ndim);

/*
 * @brief Initializes a buffer according to shape within the range low(inclusive) and high(exclusive)
 *
 * @param int* The shape of the buffer
 * @param int The number of dimensions in the shape
 * @param int The lower bound(inclusive) and high(exclusive)
 */
Buffer* uniform(int* shape, int ndim, float low, float high);

/*
 * @brief Initializes a new buffer with the shape of the input buffer and fill value passed
 *
 * @param Buffer* Pointer to the buffer to mimic the shape
 * @param int The value to fill the buffer with
 * @return Pointer to the newly created Buffer, or NULL if creation failed.
 */
Buffer* full_like(Buffer* buf, float fill_value);

#endif // BUFFER_H
