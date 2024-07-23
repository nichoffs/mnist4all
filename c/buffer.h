#ifndef BUFFER_H
#define BUFFER_H

#include "shapetracker.h"
#include <stdbool.h>

/**
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

Buffer* buffer_copy(Buffer* buf);

/**
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

/**
 * @brief Destroys a Buffer and frees associated memory.
 *
 * If the buffer's copy flag is true, this function also frees the data array.
 *
 * @param buf Pointer to the Buffer to be destroyed.
 */
void buffer_destroy(Buffer* buf);

Buffer* zeros(int* shape, int ndim);
Buffer* randint(int *shape, int ndim, int low, int high);
Buffer* uniform(int* shape, int ndim, float low, float high);
Buffer* full_like(Buffer* buf, float fill_value);


#endif // BUFFER_H
