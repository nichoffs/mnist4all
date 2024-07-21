#ifndef SHAPETRACKER_H
#define SHAPETRACKER_H

/**
 * @struct ShapeTracker
 * @brief Tracks the shape and stride information of a tensor.
 */
typedef struct {
    int* shape;   /**< Array of dimension sizes */
    int* stride;  /**< Array of strides for each dimension */
    int offset;   /**< Offset from the start of the data */
    int ndim;     /**< Number of dimensions */
    int numel;    /**< Total number of elements */
} ShapeTracker;

/**
 * @brief Creates a new ShapeTracker.
 * @param shape Array of dimension sizes
 * @param stride Array of strides for each dimension
 * @param offset Offset from the start of the data
 * @param ndim Number of dimensions
 * @return Pointer to the new ShapeTracker, or NULL if allocation fails
 */
ShapeTracker* shapetracker_create(int* shape, int* stride, int offset, int ndim);

/**
 * @brief Destroys a ShapeTracker and frees its memory.
 * @param st Pointer to the ShapeTracker to destroy
 */
void shapetracker_destroy(ShapeTracker* st);

/**
 * @brief Calculates the physical index from a logical index.
 * @param st Pointer to the ShapeTracker
 * @param numel Logical index
 * @return Physical index, or -1 if the input is invalid
 */
int view_index(ShapeTracker* st, int numel);

#endif // SHAPETRACKER_H
