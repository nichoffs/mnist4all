#ifndef SHAPETRACKER_H
#define SHAPETRACKER_H

typedef struct {
    int *shape;
    int *stride;
    int offset;
    int ndim;
    int numel;
} ShapeTracker;

ShapeTracker *shapetracker_create(int *shape, int *stride, int offset, int ndim);
ShapeTracker *shapetracker_copy(int *shape, int *stride, int offset, int ndim);
ShapeTracker *shapetracker_destroy(ShapeTracker *st);

int view_index(ShapeTracker *st, int numel);

int _numel(int *shape, int ndim);

#endif
