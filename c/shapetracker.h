#ifndef SHAPETRACKER_H
#define SHAPETRACKER_H

typedef struct ShapeTracker {
  int *shape;
  int *strides;
  int size;
  int ndim;
} ShapeTracker;

int calculate_ndim(int *shape);
int *calculate_strides(int *shape, int ndim);
ShapeTracker *createShapeTracker(int *shape, int size);
ShapeTracker *copyShapeTracker(ShapeTracker *st);
void freeShapeTracker(ShapeTracker *st);

#endif // SHAPETRACKER_H

