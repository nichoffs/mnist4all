#ifndef SHAPETRACKER_H
#define SHAPETRACKER_H

typedef struct ShapeTracker {
  int *shape;
  int *strides;
  int ndim;
  int numel;
} ShapeTracker;

int calculate_ndim(int *shape);
int calculate_numel(int *shape, int ndim);
int *calculate_strides(int *shape, int ndim);
ShapeTracker *createShapeTracker(int *shape);
void freeShapeTracker(ShapeTracker *st);

#endif // SHAPETRACKER_H

