#ifndef SHAPETRACKER_H
#define SHAPETRACKER_H

typedef struct ShapeTracker {
  int *shape;
  int *strides;
  int size;
  int ndim;
  int offset;
} ShapeTracker;

int calculate_ndim(int *shape);
int *_default_strides(int *shape, int ndim);
ShapeTracker *initShapeTracker(int *shape, int size);
void freeShapeTracker(ShapeTracker *st);

#endif // SHAPETRACKER_H

