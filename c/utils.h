#include "buffer.h"

void reportMemoryError(const char *type);
static void _printData(Buffer *buf, int dim, int offset);
void printBuffer(Buffer *buf);
