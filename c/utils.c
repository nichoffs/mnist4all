#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *safe_malloc(size_t n) {
  void *p = malloc(n);
  if (p == NULL) {
    perror("Fatal: failed to malloc bytes. Aborting program.\n");
    abort();
  }
  return p;
}

// initializes memory and copies from source
void *safe_malloc_copy(void *src, size_t n) {
  void *dest = safe_malloc(n);
  memcpy(dest, src, n);
  return dest;
}
