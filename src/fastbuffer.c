#include "fastbuffer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "collision.h"

TMNF_HD static void fail_allocation(void) {
	tmnf_abort();
}

TMNF_HD void CFastBuffer_SHmsPhysicalCollision_Init(
	CFastBuffer_SHmsPhysicalCollision *self) {
	self->count = 0;
	self->data = NULL;
	self->capacity = 0;
}

TMNF_HD void CFastBuffer_SHmsPhysicalCollision_Destroy(
	CFastBuffer_SHmsPhysicalCollision *self) {
	free(self->data);
	self->count = 0;
	self->data = NULL;
	self->capacity = 0;
}

/* 0x00536DA0  Returns the indexed 0x4c-byte collision record. */
/* UNVALIDATED */
TMNF_HD SHmsPhysicalCollision *CFastBuffer_SHmsPhysicalCollision_At(
	CFastBuffer_SHmsPhysicalCollision *self, uint32_t index) {
	return self->data + index;
}

/* 0x00537370  Grows capacity with the game's 1.5x policy. */
/* UNVALIDATED */
TMNF_HD void CFastBuffer_SHmsPhysicalCollision_SetSizeAtLeast(
	CFastBuffer_SHmsPhysicalCollision *self, uint32_t requested) {
	uint32_t capacity = self->capacity;
	if (requested <= capacity) {
		return;
	}
#if defined(__CUDA_ARCH__)
	/* Device buffers are fixed-capacity scratch bound at link time. Growing
	 * one would need a kernel allocation, so the port fails loudly instead
	 * of dropping the record. */
	tmnf_fail("collision buffer capacity exceeded");
#else
	uint32_t grown = capacity + (capacity >> 1);
	if (requested <= grown) {
		requested = grown;
	}
	SHmsPhysicalCollision *data = (SHmsPhysicalCollision *)
		malloc((size_t)requested * sizeof(*self->data));
	if (data == NULL) {
		fail_allocation();
	}
	if (self->count != 0) {
		memcpy(data, self->data, (size_t)self->count * sizeof(*data));
	}
	free(self->data);
	self->data = data;
	self->capacity = requested;
#endif
}

/* 0x00537400  Appends one uninitialized collision record. */
/* UNVALIDATED */
TMNF_HD SHmsPhysicalCollision *CFastBuffer_SHmsPhysicalCollision_AddNewElem(
	CFastBuffer_SHmsPhysicalCollision *self) {
	uint32_t index = self->count;
	CFastBuffer_SHmsPhysicalCollision_SetSizeAtLeast(self, index + 1);
	self->count = index + 1;
	return self->data + index;
}

TMNF_HD static void swap_records(unsigned char *a, unsigned char *b) {
	if (a == b) {
		return;
	}
	for (size_t i = 0; i < sizeof(SHmsPhysicalCollision); i++) {
		unsigned char value = a[i];
		a[i] = b[i];
		b[i] = value;
	}
}

TMNF_HD static void shortsort(
	unsigned char *lo, unsigned char *hi,
	SHmsPhysicalCollisionCompare compare) {
	const size_t width = sizeof(SHmsPhysicalCollision);
	while (lo < hi) {
		unsigned char *max = lo;
		for (unsigned char *scan = lo + width; scan <= hi; scan += width) {
			if (compare(
					(const SHmsPhysicalCollision *)scan,
					(const SHmsPhysicalCollision *)max) > 0) {
				max = scan;
			}
		}
		swap_records(max, hi);
		hi -= width;
	}
}

/* 0x00547DE0  Sorts records with the game's MSVC quicksort implementation. */
/* UNVALIDATED */
TMNF_HD void CFastBuffer_SHmsPhysicalCollision_QSort(
	CFastBuffer_SHmsPhysicalCollision *self,
	SHmsPhysicalCollisionCompare compare) {
	const size_t width = sizeof(SHmsPhysicalCollision);
	unsigned char *lo_stack[30];
	unsigned char *hi_stack[30];
	int stack_top = 0;

	if (self->count < 2) {
		return;
	}
	if (self->data == NULL || compare == NULL) {
		tmnf_abort();
	}

	unsigned char *lo = (unsigned char *)self->data;
	unsigned char *hi = lo + (size_t)(self->count - 1) * width;

	for (;;) {
		uint32_t count = (uint32_t)((hi - lo) / (ptrdiff_t)width + 1);
		while (count > 8) {
			unsigned char *mid = lo + (size_t)(count >> 1) * width;

			if (compare(
					(const SHmsPhysicalCollision *)lo,
					(const SHmsPhysicalCollision *)mid) > 0) {
				swap_records(lo, mid);
			}
			if (compare(
					(const SHmsPhysicalCollision *)lo,
					(const SHmsPhysicalCollision *)hi) > 0) {
				swap_records(lo, hi);
			}
			if (compare(
					(const SHmsPhysicalCollision *)mid,
					(const SHmsPhysicalCollision *)hi) > 0) {
				swap_records(mid, hi);
			}

			unsigned char *left = lo;
			unsigned char *right = hi;
			for (;;) {
				if (left < mid) {
					do {
						left += width;
						if (mid <= left) {
							break;
						}
					} while (compare(
						(const SHmsPhysicalCollision *)left,
						(const SHmsPhysicalCollision *)mid) < 1);
				}
				if (mid <= left) {
					do {
						left += width;
						if (hi < left) {
							break;
						}
					} while (compare(
						(const SHmsPhysicalCollision *)left,
						(const SHmsPhysicalCollision *)mid) < 1);
				}

				do {
					right -= width;
					if (right <= mid) {
						break;
					}
				} while (compare(
					(const SHmsPhysicalCollision *)right,
					(const SHmsPhysicalCollision *)mid) > 0);

				if (left > right) {
					break;
				}
				swap_records(left, right);
				if (mid == right) {
					mid = left;
				}
			}

			right += width;
			if (mid < right) {
				do {
					right -= width;
					if (right <= mid) {
						break;
					}
				} while (compare(
					(const SHmsPhysicalCollision *)right,
					(const SHmsPhysicalCollision *)mid) == 0);
				if (mid < right) {
					goto partitioned;
				}
			}
			if (mid <= right) {
				do {
					right -= width;
					if (right <= lo) {
						break;
					}
				} while (compare(
					(const SHmsPhysicalCollision *)right,
					(const SHmsPhysicalCollision *)mid) == 0);
			}

partitioned:
			if (right - lo < hi - left) {
				if (left < hi) {
					lo_stack[stack_top] = left;
					hi_stack[stack_top] = hi;
					stack_top++;
				}
				hi = right;
				if (right <= lo) {
					break;
				}
			} else {
				if (lo < right) {
					lo_stack[stack_top] = lo;
					hi_stack[stack_top] = right;
					stack_top++;
				}
				lo = left;
				if (hi <= left) {
					break;
				}
			}
			count = (uint32_t)((hi - lo) / (ptrdiff_t)width + 1);
		}

		if (lo < hi) {
			shortsort(lo, hi, compare);
		}
		stack_top--;
		if (stack_top < 0) {
			return;
		}
		lo = lo_stack[stack_top];
		hi = hi_stack[stack_top];
	}
}
