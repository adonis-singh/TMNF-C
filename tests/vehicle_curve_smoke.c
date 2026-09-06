#include <assert.h>
#include <stdint.h>

#include "vehicle_curve.h"

int main(void)
{
	static const float positions[] = {0.0f, 10.0f, 20.0f};
	static const float values[] = {0.0f, 100.0f, 200.0f};
	static const float one_position[] = {4.0f};
	static const float one_value[] = {7.0f};
	CFuncKeysReal linear = {
		.keys = {.count = 3, .positions = positions},
		.values = values,
		.interpolation = 0,
	};
	CFuncKeysReal step = {
		.keys = {.count = 3, .positions = positions},
		.values = values,
		.interpolation = 1,
	};
	CFuncKeysReal empty = {
		.keys = {.count = 0, .positions = NULL},
		.values = NULL,
		.interpolation = 0,
	};
	CFuncKeysReal single = {
		.keys = {.count = 1, .positions = one_position},
		.values = one_value,
		.interpolation = 0,
	};
	uint32_t index = 0;
	float value;

	CFuncKeys_Compile(&linear.keys);
	CFuncKeys_Compile(&step.keys);
	CFuncKeys_Compile(&empty.keys);
	CFuncKeys_Compile(&single.keys);
	CFuncKeysReal_GetValueOut(&linear, 5.0f, &value, &index);
	assert(value == 50.0f);
	assert(index == 0);

	CFuncKeysReal_GetValueOut(&linear, 15.0f, &value, &index);
	assert(value == 150.0f);
	assert(index == 1);

	index = 0;
	CFuncKeysReal_GetValueOut(&step, 15.0f, &value, &index);
	assert(value == 100.0f);
	assert(index == 1);

	index = 0;
	CFuncKeysReal_GetValueOut(&empty, 1.0f, &value, &index);
	assert(value == 0.0f);
	assert(index == UINT32_MAX);

	index = 0;
	CFuncKeysReal_GetValueOut(&single, 100.0f, &value, &index);
	assert(value == 7.0f);
	assert(index == 0);

	return 0;
}
