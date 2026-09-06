#include "targets.h"
#include "../../src/trace_format.h"

#define D(tag_, kind_, offset_, add_, len_) \
	{ (tag_), (kind_), (offset_), (add_), (len_) }

const struct TargetSpec g_targets[] = {
	{
		0x008E3120, "GmQuat_Normalize", 7,
		{ 0x51, 0x56, 0x8B, 0xF1, 0xD9, 0x46, 0x04 },
		1, 1,
		{ D(TAG_THIS, CAP_THIS, 0, 0, 16) },
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 16) }
	},
	{
		0x008E0C60, "GmMat3_SetTranspose", 6,
		{ 0x8B, 0x44, 0x24, 0x04, 0xD9, 0x00 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 16),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 36)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 36) }
	},
	{
		0x008E09F0, "GmMat3_Mult", 6,
		{ 0x8B, 0x44, 0x24, 0x04, 0xD9, 0x00 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 36),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 36)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 36) }
	},
	{
		0x0045BBA0, "GmVec3_SetMultIso4", 8,
		{ 0x8B, 0x44, 0x24, 0x08, 0x8B, 0x54, 0x24, 0x04 },
		2, 1,
		{
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12),
			D(TAG_ARG2, CAP_STACK_POINTER, 8, 0, 48)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 12) }
	},
	{
		0x004574A0, "GmVec3_SetMultMat3", 8,
		{ 0x8B, 0x54, 0x24, 0x08, 0x8B, 0x44, 0x24, 0x04 },
		2, 1,
		{
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12),
			D(TAG_ARG2, CAP_STACK_POINTER, 8, 0, 36)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 12) }
	},
	{
		0x0045BD40, "GmVec3_MultTranspose", 5,
		{ 0x83, 0xEC, 0x0C, 0x8B, 0x01 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 12),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 36)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 12) }
	},
	{
		0x00533510, "CHmsDyna_IntegrateStep", 6,
		{ 0x83, 0xEC, 0x40, 0x53, 0x55, 0x56 },
		4, 1,
		{
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 180),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 12, 0, 4),
			D(TAG_THIS, CAP_THIS, 0, 0, 0x344),
			D(TAG_ARG3, CAP_DEREF_THIS, 0x108, 0, 0x44)
		},
		{ D(TAG_OUT_ARG2, CAP_STACK_POINTER, 8, 0, 180) }
	},
	{
		0x00532D40, "CHmsDyna_CopyStateToTemp", 9,
		{ 0x56, 0x8B, 0xC1, 0x8B, 0xB0, 0x2C, 0x03, 0x00, 0x00 },
		2, 1,
		{
			D(TAG_ARG1, CAP_DEREF_THIS, 0x32C, 0, 180),
			D(TAG_THIS, CAP_THIS_OFFSET, 0x274, 0, 180)
		},
		{ D(TAG_OUT_THIS, CAP_THIS_OFFSET, 0x274, 0, 180) }
	},
	{
		0x005341C0, "CHmsDyna_AddLocalTorque", 7,
		{ 0x83, 0xEC, 0x0C, 0x8B, 0x54, 0x24, 0x10 },
		2, 1,
		{
			D(TAG_THIS, CAP_DEREF_THIS, 0x32C, 0, 180),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12)
		},
		{ D(TAG_OUT_THIS, CAP_DEREF_THIS, 0x32C, 0, 180) }
	},
	{
		0x00535D50, "CHmsDyna_ComputeSynthetizedReplacement", 10,
		{ 0x83, 0xEC, 0x1C, 0x53, 0x8D, 0x99, 0x30, 0x03, 0x00, 0x00 },
		3, 1,
		{
			D(TAG_THIS, CAP_THIS_OFFSET, 0x330, 0, 12),
			D(TAG_ARG1, CAP_DYNA_REPLACEMENTS, 0x334, 0, 0),
			D(TAG_ARG2, CAP_STACK_POINTER, 4, 0, 12)
		},
		{ D(TAG_OUT_THIS, CAP_STACK_POINTER, 4, 0, 12) }
	},
	{
		0x00549C90, "CHmsZoneDynamic_PhysicsStep2", 5,
		{ 0xE9 },
		1, 1,
		{ D(TAG_THIS, CAP_THIS, 0, 0, 0x180) },
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x180) }
	},
	{
		0x007BC8B0, "CSceneVehicleCar_UpdateTurbo", 7,
		{ 0x83, 0xB9, 0x00, 0x06, 0x00, 0x00, 0x00 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 4, 0, 4)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x878) }
	},
	{
		0x007C1060, "CSceneVehicleCarWheelRealTimeState_Integrate", 9,
		{ 0x83, 0xEC, 0x10, 0xD9, 0x05, 0x64, 0xEF, 0xB9, 0x00 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0xa8),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 4, 0, 4)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0xa8) }
	},
	{
		0x007C0EC0, "CSceneVehicleCar_WheelUpdateSpeedFromVehicleSpeed", 5,
		{ 0x56, 0x8B, 0x74, 0x24, 0x08 },
		7, 1,
		{
			D(TAG_THIS, CAP_THIS_OFFSET, 0x50, 0, 0x08),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0x08, 0x04),
			D(TAG_ARG2, CAP_STACK_POINTER, 4, 0x120, 0x08),
			D(TAG_ARG3, CAP_THIS_OFFSET, 0x60c, 0, 0x134),
			D(TAG_ARG4, CAP_VEHICLE_WHEEL_SPEED, 0, 0, 0x04),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 8, 0, 4),
			D(TAG_SCALAR2, CAP_STACK_SCALAR, 12, 0, 4)
		},
		{ D(TAG_OUT_ARG1, CAP_STACK_POINTER, 4, 0x120, 0x04) }
	},
	{
		0x008F46C0, "GmSpringFloat_Integrate", 6,
		{ 0xD9, 0x41, 0x0C, 0xD8, 0x61, 0x08 },
		2, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x14),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 4, 0, 4)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x14) }
	},
	{
		0x007BD090, "SDynaMath_ComputeImpulse", 8,
		{ 0x83, 0xEC, 0x18, 0x56, 0x8B, 0x74, 0x24, 0x30 },
		6, 1,
		{
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 4, 0, 4),
			D(TAG_ARG1, CAP_STACK_POINTER, 8, 0, 36),
			D(TAG_SCALAR2, CAP_STACK_SCALAR, 12, 0, 4),
			D(TAG_ARG2, CAP_STACK_POINTER, 16, 0, 12),
			D(TAG_ARG3, CAP_STACK_POINTER, 20, 0, 12),
			D(TAG_ARG4, CAP_STACK_POINTER, 24, 0, 12)
		},
		{ D(TAG_OUT_THIS, CAP_STACK_POINTER, 28, 0, 12) }
	},
	{
		0x007BD3F0, "CSceneVehicleCar_WheelIntegrate", 6,
		{ 0x83, 0xEC, 0x0C, 0x53, 0x56, 0x57 },
		4, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 0x2fc),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 8, 0, 4),
			D(TAG_ARG2, CAP_VEHICLE_TUNING, 0, 0, 0x3ac)
		},
		{ D(TAG_OUT_ARG1, CAP_STACK_POINTER, 4, 0, 0x2fc) }
	},
	{
		0x007BE310, "CSceneVehicleCar_AddVehicleCentralForce", 6,
		{ 0x56, 0x57, 0x8B, 0x7C, 0x24, 0x0C },
		3, 2,
		{
			D(TAG_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12),
			D(TAG_ARG2, CAP_VEHICLE_DYNA_STATE, 0, 0, 180)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_OUT_ARG1, CAP_VEHICLE_DYNA_STATE, 0, 0x64, 24)
		}
	},
	{
		0x007BE2C0, "CSceneVehicleCar_AddVehicleForce", 5,
		{ 0x8B, 0x44, 0x24, 0x08, 0x56 },
		5, 2,
		{
			D(TAG_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12),
			D(TAG_ARG2, CAP_STACK_POINTER, 8, 0, 12),
			D(TAG_ARG3, CAP_VEHICLE_DYNA_STATE, 0, 0, 180),
			D(TAG_ARG4, CAP_VEHICLE_DYNA_PARAMS, 0, 0, 0x44)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_OUT_ARG1, CAP_VEHICLE_DYNA_STATE, 0, 0x64, 24)
		}
	},
	{
		0x007BE390, "CSceneVehicleCar_AddVehicleImpulse", 9,
		{ 0x83, 0xEC, 0x3C, 0x53, 0x8B, 0xD9, 0x8B, 0x4B, 0x28 },
		6, 2,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 12),
			D(TAG_ARG2, CAP_STACK_POINTER, 8, 0, 12),
			D(TAG_ARG3, CAP_VEHICLE_DYNA_STATE, 0, 0, 180),
			D(TAG_ARG4, CAP_VEHICLE_DYNA_PARAMS, 0, 0, 0x44),
			D(TAG_SCALAR1, CAP_VEHICLE_TUNING, 0, 0, 0x3ac)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS_OFFSET, 0x824, 0, 12),
			D(TAG_OUT_ARG1, CAP_VEHICLE_DYNA_STATE, 0, 0, 180)
		}
	},
	{
		0x007BD700, "CSceneVehicleCar_EngineIntegrate", 7,
		{ 0x83, 0xEC, 0x14, 0xD9, 0x44, 0x24, 0x18 },
		9, 1,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_ARG1, CAP_VEHICLE_TUNING, 0, 0, 0x3ac),
			D(TAG_ARG2, CAP_VEHICLE_WHEELS, 0, 0, 0xbf0),
			D(TAG_ARG3, CAP_VEHICLE_TUNING_BUFFER, 0x2c4, 0, 0x18),
			D(TAG_ARG4, CAP_VEHICLE_TUNING_BUFFER, 0x2d4, 0, 0x18),
			D(TAG_SCALAR3, CAP_VEHICLE_TUNING_BUFFER, 0x2e0, 0, 0x18),
			D(TAG_OUT_ARG4, CAP_VEHICLE_TUNING_BUFFER, 0x304, 0, 0x18),
			D(TAG_SCALAR1, CAP_STACK_SCALAR, 4, 0, 4),
			D(TAG_SCALAR2, CAP_STACK_SCALAR, 8, 0, 4)
		},
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x878) }
	},
	{
		0x007C11D0, "CSceneVehicleCar_WheelAbsorbContact", 7,
		{ 0x83, 0xEC, 0x38, 0x53, 0x55, 0x56, 0x57 },
		8, 4,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 0x2fc),
			D(TAG_ARG2, CAP_STACK_POINTER, 8, 0, 0x50),
			D(TAG_ARG3, CAP_VEHICLE_TUNING, 0, 0, 0x3ac),
			D(TAG_ARG4, CAP_VEHICLE_DYNA_PARAMS, 0, 0, 0x44),
			D(TAG_SCALAR1, CAP_VEHICLE_DYNA_STATE, 0, 0, 0xb4),
			D(TAG_SCALAR2, CAP_VEHICLE_CORPUS_ISO, 0, 0, 0x30),
			D(TAG_SCALAR3, CAP_CONTACT_OTHER_ISO, 8, 0, 0x30)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_OUT_ARG1, CAP_STACK_POINTER, 4, 0, 0x2fc),
			D(TAG_OUT_ARG2, CAP_STACK_POINTER, 8, 0, 0x50),
			D(TAG_OUT_ARG3, CAP_VEHICLE_DYNA_STATE, 0, 0, 0xb4)
		}
	},
	{
		0x007C1810, "CSceneVehicleCar_WheelAddForceToVehicle", 5,
		{ 0x83, 0xEC, 0x24, 0x53, 0x55 },
		5, 2,
		{
			D(TAG_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_ARG1, CAP_VEHICLE_TUNING, 0, 0, 0x3ac),
			D(TAG_ARG2, CAP_STACK_POINTER, 4, 0, 0x2fc),
			D(TAG_ARG3, CAP_VEHICLE_DYNA_STATE, 0, 0, 180),
			D(TAG_ARG4, CAP_VEHICLE_DYNA_PARAMS, 0, 0, 0x44)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS_OFFSET, 0x818, 0, 12),
			D(TAG_OUT_ARG1, CAP_VEHICLE_DYNA_STATE, 0, 0x64, 24)
		}
	},
	{
		0x007C3410, "CSceneVehicleCar_AbsorbContact", 9,
		{ 0x83, 0xEC, 0x30, 0x56, 0x57, 0x8B, 0x7C, 0x24, 0x3C },
		5, 3,
		{
			D(TAG_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_ARG1, CAP_STACK_POINTER, 4, 0, 0x50),
			D(TAG_ARG2, CAP_VEHICLE_TUNING, 0, 0, 0x3ac),
			D(TAG_ARG3, CAP_VEHICLE_DYNA_PARAMS, 0, 0, 0x44),
			D(TAG_ARG4, CAP_VEHICLE_DYNA_STATE, 0, 0, 0xb4)
		},
		{
			D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x878),
			D(TAG_OUT_ARG1, CAP_STACK_POINTER, 4, 0, 0x50),
			D(TAG_OUT_ARG2, CAP_VEHICLE_DYNA_STATE, 0, 0, 0xb4)
		}
	},
	{
		0x007C3E80, "CSceneVehicleCar_ComputeForcesModel6", 6,
		{ 0x81, 0xEC, 0x30, 0x01, 0x00, 0x00 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	},
	/* Same argument ABI as Model6 (0x007C69E0 selects on tuning +0x354);
	 * the vehicle-graph capture serializes all four the same way. */
	{
		0x007FA770, "CSceneVehicleCar_ComputeForcesModel3", 6,
		{ 0x83, 0xEC, 0x6C, 0x53, 0x55, 0x56 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	},
	{
		0x007FB5F0, "CSceneVehicleCar_ComputeForcesModel4", 6,
		{ 0x81, 0xEC, 0x84, 0x00, 0x00, 0x00 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	},
	{
		0x007FC170, "CSceneVehicleCar_ComputeForcesModel5", 6,
		{ 0x81, 0xEC, 0x90, 0x00, 0x00, 0x00 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	},
	{
		0x00547E00, "GmCollision_Neg", 5,
		{ 0xD9, 0x41, 0x0C, 0xD9, 0xE0 },
		1, 1,
		{ D(TAG_THIS, CAP_THIS, 0, 0, 0x38) },
		{ D(TAG_OUT_THIS, CAP_THIS, 0, 0, 0x38) }
	},
	{
		0x00548BF0, "CHmsZoneDynamic_SolveImpulse", 6,
		{ 0x81, 0xEC, 0x8C, 0x00, 0x00, 0x00 },
		1, 0,
		{ D(TAG_SCALAR1, CAP_CURRENT_TICK, 0, 0, 4) },
		{ { 0 } }
	},
	{
		0x005497C0, "CHmsZoneDynamic_ComputeCollisionResponse", 6,
		{ 0x81, 0xEC, 0xD8, 0x00, 0x00, 0x00 },
		1, 0,
		{ D(TAG_SCALAR1, CAP_CURRENT_TICK, 0, 0, 4) },
		{ { 0 } }
	},
	{
		0x007B39F0, "internal_CSceneMobilAbsorbContact", 7,
		{ 0x8B, 0x44, 0x24, 0x04, 0x8B, 0x48, 0x40 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	},
	{
		0x0047CBA0, "internal_CTrackManiaRaceTriggerAbsorbContact", 7,
		{ 0x57, 0x8B, 0xF9, 0x83, 0x7F, 0x04, 0x00 },
		0, 0,
		{ { 0 } },
		{ { 0 } }
	}
};

const uint32_t g_target_count = sizeof(g_targets) / sizeof(g_targets[0]);
