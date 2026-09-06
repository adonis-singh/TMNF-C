#ifndef TMNF_VEHICLE_FAKE_CONTACT_MASK_H
#define TMNF_VEHICLE_FAKE_CONTACT_MASK_H
#include <stdint.h>
enum { TMNF_FAKE_CONTACT_MASK_WIDTH = 128, TMNF_FAKE_CONTACT_MASK_HEIGHT = 128 };
/* Image bytes are generated locally from the user's installation. The zero
 * array lets source-only unit tests link; World_Create refuses game simulation
 * when TMNF_HAS_GAME_MASK is false. It is never a substitute physics asset. */
#ifdef TMNF_LOCAL_GAME_MASK
#include <tmnf_local_game_mask.h>
#define TMNF_HAS_GAME_MASK 1
#else
#define TMNF_HAS_GAME_MASK 0
static const uint8_t TMNF_FAKE_CONTACT_MASK[128 * 128] = {0};
#endif
#endif
