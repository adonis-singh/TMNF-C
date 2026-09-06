/* Golden-vector trace format shared by the in-game tracer (writer) and the
 * replay harness (reader).
 *
 * A trace file records every observed call to one game function: the raw bytes
 * of each input buffer (this-pointer region, referenced args) captured at entry
 * and each output buffer captured at return. The harness feeds recorded inputs
 * to the C port and compares its outputs to the recorded output bytes.
 *
 * File: oracle/traces/<VA>_<name>.bin, little-endian.
 *   char     magic[8]      = "TMNFTRC1"
 *   uint32   function_va
 *   uint32   record_count
 *   record[record_count]
 *
 * record:
 *   uint32   seq
 *   uint16   n_in
 *   uint16   n_out
 *   buffer[n_in]           (entry snapshots)
 *   buffer[n_out]          (return snapshots)
 *
 * buffer:
 *   uint32   tag           logical role (see TraceTag)
 *   uint32   addr          original game address (for aliasing/debug)
 *   uint32   len
 *   uint8    bytes[len]
 *
 * A scalar float/int arg passed by value is recorded as a 4-byte buffer whose
 * addr is 0 and tag identifies the parameter position. */
#ifndef TMNF_TRACE_FORMAT_H
#define TMNF_TRACE_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TMNF_TRACE_MAGIC "TMNFTRC1"

/* Logical roles. THIS/OUT_THIS refer to the same object at entry vs return. */
enum TraceTag {
	TAG_THIS      = 0,   /* this-pointer object, entry snapshot            */
	TAG_ARG1      = 1,
	TAG_ARG2      = 2,
	TAG_ARG3      = 3,
	TAG_ARG4      = 4,
	TAG_SCALAR1   = 16,  /* by-value scalar params                          */
	TAG_SCALAR2   = 17,
	TAG_SCALAR3   = 18,
	TAG_OUT_THIS  = 32,  /* this-pointer object, return snapshot            */
	TAG_OUT_ARG1  = 33,
	TAG_OUT_ARG2  = 34,
	TAG_OUT_ARG3  = 35,
	TAG_OUT_ARG4  = 36,
};

#ifdef __cplusplus
}
#endif

#endif /* TMNF_TRACE_FORMAT_H */
