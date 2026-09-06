#!/bin/sh
# The exactness proof of the CUDA backend rests on every binary32 operation
# rounding once. A contracted multiply-add rounds once for two
# operations and diverges from the CPU at tick 2. -fmad=false is passed to
# nvcc, and because the device LTO backend runs inside nvlink and does not
# inherit it, CMakeLists.txt also passes -Xnvlink -Xnvvm=-fma=0 and
# -Xnvlink -Xptxas=--fmad=false. This test disassembles the linked device code
# and requires that tmnf_fma_canary_kernel (a plain a * b + c in float and
# double, src/cuda/tmnf_cuda_env.cu) contains no FFMA/DFMA and does contain
# FMUL/FADD/DMUL/DADD.
#
# The whole binary cannot be FMA-free: the correctly rounded division and
# square root sequences (__fdiv_rn, __fsqrt_rn, double /, sqrt) and CUDA's
# libm refine with explicit FMAs by design, and HFMA2 Rx, -RZ, RZ, imm is the
# compiler's 32-bit immediate move. Those counts are printed for the record.
#
#   cuda_no_fma.sh BINARY...
set -eu
status=0
for binary in "$@"; do
	sass=$(cuobjdump -sass "$binary")
	canary=$(printf '%s\n' "$sass" | awk '
		/Function :/ { inside = index($0, "tmnf_fma_canary_kernel") > 0 }
		inside { print }')
	if [ -z "$canary" ]; then
		echo "cuda_no_fma: $binary: tmnf_fma_canary_kernel not found" >&2
		status=1
		continue
	fi
	count() { printf '%s\n' "$2" | grep -cE "^\s+/\*[0-9a-f]+\*/\s+(@!?U?P[0-9T]+\s+)?$1(\.|\s)" || true; }
	fma=$(count '(U?FFMA|DFMA|FFMA2)' "$canary")
	fmul=$(count 'FMUL' "$canary")
	fadd=$(count 'FADD' "$canary")
	dmul=$(count 'DMUL' "$canary")
	dadd=$(count 'DADD' "$canary")
	if [ "$fma" -ne 0 ] || [ "$fmul" -eq 0 ] || [ "$fadd" -eq 0 ] || \
		[ "$dmul" -eq 0 ] || [ "$dadd" -eq 0 ]; then
		echo "cuda_no_fma: $binary: canary FFMA/DFMA $fma, FMUL $fmul FADD $fadd DMUL $dmul DADD $dadd: multiply-add contraction is on" >&2
		printf '%s\n' "$canary" | grep -E 'FMA|FMUL|FADD|DMUL|DADD' >&2
		status=1
		continue
	fi
	total_fma=$(count '(U?FFMA|DFMA|FFMA2)' "$sass")
	echo "cuda_no_fma: $binary: canary FMUL $fmul FADD $fadd DMUL $dmul DADD $dadd, 0 FMA; whole binary $total_fma FFMA/DFMA inside division, sqrt and libm sequences"
done
exit $status
