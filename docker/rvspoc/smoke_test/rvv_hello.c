// SPDX-License-Identifier: Apache-2.0
// RVV 1.0 smoke test: vector add of 16 floats using intrinsics.
// Verifies cross-toolchain + QEMU RVV emulation work end-to-end.

#include <riscv_vector.h>
#include <stdio.h>

int main(void) {
    const int N = 16;
    float a[16], b[16], c[16];
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(N - i); }

    size_t n = N;
    float* pa = a; float* pb = b; float* pc = c;
    for (size_t vl; n > 0; n -= vl, pa += vl, pb += vl, pc += vl) {
        vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(pa, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(pb, vl);
        vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(pc, vc, vl);
    }

    int ok = 1;
    for (int i = 0; i < N; i++) if (c[i] != (float)N) ok = 0;
    printf("rvv_hello: VLEN-discovered, computed c[i]=a[i]+b[i] for %d floats — %s\n",
           N, ok ? "PASS" : "FAIL");
    printf("c[0]=%.1f c[15]=%.1f (expect %d.0)\n", c[0], c[15], N);
    return ok ? 0 : 1;
}
