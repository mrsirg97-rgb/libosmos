// the fused kernels' matmul with C kept in f32 (fused_iq2xxs.cc's mm_f32acc): mm_bfp's 8x8 bfp16 block MACs, a C
// sub-tile's partial sums loaded from f32 memory into the accumulator and stored back. C (64 x NR) f32 in sub-tile order
// [z][j][64] (lane 8 m + n); A 64 x 128 bfp16 ([z][i]); B NR x 128 ([j][i])
#pragma once
template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_f32acc(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pC) {
  using acc_t = aie::accum<accfloat, 64>;
  for (unsigned z = 0; z < rowA; z += 2)
    for (unsigned j = 0; j < colB; j += 2) {
      float *c00 = pC + (z * colB + j) * 64, *c01 = c00 + 64, *c10 = pC + ((z + 1) * colB + j) * 64, *c11 = c10 + 64;
      acc_t a00, a01, a10, a11;
      a00.from_vector(aie::load_v<64>(c00)), a01.from_vector(aie::load_v<64>(c01));
      a10.from_vector(aie::load_v<64>(c10)), a11.from_vector(aie::load_v<64>(c11));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pAIn(pA);
      pAIn.seek(z * colA);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pBIn(pB);
      pBIn.seek(j * colA);
      AIE_LOOP_UNROLL_FULL
      for (unsigned i = 0; i < colA; ++i) {
        auto A0 = pAIn.pop_seek(colA - 1);
        auto A1 = pAIn.pop_seek(-(int)colA);
        auto B0 = pBIn.pop_seek(colA - 1);
        auto B1 = pBIn.pop_seek(-(int)colA);
        a00 = mac_8x8_8x8T(A0, B0, a00);
        a01 = mac_8x8_8x8T(A0, B1, a01);
        a10 = mac_8x8_8x8T(A1, B0, a10);
        a11 = mac_8x8_8x8T(A1, B1, a11);
      }
      aie::store_v(c00, a00.to_vector<float>()), aie::store_v(c01, a01.to_vector<float>());
      aie::store_v(c10, a10.to_vector<float>()), aie::store_v(c11, a11.to_vector<float>());
    }
}

