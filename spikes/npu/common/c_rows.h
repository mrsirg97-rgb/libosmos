#pragma once
// a half's C (64 x NRc f32 in sub-tile order: [row octet z][column octet j][8 m + n]) to rows ([m][n], NRc wide):
// a row octet's NRc / 8 sub-tiles hold the same values as its 8 rows - each octet permuted in place through a
// scratch of one octet. so the C tile leaves the core in rows, and a DMA of 4 dimensions puts it where the GPU reads
// (fused_ds.py ROWMAJOR)
// t: a scratch of 8 NRc floats, 64-byte aligned - the kernel's decoded tile (Bm), free once the last matmul has read it
template <unsigned NRc, unsigned ZO = 8> // ZO: the tile's row octets (64 rows: 8; fused_ds2.py's 32: 4)
static inline void c_rows(float *__restrict ch, float *__restrict t) {
  for (unsigned z = 0; z < ZO; z++) {
    float *b = ch + z * 8 * NRc;
    for (unsigned i = 0; i < 8 * NRc; i += 16) aie::store_v(t + i, aie::load_v<16>(b + i));
    for (unsigned j = 0; j < NRc / 8; j++)
      for (unsigned m = 0; m < 8; m++) aie::store_v(b + m * NRc + 8 * j, aie::load_v<8>(t + j * 64 + m * 8));
  }
}

