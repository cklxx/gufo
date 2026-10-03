// Quantization formats for tiny-model CPU experiments.
// Baseline: Q4_K (GGUF k-quants, 256 weights/block, 8x32 sub-blocks).
// Candidate: NVFP4-style microscaling (E2M1, 32 weights share one FP8 scale).
//
// Q4_K quantizer follows llama.cpp quantize_row_q4_K_ref (simplified:
// min/max per sub-block instead of iterative refinement).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace tinyquant {

// ---------- FP16 <-> FP32 ----------
inline float Fp16ToFloat(uint16_t h) noexcept {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t v;
  if (exp == 0) {
    float f = (float)mant / 1024.0f * std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    v = sign | 0x7F800000U | (mant << 13);
  } else {
    v = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &v, sizeof(float));
  return f;
}

inline uint16_t FloatToFp16(float f) noexcept {
  uint32_t v;
  std::memcpy(&v, &f, sizeof(float));
  uint32_t sign = (v >> 16) & 0x8000;
  int32_t exp = ((v >> 23) & 0xFF) - 112;
  uint32_t mant = (v >> 13) & 0x3FF;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7BFF);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

// ---------- Q4_K ----------
struct BlockQ4K {
  uint16_t d;        // super-block scale (fp16) = max_scale/63
  uint16_t dmin;     // super-block min (fp16) = max_min/63
  uint8_t scales[12];// 8x 6-bit scale + 6-bit min, packed
  uint8_t qs[128];   // 256x 4-bit values, sub-block pairs interleaved
};
static_assert(sizeof(BlockQ4K) == 144);

inline void Q4KScales(const uint8_t* packed, int idx,
                      uint8_t& sc, uint8_t& mn) noexcept {
  if (idx < 4) {
    sc = packed[idx] & 63;
    mn = packed[idx + 4] & 63;
  } else {
    sc = (packed[idx + 4] & 15) | ((packed[idx - 4] >> 6) << 4);
    mn = (packed[idx + 4] >> 4) | ((packed[idx] >> 6) << 4);
  }
}

// Quantize 256 floats -> BlockQ4K. Returns max abs error.
inline float QuantizeQ4K(const float* x, BlockQ4K& b) {
  float max_err = 0;
  float sub_scale[8], sub_min[8];
  for (int s = 0; s < 8; ++s) {
    float sb_max = -1e30f, sb_min = 1e30f;
    for (int i = 0; i < 32; ++i) {
      float v = x[s * 32 + i];
      sb_max = std::max(sb_max, v);
      sb_min = std::min(sb_min, v);
    }
    // llama.cpp: if min > 0, clamp min to 0 (scale then covers [0, max])
    if (sb_min > 0) sb_min = 0;
    sub_scale[s] = 0;
    sub_min[s] = 0;
    if (sb_max > sb_min) {
      sub_scale[s] = (sb_max - sb_min) / 15.0f;
      sub_min[s] = -sb_min;
    }
  }
  float max_scale = 0, max_min = 0;
  for (int s = 0; s < 8; ++s) {
    max_scale = std::max(max_scale, sub_scale[s]);
    max_min = std::max(max_min, sub_min[s]);
  }
  if (max_scale == 0) max_scale = 1e-6f;
  if (max_min == 0) max_min = 1e-6f;
  b.d = FloatToFp16(max_scale / 63.0f);
  b.dmin = FloatToFp16(max_min / 63.0f);
  float fd = Fp16ToFloat(b.d), fdmin = Fp16ToFloat(b.dmin);
  std::memset(b.scales, 0, sizeof(b.scales));
  for (int s = 0; s < 8; ++s) {
    int ls = (int)std::round(63 * sub_scale[s] / max_scale);
    int lm = (int)std::round(63 * sub_min[s] / max_min);
    ls = std::max(0, std::min(63, ls));
    lm = std::max(0, std::min(63, lm));
    if (s < 4) {
      b.scales[s] = (uint8_t)ls;
      b.scales[s + 4] = (uint8_t)lm;
    } else {
      b.scales[s + 4] = (uint8_t)((ls & 15) | ((lm & 15) << 4));
      b.scales[s - 4] |= (uint8_t)((ls >> 4) << 6);
      b.scales[s] |= (uint8_t)((lm >> 4) << 6);
    }
  }
  // q = round((x + dmin*mn) / (d*sc)); GGUF pair-interleaved packing
  uint8_t L[256];
  for (int s = 0; s < 8; ++s) {
    uint8_t sc, mn;
    Q4KScales(b.scales, s, sc, mn);
    float d = fd * sc;
    float dm = fdmin * mn;
    for (int i = 0; i < 32; ++i) {
      int q = 0;
      if (d > 0) {
        q = (int)std::round((x[s * 32 + i] + dm) / d);
        q = std::max(0, std::min(15, q));
      }
      L[s * 32 + i] = (uint8_t)q;
      float recon = d * q - dm;
      max_err = std::max(max_err, std::fabs(x[s * 32 + i] - recon));
    }
  }
  std::memset(b.qs, 0, sizeof(b.qs));
  uint8_t* q = b.qs;
  for (int j = 0; j < 256; j += 64) {
    for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
    q += 32;
  }
  return max_err;
}

inline float DequantQ4K(const BlockQ4K& b, int i) noexcept {
  int s = i / 32;
  uint8_t sc, mn;
  Q4KScales(b.scales, s, sc, mn);
  int lane = i % 32, pair = s / 2;
  uint8_t packed = b.qs[pair * 32 + lane];
  uint8_t q = (s & 1) ? (packed >> 4) : (packed & 15);
  return Fp16ToFloat(b.d) * sc * q - Fp16ToFloat(b.dmin) * mn;
}

// ---------- NVFP4-style microscaling ----------
// E2M1 values: {-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6}
// 32 weights share one FP8 (E4M3) scale. Global FP32 per-tensor scale.
struct BlockNVFP4 {
  uint8_t qs[16];   // 32x 4-bit E2M1
  uint8_t scale;    // FP8 E4M3 scale
};
static_assert(sizeof(BlockNVFP4) == 17);

inline float E2M1ToFloat(uint8_t q) noexcept {
  static const float lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return lut[q & 15];
}

inline uint8_t FloatToE2M1(float v) noexcept {
  static const float lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  int best = 0;
  float bd = std::fabs(v - lut[0]);
  for (int i = 1; i < 16; ++i) {
    float d = std::fabs(v - lut[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return (uint8_t)best;
}

// FP8 E4M3 encode/decode (bias 8)
inline uint8_t FloatToFp8E4M3(float v) noexcept {
  if (v <= 0) return 0;
  int e = (int)std::floor(std::log2(v)) + 8;
  e = std::max(1, std::min(15, e));
  float m = v / std::ldexp(1.0f, e - 8) - 1.0f;
  int mi = (int)std::round(m * 8.0f);
  mi = std::max(0, std::min(7, mi));
  return (uint8_t)((e << 3) | mi);
}

inline float Fp8E4M3ToFloat(uint8_t b) noexcept {
  int e = (b >> 3) & 15, m = b & 7;
  if (e == 0) return std::ldexp((float)m / 8.0f, -7);
  return std::ldexp(1.0f + (float)m / 8.0f, e - 8);
}

// Quantize 32 floats -> BlockNVFP4, given global scale. Returns max abs error.
inline float QuantizeNVFP4(const float* x, float global_scale, BlockNVFP4& b) {
  float amax = 0;
  for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
  float s = amax / 6.0f / global_scale;
  if (s == 0) s = 1e-8f;
  b.scale = FloatToFp8E4M3(s);
  float rs = Fp8E4M3ToFloat(b.scale) * global_scale;
  float max_err = 0;
  std::memset(b.qs, 0, sizeof(b.qs));
  for (int i = 0; i < 32; ++i) {
    uint8_t q = FloatToE2M1(x[i] / rs);
    if (i & 1) b.qs[i / 2] |= (uint8_t)(q << 4);
    else b.qs[i / 2] |= (uint8_t)(q & 15);
    float recon = E2M1ToFloat(q) * rs;
    max_err = std::max(max_err, std::fabs(x[i] - recon));
  }
  return max_err;
}

inline float DequantNVFP4(const BlockNVFP4& b, float global_scale, int i) noexcept {
  uint8_t packed = b.qs[i / 2];
  uint8_t q = (i & 1) ? (packed >> 4) : (packed & 15);
  return E2M1ToFloat(q) * Fp8E4M3ToFloat(b.scale) * global_scale;
}

// Effective bits per weight including scale overhead.
inline double BitsPerWeightQ4K() { return (144.0 * 8) / 256; }    // 4.5
inline double BitsPerWeightNVFP4() { return (17.0 * 8) / 32; }    // 4.25

}  // namespace tinyquant
