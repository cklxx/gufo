// Tiny MoE model for CPU quantization experiments.
// 8 experts, top-2 routing, hidden=256, expert intermediate=512.
// Random weights with realistic magnitude distribution (Xavier init).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "quant.h"

namespace tinyquant {

struct TinyMoE {
  int n_experts = 8;
  int top_k = 2;
  int hidden = 256;
  int inter = 512;
  // gate_w: [hidden] shared router scalars; per expert: w1 [hidden x inter], w2 [inter x hidden]
  std::vector<float> gate_w;
  std::vector<float> w1;  // n_experts * hidden * inter
  std::vector<float> w2;  // n_experts * inter * hidden

  void init(uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    auto fill = [&](std::vector<float>& v, int fan_in, int fan_out) {
      float s = std::sqrt(2.0f / (fan_in + fan_out));
      std::normal_distribution<float> d(0.0f, s);
      for (auto& x : v) x = d(rng);
    };
    gate_w.resize(hidden);
    fill(gate_w, hidden, n_experts);
    w1.resize((size_t)n_experts * hidden * inter);
    fill(w1, hidden, inter);
    w2.resize((size_t)n_experts * inter * hidden);
    fill(w2, inter, hidden);
  }

  // Reference forward (fp32). x: [hidden] -> out: [hidden]
  void forward_ref(const float* x, float* out) const {
    float logits[8] = {0};
    for (int i = 0; i < hidden; ++i)
      for (int e = 0; e < n_experts; ++e)
        logits[e] += x[i] * gate_w[i] * (0.5f + 0.1f * e);
    int top[2] = {0, 1};
    for (int e = 0; e < n_experts; ++e)
      for (int k = 0; k < top_k; ++k)
        if (logits[e] > logits[top[k]]) {
          for (int j = top_k - 1; j > k; --j) top[j] = top[j - 1];
          top[k] = e;
          break;
        }
    float wsum = 0, ws[2];
    for (int k = 0; k < top_k; ++k) {
      ws[k] = std::exp(logits[top[k]]);
      wsum += ws[k];
    }
    for (int k = 0; k < top_k; ++k) ws[k] /= wsum;

    for (int i = 0; i < hidden; ++i) out[i] = 0;
    std::vector<float> h(inter);
    for (int k = 0; k < top_k; ++k) {
      int e = top[k];
      const float* e1 = &w1[(size_t)e * hidden * inter];
      const float* e2 = &w2[(size_t)e * inter * hidden];
      for (int j = 0; j < inter; ++j) {
        float s = 0;
        for (int i = 0; i < hidden; ++i) s += x[i] * e1[i * inter + j];
        h[j] = s / (1.0f + std::exp(-s));  // SiLU
      }
      for (int i = 0; i < hidden; ++i) {
        float s = 0;
        for (int j = 0; j < inter; ++j) s += h[j] * e2[j * hidden + i];
        out[i] += ws[k] * s;
      }
    }
  }
};

// Quantized weight store: per-format blocks + dequant GEMV.
struct QuantMoE {
  const TinyMoE* ref;
  std::vector<BlockQ4K> q4k_w1, q4k_w2;      // 256 weights/block
  std::vector<BlockNVFP4> fp4_w1, fp4_w2;    // 32 weights/block
  float fp4_gscale_w1 = 1, fp4_gscale_w2 = 1; // per-tensor global scale

  explicit QuantMoE(const TinyMoE* m) : ref(m) { quantize(); }

  void quantize() {
    size_t n1 = ref->w1.size(), n2 = ref->w2.size();
    q4k_w1.resize((n1 + 255) / 256);
    q4k_w2.resize((n2 + 255) / 256);
    for (size_t b = 0; b < q4k_w1.size(); ++b) {
      float tmp[256] = {0};
      size_t n = std::min<size_t>(256, n1 - b * 256);
      for (size_t i = 0; i < n; ++i) tmp[i] = ref->w1[b * 256 + i];
      QuantizeQ4K(tmp, q4k_w1[b]);
    }
    for (size_t b = 0; b < q4k_w2.size(); ++b) {
      float tmp[256] = {0};
      size_t n = std::min<size_t>(256, n2 - b * 256);
      for (size_t i = 0; i < n; ++i) tmp[i] = ref->w2[b * 256 + i];
      QuantizeQ4K(tmp, q4k_w2[b]);
    }
    // NVFP4 global scale = amax / 448 (FP8 E4M3 max)
    float a1 = 0, a2 = 0;
    for (float v : ref->w1) a1 = std::max(a1, std::fabs(v));
    for (float v : ref->w2) a2 = std::max(a2, std::fabs(v));
    fp4_gscale_w1 = a1 / 448.0f;
    fp4_gscale_w2 = a2 / 448.0f;
    fp4_w1.resize((n1 + 31) / 32);
    fp4_w2.resize((n2 + 31) / 32);
    for (size_t b = 0; b < fp4_w1.size(); ++b) {
      float tmp[32] = {0};
      size_t n = std::min<size_t>(32, n1 - b * 32);
      for (size_t i = 0; i < n; ++i) tmp[i] = ref->w1[b * 32 + i];
      QuantizeNVFP4(tmp, fp4_gscale_w1, fp4_w1[b]);
    }
    for (size_t b = 0; b < fp4_w2.size(); ++b) {
      float tmp[32] = {0};
      size_t n = std::min<size_t>(32, n2 - b * 32);
      for (size_t i = 0; i < n; ++i) tmp[i] = ref->w2[b * 32 + i];
      QuantizeNVFP4(tmp, fp4_gscale_w2, fp4_w2[b]);
    }
  }

  // Dequant accessor: 0=Q4_K, 1=NVFP4
  float w1_at(int fmt, size_t idx) const {
    if (fmt == 0) return DequantQ4K(q4k_w1[idx / 256], (int)(idx % 256));
    return DequantNVFP4(fp4_w1[idx / 32], fp4_gscale_w1, (int)(idx % 32));
  }
  float w2_at(int fmt, size_t idx) const {
    if (fmt == 0) return DequantQ4K(q4k_w2[idx / 256], (int)(idx % 256));
    return DequantNVFP4(fp4_w2[idx / 32], fp4_gscale_w2, (int)(idx % 32));
  }

  // Forward with quantized weights (fp32 activations). fmt: 0=Q4_K, 1=NVFP4
  void forward_q(int fmt, const float* x, float* out) const {
    int H = ref->hidden, I = ref->inter, E = ref->n_experts, K = ref->top_k;
    float logits[8] = {0};
    for (int i = 0; i < H; ++i)
      for (int e = 0; e < E; ++e)
        logits[e] += x[i] * ref->gate_w[i] * (0.5f + 0.1f * e);
    int top[2] = {0, 1};
    for (int e = 0; e < E; ++e)
      for (int k = 0; k < K; ++k)
        if (logits[e] > logits[top[k]]) {
          for (int j = K - 1; j > k; --j) top[j] = top[j - 1];
          top[k] = e;
          break;
        }
    float wsum = 0, ws[2];
    for (int k = 0; k < K; ++k) {
      ws[k] = std::exp(logits[top[k]]);
      wsum += ws[k];
    }
    for (int k = 0; k < K; ++k) ws[k] /= wsum;

    for (int i = 0; i < H; ++i) out[i] = 0;
    std::vector<float> h(I);
    for (int k = 0; k < K; ++k) {
      int e = top[k];
      size_t b1 = (size_t)e * H * I, b2 = (size_t)e * I * H;
      for (int j = 0; j < I; ++j) {
        float s = 0;
        for (int i = 0; i < H; ++i) s += x[i] * w1_at(fmt, b1 + (size_t)i * I + j);
        h[j] = s / (1.0f + std::exp(-s));
      }
      for (int i = 0; i < H; ++i) {
        float s = 0;
        for (int j = 0; j < I; ++j) s += h[j] * w2_at(fmt, b2 + (size_t)j * H + i);
        out[i] += ws[k] * s;
      }
    }
  }
};

}  // namespace tinyquant
