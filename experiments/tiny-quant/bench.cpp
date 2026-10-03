// Compare Q4_K vs NVFP4-style microscaling on a tiny MoE.
// Metrics: weight SNR, end-to-end output SNR, dequant+GEMV time (CPU).
// Build: g++ -O3 -std=c++17 -o bench bench.cpp && ./bench
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "tiny_model.h"

using namespace tinyquant;
using hrc = std::chrono::high_resolution_clock;

static double snr_db(const float* ref, const float* q, int n) {
  double sig = 0, noise = 0;
  for (int i = 0; i < n; ++i) {
    sig += (double)ref[i] * ref[i];
    double e = (double)ref[i] - q[i];
    noise += e * e;
  }
  return 10 * std::log10(sig / (noise + 1e-30));
}

int main() {
  printf("bits/weight: Q4_K=%.2f  NVFP4=%.2f\n",
         BitsPerWeightQ4K(), BitsPerWeightNVFP4());

  TinyMoE model;
  model.init(42);
  QuantMoE qm(&model);

  // 1. Weight-level SNR
  {
    int n = (int)model.w1.size();
    std::vector<float> rq(n);
    for (int i = 0; i < n; ++i) rq[i] = qm.w1_at(0, i);
    double s4 = snr_db(model.w1.data(), rq.data(), n);
    for (int i = 0; i < n; ++i) rq[i] = qm.w1_at(1, i);
    double sf = snr_db(model.w1.data(), rq.data(), n);
    printf("[w1] Q4_K SNR=%.2f dB   NVFP4 SNR=%.2f dB\n", s4, sf);
  }

  // 2. End-to-end: 200 random inputs
  const int NTRIAL = 200, H = model.hidden;
  std::mt19937_64 rng(7);
  std::normal_distribution<float> d(0, 1);
  double snr_q4 = 0, snr_fp4 = 0;
  std::vector<float> x(H), o_ref(H), o_q(H);
  for (int t = 0; t < NTRIAL; ++t) {
    for (int i = 0; i < H; ++i) x[i] = d(rng) * 0.5f;
    model.forward_ref(x.data(), o_ref.data());
    qm.forward_q(0, x.data(), o_q.data());
    snr_q4 += snr_db(o_ref.data(), o_q.data(), H);
    qm.forward_q(1, x.data(), o_q.data());
    snr_fp4 += snr_db(o_ref.data(), o_q.data(), H);
  }
  printf("[e2e]  Q4_K SNR=%.2f dB   NVFP4 SNR=%.2f dB  (avg over %d)\n",
         snr_q4 / NTRIAL, snr_fp4 / NTRIAL, NTRIAL);

  // 3. Dequant+GEMV speed (single expert w1 GEMV, HxI)
  const int REPS = 50;
  volatile float sink = 0;
  for (int i = 0; i < H; ++i) x[i] = d(rng);
  std::vector<float> y(model.inter);
  for (int fmt = 0; fmt < 2; ++fmt) {
    auto t0 = hrc::now();
    for (int r = 0; r < REPS; ++r) {
      for (int j = 0; j < model.inter; ++j) {
        float s = 0;
        for (int i = 0; i < H; ++i) s += x[i] * qm.w1_at(fmt, (size_t)i * model.inter + j);
        y[j] = s;
      }
      sink += y[0];
    }
    auto t1 = hrc::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
    printf("[gemv] %s: %.1f us/rep\n", fmt == 0 ? "Q4_K " : "NVFP4", us);
  }
  (void)sink;

  // 4. Size
  size_t q4bytes = qm.q4k_w1.size() * sizeof(BlockQ4K) + qm.q4k_w2.size() * sizeof(BlockQ4K);
  size_t f4bytes = qm.fp4_w1.size() * sizeof(BlockNVFP4) + qm.fp4_w2.size() * sizeof(BlockNVFP4);
  size_t fp32bytes = (model.w1.size() + model.w2.size()) * 4;
  printf("[size] fp32=%zu KB  Q4_K=%zu KB (%.1f%%)  NVFP4=%zu KB (%.1f%%)\n",
         fp32bytes / 1024, q4bytes / 1024, 100.0 * q4bytes / fp32bytes,
         f4bytes / 1024, 100.0 * f4bytes / fp32bytes);
  return 0;
}
