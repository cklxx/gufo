# tiny-quant: CPU 量化对比实验

在 CPU 上对比 Q4_K vs NVFP4 风格 microscaling，用 tiny MoE 模型验证。

## 文件

- `quant.h`：Q4_K 和 NVFP4 的量化/反量化实现
  - Q4_K：GGUF 格式，256 weights/block，8x32 sub-block，6-bit scale/min
  - NVFP4：E2M1，32 weights/group，FP8(E4M3) scale，全局 FP32 scale
- `tiny_model.h`：Tiny MoE（8 experts, top-2, hidden=256, inter=512）
- `bench.cpp`：主对比测试

## 构建运行

```bash
g++ -O3 -std=c++17 -o bench bench.cpp && ./bench
```

## 结果（2026-10-03）

| 指标 | Q4_K | NVFP4 |
|---|---|---|
| bits/weight | 4.50 | 4.25 |
| 权重 SNR | 22.11 dB | 19.86 dB |
| 端到端 SNR | 19.08 dB | 16.77 dB |
| 大小（vs fp32） | 14.1% | 13.3% |

**结论**：Q4_K 在 tiny 模型上稳定好约 2dB。

### 为什么 Q4_K 更好

1. **位宽**：Q4_K 4.5 bits/weight，NVFP4 4.25 bits/weight，多 0.25 bits
2. **scale 精度**：Q4_K 的 6-bit sub-block scale（1.6% 相对误差）vs NVFP4 的 FP8 scale（12.5% 相对误差）
3. **min offset**：Q4_K 有 6-bit per-subblock min，NVFP4 无（对称量化）

### 试过的改进（都没反超）

- NVFP4 换 FP16 scale：19.86 dB（无提升，scale 精度不是瓶颈）
- NVFP4 最优 scale 搜索：20.73 dB（+0.9 dB，仍差 1.4 dB）
- NVFP4 + per-group min（非对称）：12.40 dB（更差，浪费码本）
- 1% outliers 分布：Q4_K 20.14 dB vs NVFP4 18.87 dB（Q4_K 仍胜）

## 下一步

- 试 Q5_K / Q6_K（更高位宽）
- 试 per-channel NVFP4（真实 LLM 用法）
- 在真实模型权重上验证（非随机权重）
