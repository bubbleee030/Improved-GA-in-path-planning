# 改進 Genetic Algorithm 在 path planning 中的應用:機制示範

[English](README.md) · **繁體中文**

計算型智慧課程的期末報告。探討 Di Chen(2024)提出的改進 Genetic Algorithm,針對傳統 GA 在 path planning 中
收斂慢、路徑不平滑兩個問題:讓 crossover rate 與 mutation rate 隨迭代調整,並在 fitness function 中獎勵平滑轉彎。

`demo.py` 單獨實作論文的核心機制:

| 機制 | 論文 | `demo.py` 中的公式 |
|---|---|---|
| Non-linear crossover rate | 式 10 | `p_c(i) = p_d − √i / (1 + i)²` |
| Decreasing mutation rate | 式 11 | `p_m(i) = p_min · (1 − √(i / G))` |
| Smoothness penalty | 式 8–9 | 依轉彎角度給予懲罰 |

執行後會畫出 50 代的兩條 rate 曲線,並為一條範例路徑計分。

## 範圍

這是機制的示範,不是完整的路徑規劃器:沒有 population、selection 或地圖。`calculate_smoothness_penalty`
的懲罰權重是簡化的替代值,並非論文中的 α、β、γ。

## 執行

```bash
pip install numpy matplotlib
python demo.py
```
