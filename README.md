# Traffic Optimization in Kobe Using Deep Q-Network (DQN)

<p align="center">
    <img width="624" height="452" alt="image" src="https://github.com/user-attachments/assets/fe8898be-38d3-4907-bb4a-91d373e1e550" />
</p>

## 概要

本プロジェクトは、**神戸市須磨区の実交通データ**を利用し、SUMO（Simulation of Urban MObility）上で交通流を再現し、**Deep Q-Network（DQN）**による信号制御最適化を行う研究プロジェクトです。

従来の固定時間制御に加え、

- Fixed-Time（Baseline）
- Self-Organizing Traffic Lights（SOTL）
- Max Pressure
- Deep Q-Network（DQN）

の4種類の信号制御手法を同一環境で比較評価できるよう設計されています。

最新版では、

- 複数日の実交通データを用いた学習・検証
- ハイパーパラメータ自動最適化
- ロバストネス評価
- 可視化ツール

まで含めた実験環境となっています。

---

# システム構成

```
実交通データ
        │
        ▼
交通量CSV
        │
        ▼
make_daily_trips.py
        │
        ▼
SUMO Route生成
        │
        ▼
SUMOシミュレーション
        │
        ├──────────────┐
        │              │
        ▼              ▼
従来制御        DQNエージェント
(Baseline /     (PyTorch)
SOTL / MP)
        │              │
        └──────┬───────┘
               ▼
        性能評価・比較
               │
               ▼
      グラフ・統計・分析
```

---

# 主な機能

- SUMOによる交通シミュレーション
- 神戸市実交通データの利用
- DQNによる交通信号制御
- Baselineとの比較
- Max Pressureとの比較
- SOTLとの比較
- ハイパーパラメータ最適化
- 学習済みモデル保存
- 検証データによる性能評価
- ロバストネス評価
- グラフ自動生成

---

# ディレクトリ構成

```
.
├── dqn_sumo_tls_all_metrics.py
├── optimize_dqn.py
├── baseline_test_all_metrics.py
├── run_mp_sotl_all_metrics.py
├── robustness_eval.py
├── compare_validation.py
├── analyze_optimization.py
├── visualization.py
├── plot_controller_comparison_graphs.py
├── make_daily_trips.py
├── osm.sumocfg
│
├── opt_trials/
│     ├── trial_00_...
│     ├── trial_01_...
│     └── ...
│
├── optimization_results.csv
├── dqn_best_config.json
├── dqn_traffic_model_best.pth
│
├── comparison_graphs/
├── dqn_graphs/
├── optimization_analysis/
└── final_comparison_graphs/
```

---

# 各プログラムの説明

| ファイル | 説明 |
|----------|------|
| `make_daily_trips.py` | 実交通データから日別のSUMO Routeを生成 |
| `dqn_sumo_tls_all_metrics.py` | DQNの学習・検証 |
| `baseline_test_all_metrics.py` | 固定時間制御の評価 |
| `run_mp_sotl_all_metrics.py` | Max Pressure・SOTLの評価 |
| `optimize_dqn.py` | ハイパーパラメータ探索 |
| `analyze_optimization.py` | 最適化結果の分析 |
| `robustness_eval.py` | 複数乱数シードによるロバストネス評価 |
| `compare_validation.py` | 各制御手法の性能比較 |
| `visualization.py` | DQN学習・検証結果の可視化 |
| `plot_controller_comparison_graphs.py` | 比較グラフ生成 |

---

# 使用技術

- Python
- PyTorch
- SUMO
- TraCI API
- NumPy
- Pandas
- Matplotlib

---

# 学習手順

### ① 実交通データからRoute生成

```bash
python make_daily_trips.py
```

---

### ② DQN学習

```bash
python dqn_sumo_tls_all_metrics.py
```

---

### ③ Baseline評価

```bash
python baseline_test_all_metrics.py
```

---

### ④ Max Pressure・SOTL評価

```bash
python run_mp_sotl_all_metrics.py
```

---

### ⑤ DQNハイパーパラメータ探索

```bash
python optimize_dqn.py
```

---

### ⑥ パラメータ分析

```bash
python analyze_optimization.py
```

---

### ⑦ ロバストネス評価

```bash
python robustness_eval.py --controller dqn
```

---

### ⑧ グラフ生成

```bash
python visualization.py
```

または

```bash
python plot_controller_comparison_graphs.py
```

---

# 評価指標

本研究では以下の交通指標を用いて性能を比較しています。

- Average Halting Vehicles
- Average Waiting Time
- Episode Mean Total Delay (EMTD)
- Average Traffic Density
- Throughput

---

# ハイパーパラメータ最適化

DQNでは以下のパラメータを自動探索します。

- Reward Weights
- Learning Rate
- Discount Factor
- Batch Size
- Target Update Frequency
- Action Interval

探索後は

- 最良モデル
- 最良設定
- ランキング
- パラメータ重要度

を自動保存します。

---

# 実験結果

各制御手法について

- Baseline
- SOTL
- Max Pressure
- DQN

を比較し、

- 学習曲線
- EMTD
- Waiting Time
- Traffic Density
- Throughput

などを自動でグラフ化できます。

---

# 今後の課題

今後は以下を予定しています。

- より広域な道路ネットワークへの適用
- 複数交差点へのDQN拡張
- 実時間交通データとの連携
- PPO・A2Cなど他の強化学習手法との比較
- より大規模な交通ネットワークでの評価

---

