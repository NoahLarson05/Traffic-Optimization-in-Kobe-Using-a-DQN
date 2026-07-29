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

## 使用データ

本プロジェクトでは、**国土交通省が公開している道路交通観測データ**を利用して交通需要を生成しています。

取得した交通量データを前処理し、SUMOで利用可能な交通需要（TripファイルおよびRouteファイル）へ変換することで、実交通に近いシミュレーション環境を構築しています。

交通需要の生成には、本リポジトリに含まれる `make_daily_trips.py` を使用しています。

---

## データ公開について

本リポジトリには、**国土交通省の元データおよびそこから生成された交通需要ファイルは含まれていません。**

これは、データ利用条件への配慮およびリポジトリの軽量化を目的としています。

プロジェクトを再現する場合は、国土交通省が公開している道路交通観測データを各自で取得し、本リポジトリに含まれる前処理スクリプトを用いて交通需要ファイルを生成してください。

---

## データ生成フロー

```text
Ministry of Land, Infrastructure, Transport and Tourism (MLIT)
Road Traffic Observation Data
                    │
                    ▼
              CSV Traffic Data
                    │
                    ▼
         make_daily_trips.py
                    │
                    ▼
          SUMO Trip Files
                    │
                    ▼
         SUMO Route Files
                    │
                    ▼
       DQN Training & Evaluation
```

---

## データソース

- **提供機関**：国土交通省（Ministry of Land, Infrastructure, Transport and Tourism）
- **データ種別**：道路交通観測データ
- **対象地域**：兵庫県神戸市須磨区（国道2号周辺）
- **利用目的**：交通需要生成および交通シミュレーション

---

## リポジトリに含まれないファイル

以下のファイルは公開データから生成されるため、本リポジトリには含まれていません。

```text
suma_traffic_max.csv
trips_*.xml
routes_*.rou.xml
```

これらのファイルは、国土交通省の公開データを取得した後、`make_daily_trips.py` を実行することで生成できます。

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

## 実験結果

本研究では、**Fixed-Time（Baseline）**、**Self-Organizing Traffic Lights（SOTL）**、**Max Pressure**、および **Deep Q-Network（DQN）** の4種類の交通信号制御手法を、同一の交通環境および実交通データを用いて比較しました。

### Average Halting Vehicles

<p align="center">
    <img width="2400" height="1500" alt="01_halting_comparison" src="https://github.com/user-attachments/assets/0075cb41-7a7c-4559-bf06-ff63c7636e83" />
</p>

停止車両数の比較では、DQNが4つの制御手法の中で最も少ない停止車両数を記録しました。これは、信号制御を動的に最適化することで、交差点での車両滞留を効果的に抑制できたことを示しています。

---

### Average Waiting Time

<p align="center">
    <img width="2400" height="1500" alt="02_waiting_time_comparison" src="https://github.com/user-attachments/assets/099d7674-7971-4324-95a6-1acd8d6c050e" />
</p>

平均待ち時間についても、DQNは他の制御手法より短い待ち時間を達成しました。特に、Fixed-Time、SOTL、および Max Pressure と比較して待機時間が減少しており、交通流の改善が確認できます。

---

### Traffic Density

<p align="center">
    <img width="2400" height="1500" alt="04_density_comparison" src="https://github.com/user-attachments/assets/86665247-4160-4641-a696-81840e276b26" />
</p>

交通密度の比較では、DQNが最も低い密度を維持しました。これは、交差点周辺の混雑を緩和し、より円滑な交通流を実現できたことを示しています。

---

### 結果のまとめ

今回の実験では、DQNは比較対象である Fixed-Time、SOTL、および Max Pressure と比較して、

- 停止車両数（Average Halting Vehicles）の削減
- 平均待ち時間（Average Waiting Time）の短縮
- 交通密度（Traffic Density）の低減

を達成し、本研究で評価した4つの制御手法の中で最も優れた交通制御性能を示しました。

# 今後の課題

今後は以下を予定しています。

- より広域な道路ネットワークへの適用
- 複数交差点へのDQN拡張
- 実時間交通データとの連携
- PPO・A2Cなど他の強化学習手法との比較
- より大規模な交通ネットワークでの評価

---

