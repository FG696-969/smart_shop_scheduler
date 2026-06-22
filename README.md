# 智能车间监测与动态重调度系统 2.0

> 基于遗传算法与 Double DQN 的作业车间调度、扰动响应和可视化仿真平台。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/Deep%20RL-Double%20DQN-EE4C2C)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Tests](https://img.shields.io/badge/tests-75%20passed-2EA44F)

## 项目亮点

- 用神经网络拟合动作价值函数，完成从表格 Q-learning 到 Double DQN 的升级。
- DQN 不直接生成调度，而是根据种群状态选择 GA 的交叉、变异和瓶颈局部搜索策略。
- 支持 FT06、LA01、LA02 与自定义 5x4 JSSP 数据。
- 支持设备故障、紧急订单、已完成工序冻结和剩余工序动态重调度。
- 提供甘特图、奖励/损失/epsilon、动作分布、Q 值、事件日志和结果导出。
- 训练模型采用版本化 checkpoint，可继续训练、替换和复现实验。

## 系统界面

### 初始调度与甘特图

![2.0 初始调度界面](docs/assets/dashboard-v2-scheduling.png)

### DQN 训练与决策监测

![2.0 DQN 训练界面](docs/assets/dashboard-v2-intelligence.png)

### 设备故障后的动态重调度

![2.0 动态重调度界面](docs/assets/dashboard-v2-rescheduled.png)

## 项目演进

```mermaid
flowchart LR
    ROOT((智能车间<br/>调度系统))

    ROOT --> V1
    ROOT --> V15
    ROOT --> V2

    subgraph V1[阶段一：基础仿真]
        direction TB
        A1[自定义 5x4 JSSP]
        A2[FIFO 基线]
        A3[固定参数 GA]
        A1 --> A2 --> A3
    end

    subgraph V15[阶段二：Q-learning]
        direction TB
        B1[JSPLIB 基准数据]
        B2[离散状态与 Q 表]
        B3[自适应 Pc / Pm]
        B1 --> B2 --> B3
    end

    subgraph V2[阶段三：DQN 最终版]
        direction TB
        C1[10 维连续状态]
        C2[Double DQN 神经网络]
        C3[5 类 GA 搜索动作]
        C4[扰动感知动态重调度]
        C1 --> C2 --> C3 --> C4
    end

    A3 -. 数据与解码器复用 .-> B1
    B3 -. 控制器升级 .-> C1

    classDef root fill:#E5484D,color:#fff,stroke:#E5484D,stroke-width:2px;
    classDef stage fill:#151A23,color:#E6EDF3,stroke:#6E7681;
    class ROOT root;
    class A1,A2,A3,B1,B2,B3,C1,C2,C3,C4 stage;
```

## 2.0 系统架构

```mermaid
flowchart LR
    subgraph INPUT[数据与事件]
        direction TB
        D1[FT06 / LA01 / LA02]
        D2[自定义 JSSP]
        D3[设备故障]
        D4[紧急订单]
    end

    subgraph CORE[调度内核]
        direction TB
        E1[染色体编码与解码]
        E2[种群评估]
        E3[交叉 / 变异 / 局部搜索]
    end

    subgraph RL[Double DQN 控制器]
        direction TB
        R1[10 维连续状态]
        R2[Online Q Network]
        R3[Target Q Network]
        R4[Replay Buffer]
        R1 --> R2
        R2 --> R3
        R4 --> R2
    end

    subgraph OUTPUT[监测与输出]
        direction TB
        O1[初始 / 扰动 / 重调度甘特图]
        O2[Cmax / 利用率 / 偏差]
        O3[学习曲线与动作分布]
        O4[CSV / HTML / 事件日志]
    end

    D1 --> E1
    D2 --> E1
    D3 --> E1
    D4 --> E1
    E1 --> E2 --> E3
    E2 --> R1
    R2 -->|选择搜索动作| E3
    E3 -->|奖励与下一状态| R4
    E3 --> O1
    E3 --> O2
    R2 --> O3
    O1 --> O4
```

## DQN-GA 决策闭环

```mermaid
flowchart LR
    S[种群状态<br/>10 个连续特征] --> Q[64-64-5<br/>Online Q Network]
    Q --> A{选择动作}
    A --> A1[强化开发]
    A --> A2[平衡搜索]
    A --> A3[增强探索]
    A --> A4[插入变异]
    A --> A5[瓶颈局部搜索]
    A1 --> G[GA 演化一代]
    A2 --> G
    A3 --> G
    A4 --> G
    A5 --> G
    G --> R["奖励：最优值、均值、<br/>多样性与动作代价"]
    R --> M[Replay Buffer]
    M --> U[Double DQN 更新]
    U --> Q
    G --> S
```

状态特征覆盖最优/平均适应度、种群离散度、停滞程度、迭代进度、剩余工序比例、机器负载离散度、故障压力和紧急订单标记。Double DQN 使用在线网络选择下一动作、目标网络评估该动作，降低 Q 值过估计。

## 实验结果

实验配置：三个 JSPLIB 实例、5 个随机种子（11/22/33/44/55）、种群规模 60、迭代 100。除 FIFO 外，所有算法使用相同搜索预算。表中为 `平均 Cmax ± 标准差`，越低越好。

| 算法 | FT06 | LA01 | LA02 |
| --- | ---: | ---: | ---: |
| FIFO | 60.00 ± 0.00 | 858.00 ± 0.00 | 904.00 ± 0.00 |
| GA | 58.20 ± 0.75 | 746.40 ± 33.58 | 767.20 ± 28.73 |
| SLGA（表格 Q-learning） | 57.80 ± 0.98 | 730.20 ± 14.63 | **739.80 ± 16.10** |
| CP-AOL-SLGA | 57.80 ± 0.75 | 748.00 ± 19.18 | 749.00 ± 15.01 |
| **DQN-AOL-GA** | **57.00 ± 1.67** | **721.00 ± 16.99** | 754.40 ± 21.36 |

| 数据集 | BKS | DQN 最优值 | DQN 平均 Gap | DQN 平均运行时间 |
| --- | ---: | ---: | ---: | ---: |
| FT06 | 55 | **55** | 3.64% | 0.2362 s |
| LA01 | 666 | 700 | 8.26% | 0.3044 s |
| LA02 | 655 | 734 | 15.18% | 0.3044 s |

结果说明：DQN-AOL-GA 在 FT06 和 LA01 的平均 Cmax 最低；LA02 上 SLGA 更稳定，说明当前训练数据量和网络规模仍有提升空间。三个数据集的平均归一化 Gap 为 9.03%，略低于 SLGA 的 9.23%。完整原始数据见 [`outputs/experiments/static_raw.csv`](outputs/experiments/static_raw.csv) 与 [`static_summary.csv`](outputs/experiments/static_summary.csv)。

### 动态故障验证

FT06、seed=42、M2 在 `t=20~40` 停机：

| 指标 | 结果 |
| --- | ---: |
| 初始计划 Cmax | 60 |
| 仅施加故障、不优化 Cmax | 74 |
| DQN 动态重调度 Cmax | **62** |
| 相对未优化方案降低 | **16.22%** |
| 调度偏差 | 120 |

## 快速开始

```powershell
git clone https://github.com/FG696-969/smart_shop_scheduler.git
cd smart_shop_scheduler
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

首次运行 DQN 前先训练模型：

```powershell
python -m training.train_dqn `
  --datasets FT06 LA01 LA02 `
  --episodes 45 `
  --checkpoint models\dqn_aol_ga.pt `
  --base-seed 84
```

启动系统：

```powershell
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。若只想快速验证训练链路，可在训练命令末尾增加 `--fast`。

## 复现实验

```powershell
python -m experiments.benchmark `
  --checkpoint models\dqn_aol_ga.pt `
  --datasets FT06 LA01 LA02 `
  --seeds 11 22 33 44 55 `
  --population 60 `
  --generations 100 `
  --output-dir outputs\experiments
```

## 目录结构

```text
smart_shop_scheduler2.0/
├─ algorithms/          # FIFO、GA、SLGA、DQN-AOL-GA
├─ rl/                  # 状态、动作、Replay Buffer、Double DQN、checkpoint
├─ services/            # 训练与调度服务
├─ training/            # DQN 命令行训练入口
├─ experiments/         # 可复现实验脚本
├─ data/                # JSSP 数据集
├─ docs/assets/         # README 页面截图
├─ outputs/experiments/ # 实验原始数据与汇总表
├─ tests/               # 单元、集成、动态重调度与 UI 测试
└─ app.py               # Streamlit 入口
```

## Q-learning 与 DQN 的区别

| 项目 | SLGA | DQN-AOL-GA |
| --- | --- | --- |
| Q 函数 | 10 状态的离散 Q 表 | 神经网络拟合连续状态 Q 值 |
| 状态表达 | 最优值、均值和多样性离散化 | 10 维连续种群/扰动特征 |
| 动作 | 调整交叉率和变异率 | 5 类搜索策略，包括局部搜索 |
| 经验利用 | 当前 episode 在线更新 | Replay Buffer 随机采样 |
| 稳定机制 | 无目标网络 | Online/Target 双网络 |
| 模型保存 | Q 表未持久化 | 版本化 PyTorch checkpoint |

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

当前测试覆盖状态编码、动作空间、Replay Buffer、Double DQN 更新、checkpoint 恢复、静态/动态调度、训练服务、实验汇总和 Streamlit 页面冒烟测试。

## 后续改进

- 扩充训练实例与 episode 数量，重点提升 LA02 泛化稳定性。
- 加入 Prioritized Experience Replay、Dueling DQN 或 n-step return。
- 引入更多动态事件、交期与总拖期目标，扩展为多目标调度。
- 在更大规模 JSPLIB 实例上进行显著性检验与消融实验。
