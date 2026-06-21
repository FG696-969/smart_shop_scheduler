# Smart Shop Scheduler 2.0: DQN-AOL-GA Design

## 1. Context

The current system combines JSSP genetic search with a tabular Q-learning-style controller. The controller compresses population features into one of ten discrete states and uses a Q table to choose crossover, mutation, and local-search strategies.

Version 2.0 replaces that table with a neural Q-function while preserving the validated chromosome representation, schedule decoder, disturbance handling, metrics, and Streamlit workflow. The resulting algorithm is named **DQN-AOL-GA**: a Double-DQN adaptive operator learning controller wrapped around the existing GA search engine.

The implementation target is `FINAL_REPORT_DIR\smart_shop_scheduler2.0`, where `FINAL_REPORT_DIR` is the user-provided final-report folder. It will be developed from the existing Git repository in an isolated branch/worktree so the 1.0 directory remains usable.

## 2. Goals

- Implement a real neural-network approximation of `Q(s, a)` using PyTorch.
- Let DQN select GA parameter/operator strategies once per generation.
- Support both initial scheduling and event-driven dynamic rescheduling.
- Preserve FIFO, GA, SLGA, and CP-AOL-SLGA as comparison baselines.
- Save and load trained model checkpoints with reproducible metadata.
- Expose DQN training and inference state in the Streamlit interface.
- Produce feasible schedules for Custom 5x4, FT06, LA01, and LA02.
- Keep CPU execution practical for classroom demonstration.

## 3. Non-goals

- The neural network will not directly construct an operation sequence.
- This version will not implement PPO, GNN scheduling, multi-agent RL, FJSP, MES/PLC integration, or a new presentation deck.
- The implementation will not claim DQN superiority unless repeated experiments support it.
- Experiment reports and README/GitHub publication follow system completion; they are not prerequisites for the first runnable 2.0 system.

## 4. Selected Architecture

```text
Streamlit UI
    |
Application service / run configuration
    |
Algorithm registry
    |-- FIFO
    |-- Fixed GA
    |-- Tabular SLGA
    |-- CP-AOL-SLGA
    `-- DQN-AOL-GA
            |
            |-- GA population and operators
            |-- Continuous state encoder
            |-- Double-DQN agent
            |-- Replay buffer and target network
            `-- Checkpoint manager
    |
Shared JSSP decoder and disturbance rescheduler
    |
Metrics, Plotly visualizations, exports
```

The GA remains responsible for generating candidate schedules. DQN observes the population and scheduling context, selects one search strategy, receives a reward after the generation, and learns from stored transitions.

## 5. DQN State

The state is a fixed-size normalized float vector so it can be used across the included datasets. It contains:

1. Current best makespan relative to the episode's initial best.
2. Current average makespan relative to the episode's initial average.
3. Relative gap between average and best fitness.
4. Population diversity ratio.
5. Normalized fitness standard deviation.
6. Consecutive stagnation generations divided by the generation budget.
7. Current generation divided by the generation budget.
8. Remaining-operation ratio.
9. Machine-load imbalance coefficient.
10. Disturbance context value: zero for initial scheduling; normalized breakdown pressure plus an emergency-job indicator for rescheduling.

All denominators use guarded minimum values. NaN and infinite values are rejected before agent inference.

## 6. DQN Actions

The first version keeps five discrete, interpretable actions to preserve comparison with CP-AOL-SLGA:

| Action | Pc | Pm | Mutation/search behavior |
| --- | ---: | ---: | --- |
| Exploit | 0.90 | 0.08 | Swap mutation |
| Balanced | 0.80 | 0.18 | Swap mutation |
| Explore | 0.65 | 0.28 | Swap mutation |
| Insertion | 0.75 | 0.22 | Insertion mutation |
| Bottleneck refine | 0.80 | 0.14 | Swap plus bottleneck local search |

The action definitions live in one shared registry so tabular and neural controllers cannot silently use different operators during comparison.

## 7. Reward

The per-generation reward is normalized and bounded:

```text
reward = 0.65 * best_improvement
       + 0.20 * average_improvement
       + 0.10 * diversity_recovery
       - 0.05 * operator_cost
```

- `best_improvement` and `average_improvement` are relative makespan decreases.
- `diversity_recovery` is positive only when diversity improves during stagnation.
- `operator_cost` discourages unnecessary local search when it produces no gain.
- A small terminal bonus is awarded when the episode improves its global best.
- Rewards are clipped to a fixed range for stable training.

Schedule deviation remains a separately reported dynamic-rescheduling metric. It will not be folded into the first training reward because that would change the optimization objective and make comparison with existing baselines ambiguous.

## 8. Agent and Training Configuration

- Framework: PyTorch on CPU.
- Online network: input -> 64 ReLU -> 64 ReLU -> five Q values.
- Target network: identical structure, synchronized periodically.
- Update: Double DQN target with Huber loss and Adam.
- Replay buffer: bounded FIFO memory with random mini-batches.
- Exploration: epsilon-greedy with configurable decay and deterministic inference mode.
- Reproducibility: Python, NumPy, and PyTorch seeds are set from one run configuration.
- Checkpoint: model weights plus JSON metadata containing state version, action version, feature normalization, seed, datasets, and training configuration.

The UI defaults to loading a compatible checkpoint for inference. Training is an explicit action and never occurs silently during ordinary schedule generation. If no checkpoint exists, the user can train a model or run a clearly labelled untrained demo policy.

## 9. Proposed Modules

```text
smart_shop_scheduler2.0/
|-- app.py
|-- algorithms/
|   |-- common.py
|   |-- ga.py
|   |-- tabular_ga.py
|   `-- dqn_ga.py
|-- rl/
|   |-- actions.py
|   |-- state_encoder.py
|   |-- replay_buffer.py
|   |-- dqn_agent.py
|   `-- checkpoint.py
|-- services/
|   |-- scheduling.py
|   `-- training.py
|-- training/
|   `-- train_dqn.py
|-- experiments/
|-- models/
|-- tests/
|-- data/
|-- scheduler_core.py
|-- disturbance.py
|-- metrics.py
|-- visualization.py
|-- requirements.txt
`-- README.md
```

The existing scheduling decoder and disturbance semantics remain shared and are changed only where tests identify a correctness issue.

## 10. Streamlit Interface

The interface keeps the existing scheduling workflow but separates it into three views:

### Scheduling

- Dataset, algorithm, seed, and fast/full mode.
- Initial schedule generation.
- Metric summary and Gantt schedule.

### Intelligence

- DQN checkpoint status and compatibility.
- Explicit train/load/reset controls.
- Episode reward, training loss, epsilon, Q-value, and action-frequency charts.
- GA convergence and selected strategy trace.

### Disturbance Analysis

- Machine breakdown and emergency-job controls.
- Initial, disturbed, and rescheduled Gantt views.
- Makespan, utilization, idle time, runtime, BKS gap, and schedule deviation comparison.

Training errors are reported without destroying the last valid schedule in session state.

## 11. Model Lifecycle and Error Handling

- A checkpoint is accepted only when its state/action schema matches the running code.
- Missing or incompatible checkpoints produce an actionable UI warning.
- Training cancellation or failure leaves the previous checkpoint untouched.
- Checkpoints are written atomically through a temporary file and rename.
- Invalid chromosomes, non-finite states, empty remaining schedules, and malformed emergency routes are explicit errors.
- DQN inference uses a deterministic greedy policy unless exploration is explicitly enabled.

## 12. Testing

Tests will cover:

- State vector shape, range, determinism, and finite values.
- Replay-buffer capacity and sample shape.
- DQN forward pass, greedy action, training update, and target synchronization.
- Checkpoint round-trip and incompatibility rejection.
- GA chromosome validity after every action.
- Feasible schedules and deterministic seeded runs on the included datasets.
- Dynamic rescheduling with breakdown, emergency job, and no remaining work.
- Streamlit service-layer behavior without requiring browser interaction for core tests.

Browser verification will then cover model controls, schedule generation, training charts, and disturbance rescheduling at desktop width.

## 13. System-first Delivery Sequence

1. Create the isolated 2.0 worktree and copy only intentional project files.
2. Add tests for state/action/reward and DQN agent behavior.
3. Extract the existing GA into a controller-neutral engine.
4. Implement Double DQN, replay memory, checkpointing, and training service.
5. Integrate DQN-AOL-GA with initial and dynamic scheduling.
6. Update the Streamlit interface and visualizations.
7. Run automated and browser verification on the complete system.
8. After system acceptance, run repeated experiments, update README, and publish GitHub changes.

## 14. Acceptance Criteria

- `DQN-AOL-GA` uses neural Q-values and contains no Q table fallback under that name.
- FIFO, GA, SLGA, CP-AOL-SLGA, and DQN-AOL-GA remain selectable.
- A model can be trained, saved, loaded in a new process, and used for inference.
- Custom 5x4, FT06, LA01, and LA02 produce feasible schedules.
- Machine breakdown and emergency-job scenarios complete without corrupting frozen operations.
- Fixed seeds reproduce action traces and final makespan within deterministic CPU execution.
- The UI clearly distinguishes trained, untrained, missing, and incompatible model states.
- The complete 2.0 system runs from the final-report folder using documented commands.

## 15. Git and Publication Strategy

- Preserve the current 1.0 checkout and commit history.
- Build 2.0 in `FINAL_REPORT_DIR\smart_shop_scheduler2.0` on an isolated feature branch/worktree.
- Commit in reviewable stages: architecture/tests, DQN core, integration, UI, and final verification.
- Do not push incomplete or failing work to `main`.
- After experiments and documentation pass, update the existing public GitHub repository and tag the final release as `v2.0.0`.
