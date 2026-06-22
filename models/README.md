# DQN checkpoints

PyTorch checkpoint files are generated locally and intentionally excluded from Git.

Train the default model from the repository root:

```powershell
python -m training.train_dqn --datasets FT06 LA01 LA02 --episodes 45 --checkpoint models\dqn_aol_ga.pt --base-seed 84
```

The checkpoint stores network weights, optimizer state, replay memory, epsilon, random-number states, DQN configuration, training datasets, episode count, and schema versions. Retrain the model if the state or action schema changes.
