# DQN checkpoints

The repository includes `dqn_aol_ga.pt`, the validated 90-episode CPU checkpoint used by the Streamlit demo and benchmark tables. Other candidate checkpoints remain excluded from Git.

Train the default model from the repository root:

```powershell
python -m training.train_dqn --datasets FT06 LA01 LA02 --episodes 90 --checkpoint models\dqn_aol_ga.pt --base-seed 42
```

The checkpoint stores network weights, optimizer state, replay memory, epsilon, random-number states, DQN configuration, training datasets, episode count, and schema versions. Retrain the model if the state or action schema changes.
