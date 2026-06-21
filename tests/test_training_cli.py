from pathlib import Path

from training.train_dqn import main


def test_training_cli_prints_episode_and_checkpoint_summary(
    tmp_path: Path, capsys
):
    checkpoint = tmp_path / "cli.pt"

    exit_code = main(
        [
            "--datasets",
            "Custom 5x4",
            "--episodes",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--fast",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Episode 1/1" in output
    assert "Checkpoint saved:" in output
    assert checkpoint.exists()


def test_training_cli_returns_nonzero_without_replacing_checkpoint(
    tmp_path: Path, capsys
):
    checkpoint = tmp_path / "cli.pt"
    checkpoint.write_bytes(b"last-known-good")

    exit_code = main(
        [
            "--datasets",
            "Missing Dataset",
            "--episodes",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--fast",
        ]
    )

    error = capsys.readouterr().err
    assert exit_code == 1
    assert "Training failed:" in error
    assert checkpoint.read_bytes() == b"last-known-good"
