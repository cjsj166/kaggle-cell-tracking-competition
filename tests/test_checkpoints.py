from pathlib import Path

import pytest
import torch

from tracking_cellmot.checkpoints import CheckpointInfo, load_training_checkpoint


def test_load_training_checkpoint_restores_model_optimizer_and_metadata(
    tmp_path: Path,
) -> None:
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.012)
    source(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    expected_model = {key: value.detach().clone() for key, value in source.state_dict().items()}
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "epoch": 4,
            "global_step": 37,
        },
        path,
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
    info = load_training_checkpoint(path, model, optimizer)

    assert info == CheckpointInfo(epoch=4, global_step=37)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[key])
    assert optimizer.param_groups[0]["lr"] == 0.012
    assert len(optimizer.state) == len(source_optimizer.state)


def test_load_training_checkpoint_requires_optimizer_state_when_requested(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pth"
    model = torch.nn.Linear(1, 1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "global_step": 2,
        },
        path,
    )

    with pytest.raises(ValueError, match="optimizer_state_dict"):
        load_training_checkpoint(
            path,
            model,
            torch.optim.AdamW(model.parameters()),
        )


def test_load_training_checkpoint_rejects_non_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    torch.save(torch.nn.Linear(1, 1).state_dict(), path)

    with pytest.raises(ValueError, match="Not a training checkpoint"):
        load_training_checkpoint(path, torch.nn.Linear(1, 1))


def test_load_training_checkpoint_reports_incompatible_model_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {"unexpected": torch.ones(1)},
            "epoch": 1,
            "global_step": 2,
        },
        path,
    )

    with pytest.raises(ValueError, match="Invalid training checkpoint state"):
        load_training_checkpoint(path, torch.nn.Linear(1, 1))
