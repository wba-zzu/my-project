from pathlib import Path


def increment_experiment_dir(base_dir: str, name: str) -> Path:
    """Return a non-existing experiment directory.

    Examples:
        base/exp   -> if missing, use base/exp
        base/exp   -> if exists, use base/exp2
        base/exp2  -> if exists too, use base/exp3
    """
    base_dir = Path(base_dir).expanduser().resolve()

    if not name or not name.strip():
        raise ValueError("Experiment name must not be empty.")

    name = name.strip()
    candidate = base_dir / name

    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_dir / f"{name}{index}"
        if not candidate.exists():
            return candidate
        index += 1


def experiment_dir_from_resume(resume_path: str) -> Path:
    """Infer the original experiment directory from a resume checkpoint.

    Expected layout after the logging patch:
        exp/
        ├── results.csv
        ├── log.txt
        └── weights/
            ├── last.pth
            ├── best.pth
            └── epoch005.pth

    For a checkpoint under exp/weights/, this returns exp/.
    For a checkpoint directly under an experiment directory, this returns
    the checkpoint's parent directory.
    """
    checkpoint = Path(resume_path).expanduser().resolve()

    if checkpoint.parent.name == "weights":
        return checkpoint.parent.parent

    return checkpoint.parent


def resolve_experiment_dir(base_dir: str, name: str, resume: str = None) -> Path:
    """Resolve the actual output directory for this run.

    Rules:
      1. Resume training always continues writing into the original experiment.
      2. Fresh training uses base_dir/name.
      3. If base_dir/name already exists, automatically use name2, name3, ...
    """
    if resume:
        return experiment_dir_from_resume(resume)

    return increment_experiment_dir(base_dir, name)