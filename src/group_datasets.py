"""
Group dataset folders under one root so `generate_combined_data_set` can pool them.

The user specifies:
  --from   where the source datasets live (optional; if omitted, dataset args are full paths)
  --out    where to save the grouped root (required; created if missing)
  positional args  which datasets to include (paths, or names relative to --from)

Each dataset is copied into --out under its original folder name.

Examples:
    # All datasets under a common parent (cleanest, no path repetition):
    python -m group_datasets --from data/raw --out data/raw/my_combined \\
        2_livernuclei 6_Drosophila_denoised 7_liver_hcc_dataset

    # Datasets scattered across the filesystem:
    python -m group_datasets --out /scratch/me/my_combined \\
        /scratch/me/liver /lab/shared/brain /home/me/heart
"""

from pathlib import Path
import shutil
from typing import List, Optional

import typer


def main(
    datasets: List[str] = typer.Argument(
        ..., help="Dataset paths (or names, when --from is set)."
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output root folder for the grouped dataset (created if missing).",
    ),
    from_dir: Optional[Path] = typer.Option(
        None,
        "--from",
        help="Resolve dataset arguments as names relative to this directory.",
    ),
):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for ds_arg in datasets:
        ds = (from_dir / ds_arg) if from_dir is not None else Path(ds_arg)
        ds = ds.resolve()
        if not ds.is_dir():
            raise typer.BadParameter(f"Not a directory: {ds}")
        dest = out / ds.name
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ds, dest)
        typer.echo(f"copied  {dest}  <-  {ds}")


if __name__ == "__main__":
    typer.run(main)
