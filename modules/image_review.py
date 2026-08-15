from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import shutil

from fastai.interpret import ClassificationInterpretation


def _unique_destination(path: Path, reserved: set[Path]) -> Path:
    """Return a destination that will not overwrite an existing or planned file."""
    candidate = path
    counter = 1
    while candidate.exists() or candidate in reserved:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    reserved.add(candidate)
    return candidate


class TopLossCleaner:
    """Widget-free review helper for folder-labeled image classifiers."""

    def __init__(self, learn, k: int | None = 60, ds_idx: int = 1, dl=None):
        self.interp = ClassificationInterpretation.from_learner(
            learn,
            ds_idx=ds_idx,
            dl=dl,
        )

        total = len(self.interp.losses)
        if total == 0:
            raise ValueError("The selected dataset contains no examples.")

        if k is None:
            k = total
        elif k < 1:
            raise ValueError("k must be at least 1 or None.")
        else:
            k = min(int(k), total)

        self.losses, self.dataset_idxs, self.items = self.interp.top_losses(
            k=k,
            items=True,
        )

    def __len__(self) -> int:
        return len(self.items)

    def show(
        self,
        page: int = 1,
        page_size: int = 12,
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """Show one static page and print its stable review IDs and file paths."""
        if page < 1:
            raise ValueError("page is 1-based and must be at least 1.")
        if page_size < 1:
            raise ValueError("page_size must be at least 1.")

        start = (page - 1) * page_size
        stop = min(start + page_size, len(self))

        if start >= len(self):
            last_page = (len(self) + page_size - 1) // page_size
            raise ValueError(f"Page {page} is out of range. Last page: {last_page}.")

        self.interp.plot_top_losses(range(start, stop), figsize=figsize)

        last_page = (len(self) + page_size - 1) // page_size
        print(f"Page {page}/{last_page}. Use the IDs below when recording decisions.")
        for review_id in range(start, stop):
            loss = float(self.losses[review_id].item())
            print(f"{review_id:>3} | loss={loss:.4f} | {self.items[review_id]}")

    def apply(
        self,
        dataset_root: str | Path,
        *,
        delete: Iterable[int] = (),
        relabel: Mapping[int, str] | None = None,
        dry_run: bool = True,
        rejected_dir: str | Path | None = None,
    ) -> list[dict[str, str | int]]:
        """
        Preview or apply review decisions.

        `delete` moves files to a reversible quarantine directory.
        `relabel` moves files to another class folder beneath `dataset_root`.
        """
        delete_ids = {int(review_id) for review_id in delete}
        relabel_map = {
            int(review_id): str(label).strip()
            for review_id, label in (relabel or {}).items()
        }

        overlap = delete_ids & relabel_map.keys()
        if overlap:
            raise ValueError(f"IDs cannot be both deleted and relabeled: {sorted(overlap)}")

        requested_ids = delete_ids | relabel_map.keys()
        invalid_ids = sorted(
            review_id
            for review_id in requested_ids
            if review_id < 0 or review_id >= len(self)
        )
        if invalid_ids:
            raise IndexError(f"Unknown review IDs: {invalid_ids}")

        root = Path(dataset_root).expanduser().resolve()
        rejected_root = (
            Path(rejected_dir).expanduser().resolve()
            if rejected_dir is not None
            else root.parent / f"{root.name}_rejected"
        )

        actions: list[dict[str, str | int]] = []
        reserved: set[Path] = set()

        for review_id in sorted(requested_ids):
            item = self.items[review_id]
            if not isinstance(item, (str, Path)):
                raise TypeError(
                    "This helper expects image items to be file paths. "
                    f"ID {review_id} contains {type(item).__name__}."
                )

            source = Path(item).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"Source file no longer exists: {source}")

            try:
                relative_source = source.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Source file is outside dataset_root: {source}"
                ) from exc

            if review_id in delete_ids:
                action = "reject"
                destination = rejected_root / relative_source
            else:
                new_label = relabel_map[review_id]
                label_path = Path(new_label)
                if not new_label or new_label in {".", ".."} or len(label_path.parts) != 1:
                    raise ValueError(
                        f"Relabel target must be one class-folder name: {new_label!r}"
                    )
                action = f"relabel:{new_label}"
                destination = root / new_label / source.name

            destination = _unique_destination(destination, reserved)
            actions.append(
                {
                    "review_id": review_id,
                    "action": action,
                    "source": str(source),
                    "destination": str(destination),
                }
            )

        for action in actions:
            print(
                f"[{action['review_id']}] {action['action']}:\n"
                f"  {action['source']}\n"
                f"  -> {action['destination']}"
            )

            if not dry_run:
                destination = Path(str(action["destination"]))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(action["source"]), str(destination))

        if not actions:
            print("No actions supplied. Everything is being kept.")
        elif dry_run:
            print("\nDry run only. Run again with dry_run=False to apply these moves.")
        else:
            print("\nChanges applied. Rebuild your DataLoaders before retraining.")

        return actions
