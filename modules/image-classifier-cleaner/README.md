# Image Review

`image_review.py` provides a lightweight, widget-free workflow for reviewing the
highest-loss images from a fastai image classifier. It displays images in static,
paginated groups and uses stable review IDs to record files that should be
quarantined or moved to another class folder.

## Why I built this

fastai provides `ClassificationInterpretation` for identifying images that a
model finds difficult to classify. These high-loss examples can then be reviewed
and removed from, or relabeled within, the training data.

I initially ran into a bug with fastai's widget-based image cleaner. Running the
notebook in JupyterLab on Paperspace appeared to fix it, based on the advice in
this [fast.ai forum thread](https://forums.fast.ai/t/lesson-2-imageclassifiercleaner-error-displaying-widget/109371).
However, JupyterLab repeatedly lost its connection, so the workflow remained
unreliable. The experience suggested that fastai's widget-based cleaning tools
could still use some attention across notebook environments.

As a result, I built this widget-free module to review and clean both my training
and validation data. It uses `ClassificationInterpretation` to find high-loss
images, presents them in static pages, and applies deletion or relabeling
decisions through ordinary Python collections and filesystem operations.

## Requirements

- Python 3.10+
- [fastai](https://docs.fast.ai/)
- A trained fastai `Learner`
- A folder-labeled image dataset

The module must be importable from the current Python environment. For example,
run a notebook from this directory or add `modules` to `PYTHONPATH`.

## Usage

Create a reviewer from a trained learner. By default, `ds_idx=1` reviews the
validation dataset and `k=60` selects the 60 examples with the highest loss.

```python
from image_review import TopLossCleaner

review = TopLossCleaner(
    learner,
    k=60,
    ds_idx=1,
)
```

Display a page of results:

```python
review.show(
    page=1,
    page_size=12,
    figsize=(12, 10),
)
```

Along with the image grid, `show()` prints each image's stable review ID, loss,
and file path. Use those IDs to record decisions:

```python
delete_ids = set()
relabels = {}

delete_ids.update([3])

relabels.update({
    0: "diseased",
})
```

Preview the resulting file moves before changing the dataset:

```python
actions = review.apply(
    dataset_root=path,
    delete=delete_ids,
    relabel=relabels,
    dry_run=True,
)
```

Inspect the printed destinations and the returned `actions` list. When the
preview is correct, apply the moves:

```python
actions = review.apply(
    dataset_root=path,
    delete=delete_ids,
    relabel=relabels,
    dry_run=False,
)
```

After applying changes, rebuild the fastai `DataLoaders` before retraining.

## How decisions are applied

### Delete

Files passed through `delete` are moved rather than permanently deleted. The
default quarantine directory is a sibling of the dataset:

```text
parent/
├── dataset/
└── dataset_rejected/
```

The original path beneath `dataset_root` is preserved. A custom quarantine
location can be supplied with `rejected_dir`:

```python
review.apply(
    dataset_root=path,
    delete={3},
    rejected_dir="reviewed/rejected",
    dry_run=False,
)
```

### Relabel

Relabeling moves an image into another class folder directly beneath
`dataset_root`. For example, relabeling an image as `"diseased"` moves it to:

```text
dataset/diseased/image_name.jpg
```

The target must be a single class-folder name, not a nested or absolute path.

If a destination filename already exists, the cleaner adds a numeric suffix
instead of overwriting it.

## API

### `TopLossCleaner(learn, k=60, ds_idx=1, dl=None)`

- `learn`: trained fastai `Learner`.
- `k`: number of highest-loss examples to review. Use `None` for all examples.
- `ds_idx`: dataset index passed to fastai's
  `ClassificationInterpretation.from_learner`.
- `dl`: optional `DataLoader` to review instead of the dataset selected by
  `ds_idx`.

### `show(page=1, page_size=12, figsize=(12, 10))`

Displays one 1-based page of highest-loss examples and prints their review IDs,
losses, and paths.

### `apply(dataset_root, *, delete=(), relabel=None, dry_run=True, rejected_dir=None)`

Previews or performs review decisions.

- `dataset_root`: root directory containing the class folders.
- `delete`: iterable of review IDs to quarantine.
- `relabel`: mapping of review IDs to target class names.
- `dry_run`: when `True`, only prints and returns the planned actions.
- `rejected_dir`: optional quarantine directory.

The method returns a list of action dictionaries containing `review_id`,
`action`, `source`, and `destination`.

An ID cannot be both deleted and relabeled in the same call.
