# VideoUnsam

Workspace for the video extension of UnSAM. For now it holds a Gradio explorer
for the UnSAMv2 granular-mask training data.

## Mask explorer

```bash
cd VideoUnsam
../.venv/bin/python app.py                      # http://127.0.0.1:7860
../.venv/bin/python app.py --image-dir ../images --port 7861
```

Flags: `--ann-dir` (default `/home/sebastian/Documents/Uni/Practical/UnSAMv2_train_data`),
`--image-dir`, `--port`, `--share`.

The app runs **without the SA-1B images** — masks are drawn on a grey canvas
until the shard is extracted. Once `sa_000023.tar` is unpacked, either restart
with `--image-dir`, or paste the path into the *image directory* box in the UI
and it picks it up live. These locations are auto-detected:

- `../../UnSAMv2_train_images`
- `../images`, `../sa_000023`
- `<ann-dir>/images`

### What you can inspect

- **Granularity range** (min/max sliders) — the core UnSAMv2 axis. `1.0` is
  whole instances, `0.1` the finest parts. Narrow the range to isolate one
  level of the hierarchy.
- **Mask kind** — `instance` (`is_part=False`), `part`, `part+conquer`
  (`is_conquer=True`), plus an instance-id filter to look at one object's
  decomposition alone.
- **Area filter** in % of frame, to drop the very small masks (most images are
  dominated by `very_small` ones per `instance_part_stats`).
- **Click the image** → reports the smallest mask under that pixel and every
  coarser mask covering it, with granularities. This is the quickest way to see
  the nesting at a point.
- **Mask table** — id, granularity, instance, kind, `is_divide`, area %, bbox.
- **Per-mask crops** tab — bbox thumbnail per mask (off by default, it is the
  slow part).

Colour by `granularity` (turbo colormap: blue = fine, red = coarse), by
`instance`, or by `index` for maximum contrast between neighbours.

## Files

- `unsamv2_data.py` — dataset loading, RLE decoding, image lookup. No Gradio
  dependency, meant to be imported by the experiments that follow.
- `app.py` — the Gradio UI.

## Data notes

- 6072 annotated images, ids `257318`–`268501`, all inside SA-1B shard
  `sa_000023.tar`.
- One JSON per image: `image`, `annotations`, and `instance_part_stats`
  (instance/part counts, per-instance part mapping, area histogram).
- Masks are COCO RLE at full image resolution and overlap heavily by design —
  the same pixel belongs to an instance and to each of its parts.
- Rendering downscales to 900 px on the long side; a full-resolution stack of
  ~150 masks at 1500×2258 would be ~500 MB.
