# Stress-test images (out-of-distribution check)

The leaf model only knows 3 classes. These images test what it does with
inputs that are **not** clean bean-leaf photos.

| Folder | What | Committed? |
|---|---|---|
| `non_leaf/` | `coffee_cup.jpg` (Rachel Michetti, CC0) and `cat.jpg` ("Chelsea", Stefan van der Walt, CC0), both from the scikit-image sample data | yes |
| `instructor_photos/` | 3–4 photos the instructor takes (house-plant leaf, tree leaf, something green that is not a leaf). Processed by `scripts/prepare_stress_images.py`: EXIF/GPS metadata removed, resized to ≤ 800 px | yes, after processing |
| `_incoming/` | Drop raw phone photos here first. **Never committed** (git-ignored) because raw phone photos carry GPS location in their metadata | no |
| `my_photos/` | Students' own uploads when running locally | no |

Corrupted bean-leaf images (blur, dark, glare, compression, zoom) and synthetic
images (noise, flat green) are generated in code at run time
(`dlhandson.ood`), so they are not stored here.

## Adding the instructor photos

```bash
# 1. Copy the phone photos (JPG, PNG or iPhone HEIC) into data/stress_test/_incoming/
# 2. With the project's .venv active:
python scripts/prepare_stress_images.py
# 3. Check data/stress_test/instructor_photos/ and commit those files.
```
Name the files descriptively (e.g. `maple_leaf.jpg`, `house_plant_leaf.jpg`,
`green_backpack.jpg`) — the file name becomes the caption in the notebook.
