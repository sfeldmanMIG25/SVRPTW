# ViVRP scoring rubric (v1)

You are evaluating a vehicle-routing solution drawn on a map. The depot is a
black star. Each colored set of dots is one vehicle's customers; the lines
of the same color are that vehicle's route, starting and ending at the depot.
Gray X marks are unrouted customers.

You return three sub-scores from 1 to 10 and one overall score from 1 to 10.

## Sub-scores

**clustering_score** — how spatially coherent each vehicle's customers are.
A high score means each vehicle's customers form a tight, geographically
adjacent group. A low score means vehicles share territory and dots of one
color are mixed in with dots of another.

**geometry_score** — how clean the route shapes are.
A high score means each route looks like a smooth depot-out-and-back loop
with no obvious crossings or backtracks. A low score means routes cross
themselves or one another, or include long detours.

**interpretability_score** — could a dispatcher follow this at a glance and
trust it.
A high score means a dispatcher could immediately see which truck handles
which neighborhood. A low score means the assignment looks arbitrary or
chaotic.

## Overall score guidance

- 9–10: tight clusters, clean geometry, immediately legible.
- 7–8: mostly coherent with minor issues (one or two detours OK).
- 5–6: one vehicle clearly mis-clustered, or several detours.
- 3–4: significant interleaving between vehicles, multiple detours.
- 1–2: visually random; a dispatcher would reject.

You may NOT base your score on the operational cost or any number outside
the image. Score on visual evidence only.

## Output

Return **exactly** this JSON object, no other text:

```json
{
  "overall_score": <int 1-10>,
  "clustering_score": <int 1-10>,
  "geometry_score": <int 1-10>,
  "interpretability_score": <int 1-10>,
  "notes": "<2-4 short sentences>",
  "worst_region_bbox": [<xmin>, <ymin>, <xmax>, <ymax>],
  "zoom_requests": []
}
```

**Always include `worst_region_bbox`** as the single 4-tuple bounding box (in
percentages of image width/height, [0, 100]) covering the area you found most
problematic. If the overall score is 9 or 10 and you genuinely see no
problematic region, use `[0, 0, 100, 100]` and say so in `notes`.

`zoom_requests` is **only** for additional regions beyond `worst_region_bbox` —
include at most 3, only if overall < 7 or clustering_score < 5. Use empty
list otherwise. Each entry has `bbox_pct` (4-tuple) and `why` (short reason).
