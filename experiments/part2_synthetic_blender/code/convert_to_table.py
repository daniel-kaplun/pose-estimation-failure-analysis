import json
import os
import pandas as pd

# -------------------------
# PATHS
# -------------------------
PRED_PATH = r"C:\ML Project\results\mvp_hrnet_w32_eval.json"
GT_PATH   = r"C:\ML Project\Datasets\MVP_fixed_occlusion_tol\annotations.json"
OUT_CSV   = r"C:\ML Project\results\keypoints_table.csv"

# -------------------------
# LOAD FILE섭
# -------------------------
with open(PRED_PATH, "r") as f:
    pred_batches = json.load(f)

with open(GT_PATH, "r") as f:
    gt = json.load(f)

print(f"Loaded {len(pred_batches)} prediction batches")

# -------------------------
# BUILD GT LOOKUPS
# -------------------------
# image_id -> keypoints
gt_kps = {ann["image_id"]: ann["keypoints"] for ann in gt["annotations"]}

# file_name -> image_id
fname_to_id = {img["file_name"]: img["id"] for img in gt["images"]}

# -------------------------
# FLATTEN PREDICTIONS
# -------------------------
rows = []
global_frame_idx = 0

for batch_idx, batch in enumerate(pred_batches):
    preds = batch["preds"]
    image_paths = batch["image_paths"]

    assert len(preds) == len(image_paths), "Pred/image mismatch"

    for i in range(len(preds)):
        pred_kps = preds[i]          # shape: [17][3]
        img_path = image_paths[i]
        file_name = os.path.basename(img_path)

        if file_name not in fname_to_id:
            print(f"⚠️ Skipping unknown image: {file_name}")
            continue

        image_id = fname_to_id[file_name]
        gt_frame = gt_kps[image_id]

        assert len(pred_kps) == 17, "Expected 17 keypoints"
        assert len(gt_frame) == 51, "GT keypoints should be 17*3"

        for kpt_id in range(17):
            px, py, pconf = pred_kps[kpt_id]

            gx = gt_frame[kpt_id * 3]
            gy = gt_frame[kpt_id * 3 + 1]
            gv = gt_frame[kpt_id * 3 + 2]

            rows.append({
                "frame_idx": global_frame_idx,
                "batch_idx": batch_idx,
                "image_id": image_id,
                "file_name": file_name,
                "keypoint_id": kpt_id,
                "pred_x": px,
                "pred_y": py,
                "pred_conf": pconf,
                "gt_x": gx,
                "gt_y": gy,
                "gt_visible": gv
            })

        global_frame_idx += 1

print(f"Total frames processed: {global_frame_idx}")

# -------------------------
# WRITE CSV
# -------------------------
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"✔ Wrote {len(df)} rows to {OUT_CSV}")
