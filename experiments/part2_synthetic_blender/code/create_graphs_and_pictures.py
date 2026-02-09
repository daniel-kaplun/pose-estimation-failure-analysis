import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = r"C:\ML Project\results\keypoints_table.csv"
IMAGE_ROOT = r"C:\ML Project\Datasets\MVP_fixed_occlusion_tol\images"

OUT_ROOT = r"C:\ML Project\analysis\final_quant_data"
GRAPH_DIR = os.path.join(OUT_ROOT, "graphs")
TABLE_DIR = os.path.join(OUT_ROOT, "tables")
VIZ_DIR = os.path.join(OUT_ROOT, "visualizations")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ================= FIND FILENAME COLUMN =================
file_col = [c for c in df.columns if df[c].astype(str).str.contains(".png").any()][0]

# ================= PARSE ITERATION & FRAME =================
df["iteration"] = df[file_col].str.extract(r"iter_(\d+)").astype(int)
df["frame"] = df[file_col].str.extract(r"frame_(\d+)").astype(int)

# Only frames 0–70
df = df[df["frame"] <= 70]

iterations = sorted(df["iteration"].unique())

# ================= LEFT SIDE =================
LEFT_KPTS = [5, 7, 9, 11, 13, 15]
LEFT_SHOULDER = 5

# ================= AP (PCK@20px) =================
df["error"] = np.sqrt((df["pred_x"] - df["gt_x"])**2 + (df["pred_y"] - df["gt_y"])**2)
THRESH = 20
df["AP"] = (df["error"] < THRESH).astype(float)

left_df = df[df["keypoint_id"].isin(LEFT_KPTS)]
shoulder_df = df[df["keypoint_id"] == LEFT_SHOULDER]

# =========================================================
# 1️⃣ LEFT SIDE AP vs FRAME (SEPARATE GRAPH PER ITERATION)
# =========================================================
for i in iterations:
    g = left_df[left_df["iteration"] == i].groupby("frame")["AP"].mean()
    plt.figure()
    plt.plot(g.index, g.values, marker="o")
    plt.title(f"Left Side AP vs Frame (Iteration {i})")
    plt.xlabel("Frame")
    plt.ylabel("AP (0–1)")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(GRAPH_DIR, f"left_AP_vs_frame_iter_{i}.png"))
    plt.close()

# =========================================================
# 2️⃣ LEFT SIDE AP vs ITERATION
# =========================================================
left_iter_ap = left_df.groupby("iteration")["AP"].mean().reset_index()
plt.figure()
plt.plot(left_iter_ap["iteration"], left_iter_ap["AP"], marker="o")
plt.title("Left Side AP vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("AP (0–1)")
plt.ylim(0, 1)
plt.grid()
plt.savefig(os.path.join(GRAPH_DIR, "left_AP_vs_iteration.png"))
plt.close()

# =========================================================
# 3️⃣ LEFT SHOULDER AP vs FRAME (SEPARATE)
# =========================================================
for i in iterations:
    g = shoulder_df[shoulder_df["iteration"] == i].groupby("frame")["error"].mean()
    plt.figure()
    plt.plot(g.index, g.values, marker="o")
    plt.title(f"Left Shoulder Mean Error vs Frame (Iteration {i})")
    plt.xlabel("Frame")
    plt.ylabel("Mean Error (pixels)")
    plt.ylim(0, 150)
    plt.grid()
    plt.savefig(os.path.join(GRAPH_DIR, f"left_shoulder_error_vs_frame_iter_{i}.png"))
    plt.close()

# =========================================================
# 4️⃣ LEFT SHOULDER AP vs ITERATION
# =========================================================
shoulder_iter_err = shoulder_df.groupby("iteration")["error"].mean().reset_index()
plt.figure()
plt.plot(shoulder_iter_err["iteration"], shoulder_iter_err["error"], marker="o")
plt.title("Left Shoulder Mean Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Mean Error (pixels)")
plt.ylim(0, 150)
plt.grid()
plt.savefig(os.path.join(GRAPH_DIR, "left_shoulder_error_vs_iteration.png"))
plt.close()
# =========================================================
# 5️⃣ OVERALL AP vs ITERATION
# =========================================================
overall_ap = df.groupby("iteration")["AP"].mean().reset_index()
plt.figure()
plt.plot(overall_ap["iteration"], overall_ap["AP"], marker="o")
plt.title("Overall AP vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("AP (0–1)")
plt.ylim(0, 1)
plt.grid()
plt.savefig(os.path.join(GRAPH_DIR, "overall_AP_vs_iteration.png"))
plt.close()

# =========================================================
# IMAGE VISUALIZATION
# =========================================================
def draw_points(img, sub_df, color):
    for _, r in sub_df.iterrows():
        cv2.circle(img, (int(r["pred_x"]), int(r["pred_y"])), 4, color, -1)
    return img

def draw_pred_gt(img, sub_df):
    for _, r in sub_df.iterrows():
        cv2.circle(img, (int(r["pred_x"]), int(r["pred_y"])), 4, (0,0,255), -1)
        cv2.circle(img, (int(r["gt_x"]), int(r["gt_y"])), 4, (0,255,0), -1)
    return img

for i in iterations:
    iter_df = df[df["iteration"] == i]
    frames = sorted(iter_df["frame"].unique())
    if len(frames) < 15:
        continue

    selected = frames[:5] + frames[len(frames)//2-2:len(frames)//2+3] + frames[-5:]

    for f in selected:
        frame_df = iter_df[iter_df["frame"] == f]
        img_name = frame_df[file_col].iloc[0]
        img_path = os.path.join(IMAGE_ROOT, img_name)
        img = cv2.imread(img_path)

        left_sub = frame_df[frame_df["keypoint_id"].isin(LEFT_KPTS)]

        cv2.imwrite(os.path.join(VIZ_DIR, f"iter{i}_frame{f}_left_pred.png"),
                    draw_points(img.copy(), left_sub, (255,0,0)))

        cv2.imwrite(os.path.join(VIZ_DIR, f"iter{i}_frame{f}_all_pred.png"),
                    draw_points(img.copy(), frame_df, (0,0,255)))

        cv2.imwrite(os.path.join(VIZ_DIR, f"iter{i}_frame{f}_pred_vs_gt.png"),
                    draw_pred_gt(img.copy(), frame_df))

print("✅ ALL REQUIREMENTS SAT")
