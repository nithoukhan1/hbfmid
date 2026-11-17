"""Per-class image / group / box statistics. READ-ONLY on dataset; writes only into --out."""
import argparse, os, glob, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
root, out = os.path.realpath(a.data_root), os.path.realpath(a.out)
assert not out.startswith(root), "out must be outside dataset"
names = ["Comminuted","Greenstick","Healthy","Linear","Oblique Displaced","Oblique",
         "Segmental","Spiral","Transverse Displaced","Transverse"]
g = pd.read_csv(a.groups)
g["classes"] = g["classes"].astype(str)
lab_dir = {"train":"train","valid":"valid","test":"test"}
rows = []
for _, r in g.iterrows():
    stem = os.path.splitext(r["image"])[0]
    p = os.path.join(root, lab_dir[r["split"]], "labels", stem + ".txt")
    if not os.path.exists(p): continue
    for ln in open(p):
        t = ln.split()
        if len(t) >= 5:
            c = int(t[0]); w = float(t[3]); h = float(t[4])
            rows.append((r["split"], r["image"], r["group_id"], c, w*h, w/max(h,1e-9)))
b = pd.DataFrame(rows, columns=["split","image","group","cls","area","ar"])
b["name"] = b["cls"].map(dict(enumerate(names)))
res = []
for c, d in b.groupby("name"):
    imgs = d["image"].nunique(); grp = d["group"].nunique()
    tr = d[d.split=="train"]
    res.append(dict(cls=c, boxes_all=len(d), images_all=imgs, groups_all=grp,
        mult_img_per_group=round(imgs/grp,2), boxes_train=len(tr),
        groups_train=tr["group"].nunique(), groups_valid=d[d.split=="valid"]["group"].nunique(),
        groups_test=d[d.split=="test"]["group"].nunique(),
        med_area_pct=round(100*d["area"].median(),2), med_aspect_wh=round(d["ar"].median(),2),
        p10_area_pct=round(100*d["area"].quantile(.1),2)))
df = pd.DataFrame(res).sort_values("groups_all", ascending=False)
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "class_group_stats.csv"), index=False)
print(df.to_string(index=False))
# co-occurrence of classes within an image
co = b.groupby("image")["name"].apply(lambda s: tuple(sorted(set(s))))
print("\nimages with >1 distinct class:", int((co.apply(len)>1).sum()), "of", len(co))
print(co[co.apply(len)>1].value_counts().head(8).to_string())
