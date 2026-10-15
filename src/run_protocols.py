import sys, os, json, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import *
FS = os.environ.get("FS", "hogcolor")
d = np.load("data/cache/coil100.npz")
obj, ang = d["obj"], d["ang"]
if FS == "hogcolor":
    f = np.load("data/cache/features.npz"); X = np.hstack([f["hog"], f["color"]])
else:
    X = np.load("data/cache/features_pix.npz")["pix"]
P, arc, unk = make_protocols(obj, ang)
sfx = "" if FS == "hogcolor" else f"_{FS}"
for p in sys.argv[1:]:
    R, out, final, iso = run_protocol(p, X, obj, ang, P, log=lambda s: print(s, flush=True))
    R["features"] = FS
    json.dump(R, open(f"results/{p}{sfx}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(f"results/{p}{sfx}_out.npz", **out)
    print(json.dumps(R["test"], indent=1), flush=True)
