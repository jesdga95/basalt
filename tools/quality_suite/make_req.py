"""Reference file -> llama_project request (token sequences + scored rows with the reference top-K ids)."""
import struct, sys, numpy as np
from safetensors.numpy import load_file
if len(sys.argv) < 3: sys.exit("usage: make_req.py REF.safetensors OUT.bin [--smoke]")
ref, out = sys.argv[1], sys.argv[2]
smoke = "--smoke" in sys.argv
from common import smoke_convs
t = load_file(ref); off, ro = t["conv_off"], t["row_off"]
with open(out, "wb") as o:
    convs = smoke_convs(off) if smoke else range(len(off) - 1)
    o.write(struct.pack("<i", len(convs)))
    for c in convs:
        a, b = int(off[c]), int(off[c + 1])
        o.write(struct.pack("<i", b - a)); o.write(t["tokens"][a:b].astype(np.int32).tobytes())
        rows = [i for i in range(a, b) if ro[i + 1] > ro[i]]
        o.write(struct.pack("<i", len(rows)))
        for i in rows:
            ids = t["topk_ids"][ro[i]:ro[i + 1]]
            o.write(struct.pack("<ii", i - a, len(ids))); o.write(ids.astype(np.int32).tobytes())
