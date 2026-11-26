"""Generate a tiny synthetic glm5-next .basalt pack for session tests and dev, at a fixed path.

Reuses the GGUF builder and the conversion path that tools/convert/test_convert_glm5next.py already proves the
converter handles end to end, so this fixture is exactly a small real pack: KDA + sparse-MLA layers, a dense layer, a
drafter (mtp) block, routed experts, and the tokenizer. The tokenizer export is stubbed (a geometry/section fixture
needs no real vocabulary). A C++ session test opens the produced pack with the same loader the engine uses.

    python3 tools/convert/glm5next_tiny_pack.py [out.basalt]   # default /tmp/glm5next_tiny.basalt
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
from model_file import ModelFile  # noqa: E402
from test_convert_glm5next import Glm5NextTest  # noqa: E402


def build(out: str, quant_dense: bool = False, embed_type: str = "F32") -> Path:
    out = Path(out)
    tmp = Path(tempfile.mkdtemp())
    # runnable=True: the geometry-shaped form, so a session test can run a real forward on it. The converter's own
    # test keeps the placeholder blobs it asserts byte layouts against. Its routed experts are i-quant (IQ2_XXS
    # gate/up, IQ3_XXS down) like the release pack's, because those are the only ones the tier can compute.
    # --quant-dense stores the dense projections as Q8_0 as well, which is what the release pack does to most of
    # its tensors and is the only way the mmvq product gets exercised. --embed-type Q8_0 does the same to the token
    # embedding, which the release pack keeps in the file and the forward pass dequantizes a row at a time.
    gguf = Glm5NextTest("gguf").gguf(tmp / "GLM-5.3-Flash-UD-IQ1_S.gguf", runnable=True, quant_dense=quant_dense,
                                     embed_type=embed_type)

    def tokenizer_files(first, out_dir):
        d = Path(out_dir) / "tokenizer"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokenizer.json").write_bytes(b"{}")
        return sorted(d.iterdir())
    convert.export_tokenizer = tokenizer_files

    args = type("A", (), dict(
        gguf=[str(gguf)], mmproj=None, out=str(out), mtp_source=None, mtp_dense=None, draft_vocab=None,
        profile=None, profile_from=None, hc8=None, name="GLM-Tiny", compat_bf16=False, no_vision=False, force=True,
        sections_from=None))()
    convert.convert(args)
    return out


def main(argv):
    args = argv[1:]
    quant_dense = "--quant-dense" in args
    embed_type = "F32"
    if "--embed-type" in args:
        i = args.index("--embed-type")
        if i + 1 >= len(args):
            raise SystemExit("--embed-type wants a type")
        embed_type = args[i + 1].upper()
        if embed_type not in ("F32", "Q8_0"):
            raise SystemExit("--embed-type takes F32 or Q8_0")
        del args[i:i + 2]
    rest = [a for a in args if not a.startswith("--")]
    out = build(rest[0] if rest else "/tmp/glm5next_tiny.basalt", quant_dense, embed_type)
    m = ModelFile(str(out))
    g = json.loads(m.read("meta"))["geometry"]
    print(f"tiny GLM pack -> {out}  ({out.stat().st_size} bytes), routed experts IQ2_XXS/IQ3_XXS"
          + (", dense projections Q8_0" if quant_dense else "")
          + (f", token embedding {embed_type}" if embed_type != "F32" else ""))
    print("sections:", ", ".join(m.names()))
    print("geometry: arch=%s block_count=%s embedding_length=%s kpool=%s hc.count=%s kinds=%s" % (
        json.loads(m.read("meta"))["arch"], g.get("block_count"), g.get("embedding_length"),
        g.get("attention.indexer.kpool"), g.get("hyper_connection.count"), g.get("attention.head_count_kv")))


if __name__ == "__main__":
    main(sys.argv)
