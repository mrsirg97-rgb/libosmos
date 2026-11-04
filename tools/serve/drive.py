#!/usr/bin/env python3
"""tools/serve/drive.py - a small driver for osmos-serve's protocol: starts the engine, sends each request, and
prints each answer's token hash (dbench's: FNV-1a 64 over the tokens), its reuse and its times.

    python tools/serve/drive.py --exe bin/osmos-serve -- <engine args>  <  requests (one per line: max_new spec ids_file)
"""
import argparse, subprocess, sys, time

FNV_SEED, FNV_PRIME, MASK = 1469598103934665603, 1099511628211, (1 << 64) - 1


def hash_tokens(toks):
    h = FNV_SEED
    for t in toks:
        h = ((h ^ t) * FNV_PRIME) & MASK
    return h


class Engine:
    def __init__(self, exe, args, env=None):
        self.p = subprocess.Popen([exe, *args, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1, env=env)
        self.notes = []
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("NOTE "):
                self.notes.append(line)
            if line.startswith("READY "):
                return
        raise SystemExit("the engine did not start")

    def gen(self, ids, max_new, spec=0, extra=""):
        t0 = time.monotonic()
        self.p.stdin.write(f"GEN {max_new} spec={spec}{extra} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        out = {"tokens": [], "pp": [], "first_s": None}
        for line in self.p.stdout:
            f = line.split()
            if not f:
                continue
            if f[0] == "T":
                if out["first_s"] is None:
                    out["first_s"] = time.monotonic() - t0
                out["tokens"].append(int(f[-1]))
            elif f[0] == "PP":
                out["pp"].append(line.strip())
            elif f[0] == "DONE":
                out.update(done=f, reused=int(f[8]), finish=f[5], s=time.monotonic() - t0)
                return out
            elif f[0] == "ERR":
                out.update(err=line.strip(), s=time.monotonic() - t0)
                return out
            elif f[0] == "NOTE":
                self.notes.append(line.strip())
        raise SystemExit("the engine ended")

    def quit(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        return self.p.wait(timeout=600)


def read_ids(path):
    return [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    e = Engine(a.exe, args)
    for req in sys.stdin:
        if not req.strip():
            continue
        max_new, spec, path = req.split()
        r = e.gen(read_ids(path), int(max_new), int(spec))
        if "err" in r:
            print(r["err"], flush=True)
            continue
        print(f"{path}: {len(r['tokens'])} tokens, hash {hash_tokens(r['tokens']):016x}, reused {r['reused']}, "
              f"finish {r['finish']}, first token {r['first_s']:.2f} s, all {r['s']:.1f} s", flush=True)
    e.quit()


class Multi:
    """osmos-serve --parallel N: requests sent at once, their lines (T / PP / DONE / ERR <rid> ...) routed back by a
    reader thread; wait(rid) returns a request's record."""

    def __init__(self, exe, args, env=None):
        import threading
        self.p = subprocess.Popen([exe, *args, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1, env=env)
        self.notes, self.req, self.cv, self.log = [], {}, threading.Condition(), []
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("NOTE "):
                self.notes.append(line)
            if line.startswith("READY "):
                break
        else:
            raise SystemExit("the engine did not start")
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            f = line.split()
            if not f or f[0] not in ("T", "PP", "DONE", "ERR"):
                continue
            with self.cv:
                self.log.append((time.monotonic(), f[0], int(f[1]) if f[1].lstrip("-").isdigit() else -1))
                r = self.req.get(int(f[1])) if f[1].lstrip("-").isdigit() else None
                if r is None:
                    continue
                if f[0] == "T":
                    r["tokens"].append(int(f[2]))
                elif f[0] == "PP":
                    r["pp"].append(f)
                elif f[0] == "DONE":
                    r.update(done=f, reused=int(f[9]), finish=f[6], s=time.monotonic() - r["t0"])
                else:
                    r.update(err=line.strip(), s=time.monotonic() - r["t0"])
                self.cv.notify_all()

    def send(self, rid, ids, max_new, spec=0):
        with self.cv:
            self.req[rid] = {"tokens": [], "pp": [], "t0": time.monotonic()}
        self.p.stdin.write(f"GEN {max_new} rid={rid} spec={spec} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()

    def stop(self, rid):
        self.p.stdin.write(f"STOP {rid}\n")
        self.p.stdin.flush()

    def wait(self, rid, until=lambda r: "done" in r or "err" in r, timeout=3600):
        with self.cv:
            if not self.cv.wait_for(lambda: until(self.req[rid]), timeout):
                raise SystemExit(f"request {rid}: no answer in {timeout} s")
            return self.req[rid]

    def quit(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        return self.p.wait(timeout=600)
