#!/usr/bin/env python3
"""osmos-serve's protocol, scripted, for osmos-api's tests: READY after FAKE_DIR/ready appears; a GEN answered with
the ids in FAKE_DIR/reply (its first line; a second line, if any, the finish: stop by default, "exit" to die
mid-answer), a token every FAKE_DIR/pace seconds (0.01); a STOP ends its run (cancel). every line in, and the argv,
logged to FAKE_DIR/log"""
import os
import sys
import threading
import time

D = os.environ["FAKE_DIR"]
lock = threading.Lock()
stopped = set()


def log(s):
    with open(os.path.join(D, "log"), "a") as f:
        f.write(s + "\n")


def out(s):
    with lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def answer(rid, prompt_n):
    lines = open(os.path.join(D, "reply")).read().split("\n")
    ids = [int(x) for x in lines[0].split(",") if x]
    finish = lines[1] if len(lines) > 1 and lines[1] else "stop"
    pace = float(open(os.path.join(D, "pace")).read()) if os.path.exists(os.path.join(D, "pace")) else 0.01
    out(f"PP {rid} {prompt_n} {prompt_n} 12.5 1000.0 0")
    n = 0
    for t in ids:
        if rid in stopped:
            out(f"DONE {rid} {n} {prompt_n} 12.5 {n * 10.0} cancel 0 0 0 0 0")
            return
        if finish == "exit" and n == len(ids) // 2:
            os._exit(3)
        out(f"T {rid} {t}")
        n += 1
        time.sleep(pace)
    while finish == "hold" and rid not in stopped:
        time.sleep(0.01)
    out(f"DONE {rid} {n} {prompt_n} 12.5 {n * 10.0} {'cancel' if finish == 'hold' else finish} 0 0 0 0 0")


log("ARGV " + " ".join(sys.argv[1:]))
while not os.path.exists(os.path.join(D, "ready")):
    time.sleep(0.02)
out("READY 4096 stop multi=1")
for line in sys.stdin:
    line = line.rstrip("\n")
    log(line)
    if line.startswith("GEN "):
        parts = line.split()
        rid = parts[2].split("=")[1]
        threading.Thread(target=answer, args=(rid, len(parts[-1].split(","))), daemon=True).start()
    elif line.startswith("STOP "):
        stopped.add(line.split()[1])
    elif line == "QUIT":
        break
