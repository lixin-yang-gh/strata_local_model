#!/usr/bin/env python3
"""Set up or start Strata with a pre-downloaded, Strata-compatible model.

Thin wrapper around setup.py: it registers a "local" model family built from a
GGUF you already have, then hands off to setup.py for the engine, the pack, the
MTP draft layer and the start. Everything setup.py does is unchanged.

Start an already-installed model (same as START-HERE.bat):
    start_local_model.bat
    ./setup_local_model.py

Set up a single pre-downloaded GGUF (any name, e.g. abc.gguf):
    start_local_model.bat --gguf C:/models/abc.gguf --yes
    ./setup_local_model.py --gguf /models/abc.gguf --yes

Set up a folder that holds every shard of one model:
    ./setup_local_model.py --gguf /models/mymodel --yes

Only Qwen3.8-Flash-Next GGUFs run: setup.py checks for the per_layer_token_embd
tensor at the end and stops if it is missing. A single file must be the whole
model (all experts plus the PLE table), not one shard of a split.

Wrapper options (anything else is passed straight through to setup.py -
--context, --kv, --gpu, --gpus, --host, --api-key, --low-ram, --build, --yes,
--no-start, ...):
    --gguf PATH      the model: a single .gguf file, or a folder of shards
    --name NAME      a label for the model (default: from the file name)
    --arena-gb N     experts size estimate in GB (default: ~92%% of the file size)
    --ram-gb N       RAM estimate in GB (default: arena + 10)
    --compat-bf16    pass --compat-bf16 to the pack step (Q8_0 projections -> BF16)
    --stage-dir DIR  where to link a single renamed shard (default: beside the file)

Images are not wired up for local models here, so --vision is left off.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup  # noqa: E402


LOCAL_FAMILY = "local"
LOCAL_MODEL = "LOCAL"
DROP_OPTS = ("--family", "--model", "--gguf-dir")


def file_gb(path: Path) -> float:
    return path.stat().st_size / 1e9


def drop_conflicts(passthrough):
    """Remove options the wrapper sets itself, so a stray copy in the passthrough does
    not override them (argparse would take the last occurrence)."""
    out, skip = [], False
    for tok in passthrough:
        if skip:
            skip = False
            continue
        if tok in DROP_OPTS:
            skip = True
            continue
        if any(tok.startswith(o + "=") for o in DROP_OPTS):
            continue
        out.append(tok)
    return out


def stage_single_file(src: Path, stem: str, stage_dir: Path) -> Path:
    """Link (or copy) a single GGUF into stage_dir under the name setup.py expects of a
    one-shard model (<stem>-LOCAL-00001-of-00001.gguf), so --gguf-dir finds it."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / f"{stem}-{LOCAL_MODEL}-00001-of-00001.gguf"
    if target.exists():
        if target.stat().st_size == src.stat().st_size:
            return target
        target.unlink()
    for link in (os.link, getattr(os, "symlink", None)):
        if link is None:
            continue
        try:
            link(str(src), str(target))
            return target
        except (OSError, NotImplementedError):
            continue
    print(f"  (could not link {src.name}; copying {file_gb(src):.1f} GB into {stage_dir} - this takes a while)",
          flush=True)
    shutil.copyfile(src, target)
    return target


def register_local(stem: str, name: str, arena_gb: float, ram_gb: float, compat_bf16: bool) -> None:
    """Add a 'local' family and a 'LOCAL' size to setup.py's tables, before it builds its
    argument parser and runs. Only Qwen3.8-Flash-Next GGUFs actually load; the check is
    setup.py's own per_layer_token_embd gate at the end."""
    fam = {
        "title": name,
        "by": "a pre-downloaded model",
        "about": "your own Strata-compatible GGUF",
        "hf": "",                               # never used: --gguf-dir skips every download
        "file": stem + "-{q}-0000{i}-of-00001.gguf",
        "tag": "local-",
        "mmproj_hf": setup.hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
        "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "name": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "local",
        "vision": False,
    }
    if compat_bf16:
        fam["pack_args"] = ["--compat-bf16"]
    setup.FAMILIES[LOCAL_FAMILY] = fam
    setup.MODELS[LOCAL_MODEL] = {
        "about": "a pre-downloaded model",
        "download_gb": arena_gb,
        "ram_gb": ram_gb,
        "arena_gb": arena_gb,
        "families": (LOCAL_FAMILY,),
    }


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="a single .gguf file, or a folder of shards")
    ap.add_argument("--name", help="a label for the model")
    ap.add_argument("--arena-gb", type=float, help="experts size estimate (GB)")
    ap.add_argument("--ram-gb", type=float, help="RAM estimate (GB)")
    ap.add_argument("--compat-bf16", action="store_true", help="pass --compat-bf16 to the pack step")
    ap.add_argument("--stage-dir", help="where to link a single renamed shard")
    known, passthrough = ap.parse_known_args()
    passthrough = drop_conflicts(passthrough)

    # No --gguf: this is a plain start (or the pick-a-model menu). setup.py finds the
    # installed config and starts it; a start reads the config directly, so the 'local'
    # family does not need to be registered for it.
    if not known.gguf:
        sys.argv = [sys.argv[0]] + passthrough
        return setup.main()

    gguf = Path(known.gguf).expanduser().resolve()
    if not gguf.exists():
        setup.fail(f"not found: {gguf}", "pass --gguf with the path to your .gguf file or its folder")
    name = known.name or (gguf.stem if gguf.is_file() else gguf.name) or "Local model"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "local"

    if gguf.is_dir():
        gguf_dir = gguf
        total = sum(file_gb(p) for p in gguf.glob("*.gguf"))
        arena = known.arena_gb if known.arena_gb else round(max(total * 0.92, 1.0), 1)
        if not any(gguf.glob("*-00001-of-*.gguf")):
            setup.warn("no '<name>-00001-of-0000N.gguf' shard in that folder: if setup cannot find the model, "
                       "pass the single file itself (--gguf <file>.gguf) so it is staged with the right name")
    else:
        stage = Path(known.stage_dir).expanduser().resolve() if known.stage_dir \
            else gguf.parent / "strata-local-staging"
        staged = stage_single_file(gguf, stem, stage)
        gguf_dir = staged.parent
        arena = known.arena_gb if known.arena_gb else round(max(file_gb(gguf) * 0.92, 1.0), 1)
        setup.ok(f"using {gguf.name} as {staged.name}")

    ram = known.ram_gb if known.ram_gb else float(int(math.ceil(arena)) + 10)
    register_local(stem, name, arena, ram, known.compat_bf16)

    sys.argv = [sys.argv[0], "--setup", "--family", LOCAL_FAMILY, "--model", LOCAL_MODEL,
                "--gguf-dir", str(gguf_dir)] + passthrough
    return setup.main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        setup.say("\nstopped.")
        sys.exit(1)
