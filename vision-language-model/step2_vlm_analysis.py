#!/usr/bin/env python3
"""Ask a Qwen vision-language model (via Ollama) what each simulation's eta images show.

For each case, a few key frames of eta (first, just before / at the onset found by step1, last)
are tiled into one labeled image (F1, F2, ...). The model answers a fixed set of narrow visual
questions, constrained to a JSON schema. Its answers are then checked against the numbers from
step1_classify_morphology.py; disagreements are listed for human review.

The model sees only the images (no metrics) unless --with-metrics is given, so the check
measures what it can judge visually.

python step2_vlm_analysis.py
python step2_vlm_analysis.py --cases case_001 case_114 --overwrite
python step2_vlm_analysis.py --outcomes dendritic transitional --model qwen3-vl:8b-instruct
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Literal, Optional

import matplotlib
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MORPH_TO_CLASS = {"uniform": "uniform", "wavy": "transitional", "dendritic": "dendritic"}

PROMPT = """你在看一个锂金属电沉积相场模拟的 eta 场（相场序参量）。
图中从左到右是同一次模拟的 {n} 个时刻，标签 {labels}，时间依次为 {times}（模拟时间单位）。
每张图：x 向右，y 向上；颜色为 jet 色标，红色 = 金属锂（eta≈1），蓝色 = 电解液（eta≈0）。
锂层从左边界开始向右生长。注意：每张图的色标是按该帧自身的最小值和最大值拉伸的，
所以几乎均匀的场也可能显示出明显的条纹或噪点，这类情况属于图像伪影，不是形貌。
{metrics}
只根据图像回答下面的问题，严格按给定 JSON 格式输出，文字用中文：
1. morphology_last：最后一帧中，红蓝界面是 uniform（基本平直）、wavy（有起伏但没有明显突出的指状结构）还是 dendritic（有明显指状或树枝状突出）。
2. protrusions_visible：最后一帧是否能看到向右伸出的指状突出。
3. first_unstable_frame：界面第一次明显不再平直的是哪一帧，填标签（如 F2）；如果到最后一帧都是平直的，填 none。
4. detached_regions_visible：最后一帧是否有与左侧锂层不相连的孤立红色区域。
5. front_reached_right_edge：最后一帧红色区域是否已经接近或到达右边界。
6. image_artifacts：列出看到的图像问题（条纹、噪点、整幅几乎单色、色标异常等），没有就给空列表。
7. evidence：对每一帧写一句你看到了什么，frame 填标签。
8. summary：两三句话总结这个模拟的形貌演化。只描述能看到的现象，不要推测原因，不要估计数值或速度，不要数枝晶个数。
"""


class FrameNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: str
    observation: str


class VLMReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    morphology_last: Literal["uniform", "wavy", "dendritic"]
    protrusions_visible: bool
    first_unstable_frame: str
    detached_regions_visible: bool
    front_reached_right_edge: bool
    image_artifacts: List[str]
    evidence: List[FrameNote] = Field(min_length=1)
    summary: str


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--morphology", default="data/summary/morphology/morphology.json",
                   help="step1 output; used to pick frames and to check the answers")
    p.add_argument("--output", default="data/summary/vlm",
                   help="summary goes to <output>/<model>; per-case files to <case>/vlm/<model>")
    p.add_argument("--model", default="qwen3.8:27b ")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--cases", nargs="*", help="only these cases")
    p.add_argument("--outcomes", nargs="*", help="only cases with these step1 outcomes, e.g. dendritic transitional")
    p.add_argument("--limit", type=int, help="analyze at most this many cases")
    p.add_argument("--frames", type=int, default=4, help="frames per case when step1 found no onset")
    p.add_argument("--tile", type=int, default=384, help="pixel size of each frame in the tiled image")
    p.add_argument("--with-metrics", action="store_true", help="also give the model step1's numbers")
    p.add_argument("--overwrite", action="store_true", help="redo cases that already have a report")
    p.add_argument("--timeout", type=int, default=600, help="seconds per request")
    p.add_argument("--detached-threshold", type=float, default=0.01,
                   help="step1 detached_area above this counts as detached regions present")
    return p.parse_args()


def frame_files(case_dir):
    out = []
    for f in glob.glob(os.path.join(case_dir, "image_files", "eta", "eta_t*.*")):
        m = re.search(r"eta_t([-+0-9.eE]+)\.(?:jpg|png)$", os.path.basename(f))
        if m:
            out.append((float(m.group(1)), f))
    return sorted(out)


def pick_frames(frames, info, n):
    """Indices of the frames to show: first, around the step1 onset, last; else evenly spaced."""
    last = len(frames) - 1
    if info and info.get("onset_time") is not None:
        onset = min(range(len(frames)), key=lambda i: abs(frames[i][0] - info["onset_time"]))
        idx = {0, max(onset - 1, 0), onset, last}
    else:
        idx = {round(k * last / max(n - 1, 1)) for k in range(n)}
    return sorted(idx)


def tile(frames, idx, size):
    font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, max(14, size // 16))
    bar = size // 9
    gap = 8
    canvas = Image.new("RGB", (len(idx) * size + (len(idx) - 1) * gap, size + bar), "white")
    draw = ImageDraw.Draw(canvas)
    labels = []
    for k, i in enumerate(idx):
        t, path = frames[i]
        label = f"F{k + 1}"
        labels.append(label)
        x = k * (size + gap)
        canvas.paste(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), (x, bar))
        draw.text((x + 6, 4), f"{label}  t={t:.4g}", fill="black", font=font)
    return canvas, labels


def ask(args, prompt, image_bytes):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(image_bytes).decode()]}],
        "format": VLMReport.model_json_schema(),
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
    }
    req = urllib.request.Request(f"{args.host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        return json.loads(r.read())


def metrics_text(info):
    if not info:
        return ""
    onset = f"{info['onset_time']:.4g}" if info.get("onset_time") is not None else "无"
    return (f"程序测得的参考数值：最后一帧前沿起伏幅度 {info['amplitude']:.3f}（占域宽比例），"
            f"界面长度/域高 {info['length_ratio']:.2f}，界面开始不平直的时间 {onset}。\n")


def check(report, info, labels, frames, idx, args):
    """Compare the model's answers with step1; returns {question: agree / disagree / n/a}."""
    if not info:
        return {}
    out = {"morphology": MORPH_TO_CLASS[report.morphology_last] == info["class"]}
    if info.get("onset_time") is not None:
        expect = next((lab for lab, i in zip(labels, idx) if frames[i][0] >= info["onset_time"] - 1e-9), "none")
    else:
        expect = "none"
    out["first_unstable_frame"] = report.first_unstable_frame.strip().upper() == expect.upper()
    out["detached_regions"] = report.detached_regions_visible == (info["detached_area"] > args.detached_threshold)
    out["front_reached_right_edge"] = report.front_reached_right_edge == (info["front_mean"] >= 0.85)
    return {"agree": out, "expected_first_unstable_frame": expect}


def model_dir_name(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def analyze_case(name, case_dir, info, params, args):
    out_dir = os.path.join(case_dir, "vlm", model_dir_name(args.model) + ("_metrics" if args.with_metrics else ""))
    if os.path.isfile(os.path.join(out_dir, "report.json")) and not args.overwrite:
        with open(os.path.join(out_dir, "check.json")) as f:
            return json.load(f), "cached"
    frames = frame_files(case_dir)
    if not frames:
        raise ValueError("no eta images")
    idx = pick_frames(frames, info, args.frames)
    canvas, labels = tile(frames, idx, args.tile)
    os.makedirs(out_dir, exist_ok=True)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    with open(os.path.join(out_dir, "input.png"), "wb") as f:
        f.write(image_bytes)

    prompt = PROMPT.format(n=len(idx), labels="、".join(labels),
                           times="、".join(f"{frames[i][0]:.4g}" for i in idx),
                           metrics=metrics_text(info) if args.with_metrics else "")
    request = {"model": args.model, "host": args.host, "prompt": prompt, "with_metrics": args.with_metrics,
               "frames": [{"label": lab, "index": i, "time": frames[i][0], "path": frames[i][1]}
                          for lab, i in zip(labels, idx)],
               "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "schema": VLMReport.model_json_schema()}
    with open(os.path.join(out_dir, "request.json"), "w") as f:
        json.dump(request, f, indent=2, ensure_ascii=False)

    t0 = time.time()
    raw = ask(args, prompt, image_bytes)
    with open(os.path.join(out_dir, "response.raw.json"), "w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    report = VLMReport.model_validate_json(raw["message"]["content"])
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    result = {"case": name, "model": args.model, "seconds": round(time.time() - t0, 1),
              "vlm": report.model_dump(exclude={"evidence"}),
              "step1": {k: info.get(k) for k in ("class", "outcome", "onset_time", "amplitude", "length_ratio",
                                                 "detached_area", "front_mean")} if info else None,
              "params": params, **check(report, info, labels, frames, idx, args)}
    with open(os.path.join(out_dir, "check.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result, "new"


def write_summary(results, args):
    os.makedirs(args.output, exist_ok=True)
    rows = sorted(results.values(), key=lambda r: r["case"])
    keys = ["morphology", "first_unstable_frame", "detached_regions", "front_reached_right_edge"]
    with open(os.path.join(args.output, "vlm_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "step1_class", "step1_outcome", "vlm_morphology", "vlm_first_unstable",
                    "expected_first_unstable", "protrusions", "detached", "reached_right", "artifacts"]
                   + [f"agree_{k}" for k in keys] + ["summary"])
        for r in rows:
            v, s, a = r["vlm"], r.get("step1") or {}, r.get("agree", {})
            w.writerow([r["case"], s.get("class"), s.get("outcome"), v["morphology_last"], v["first_unstable_frame"],
                        r.get("expected_first_unstable_frame"), v["protrusions_visible"],
                        v["detached_regions_visible"], v["front_reached_right_edge"], "; ".join(v["image_artifacts"])]
                       + [a.get(k) for k in keys] + [v["summary"]])

    checked = [r for r in rows if r.get("agree")]
    agreement = {k: (sum(r["agree"][k] for r in checked) / len(checked) if checked else None) for k in keys}
    confusion = {}
    for r in checked:
        key = f"{r['step1']['class']} -> {r['vlm']['morphology_last']}"
        confusion[key] = confusion.get(key, 0) + 1
    disagree = {r["case"]: [k for k in keys if not r["agree"][k]] for r in checked if not all(r["agree"].values())}
    summary = {"model": args.model, "with_metrics": args.with_metrics, "cases": len(rows),
               "agreement_with_step1": agreement, "morphology_confusion (step1 -> vlm)": confusion,
               "disagreements": disagree}
    with open(os.path.join(args.output, "vlm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    args.output = os.path.join(args.output, model_dir_name(args.model) + ("_metrics" if args.with_metrics else ""))
    morph = {}
    if os.path.isfile(args.morphology):
        with open(args.morphology) as f:
            morph = json.load(f)["cases"]
    else:
        print(f"[WARN] {args.morphology} not found: frames are evenly spaced and answers are not checked. "
              f"Run step1_classify_morphology.py first.")
    case_dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                       if os.path.isdir(os.path.join(d, "image_files", "eta")))
    names = [os.path.basename(d) for d in case_dirs]
    if args.cases:
        names = [n for n in names if n in args.cases]
    if args.outcomes:
        names = [n for n in names if morph.get(n, {}).get("outcome") in args.outcomes]
    if args.limit:
        names = names[:args.limit]
    if not names:
        raise SystemExit("no cases selected")

    try:
        urllib.request.urlopen(f"{args.host}/api/tags", timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"cannot reach Ollama at {args.host} ({exc}); start it with `ollama serve`")

    results, errors = {}, {}
    for k, name in enumerate(names, 1):
        case_dir = os.path.join(args.data, name)
        params = None
        if os.path.isfile(os.path.join(case_dir, "params.json")):
            with open(os.path.join(case_dir, "params.json")) as f:
                params = json.load(f)
        try:
            r, how = analyze_case(name, case_dir, morph.get(name), params, args)
            results[name] = r
            agree = r.get("agree")
            mark = "" if not agree else ("  agree" if all(agree.values()) else
                                          "  DISAGREE: " + ", ".join(q for q, ok in agree.items() if not ok))
            took = "  [cached]" if how == "cached" else f"  {r['seconds']}s"
            print(f"[{k}/{len(names)}] {name}: {r['vlm']['morphology_last']:<9} "
                  f"(step1 {(r.get('step1') or {}).get('class', '-')}){mark}{took}")
        except (ValidationError, KeyError, ValueError, urllib.error.URLError, OSError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"[{k}/{len(names)}] {name}: ERROR {errors[name][:200]}")

    if results:
        s = write_summary(results, args)
        print()
        for q, v in s["agreement_with_step1"].items():
            if v is not None:
                print(f"agreement with step1, {q}: {v:.0%}")
        for pair, c in sorted(s["morphology_confusion (step1 -> vlm)"].items()):
            print(f"  {pair}: {c}")
        if s["disagreements"]:
            print(f"cases to review ({len(s['disagreements'])}): {' '.join(s['disagreements'])}")
    if errors:
        print(f"errors ({len(errors)}): {' '.join(errors)}")
    print(f"per-case files in <case>/vlm/{os.path.basename(args.output)}/, summary in {args.output}")


if __name__ == "__main__":
    main()
