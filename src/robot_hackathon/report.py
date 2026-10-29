from collections import Counter, defaultdict
from html import escape
from .local_store import atomic_write


def summarize(records):
    groups = defaultdict(Counter)
    for row in records:
        groups[(row["object_label"], row["strategy"])][row["observed_outcome"]] += 1
    result = []
    for (obj, strategy), counts in sorted(groups.items()):
        known = counts["success"] + counts["failure"]
        result.append(dict(object=obj, strategy=strategy, total=sum(counts.values()),
                           success=counts["success"], failure=counts["failure"], unknown=counts["unknown"],
                           rate=counts["success"] / known if known else None))
    return result


def render(records, path, mode):
    if any(row["mode"] != mode for row in records):
        raise ValueError("Mixed-mode reports are prohibited")
    groups = summarize(records)
    totals = Counter(r["observed_outcome"] for r in records)
    known = totals["success"] + totals["failure"]
    rate = f"{100 * totals['success'] / known:.1f}%" if known else "N/A"
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in
           (g["object"], g["strategy"], g["total"], g["success"], g["failure"], g["unknown"],
            f"{g['rate']:.1%}" if g['rate'] is not None else "N/A")) + "</tr>" for g in groups)
    attempts = "".join(f"<tr><td>{escape(r['attempt_id'])}</td><td>{escape(r['strategy'])}</td><td>{escape(r['observed_outcome'])}</td><td>{r['duration_ms']}</td></tr>" for r in records)
    label = "MOCK DATA · SYNTHETIC ONLY" if mode == "mock" else "LIVE OBSERVATIONS"
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robot attempt inspector — {mode}</title><style>
body{{font:16px system-ui;background:#101b25;color:#eaf3f5;margin:0;padding:5vw;max-width:1100px}}h1{{font-size:clamp(32px,5vw,56px);margin:12px 0}}.tag{{color:#75e2c3;font-weight:700;letter-spacing:.1em}}.sub{{color:#b8c9d3;max-width:740px;line-height:1.6}}.cards{{display:flex;flex-wrap:wrap;gap:16px;margin:32px 0}}.card{{background:#1c2d3b;padding:24px;border-radius:14px;flex:1;min-width:140px}}.card strong{{display:block;font-size:36px}}table{{width:100%;border-collapse:collapse;background:#172734}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #304451}}th{{color:#75e2c3}}.scroll{{overflow:auto}}summary{{padding:24px 0;cursor:pointer}}footer{{margin-top:36px;color:#b8c9d3}}</style>
<p class="tag">{label}</p><h1>Every attempt tells a story.</h1><p class="sub">A local record of proposals and observed results. This report contains {mode} data only. Command acceptance is never counted as task success.</p>
<div class="cards"><div class="card"><strong>{len(records)}</strong>Attempts</div><div class="card"><strong>{rate}</strong>Known-outcome success</div><div class="card"><strong>{totals['unknown']}</strong>Unknown outcomes</div></div>
<h2>Object & strategy</h2><div class="scroll"><table><thead><tr><th>Object</th><th>Strategy</th><th>Total</th><th>Success</th><th>Failure</th><th>Unknown</th><th>Success rate</th></tr></thead><tbody>{rows}</tbody></table></div><p class="sub">Success rate = successes ÷ (successes + failures). Unknown outcomes are shown separately. Synthetic rates do not predict real robot performance.</p>
<details><summary>Inspect all {len(records)} attempts</summary><div class="scroll"><table><thead><tr><th>Attempt ID</th><th>Strategy</th><th>Observed outcome</th><th>Duration (ms)</th></tr></thead><tbody>{attempts}</tbody></table></div></details>
<footer>Viam × MongoDB hackathon preparation · Offline fallback · No physical action implemented</footer></html>"""
    atomic_write(path, page)
    return {"mode": mode, "attempts": len(records), "success": totals["success"], "failure": totals["failure"], "unknown": totals["unknown"], "known_outcome_success_rate": rate}
