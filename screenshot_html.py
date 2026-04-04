"""Generate a self-contained HTML (no CDN) and take a screenshot."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from analyze_alerts import (
    load_demo_data, analyze_alerts, EVER_HAYARKON, DAN_AREAS
)
from datetime import datetime, timedelta

all_alerts = load_demo_data()
cat1, cat2, cat3, recent = analyze_alerts(all_alerts, days=7)

today = datetime.now().date()
date_range = [today - timedelta(days=i) for i in range(6, -1, -1)]
labels = [d.strftime("%a %d/%m") for d in date_range]
c1 = [cat1.get(d, 0) for d in date_range]
c2 = [cat2.get(d, 0) for d in date_range]
c3 = [cat3.get(d, 0) for d in date_range]

max_val = max(max(c1), max(c2), max(c3), 1)
chart_h = 300
chart_w = 700
bar_w = 22
group_gap = 70
y_scale = (chart_h - 40) / max_val

# Build SVG bars
bars_svg = ""
for i in range(7):
    gx = 60 + i * group_gap
    for val, color, offset in [(c1[i], "#3b82f6", 0), (c2[i], "#ef4444", bar_w+2), (c3[i], "#eab308", 2*(bar_w+2))]:
        bh = val * y_scale
        by = chart_h - 30 - bh
        bx = gx + offset
        bars_svg += f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" rx="3"/>\n'
        if val > 0:
            bars_svg += f'<text x="{bx + bar_w//2}" y="{by - 4}" text-anchor="middle" fill="#e2e8f0" font-size="12">{val}</text>\n'
    # x-axis label
    lx = gx + (3 * bar_w + 4) // 2
    bars_svg += f'<text x="{lx}" y="{chart_h - 5}" text-anchor="middle" fill="#94a3b8" font-size="11">{labels[i]}</text>\n'

# y-axis
y_lines = ""
for yv in range(0, max_val + 1):
    yy = chart_h - 30 - yv * y_scale
    y_lines += f'<line x1="55" y1="{yy}" x2="{60 + 7*group_gap}" y2="{yy}" stroke="#334155" stroke-width="1"/>\n'
    y_lines += f'<text x="50" y="{yy + 4}" text-anchor="end" fill="#94a3b8" font-size="11">{yv}</text>\n'

# Build event table rows
cutoff = datetime.now() - timedelta(days=7)
relevant = [a for a in all_alerts if a["time"] >= cutoff and (a["area"] == EVER_HAYARKON or a["area"] in DAN_AREAS)]
relevant.sort(key=lambda r: r["time"], reverse=True)

table_rows = ""
for a in relevant[:30]:  # limit for screenshot
    bg = "rgba(239,68,68,0.12)" if a["area"] == EVER_HAYARKON else "rgba(59,130,246,0.08)"
    table_rows += f'<tr style="background:{bg}"><td>{a["time"].strftime("%Y-%m-%d %H:%M")}</td><td>{a["area"]}</td><td>{a["title"]}</td></tr>\n'

html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',Tahoma,Arial,sans-serif; background:#0f172a; color:#e2e8f0; padding:24px; direction:rtl; }}
.container {{ max-width:1100px; margin:0 auto; }}
h1 {{ text-align:center; font-size:1.7em; margin-bottom:4px; color:#f8fafc; }}
.subtitle {{ text-align:center; color:#94a3b8; margin-bottom:22px; font-size:.95em; }}
.stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:22px; }}
.stat-card {{ background:#1e293b; border-radius:12px; padding:18px; text-align:center; }}
.stat-card .number {{ font-size:2.4em; font-weight:bold; margin-bottom:4px; }}
.stat-card .label {{ font-size:.82em; color:#94a3b8; line-height:1.4; }}
.stat-card.blue {{ border-top:4px solid #3b82f6; }} .stat-card.blue .number {{ color:#3b82f6; }}
.stat-card.red {{ border-top:4px solid #ef4444; }} .stat-card.red .number {{ color:#ef4444; }}
.stat-card.yellow {{ border-top:4px solid #eab308; }} .stat-card.yellow .number {{ color:#eab308; }}
.legend-box {{ background:#1e293b; border-radius:12px; padding:18px; margin-bottom:22px; }}
.legend-item {{ display:flex; align-items:center; gap:10px; margin-bottom:7px; font-size:.92em; }}
.legend-dot {{ width:14px; height:14px; border-radius:3px; flex-shrink:0; }}
.chart-container {{ background:#1e293b; border-radius:16px; padding:22px; margin-bottom:22px; text-align:center; }}
.table-container {{ background:#1e293b; border-radius:16px; padding:22px; }}
.table-container h2 {{ margin-bottom:12px; font-size:1.15em; }}
table {{ width:100%; border-collapse:collapse; font-size:.88em; }}
th {{ background:#334155; padding:9px 12px; text-align:right; font-weight:600; }}
td {{ padding:7px 12px; border-bottom:1px solid #334155; }}
tr:hover {{ background:rgba(255,255,255,.05); }}
.footer {{ text-align:center; color:#64748b; margin-top:18px; font-size:.78em; }}
</style>
</head>
<body>
<div class="container">
  <h1>ניתוח התראות ואזעקות</h1>
  <p class="subtitle">עבר הירקון, תל אביב — 7 ימים אחרונים</p>

  <div class="stats">
    <div class="stat-card blue"><div class="number">{sum(c1)}</div><div class="label">התראה באזור דן<br>+ אזעקה בעבר הירקון</div></div>
    <div class="stat-card red"><div class="number">{sum(c2)}</div><div class="label">אזעקה בעבר הירקון<br>ללא התראה מקדימה</div></div>
    <div class="stat-card yellow"><div class="number">{sum(c3)}</div><div class="label">התראה באזור דן<br>ללא אזעקה בעבר הירקון</div></div>
  </div>

  <div class="legend-box">
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div><span><strong>התראה + אזעקה:</strong> התראה באזור דן שאחריה הגיעה אזעקה בעבר הירקון (תוך 10 דקות)</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div><span><strong>אזעקה ללא התראה:</strong> אזעקה בעבר הירקון ללא התראה מקדימה באזור דן</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#eab308"></div><span><strong>התראה ללא אזעקה:</strong> התראה באזור דן שלא הגיעה אחריה אזעקה בעבר הירקון</span></div>
  </div>

  <div class="chart-container">
    <svg width="{60 + 7*group_gap + 20}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg">
      {y_lines}
      {bars_svg}
    </svg>
  </div>

  <div class="table-container">
    <h2>פירוט התראות</h2>
    <table>
      <thead><tr><th>זמן</th><th>אזור</th><th>סוג התראה</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <p class="footer">נתונים מפיקוד העורף — oref.org.il &bull; נוצר ב-{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
</body>
</html>"""

out = "/home/user/WeDLM/alerts_selfcontained.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Self-contained HTML saved to {out}")
