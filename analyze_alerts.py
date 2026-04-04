#!/usr/bin/env python3
"""
Analyze Pikud HaOref alerts for "עבר הירקון" (Tel Aviv) area.
Categorizes alerts into 3 types and creates a bar chart.

Categories:
1. התראה + אזעקה: Alert in "דן" district followed by siren in "עבר הירקון" within 10 min
2. אזעקה ללא התראה: Siren in "עבר הירקון" without preceding alert in "דן"
3. התראה ללא אזעקה: Alert in "דן" without following siren in "עבר הירקון"

Usage:
    python analyze_alerts.py

Note: Must be run from an Israeli IP (oref.org.il is geo-blocked).
"""

import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# ── API Configuration ──────────────────────────────────────────────────────

HEADERS = {
    "Referer": "https://www.oref.org.il/",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

HISTORY_URL = "https://www.oref.org.il/warningMessages/alert/History/AlertsHistory.json"
HISTORY2_URL = "https://alerts-history.oref.org.il/Shared/Ajax/GetAlarmsHistory.aspx?lang=he&mode={mode}"

# ── Area definitions ───────────────────────────────────────────────────────

EVER_HAYARKON = "תל אביב - עבר הירקון"

# All areas in "דן" district (excluding עבר הירקון itself)
DAN_AREAS = {
    "אור יהודה",
    "אזור",
    "בני ברק",
    "בת ים",
    "גבעת שמואל",
    "גבעתיים",
    "גני תקווה",
    "חולון",
    "יהוד מונוסון",
    "מקווה ישראל",
    "פארק אריאל שרון",
    "קריית אונו",
    "רמת גן - מזרח",
    "רמת גן - מערב",
    "תל אביב - דרום העיר ויפו",
    "תל אביב - מזרח",
    "תל אביב - מרכז העיר",
    # Note: "תל אביב - עבר הירקון" is excluded - it's the siren target
}

PAIR_WINDOW = timedelta(minutes=10)


# ── Data fetching ──────────────────────────────────────────────────────────

def fetch_history1():
    """Fetch from AlertsHistory.json (recent hours)."""
    try:
        resp = requests.get(HISTORY_URL, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8-sig"
        text = resp.text.replace("\x00", "").strip()
        if not text:
            return []
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"  [!] AlertsHistory.json fetch failed: {e}")
        return []


def fetch_history2(mode=1):
    """Fetch from GetAlarmsHistory.aspx (extended history)."""
    try:
        url = HISTORY2_URL.format(mode=mode)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8-sig"
        text = resp.text.replace("\x00", "").strip()
        if not text:
            return []
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"  [!] GetAlarmsHistory mode={mode} fetch failed: {e}")
        return []


def parse_alert_date(record):
    """Parse alertDate from various formats used by oref APIs."""
    date_str = record.get("alertDate", "")
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y, %H:%M",
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def normalize_records(records, source_name=""):
    """Convert raw API records to normalized format: (datetime, area_name, title, category)."""
    normalized = []
    for rec in records:
        dt = parse_alert_date(rec)
        if dt is None:
            continue
        area = rec.get("data", "").strip()
        title = rec.get("title", rec.get("category_desc", "")).strip()
        cat = rec.get("category", 0)
        if area:
            normalized.append({
                "time": dt,
                "area": area,
                "title": title,
                "category": cat,
                "source": source_name,
            })
    return normalized


# ── Data collection ────────────────────────────────────────────────────────

def fetch_all_alerts():
    """Fetch alerts from all available endpoints and merge."""
    all_records = []

    print("[*] Fetching alerts from oref.org.il...")

    # Fetch from primary history endpoint
    print("  Fetching AlertsHistory.json...")
    h1 = fetch_history1()
    print(f"  Got {len(h1)} records from AlertsHistory.json")
    all_records.extend(normalize_records(h1, "history1"))

    # Fetch from extended history endpoint with different modes
    for mode in [1, 2, 3, 4]:
        print(f"  Fetching GetAlarmsHistory mode={mode}...")
        h2 = fetch_history2(mode=mode)
        if h2:
            print(f"  Got {len(h2)} records from mode={mode}")
            all_records.extend(normalize_records(h2, f"history2_mode{mode}"))
        else:
            print(f"  No data from mode={mode}")

    # Deduplicate by (time, area)
    seen = set()
    unique = []
    for rec in all_records:
        key = (rec["time"], rec["area"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    unique.sort(key=lambda r: r["time"])
    print(f"\n[*] Total unique records: {len(unique)}")
    return unique


# ── Analysis ───────────────────────────────────────────────────────────────

def analyze_alerts(all_alerts, days=7):
    """
    Categorize alerts into 3 types.

    Returns dict with daily counts for each category.
    """
    cutoff = datetime.now() - timedelta(days=days)

    # Filter to last N days
    recent = [a for a in all_alerts if a["time"] >= cutoff]
    print(f"[*] Alerts in last {days} days: {len(recent)}")

    # Separate עבר הירקון alerts and דן alerts
    ever_hayarkon_alerts = [a for a in recent if a["area"] == EVER_HAYARKON]
    dan_alerts = [a for a in recent if a["area"] in DAN_AREAS]

    print(f"  Alerts in עבר הירקון: {len(ever_hayarkon_alerts)}")
    print(f"  Alerts in אזור דן (excluding עבר הירקון): {len(dan_alerts)}")

    # Category 1: התראה (דן) + אזעקה (עבר הירקון) within 10 min
    # Category 2: אזעקה (עבר הירקון) without preceding דן alert
    # Category 3: התראה (דן) without following עבר הירקון alert

    paired_dan = set()  # indices of dan_alerts that were paired
    paired_eh = set()   # indices of ever_hayarkon_alerts that were paired

    # For each עבר הירקון alert, check if there was a preceding דן alert
    for i, eh_alert in enumerate(ever_hayarkon_alerts):
        for j, dan_alert in enumerate(dan_alerts):
            time_diff = eh_alert["time"] - dan_alert["time"]
            if timedelta(0) <= time_diff <= PAIR_WINDOW:
                paired_eh.add(i)
                paired_dan.add(j)
                break  # one match is enough

    # Daily counts for each category
    cat1_daily = defaultdict(int)  # התראה + אזעקה
    cat2_daily = defaultdict(int)  # אזעקה ללא התראה
    cat3_daily = defaultdict(int)  # התראה ללא אזעקה

    for i, eh_alert in enumerate(ever_hayarkon_alerts):
        day = eh_alert["time"].date()
        if i in paired_eh:
            cat1_daily[day] += 1
        else:
            cat2_daily[day] += 1

    for j, dan_alert in enumerate(dan_alerts):
        day = dan_alert["time"].date()
        if j not in paired_dan:
            cat3_daily[day] += 1

    return cat1_daily, cat2_daily, cat3_daily, recent


# ── Visualization ──────────────────────────────────────────────────────────

def create_chart(cat1_daily, cat2_daily, cat3_daily, days=7):
    """Create a stacked bar chart of alert categories per day."""

    # Configure Hebrew/RTL support
    rcParams['font.family'] = 'DejaVu Sans'

    # Generate all days in the range
    today = datetime.now().date()
    date_range = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]

    cat1_values = [cat1_daily.get(d, 0) for d in date_range]
    cat2_values = [cat2_daily.get(d, 0) for d in date_range]
    cat3_values = [cat3_daily.get(d, 0) for d in date_range]

    # Print summary
    print(f"\n{'='*60}")
    print("Summary per day:")
    print(f"{'='*60}")
    print(f"{'Date':<14} {'Cat1':>6} {'Cat2':>6} {'Cat3':>6} {'Total':>6}")
    print(f"{'-'*14} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for d, c1, c2, c3 in zip(date_range, cat1_values, cat2_values, cat3_values):
        total = c1 + c2 + c3
        print(f"{d.strftime('%Y-%m-%d'):<14} {c1:>6} {c2:>6} {c3:>6} {total:>6}")
    print(f"{'='*60}")
    print(f"{'TOTAL':<14} {sum(cat1_values):>6} {sum(cat2_values):>6} {sum(cat3_values):>6} {sum(cat1_values)+sum(cat2_values)+sum(cat3_values):>6}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(date_range))
    bar_width = 0.25

    bars1 = ax.bar([i - bar_width for i in x], cat1_values, bar_width,
                   label='Alert (Dan) + Siren (Ever HaYarkon)',
                   color='#2196F3', edgecolor='white', linewidth=0.5)

    bars2 = ax.bar(x, cat2_values, bar_width,
                   label='Siren without preceding alert',
                   color='#F44336', edgecolor='white', linewidth=0.5)

    bars3 = ax.bar([i + bar_width for i in x], cat3_values, bar_width,
                   label='Alert (Dan) without siren',
                   color='#FFC107', edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Labels and formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of alerts', fontsize=12)
    ax.set_title(
        'Air Raid Alerts Analysis - Ever HaYarkon (Tel Aviv)\n'
        'התראות ואזעקות - עבר הירקון, תל אביב',
        fontsize=14, fontweight='bold'
    )

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%a\n%d/%m') for d in date_range], fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

    # Add Hebrew labels as text annotations
    fig.text(0.02, 0.02,
             'Cat 1 (Blue): התראה באזור דן + אזעקה בעבר הירקון\n'
             'Cat 2 (Red): אזעקה ללא התראה מקדימה\n'
             'Cat 3 (Yellow): התראה ללא אזעקה',
             fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = "alerts_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[*] Chart saved to: {output_path}")
    plt.close()

    return output_path


# ── Demo mode (for testing without API access) ────────────────────────────

def load_demo_data():
    """Generate demo data for testing when API is not accessible."""
    import random
    random.seed(42)

    print("[*] Running in DEMO mode with simulated data")
    print("    (Use --live flag to fetch real data from oref.org.il)")

    records = []
    now = datetime.now()

    for day_offset in range(7):
        day_base = now - timedelta(days=day_offset)

        # Simulate 3-8 alert events per day
        num_events = random.randint(3, 8)
        for _ in range(num_events):
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            event_time = day_base.replace(hour=hour, minute=minute, second=0)

            event_type = random.choices(
                ["paired", "siren_only", "alert_only"],
                weights=[0.5, 0.2, 0.3],
                k=1
            )[0]

            if event_type == "paired":
                # Dan alert first, then Ever HaYarkon siren
                dan_area = random.choice(list(DAN_AREAS))
                records.append({
                    "time": event_time,
                    "area": dan_area,
                    "title": "ירי רקטות וטילים",
                    "category": 1,
                    "source": "demo",
                })
                siren_time = event_time + timedelta(minutes=random.randint(1, 8))
                records.append({
                    "time": siren_time,
                    "area": EVER_HAYARKON,
                    "title": "ירי רקטות וטילים",
                    "category": 1,
                    "source": "demo",
                })
            elif event_type == "siren_only":
                # Direct siren in Ever HaYarkon, no Dan alert
                records.append({
                    "time": event_time,
                    "area": EVER_HAYARKON,
                    "title": "ירי רקטות וטילים",
                    "category": 1,
                    "source": "demo",
                })
            else:
                # Dan alert without Ever HaYarkon follow-up
                dan_area = random.choice(list(DAN_AREAS))
                records.append({
                    "time": event_time,
                    "area": dan_area,
                    "title": "ירי רקטות וטילים",
                    "category": 1,
                    "source": "demo",
                })

    records.sort(key=lambda r: r["time"])
    return records


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    live_mode = "--live" in sys.argv

    if live_mode:
        all_alerts = fetch_all_alerts()
        if not all_alerts:
            print("\n[!] No alerts fetched. Possible reasons:")
            print("    - Not running from an Israeli IP (oref.org.il is geo-blocked)")
            print("    - API temporarily unavailable")
            print("    - No alerts in the current period")
            print("\n    Run without --live to see a demo with simulated data.")
            sys.exit(1)
    else:
        all_alerts = load_demo_data()

    cat1, cat2, cat3, recent = analyze_alerts(all_alerts, days=7)
    output_path = create_chart(cat1, cat2, cat3, days=7)

    print(f"\nDone! Open {output_path} to view the chart.")
    if not live_mode:
        print("\nNote: This was DEMO data. Run with --live to use real Pikud HaOref data.")


if __name__ == "__main__":
    main()
