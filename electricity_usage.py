
'''
1.read latest file by date from path folder
2. plot on dashboard:
    plot_daily_trend
    plot_hourly_distribution
    plot_daily_load_profile
    plot_heatmap
    plot_day_of_week_pattern
    plot_cumulative_sum
    table_top_consumption_days
3.export to html
'''
"""
Electricity Consumption Dashboard
==================================
1. Finds & reads the latest CSV file from a folder (by filename date or by
   file modified time).
2. Builds a set of interactive Plotly charts + a table.
3. Combines everything into a single standalone interactive HTML dashboard.

Run:
    python electricity_dashboard.py
"""
"""
Electricity Consumption Dashboard
==================================
1. Finds & reads the latest CSV file from a folder (by filename date or by
   file modified time).
2. Builds a set of interactive Plotly charts + a table.
3. Combines everything into a single standalone interactive HTML dashboard.

Run:
    python electricity_dashboard.py
"""

import glob
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

pd.options.mode.chained_assignment = None


# =============================================================================
# 1. FILE LOADING
# =============================================================================

def find_latest_file(folder: str, pattern: str = "*.csv") -> str:
    """
    Find the most recent file in `folder` matching `pattern`.

    Tries to parse a date from the filename first (e.g. 'Valandiniai[2026-06 - 2026-08].csv'
    -> picks the file whose latest date-in-name is greatest). Falls back to
    file modification time if no date can be parsed from any filename.

    Parameters
    ----------
    folder : path to folder containing the files
    pattern : glob pattern, e.g. '*.csv' or 'Valandiniai*.csv'
    """
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {folder}")

    date_pattern = re.compile(r"(\d{4}-\d{2}(?:-\d{2})?)")

    def latest_date_in_name(fp: str):
        name = os.path.basename(fp)
        dates = date_pattern.findall(name)
        if not dates:
            return None
        parsed = []
        for d in dates:
            try:
                parsed.append(pd.to_datetime(d))
            except Exception:
                pass
        return max(parsed) if parsed else None

    dated = [(f, latest_date_in_name(f)) for f in files]
    if all(d is not None for _, d in dated):
        # pick file with the latest date found in its name
        best = max(dated, key=lambda x: x[1])[0]
    else:
        # fallback: most recently modified file
        best = max(files, key=os.path.getmtime)

    return best


def load_data(filepath: str, sep: str = ";") -> pd.DataFrame:
    """
    Load the electricity CSV and standardize to ['datetime', 'volume'] columns.

    Expects columns 'Data, valanda' (datetime) and 'Kiekis, kWh' (volume),
    matching the export format used here. Adjust column names below if needed.
    """
    raw = pd.read_csv(filepath, sep=sep)

    datetime_col = "Data, valanda"
    volume_col = "Kiekis, kWh"

    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(raw[datetime_col]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(
        raw[volume_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df = df.dropna(subset=["datetime", "volume"]).sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# 2. CHART FUNCTIONS
# =============================================================================

FREQ_MAP = {
    '15min': '15min', '30min': '30min', '1h': '1h',
    '1d': '1D', '1w': '1W', '1m': 'MS',
}


def plot_daily_trend(df, freq: str = '1d', export_html: str | None = None) -> go.Figure:
    """Time series line chart of electricity volume, resampled (sum) to `freq`."""
    if freq not in FREQ_MAP:
        raise ValueError(f"freq must be one of {list(FREQ_MAP)}")

    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d = d.sort_values('datetime').set_index('datetime')
    d = d['volume'].resample(FREQ_MAP[freq]).sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d['datetime'], y=d['volume'], mode='lines', name='Volume',
        line=dict(width=1.5, color='#2E86AB'),
        hovertemplate='%{x}<br>Volume: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'Electricity Volume Trend ({freq})',
        xaxis_title='Datetime', yaxis_title='Volume (kWh)',
        template='plotly_white', hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        height=450
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def plot_hourly_distribution(
    df, exclude_days: list | None = None, day_type: str = 'all',
    show_points: str = 'outliers', export_html: str | None = None
) -> go.Figure:
    """Boxplot of interval volume distribution per hour of day, with date on hover."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['date'] = d['datetime'].dt.date
    d['hour'] = d['datetime'].dt.hour
    d['weekday_name'] = d['datetime'].dt.day_name()

    if exclude_days:
        exclude_dates = {pd.to_datetime(e).date() for e in exclude_days}
        d = d[~d['date'].isin(exclude_dates)]
    if day_type == 'weekday':
        d = d[d['datetime'].dt.weekday < 5]
    elif day_type == 'weekend':
        d = d[d['datetime'].dt.weekday >= 5]
    elif day_type != 'all':
        raise ValueError("day_type must be 'all', 'weekday', or 'weekend'")
    if d.empty:
        raise ValueError("No data left after applying filters.")

    customdata = d[['date', 'weekday_name']].astype(str).values

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=d['hour'], y=d['volume'], customdata=customdata,
        boxpoints=show_points,
        marker=dict(color='#2E86AB', size=3, opacity=0.5),
        line=dict(color='#2E86AB'), fillcolor='rgba(46, 134, 171, 0.3)',
        hovertemplate='Date: %{customdata[0]} (%{customdata[1]})<br>Hour: %{x}<br>Volume: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title='Hourly Distribution of Volume (Boxplot)',
        xaxis_title='Hour of Day', yaxis_title='Volume (kWh)',
        template='plotly_white',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        height=500
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def plot_daily_load_profile(
    df, time_resolution: str = '1h', band: str = 'minmax',
    export_html: str | None = None
) -> go.Figure:
    """Average daily load profile with a min/max or P10-P90 variability band."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])

    if time_resolution == '1h':
        d['time_of_day'] = d['datetime'].dt.hour
        x_label = 'Hour of Day'
    elif time_resolution == '15min':
        d['time_of_day'] = d['datetime'].dt.hour + d['datetime'].dt.minute / 60
        x_label = 'Time of Day (hour)'
    else:
        raise ValueError("time_resolution must be '15min' or '1h'")

    grouped = d.groupby('time_of_day')['volume']
    mean = grouped.mean()

    if band == 'minmax':
        lower, upper = grouped.min(), grouped.max()
        band_label = 'Min–Max'
    elif band == 'p10p90':
        lower, upper = grouped.quantile(0.10), grouped.quantile(0.90)
        band_label = 'P10–P90'
    else:
        raise ValueError("band must be 'minmax' or 'p10p90'")

    x = mean.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]), y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(46, 134, 171, 0.15)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip',
        name=band_label, showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode='lines+markers', name='Average',
        line=dict(width=2.5, color='#2E86AB'), marker=dict(size=4),
        hovertemplate='%{x}<br>Avg Volume: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'Average Daily Load Profile ({band_label} band)',
        xaxis_title=x_label, yaxis_title='Volume (kWh)',
        template='plotly_white', hovermode='x unified', height=450
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def plot_heatmap(
    df, time_resolution: str = '1h', agg: str = 'sum', x_tickangle: int = 280,
    y_bottom_to_top: bool = True, start_hour: int = 0, end_hour: int = 24,
    day_type: str = 'all', exclude_days: list | None = None,
    export_html: str | None = None
) -> go.Figure:
    """Heatmap of volume: time-of-day (y) vs. date (x)."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['date'] = d['datetime'].dt.date
    d['hour'] = d['datetime'].dt.hour

    d = d[(d['hour'] >= start_hour) & (d['hour'] < end_hour)]

    if day_type == 'weekday':
        d = d[d['datetime'].dt.weekday < 5]
    elif day_type == 'weekend':
        d = d[d['datetime'].dt.weekday >= 5]
    elif day_type != 'all':
        raise ValueError("day_type must be 'all', 'weekday', or 'weekend'")

    if exclude_days:
        exclude_dates = {pd.to_datetime(e).date() for e in exclude_days}
        d = d[~d['date'].isin(exclude_dates)]
    if d.empty:
        raise ValueError("No data left after applying filters.")

    if time_resolution == '1h':
        d['time_of_day'] = d['hour']
        y_label = 'Hour of Day'
    elif time_resolution == '15min':
        d['time_of_day'] = d['datetime'].dt.strftime('%H:%M')
        y_label = 'Time of Day'
    else:
        raise ValueError("time_resolution must be '15min' or '1h'")

    pivot = d.pivot_table(index='time_of_day', columns='date', values='volume', aggfunc=agg)
    pivot = pivot.sort_index(ascending=y_bottom_to_top)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
        colorscale='YlOrRd', colorbar=dict(title='Volume (kWh)'),
        hovertemplate='Date: %{x}<br>Time: %{y}<br>Volume: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'Electricity Volume Heatmap ({agg} per {time_resolution} bucket)',
        xaxis_title='Date', yaxis_title=y_label, template='plotly_white', height=500,
        xaxis=dict(type='category', tickangle=x_tickangle), yaxis=dict(type='category')
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def plot_day_of_week_pattern(
    df, agg: str = 'mean', start_hour: int = 0, end_hour: int = 24,
    exclude_days: list | None = None, bargap: float = 0.4,
    log_scale: bool = False, export_html: str | None = None
) -> go.Figure:
    """Bar chart of volume aggregated by day of week."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['date'] = d['datetime'].dt.date
    d['hour'] = d['datetime'].dt.hour

    d = d[(d['hour'] >= start_hour) & (d['hour'] < end_hour)]
    if exclude_days:
        exclude_dates = {pd.to_datetime(e).date() for e in exclude_days}
        d = d[~d['date'].isin(exclude_dates)]
    if d.empty:
        raise ValueError("No data left after applying filters.")

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    d['weekday_name'] = d['datetime'].dt.day_name()

    if agg == 'mean':
        grouped = d.groupby('weekday_name')['volume'].mean().reindex(weekday_order)
        y_label = 'Average Volume (kWh)'
    elif agg == 'sum':
        daily_totals = d.groupby(['date', 'weekday_name'])['volume'].sum().reset_index()
        grouped = daily_totals.groupby('weekday_name')['volume'].mean().reindex(weekday_order)
        y_label = 'Average Daily Total Volume (kWh)'
    else:
        raise ValueError("agg must be 'mean' or 'sum'")

    colors = ['#2E86AB'] * 5 + ['#E76F51'] * 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped.index, y=grouped.values, marker_color=colors,
        hovertemplate='%{x}<br>Volume: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title='Electricity Volume by Day of Week',
        xaxis_title='Day of Week', yaxis_title=y_label,
        yaxis_type='log' if log_scale else 'linear',
        template='plotly_white', bargap=bargap, height=450
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def plot_cumulative_sum(
    df, period: str = 'daily', exclude_days: list | None = None,
    max_periods: int | None = None, sort_by: str = 'recent',
    export_html: str | None = None
) -> go.Figure:
    """Cumulative volume line chart, resetting to 0 at the start of each period."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['date'] = d['datetime'].dt.date

    if exclude_days:
        exclude_dates = {pd.to_datetime(e).date() for e in exclude_days}
        d = d[~d['date'].isin(exclude_dates)]
    if d.empty:
        raise ValueError("No data left after applying filters.")

    d = d.sort_values('datetime')

    if period == 'daily':
        d['period_id'] = d['datetime'].dt.date.astype(str)
    elif period == 'weekly':
        iso = d['datetime'].dt.isocalendar()
        d['period_id'] = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
    elif period == 'monthly':
        d['period_id'] = d['datetime'].dt.strftime('%Y-%m')
    else:
        raise ValueError("period must be 'daily', 'weekly', or 'monthly'")

    period_start = d.groupby('period_id')['datetime'].transform('min')
    d['elapsed_hours'] = (d['datetime'] - period_start).dt.total_seconds() / 3600
    x_label = f'Elapsed Hours (since start of {period[:-2] if period != "daily" else "day"})'

    d['cumsum'] = d.groupby('period_id')['volume'].cumsum()

    if sort_by == 'recent':
        period_ids = list(d['period_id'].drop_duplicates())
        if max_periods:
            period_ids = period_ids[-max_periods:]
    elif sort_by in ('max_cumsum', 'min_cumsum'):
        totals = d.groupby('period_id')['cumsum'].max().sort_values(ascending=(sort_by == 'min_cumsum'))
        period_ids = list(totals.index)
        if max_periods:
            period_ids = period_ids[:max_periods]
    else:
        raise ValueError("sort_by must be 'recent', 'max_cumsum', or 'min_cumsum'")

    d = d[d['period_id'].isin(period_ids)]
    colors = pc.sample_colorscale('Viridis', [i / max(len(period_ids) - 1, 1) for i in range(len(period_ids))])

    fig = go.Figure()
    for pid, color in zip(period_ids, colors):
        group = d[d['period_id'] == pid]
        fig.add_trace(go.Scatter(
            x=group['elapsed_hours'], y=group['cumsum'], mode='lines', name=pid,
            line=dict(width=1.5, color=color),
            hovertemplate=f'{pid}<br>Elapsed: %{{x:.2f}}h<br>Cumulative: %{{y:.3f}}<extra></extra>'
        ))
    fig.update_layout(
        title=f'Cumulative Volume by {period.capitalize()} Period ({sort_by})',
        xaxis_title=x_label, yaxis_title='Cumulative Volume (kWh)',
        template='plotly_white', hovermode='closest', height=500,
        legend=dict(title=period.capitalize())
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


def table_top_consumption_days(
    df, top_n: int = 10, order: str = 'highest',
    exclude_days: list | None = None, export_html: str | None = None
) -> go.Figure:
    """Table of top/bottom consumption days with weekday, vs-average, and peak hour info."""
    d = df.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['date'] = d['datetime'].dt.date
    d['hour'] = d['datetime'].dt.hour

    if exclude_days:
        exclude_dates = {pd.to_datetime(e).date() for e in exclude_days}
        d = d[~d['date'].isin(exclude_dates)]
    if d.empty:
        raise ValueError("No data left after applying filters.")

    daily = d.groupby('date')['volume'].sum().reset_index()
    daily['weekday'] = pd.to_datetime(daily['date']).dt.day_name()

    avg_total = daily['volume'].mean()
    daily['vs_avg_pct'] = (daily['volume'] / avg_total - 1) * 100

    hourly_per_day = d.groupby(['date', 'hour'])['volume'].sum().reset_index()
    peak_idx = hourly_per_day.groupby('date')['volume'].idxmax()
    peak_hours = hourly_per_day.loc[peak_idx].rename(columns={'hour': 'peak_hour', 'volume': 'peak_hour_volume'})

    hour_avg = hourly_per_day.groupby('hour')['volume'].mean().rename('hour_avg_volume')
    peak_hours = peak_hours.merge(hour_avg, left_on='peak_hour', right_index=True, how='left')
    peak_hours['peak_vs_hour_avg_pct'] = (peak_hours['peak_hour_volume'] / peak_hours['hour_avg_volume'] - 1) * 100

    daily = daily.merge(
        peak_hours[['date', 'peak_hour', 'peak_hour_volume', 'hour_avg_volume', 'peak_vs_hour_avg_pct']],
        on='date', how='left'
    )

    ascending = (order == 'lowest')
    daily = daily.sort_values('volume', ascending=ascending).head(top_n).reset_index(drop=True)
    daily.insert(0, 'rank', daily.index + 1)

    day_avg_colors = ['#e8f4ea' if v >= 0 else '#fde8e8' for v in daily['vs_avg_pct']]
    hour_avg_colors = ['#e8f4ea' if v >= 0 else '#fde8e8' for v in daily['peak_vs_hour_avg_pct']]

    fig = go.Figure(data=[go.Table(
        columnwidth=[35, 85, 90, 100, 90, 75, 95, 105, 100],
        header=dict(
            values=['Rank', 'Date', 'Weekday', 'Daily Total (kWh)', 'vs. Daily Avg',
                    'Peak Hour', 'Peak Volume (kWh)', 'Typical Avg for that Hour', 'Peak vs. Hour Avg'],
            fill_color='#2E86AB', font=dict(color='white', size=12), align='center', height=36
        ),
        cells=dict(
            values=[
                daily['rank'], daily['date'].astype(str), daily['weekday'],
                daily['volume'].round(3), daily['vs_avg_pct'].apply(lambda x: f'{x:+.1f}%'),
                daily['peak_hour'].apply(lambda h: f'{int(h):02d}:00'),
                daily['peak_hour_volume'].round(3), daily['hour_avg_volume'].round(3),
                daily['peak_vs_hour_avg_pct'].apply(lambda x: f'{x:+.1f}%')
            ],
            fill_color=[
                ['white'] * len(daily), ['white'] * len(daily), ['white'] * len(daily),
                ['white'] * len(daily), day_avg_colors, ['white'] * len(daily),
                ['white'] * len(daily), ['white'] * len(daily), hour_avg_colors
            ],
            align='center', height=28
        )
    )])
    fig.update_layout(
        title=f'{"Top" if order == "highest" else "Bottom"} {top_n} Consumption Days '
              f'(Average daily total: {avg_total:.3f} kWh)',
        height=100 + 30 * top_n, margin=dict(t=60, b=10, l=10, r=10)
    )
    if export_html:
        fig.write_html(export_html, include_plotlyjs='cdn')
    return fig


# =============================================================================
# 3. DASHBOARD ASSEMBLY
# =============================================================================

def build_dashboard(df: pd.DataFrame, output_path: str = "electricity_dashboard.html") -> str:
    """
    Build all charts and combine them into a single standalone interactive HTML dashboard.

    Charts are arranged by priority:
      Tier 1 (full width) - Daily Trend, Top Consumption Days table
      Tier 2 (2 per row)  - Daily Load Profile, Day of Week Pattern
      Tier 3 (2 per row)  - Heatmap, Hourly Distribution
      Tier 4 (full width) - Cumulative Sum
    """
    # each entry: (title, figure, width) where width is 'full' or 'half'
    charts = [
        ("Daily Trend", plot_daily_trend(df, freq='1d'), 'half'),
        ("Day of Week Pattern", plot_day_of_week_pattern(df, log_scale=True), 'half'),
        ("Top 10 Consumption Days", table_top_consumption_days(df, top_n=10), 'full'),
        ("Average Daily Load Profile", plot_daily_load_profile(df), 'full'),        
        ("Heatmap (Hour vs Date)", plot_heatmap(df), 'full'),
        ("Hourly Distribution (Boxplot)", plot_hourly_distribution(df), 'full'),
        ("Cumulative Sum (Daily, last 14 days)", plot_cumulative_sum(df, sort_by='max_cumsum', period='daily', max_periods=14), 'full'),
    ]

    date_min = pd.to_datetime(df['datetime']).min()
    date_max = pd.to_datetime(df['datetime']).max()

    sections_html = []
    for i, (title, fig, width) in enumerate(charts):
        div = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{i}")
        css_class = "chart-card full" if width == 'full' else "chart-card half"
        sections_html.append(f"""
        <div class="{css_class}">
            <h2>{title}</h2>
            {div}
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Electricity Consumption Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
    body {{
        font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: #f4f6f8;
        margin: 0;
        padding: 0;
        color: #1a1a1a;
    }}
    header {{
        background: #2E86AB;
        color: white;
        padding: 24px 32px;
    }}
    header h1 {{ margin: 0; font-size: 24px; }}
    header p {{ margin: 6px 0 0; opacity: 0.9; font-size: 14px; }}
    .container {{
        max-width: 1300px;
        margin: 0 auto;
        padding: 24px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}
    .chart-card {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        min-width: 0; /* prevents plotly divs from overflowing grid cells */
    }}
    .chart-card.full {{ grid-column: 1 / -1; }}
    .chart-card.half {{ grid-column: span 1; }}
    .chart-card h2 {{
        font-size: 16px;
        color: #2E86AB;
        margin: 0 0 12px;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
    }}
    @media (max-width: 900px) {{
        .container {{ grid-template-columns: 1fr; }}
        .chart-card.half {{ grid-column: 1 / -1; }}
    }}
</style>
</head>
<body>
<header>
    <h1>⚡ Electricity Consumption Dashboard</h1>
    <p>Data range: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d} &nbsp;|&nbsp; Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
</header>
<div class="container">
    {''.join(sections_html)}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    FOLDER = r"C:\Users\linas\OneDrive\Documents\PROJECT\elektros_suvartojimas"
    PATTERN = "Valandiniai*.csv"
    OUTPUT_HTML = "electricity_dashboard.html"

    latest_file = find_latest_file(FOLDER, PATTERN)
    print(f"Loading latest file: {latest_file}")

    df = load_data(latest_file)
    print(f"Loaded {len(df)} rows, from {df['datetime'].min()} to {df['datetime'].max()}")

    out_path = build_dashboard(df, OUTPUT_HTML)
    print(f"Dashboard saved to: {out_path}")