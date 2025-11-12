# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:25:06 2025

@author: gabri
"""

import datetime as dt
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import List
import networkx as nx
import boto3
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title="Grain movement by Sea"
server = app.server

# ... 
access_ll = os.environ.get("ACCESS_LL") 
ws_llave = os.environ.get("WS_LLAVE")
BUCKET = "graindisruption"

# ---- File paths in your S3 bucket ----
month_imp_path = f"s3://{BUCKET}/imp_pair_monthly_jan_2019_dec_2024.parquet"
month_exp_path = f"s3://{BUCKET}/exp_pair_monthly_jan_2019_dec_2024.parquet"
week_imp_path  = f"s3://{BUCKET}/imp_pair_weekly_jan_2019_dec_2024.parquet"
week_exp_path  = f"s3://{BUCKET}/exp_pair_weekly_jan_2019_dec_2024.parquet"

# ---- Read parquet files ----
month_vol_imp = pd.read_parquet(month_imp_path, 
    storage_options={
        "key": access_ll, 
        "secret": ws_llave
    })

month_vol_exp = pd.read_parquet(month_exp_path, 
    storage_options={
        "key": access_ll, 
        "secret": ws_llave
    })

week_vol_imp = pd.read_parquet(week_imp_path, 
    storage_options={
        "key": access_ll, 
        "secret": ws_llave
    })

week_vol_exp = pd.read_parquet(week_exp_path, 
    storage_options={
        "key": access_ll, 
        "secret": ws_llave
    })

####------------------------------
CATEGORY_OPTIONS = ['Developing', 'Developed', 'LDC', 'SIDS', 'Territories']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Grain and Agripods transported by sea"

# ----------------------------
# DF selection + period helpers
# ----------------------------
def select_df(flow: str, freq: str):
    """
    Returns:
      df_active, period_col,
      side_country_col (discharge for Import, load for Export),
      side_category_col (matching side)
    """
    flow = (flow or "Import").title()
    freq = (freq or "weekly").lower()

    if flow == "Import" and freq == "weekly":
        df_active = week_vol_imp
        period_col = "import_week"
        side_country_col = "discharge_country"
        side_category_col = "category_discharge_country"
    elif flow == "Import" and freq == "monthly":
        df_active = month_vol_imp
        period_col = "import_month"
        side_country_col = "discharge_country"
        side_category_col = "category_discharge_country"
    elif flow == "Export" and freq == "weekly":
        df_active = week_vol_exp
        period_col = "export_week"
        side_country_col = "load_country"
        side_category_col = "category_load_country"
    else:  # Export + Monthly
        df_active = month_vol_exp
        period_col = "export_month"
        side_country_col = "load_country"
        side_category_col = "category_load_country"

    if period_col not in df_active.columns:
        raise ValueError(f"Expected period column '{period_col}' not found in selected DF.")
    return df_active, period_col, side_country_col, side_category_col


def _parse_period_parts(label: str):
    """
    Returns (year:int, part:int) from a period label that may look like:
      - YYYY-MM   (e.g., 2024-03)
      - MM-YYYY   (e.g., 03-2024)
      - YYYY-WW   (e.g., 2024-12)  # weeks
    Falls back to (9999, 9999) if it can't parse.
    """
    try:
        a, b = str(label).split("-")
        # MM-YYYY (month first)
        if len(a) == 2 and len(b) == 4:
            return int(b), int(a)
        # YYYY-MM or YYYY-WW (year first)
        if len(a) == 4:
            return int(a), int(b)
        # Fallback guess: if b looks like year
        if len(b) == 4:
            return int(b), int(a)
        return 9999, 9999
    except Exception:
        return 9999, 9999

def periods_sorted(df: pd.DataFrame, period_col: str) -> list:
    """Unique period labels sorted chronologically regardless of their format."""
    vals = df[period_col].dropna().astype(str).unique().tolist()
    return sorted(vals, key=lambda x: _parse_period_parts(x))

def ordered_category_array(series_list) -> list:
    """
    Build a single ordered category array from one or more Series of period labels.
    Uses the same chronological sorter as periods_sorted.
    """
    import pandas as pd
    all_vals = pd.concat(
        [s.astype(str) for s in series_list if s is not None and len(s) > 0],
        ignore_index=True
    ).drop_duplicates().tolist()
    return sorted(all_vals, key=lambda x: _parse_period_parts(x))

def clamp_periods_h1_2025(periods: List[str], freq: str) -> List[str]:
    """
    Keep all periods before 2025, and only H1 of 2025:
      - monthly: months <= 06
      - weekly : ISO weeks <= 26
    Works with YYYY-MM, MM-YYYY, YYYY-WW.
    """
    def keep(p):
        y, part = _parse_period_parts(p)
        if y < 2025:
            return True
        if y > 2025:
            return False
        # y == 2025
        if freq == "monthly":
            return part <= 6
        else:
            return part <= 26
    filtered = [p for p in periods if keep(p)]
    return filtered or periods

# ----------------------------
# Filtering helpers
# ----------------------------
def _category_mask(series: pd.Series, selected: list) -> pd.Series:
    """True if cell matches ANY selected token (exact or substring, case-insensitive)."""
    if not selected:
        return pd.Series(True, index=series.index)
    s = series.astype("string").fillna("")
    sel_lower = [x.lower() for x in selected]
    exact = s.str.strip().str.lower().isin(sel_lower)
    substr = pd.Series(False, index=series.index)
    for token in sel_lower:
        substr |= s.str.lower().str.contains(token, na=False)
    return exact | substr

def filter_df(
    df_active: pd.DataFrame,
    period_col: str,
    side_country_col: str,
    side_category_col: str,
    start_period: str,
    end_period: str,
    categories_selected: list,
    countries_selected: list,
    commodities_selected: list
):
    d = df_active.copy()
    d[period_col] = d[period_col].astype(str)

    # Period window
    all_periods = periods_sorted(d, period_col)
    if not all_periods:
        return d.iloc[0:0]
    i0 = all_periods.index(start_period) if start_period in all_periods else 0
    i1 = all_periods.index(end_period) if end_period in all_periods else len(all_periods) - 1
    if i0 > i1:
        i0, i1 = i1, i0
    chosen = set(all_periods[i0:i1+1])
    d = d[d[period_col].isin(chosen)]

    # Category on chosen side
    d = d[_category_mask(d[side_category_col], categories_selected)]

    # Country on chosen side
    if countries_selected and "All" not in countries_selected:
        d = d[d[side_country_col].astype(str).isin(countries_selected)]

    # Commodity
    if commodities_selected and "All" not in commodities_selected:
        d = d[d["commodity"].astype(str).isin(commodities_selected)]

    return d

# ----------------------------
# Fig 1: Sankey builder
# ----------------------------
def build_sankey(dff: pd.DataFrame, top_n_right: int = 5, valuesuffix=" mt"):
    """
    Build Sankey load_country -> discharge_country from filtered df.
    Keeps top_n_right discharge countries by total volume on RIGHT.
    """
    if dff.empty:
        fig = go.Figure(go.Sankey(node=dict(label=[]), link=dict(source=[], target=[], value=[])))
        fig.update_layout(title="No data for current filter")
        return fig

    # Top-N discharge countries (right side)
    top_right = (dff.groupby("discharge_country", as_index=False)["voy_intake_mt"].sum()
                   .sort_values("voy_intake_mt", ascending=False)
                   .head(top_n_right)["discharge_country"]
                   .tolist())

    d = dff[dff["discharge_country"].isin(top_right)].copy()

    # Aggregate flows
    flows = d.groupby(["load_country", "discharge_country"], as_index=False)["voy_intake_mt"].sum()

    # Labels: left loads, right discharges (suffix to disambiguate)
    loads = flows["load_country"].astype(str).unique().tolist()
    rights = (flows["discharge_country"].astype(str) + " (dest)").unique().tolist()
    labels = loads + rights

    idx = {lab: i for i, lab in enumerate(labels)}
    sources = flows["load_country"].map(idx).tolist()
    targets = (flows["discharge_country"] + " (dest)").map(idx).tolist()
    values  = flows["voy_intake_mt"].tolist()

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        valuesuffix=valuesuffix,
        node=dict(
            label=labels,
            pad=15,
            thickness=16,
            line=dict(color="rgba(0,0,0,0.15)", width=1),
            color="rgba(60,99,245,0.85)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(60,99,245,0.25)",
            hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f}"+valuesuffix+"<extra></extra>",
        ),
    ))
    fig.update_layout(
        title="Tonnes exchange between countries",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=12)
    )
    return fig

# ----------------------------
# Fig 2: Average series helper
# ----------------------------
def period_sum_series(d: pd.DataFrame, period_col: str, side_country_col: str) -> pd.DataFrame:
    """
    Sum voy_intake_mt per period (optionally after summing per country),
    and return it chronologically sorted (handles YYYY-MM, MM-YYYY, YYYY-WW).
    """
    if d is None or d.empty:
        return pd.DataFrame({period_col: [], "sum_voy_intake_mt": []})

    # Sum per (period, country) -> then total per period (equivalent to direct period sum)
    per_cty = d.groupby([period_col, side_country_col], as_index=False)["voy_intake_mt"].sum()
    out = (per_cty.groupby(period_col, as_index=False)["voy_intake_mt"]
                 .sum()
                 .rename(columns={"voy_intake_mt": "sum_voy_intake_mt"}))

    # Order by our robust chronological key
    out[period_col] = out[period_col].astype(str)
    cats = ordered_category_array([out[period_col]])
    out[period_col] = pd.Categorical(out[period_col], categories=cats, ordered=True)
    out = out.sort_values(period_col)
    return out

# ----------------------------
# Controls
# ----------------------------
controls = dbc.Card(
    [
        html.H5("Filters", className="mb-3"),

        dbc.Label("Trade Flow"),
        dcc.RadioItems(
            id="flow-check",
            options=[{"label": "Import", "value": "Import"},
                     {"label": "Export", "value": "Export"}],
            value="Import",
            labelStyle={"display": "block"},
        ),

        html.Hr(className="my-3"),

        dbc.Label("Frequency"),
        dcc.RadioItems(
            id="freq-check",
            options=[{"label": "Weekly", "value": "weekly"},
                     {"label": "Monthly", "value": "monthly"}],
            value="weekly",
            labelStyle={"display": "block"},
        ),

        html.Div(
            [
                dbc.Label("Start period"),
                dcc.Dropdown(id="period-start", options=[], value=None, clearable=False),
                dbc.Label("End period", className="mt-2"),
                dcc.Dropdown(id="period-end", options=[], value=None, clearable=False),
            ],
            className="mb-2"
        ),

        html.Hr(className="my-3"),

        dbc.Label("Country category"),
        dcc.Checklist(
            id="cat-check",
            options=[{"label": c, "value": c} for c in CATEGORY_OPTIONS],
            value=[],  # no category filter by default
            labelStyle={"display": "block"}
        ),

        html.Hr(className="my-3"),

        dbc.Label("Country"),
        dcc.Dropdown(
            id="country-dd",
            options=[{"label": "All", "value": "All"}],  # populated dynamically
            value=["All"],
            multi=True,
            clearable=False,
        ),

        dbc.Label("Commodity", className="mt-2"),
        dcc.Dropdown(
            id="commodity-dd",
            options=[{"label": "All", "value": "All"}],  # populated dynamically
            value=["All"],
            multi=True,
            clearable=False,
        ),
    ],
    body=True,
    className="shadow-sm"
)

def _elnino_bounds_for_freq(freq: str):
    # Hovmöller-ish window: Jun 2023 – Apr 2024
    if freq == "monthly":
        return (2023, 6), (2024, 4)     # months
    else:
        return (2023, 22), (2024, 17)   # ISO weeks (≈ Jun–Apr)

def _elnino_span_labels(cats: list, freq: str):
    """Return (x0_label, x1_label) within the CURRENT category list."""
    (y0, p0), (y1, p1) = _elnino_bounds_for_freq(freq)
    idxs = []
    for i, lab in enumerate(cats):
        y, p = _parse_period_parts(lab)
        if y == 9999: 
            continue
        if (y == 2023 and ((freq == "monthly" and p >= p0) or (freq != "monthly" and p >= p0))) \
           or (y == 2024 and ((freq == "monthly" and p <= p1) or (freq != "monthly" and p <= p1))):
            idxs.append(i)
    if not idxs:
        return None, None
    return cats[min(idxs)], cats[max(idxs)]


def _year_part_from_dt(d: dt.datetime, freq: str):
    if freq == "monthly":
        return d.year, d.month
    y, w, _ = d.isocalendar()
    return y, w

def _range_endpoints_in_cats(cats, start_dt, end_dt, freq: str):
    """
    Find x0/x1 labels that exist in current x-axis categories for a date span.
    Works with YYYY-MM, MM-YYYY, and YYYY-WW because it uses _parse_period_parts().
    """
    if not cats:
        return None, None

    ys, ps = _year_part_from_dt(start_dt, freq)
    ye, pe = _year_part_from_dt(end_dt,   freq)

    # normalize order
    if (ys, ps) > (ye, pe):
        ys, ps, ye, pe = ye, pe, ys, ps

    # keep all labels whose parsed (year,part) lie in the range
    hits = []
    for lab in cats:
        y, p = _parse_period_parts(lab)
        if y == 9999:
            continue
        if (ys, ps) <= (y, p) <= (ye, pe):
            hits.append(lab)

    if not hits:
        return None, None
    return hits[0], hits[-1]

def add_elnino_band(fig, cats, freq: str):
    # El Niño: Jun 2023 – Apr 2024
    x0, x1 = _range_endpoints_in_cats(
        cats, dt.datetime(2023, 6, 1), dt.datetime(2024, 4, 30), freq
    )
    if x0 is None:
        return
    fig.add_vrect(
        x0=x0, x1=x1, xref="x",
        fillcolor="rgba(220, 20, 60, 0.12)",  # light red
        line_width=0, layer="below"
    )
    # legend marker
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(220,20,60,0.25)"),
        name="El Niño"
    ))

def add_houthi_band(fig, cats, freq: str):
    # Houthi attacks: Oct 2023 – Oct 2025
    x0, x1 = _range_endpoints_in_cats(
        cats, dt.datetime(2023, 10, 1), dt.datetime(2025, 10, 31), freq
    )
    if x0 is None:
        return
    fig.add_vrect(
        x0=x0, x1=x1, xref="x",
        fillcolor="rgba(255, 180, 80, 0.15)",  # amber
        line_width=0, layer="below"
    )
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(255,180,80,0.35)"),
        name="Houthi attacks"
    ))

# ---- Chart card with Expand button ----
def chart_card(chart_id: str, btn_id: str, height="400px"):
    return dbc.Card(
        [
            dbc.Button(
                "Expand",
                id=btn_id,
                size="sm",
                color="secondary",
                outline=True,
                className="position-absolute top-0 end-0 m-2",
                style={"zIndex": 5},
            ),
            dcc.Graph(id=chart_id, config={"displayModeBar": False}, style={"height": "100%"}),
        ],
        className="p-2 position-relative",
        style={"height": height},
    )

grid = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(chart_card("chart-1", "expand-1"), md=6),
                dbc.Col(chart_card("chart-2", "expand-2"), md=6),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(chart_card("chart-3", "expand-3"), md=6),
                dbc.Col(chart_card("chart-4", "expand-4"), md=6),
            ],
            className="g-3 mt-1",
        ),
    ]
)

# ---- Modals for enlarged view ----
modals = [
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(f"Expanded View {i}")),
            dbc.ModalBody(dcc.Graph(id=f"modal-chart-{i}", style={"height": "80vh"})),
        ],
        id=f"modal-{i}",
        size="xl",
        centered=True,
        is_open=False,
        backdrop=True,
        scrollable=True,
    )
    for i in (1, 2, 3, 4)
]

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(controls, xs=12, sm=5, md=4, lg=3, xl=3),
                dbc.Col(grid, xs=12, sm=7, md=8, lg=9, xl=9)
            ],
            className="g-3 my-2",
        ),
        *modals,
    ],
)

# ----------------------------
# “All” logic for Country & Commodity
# ----------------------------
@app.callback(
    Output("country-dd", "value"),
    Input("country-dd", "value"),
    prevent_initial_call=True
)
def country_all_logic(value):
    if not value: return ["All"]
    if isinstance(value, str): value = [value]
    if "All" in value and len(value) > 1:
        value = [v for v in value if v != "All"]
    if len(value) == 0: value = ["All"]
    return value

@app.callback(
    Output("commodity-dd", "value"),
    Input("commodity-dd", "value"),
    prevent_initial_call=True
)
def commodity_all_logic(value):
    if not value: return ["All"]
    if isinstance(value, str): value = [value]
    if "All" in value and len(value) > 1:
        value = [v for v in value if v != "All"]
    if len(value) == 0: value = ["All"]
    return value

# ----------------------------
# Update period, country, commodity options when flow/freq change
# (and clamp periods to H1-2025)
# ----------------------------
@app.callback(
    Output("period-start", "options"),
    Output("period-end", "options"),
    Output("period-start", "value"),
    Output("period-end", "value"),
    Output("country-dd", "options"),
    Output("commodity-dd", "options"),
    Input("flow-check", "value"),
    Input("freq-check", "value"),
)
def update_period_country_commodity_options(flow, freq):
    df_active, period_col, side_country_col, _ = select_df(flow, freq)

    periods = periods_sorted(df_active, period_col) or ["N/A"]
    periods = clamp_periods_h1_2025(periods, freq)  # ⬅️ clamp to H1-2025

    period_opts = [{"label": x, "value": x} for x in periods]
    start_val, end_val = periods[0], periods[-1]

    # Country options (chosen side)
    cvals = sorted(df_active[side_country_col].dropna().astype(str).unique().tolist())
    country_opts = [{"label": "All", "value": "All"}] + [{"label": c, "value": c} for c in cvals]

    # Commodity options
    com_vals = sorted(df_active["commodity"].dropna().astype(str).unique().tolist())
    commodity_opts = [{"label": "All", "value": "All"}] + [{"label": c, "value": c} for c in com_vals]

    return period_opts, period_opts, start_val, end_val, country_opts, commodity_opts

# ----------------------------
# Charts (Fig1 = Sankey, Fig2 = averages)
# ----------------------------
@app.callback(
    Output("chart-1", "figure"),
    Output("chart-2", "figure"),
    Output("chart-3", "figure"),
    Output("chart-4", "figure"),
    Input("flow-check", "value"),
    Input("freq-check", "value"),
    Input("period-start", "value"),
    Input("period-end", "value"),
    Input("cat-check", "value"),
    Input("country-dd", "value"),
    Input("commodity-dd", "value"),
)
def update_charts(flow, freq, start_p, end_p, categories_selected, countries_selected, commodities_selected):
    # Active DF (single flow/freq)
    df_active, period_col, side_country_col, side_category_col = select_df(flow, freq)

    # Defensive clamp: ensure start/end within allowed H1-2025 set
    labels_all = clamp_periods_h1_2025(periods_sorted(df_active, period_col), freq)
    if labels_all:
        if start_p not in labels_all: start_p = labels_all[0]
        if end_p   not in labels_all: end_p   = labels_all[-1]
        if labels_all.index(start_p) > labels_all.index(end_p):
            start_p, end_p = end_p, start_p

    # Filtered subset for the chosen flow (selection series)
    dff = filter_df(df_active, period_col, side_country_col, side_category_col,
                    start_p, end_p, categories_selected, countries_selected, commodities_selected)

    # FIG 1: Sankey (Top-5 right)
    fig1 = build_sankey(dff, top_n_right=5, valuesuffix=" mt")

    # For scatter and comparisons: both flows at SAME frequency
    if freq == "weekly":
        df_imp = week_vol_imp; imp_period = "import_week"; imp_side_country = "discharge_country"; imp_side_cat = "category_discharge_country"
        df_exp = week_vol_exp; exp_period = "export_week"; exp_side_country = "load_country";       exp_side_cat = "category_load_country"
    else:
        df_imp = month_vol_imp; imp_period = "import_month"; imp_side_country = "discharge_country"; imp_side_cat = "category_discharge_country"
        df_exp = month_vol_exp; exp_period = "export_month"; exp_side_country = "load_country";       exp_side_cat = "category_load_country"

    dff_imp = filter_df(df_imp, imp_period, imp_side_country, imp_side_cat,
                        start_p, end_p, categories_selected, countries_selected, commodities_selected)
    dff_exp = filter_df(df_exp, exp_period, exp_side_country, exp_side_cat,
                        start_p, end_p, categories_selected, countries_selected, commodities_selected)

    # --- FIGURE 2: Average over time (Overall vs Selection) ---
    # Overall avg must IGNORE Category and Country; respect Flow/Freq/Period/Commodity
    dff_overall = filter_df(
    df_active, period_col, side_country_col, side_category_col,
    start_p, end_p,
    categories_selected=[],         # ignore category for gray baseline
    countries_selected=["All"],     # ignore country for gray baseline
    commodities_selected=commodities_selected
    )
    
    sel_series = period_sum_series(dff,         period_col, side_country_col)
    all_series = period_sum_series(dff_overall, period_col, side_country_col)
    
    if sel_series.empty and all_series.empty:
        fig2 = px.line(title="No data for current filter")
    else:
        fig2 = px.line(title="Tonnes over time")
    
        # Build a single ordered category array and apply it to both series
        cats = ordered_category_array([
            all_series[period_col] if not all_series.empty else None,
            sel_series[period_col] if not sel_series.empty else None,
        ])
    
        if not all_series.empty:
            s = all_series.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats, ordered=True)
            s = s.sort_values(period_col)
            fig2.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Overall sum",
                line=dict(color="#9aa0a6", width=2),
            )
    
        if not sel_series.empty:
            s = sel_series.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats, ordered=True)
            s = s.sort_values(period_col)
            fig2.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Selection sum",
                line=dict(width=3),
            )
    
        fig2.update_xaxes(type="category", categoryorder="array", categoryarray=cats)
        fig2.update_layout(yaxis_title="Sum voy_intake_mt")
        
        add_elnino_band(fig2, cats, "monthly" if freq == "monthly" else "weekly")

# --- FIGURE 3: Suez-only sum over time (Overall vs Selection) ---
    # Safe transit masks
    if "suez_transit" not in dff.columns:
        dff["suez_transit"] = 0
    dff_suez = dff.loc[dff["suez_transit"].fillna(0).astype(int) == 1].copy()

    dff_overall_suez = filter_df(
        df_active, period_col, side_country_col, side_category_col,
        start_p, end_p,
        categories_selected=[],         # ignore category for gray baseline
        countries_selected=["All"],     # ignore country for gray baseline
        commodities_selected=commodities_selected
    ).copy()
    if "suez_transit" not in dff_overall_suez.columns:
        dff_overall_suez["suez_transit"] = 0
    dff_overall_suez = dff_overall_suez.loc[dff_overall_suez["suez_transit"].fillna(0).astype(int) == 1]

    sel_sum_sz = period_sum_series(dff_suez,           period_col, side_country_col)
    all_sum_sz = period_sum_series(dff_overall_suez,   period_col, side_country_col)

    if sel_sum_sz.empty and all_sum_sz.empty:
        fig3 = px.line(title="No Suez transit data for current filter")
    else:
        fig3 = px.line(title="Suez Canal transit tonnes")

        cats3 = ordered_category_array([
            all_sum_sz[period_col] if not all_sum_sz.empty else None,
            sel_sum_sz[period_col] if not sel_sum_sz.empty else None,
        ])

        if not all_sum_sz.empty:
            s = all_sum_sz.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats3, ordered=True)
            s = s.sort_values(period_col)
            fig3.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Overall (Suez)",
                line=dict(color="#9aa0a6", width=2),
            )

        if not sel_sum_sz.empty:
            s = sel_sum_sz.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats3, ordered=True)
            s = s.sort_values(period_col)
            fig3.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Selection (Suez)",
                line=dict(width=3),
            )

    fig3.update_xaxes(type="category", categoryorder="array", categoryarray=cats3)
    fig3.update_layout(yaxis_title="Sum voy_intake_mt")
    # FIG 4: stacked area by country over period (active side)
    
    # --- FIGURE 4: Transit-only sum over time (Overall vs Selection) ---
    # Safe transit masks
    if "pc_transit" not in dff.columns:
        dff["pc_transit"] = 0
    dff_transit = dff.loc[dff["pc_transit"].fillna(0).astype(int) == 1].copy()
    
    dff_overall_transit = filter_df(
        df_active, period_col, side_country_col, side_category_col,
        start_p, end_p,
        categories_selected=[],         # ignore category for gray baseline
        countries_selected=["All"],     # ignore country for gray baseline
        commodities_selected=commodities_selected
    ).copy()
    if "pc_transit" not in dff_overall_transit.columns:
        dff_overall_transit["pc_transit"] = 0
    dff_overall_transit = dff_overall_transit.loc[dff_overall_transit["pc_transit"].fillna(0).astype(int) == 1]
    
    sel_sum = period_sum_series(dff_transit,           period_col, side_country_col)
    all_sum = period_sum_series(dff_overall_transit,   period_col, side_country_col)
    
    if sel_sum.empty and all_sum.empty:
        fig4 = px.line(title="No transit data for current filter")
    else:
        fig4 = px.line(title="Panama Canal transit tonnes")
    
        cats4 = ordered_category_array([
            all_sum[period_col] if not all_sum.empty else None,
            sel_sum[period_col] if not sel_sum.empty else None,
        ])
    
        if not all_sum.empty:
            s = all_sum.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats4, ordered=True)
            s = s.sort_values(period_col)
            fig4.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Overall (transit)",
                line=dict(color="#9aa0a6", width=2),
            )
    
        if not sel_sum.empty:
            s = sel_sum.copy()
            s[period_col] = pd.Categorical(s[period_col].astype(str), categories=cats4, ordered=True)
            s = s.sort_values(period_col)
            fig4.add_scatter(
                x=s[period_col].astype(str),
                y=s["sum_voy_intake_mt"],
                mode="lines",
                name="Selection (transit)",
                line=dict(width=3),
            )
    
        fig4.update_xaxes(type="category", categoryorder="array", categoryarray=cats4)
        fig4.update_layout(yaxis_title="Sum voy_intake_mt")
        add_elnino_band(fig4, cats4, "monthly" if freq == "monthly" else "weekly")
            
    for fig in (fig1, fig2, fig3, fig4):
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title=None)
    return fig1, fig2, fig3, fig4

# ----------------------------
# Expand modals
#   Fig 1 recomputes Sankey with Top-10 right using current filters.
#   Figs 2-4 reuse their small figures.
# ----------------------------
@app.callback(
    Output("modal-1", "is_open"),
    Output("modal-chart-1", "figure"),
    Input("expand-1", "n_clicks"),
    State("modal-1", "is_open"),
    # pass current filters to recompute Sankey at Top-10
    State("flow-check", "value"),
    State("freq-check", "value"),
    State("period-start", "value"),
    State("period-end", "value"),
    State("cat-check", "value"),
    State("country-dd", "value"),
    State("commodity-dd", "value"),
    prevent_initial_call=True
)
def open_modal_1(n, is_open, flow, freq, start_p, end_p, categories_selected, countries_selected, commodities_selected):
    if not n:
        return is_open, dash.no_update
    # Build filtered df (same logic as main callback)
    df_active, period_col, side_country_col, side_category_col = select_df(flow, freq)

    # Clamp defensively
    labels_all = clamp_periods_h1_2025(periods_sorted(df_active, period_col), freq)
    if labels_all:
        if start_p not in labels_all: start_p = labels_all[0]
        if end_p   not in labels_all: end_p   = labels_all[-1]
        if labels_all.index(start_p) > labels_all.index(end_p):
            start_p, end_p = end_p, start_p

    dff = filter_df(df_active, period_col, side_country_col, side_category_col,
                    start_p, end_p, categories_selected, countries_selected, commodities_selected)
    # Sankey Top-10 right
    fig_big = build_sankey(dff, top_n_right=10, valuesuffix=" mt")
    return True, fig_big

@app.callback(
    Output("modal-2", "is_open"),
    Output("modal-chart-2", "figure"),
    Input("expand-2", "n_clicks"),
    State("modal-2", "is_open"),
    State("chart-2", "figure"),
    prevent_initial_call=True
)
def open_modal_2(n, is_open, fig):
    if n: return True, fig
    return is_open, fig

@app.callback(
    Output("modal-3", "is_open"),
    Output("modal-chart-3", "figure"),
    Input("expand-3", "n_clicks"),
    State("modal-3", "is_open"),
    State("chart-3", "figure"),
    prevent_initial_call=True
)
def open_modal_3(n, is_open, fig):
    if n: return True, fig
    return is_open, fig

@app.callback(
    Output("modal-4", "is_open"),
    Output("modal-chart-4", "figure"),
    Input("expand-4", "n_clicks"),
    State("modal-4", "is_open"),
    State("chart-4", "figure"),
    prevent_initial_call=True
)
def open_modal_4(n, is_open, fig):
    if n: return True, fig
    return is_open, fig

# ----------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
# ---- Configuration ----

