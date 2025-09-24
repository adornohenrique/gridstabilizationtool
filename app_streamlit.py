# app_streamlit.py
import io
from typing import Optional
import pandas as pd
import streamlit as st
from dispatch_core import optimize_dispatch

st.set_page_config(page_title="Dispatch Optimizer", layout="wide")
st.title("Quarter-hour Dispatch Optimizer (Profit-Max)")

# ---------- Helpers ----------
def _standardize_cols(df0: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Try to find timestamp and price columns; if only 2 cols, assume ts+price."""
    if df0 is None or df0.empty:
        return None
    cols_map = {str(c).strip().lower(): c for c in df0.columns}
    ts_key = next((k for k in cols_map if any(x in k for x in ["timestamp","time","datetime","interval","start"])), None)
    pr_key = next((k for k in cols_map if any(x in k for x in ["price","lmp","eur_per_mwh","usd_per_mwh","$/mwh","€/mwh"])), None)
    if ts_key and pr_key:
        out = df0[[cols_map[ts_key], cols_map[pr_key]]].copy()
        out.columns = ["timestamp", "price_eur_per_mwh"]
        return out
    if df0.shape[1] == 2:
        out = df0.copy()
        out.columns = ["timestamp", "price_eur_per_mwh"]
        return out
    return None

def load_prices(uploaded) -> pd.DataFrame:
    """Accept CSV (any common separator) or Excel; return df with [timestamp, price_eur_per_mwh]."""
    name = uploaded.name.lower()
    df = None

    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded)
        for sh in xls.sheet_names:
            try:
                tmp = pd.read_excel(xls, sheet_name=sh)
                df = _standardize_cols(tmp)
                if df is not None:
                    break
            except Exception:
                continue
        if df is None:
            raise ValueError("Could not find timestamp/price columns in the Excel file.")
    else:
        content = uploaded.read()
        uploaded.seek(0)

        # Try pandas automatic sniffing
        try:
            tmp = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            df = _standardize_cols(tmp)
        except Exception:
            df = None

        # Fallbacks: semicolon, tab, comma
        if df is None:
            for sep in [";", "\t", ","]:
                try:
                    tmp = pd.read_csv(io.BytesIO(content), sep=sep)
                    df = _standardize_cols(tmp)
                    if df is not None:
                        break
                except Exception:
                    continue
        if df is None:
            raise ValueError("CSV must contain timestamp and price columns. "
                             "Save as CSV (UTF-8) with headers: timestamp, price_eur_per_mwh.")

    # Clean types
    df = df.dropna(how="all")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["price_eur_per_mwh"].dtype == object:
        # handle comma decimals like "123,45"
        df["price_eur_per_mwh"] = df["price_eur_per_mwh"].astype(str).str.replace(",", ".", regex=False)
    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price_eur_per_mwh"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after parsing. Check your timestamp and price columns.")
    return df

# ---------- Sidebar (input mask) ----------
with st.sidebar:
    st.header("Inputs — Operations")
    uploaded = st.file_uploader("15-min price file (CSV or Excel)", type=["csv","xlsx","xls"])
    st.caption("Needs columns (or autodetected): timestamp and price.")

    plant_capacity_mw = st.number_input("Plant capacity (MW)", value=20.0, min_value=0.1, step=1.0)
    min_load_pct = st.slider("Min load (%)", 0.0, 100.0, 10.0, step=1.0) / 100.0
    max_load_pct = st.slider("Max load (%)", 0.0, 100.0, 100.0, step=1.0) / 100.0
    break_even = st.number_input("Break-even power price (€/MWh)", value=50.0, step=1.0)
    ramp_limit = st.number_input("Ramp limit (MW per 15-min) (optional)", value=2.0, step=0.5)
    always_on = st.checkbox("Always on (≥ min load)", value=True)

    st.header("Inputs — Production & Economics")
    mwh_per_ton = st.number_input("Electricity per ton (MWh/t)", value=11.0, step=0.1)
    methanol_price = st.number_input("Methanol price (€/t)", value=1000.0, step=10.0)
    co2_price = st.number_input("CO₂ price (€/t)", value=40.0, step=1.0)
    co2_intensity = st.number_input("CO₂ needed (t CO₂ per t MeOH)", value=1.375, step=0.025)
    maint_pct = st.number_input("Maintenance (% of revenue)", value=3.0, step=0.5) / 100.0
    sga_pct   = st.number_input("SG&A (% of revenue)", value=2.0, step=0.5) / 100.0
    ins_pct   = st.number_input("Insurance (% of revenue)", value=1.0, step=0.5) / 100.0

    st.header("Target margin control")
    margin_method = st.radio(
        "Margin method",
        ["Power-only (vs BE)", "Full-economics"],
        index=0,
        help="Choose how to compute the price cap used for dispatch."
    )
    target_margin_pct = st.number_input("Target margin (%)", value=30.0, step=1.0, min_value=0.0, max_value=95.0)

    run = st.button("Run Optimization")

# ---------- Compute price cap from target margin ----------
def compute_price_cap():
    p = float(target_margin_pct) / 100.0
    # Power-only rule: Price <= (1-p) * BE
    if margin_method.startswith("Power"):
        return max(0.0, (1.0 - p) * break_even), "power-only"

    # Full-economics rule:
    # x <= [ M*(1 - p - o) - C*k ] / E
    # where o = maint + sga + ins (fractions of revenue)
    o = float(maint_pct or 0) + float(sga_pct or 0) + float(ins_pct or 0)
    if mwh_per_ton <= 0:
        st.error("Full-economics margin requires Electricity per ton (MWh/t) > 0.")
        return None, "full-econ"
    cap = (methanol_price * (1.0 - p - o) - co2_price * co2_intensity) / mwh_per_ton
    return max(0.0, cap), "full-econ"

# ---------- Run ----------
if run:
    if uploaded is None:
        st.error("Please upload a CSV or Excel with timestamp and price.")
        st.stop()

    try:
        df = load_prices(uploaded)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Compute dispatch price cap from target margin
    price_cap, method_tag = compute_price_cap()
    if price_cap is None:
        st.stop()

    st.info(f"Applied dispatch price cap: **{price_cap:,.2f} €/MWh**  (method: {method_tag}, target margin: {target_margin_pct:.1f}%)")

    tmp_csv = "/tmp/_prices.csv"
    df.to_csv(tmp_csv, index=False)
    out_xlsx = "/tmp/dispatch_output.xlsx"

    results, kpis = optimize_dispatch(
        input_csv=tmp_csv,
        output_xlsx=out_xlsx,
        plant_capacity_mw=plant_capacity_mw,
        min_load_pct=min_load_pct,
        max_load_pct=max_load_pct,
        break_even_eur_per_mwh=break_even,                      # still used for proxy economics
        ramp_limit_mw_per_step=(ramp_limit if ramp_limit > 0 else None),
        always_on=always_on,
        dispatch_threshold_eur_per_mwh=price_cap,               # <-- dispatch uses this cap
        mwh_per_ton=(mwh_per_ton if mwh_per_ton > 0 else None),
        methanol_price_eur_per_ton=methanol_price,
        co2_price_eur_per_ton=co2_price,
        co2_t_per_ton_meoh=co2_intensity,
        maintenance_pct_of_revenue=maint_pct,
        sga_pct_of_revenue=sga_pct,
        insurance_pct_of_revenue=ins_pct,
        target_margin_fraction=float(target_margin_pct)/100.0,
        margin_method=method_tag,
    )

    st.success("Optimization complete.")

    # KPIs table (overview)
    st.subheader("KPIs (project overview)")
    show_cols = [
        "dispatch_threshold_eur_per_mwh",
        "target_margin_fraction",
        "margin_method",
        "total_energy_mwh",
        "weighted_avg_price_eur_per_mwh",
        "total_power_cost_eur",
        "total_tons",
        "total_methanol_revenue_eur",
        "total_co2_cost_eur",
        "total_opex_misc_eur",
        "total_true_profit_eur",
        "total_profit_proxy_eur",
    ]
    kpis_view = {k: kpis.get(k, None) for k in show_cols}
    st.dataframe(pd.DataFrame([kpis_view]))

    if "total_tons" in kpis and kpis.get("total_tons") is not None:
        st.metric("Total production (t)", f"{kpis['total_tons']}")

    st.subheader("Dispatch (first 200 rows)")
    st.dataframe(results.head(200))

    st.download_button("Download Excel (full results)",
                       data=open(out_xlsx, "rb").read(),
                       file_name="dispatch_plan.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.download_button("Download CSV (full results)",
                       data=results.to_csv(index=False).encode("utf-8"),
                       file_name="dispatch_plan.csv",
                       mime="text/csv")
else:
    st.info("Upload your 15-min price file and click **Run Optimization**.")
    st.caption("This app autodetects CSV/Excel, separators, and comma-decimal formats.")