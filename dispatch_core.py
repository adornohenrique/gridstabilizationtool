# dispatch_core.py
import numpy as np
import pandas as pd
from typing import Optional

def optimize_dispatch(
    input_csv: str,
    output_xlsx: str,
    plant_capacity_mw: float,
    min_load_pct: float,
    max_load_pct: float,
    break_even_eur_per_mwh: float,
    ramp_limit_mw_per_step: Optional[float],
    always_on: bool = True,
    # NEW: let dispatch use a price cap different from BE if provided
    dispatch_threshold_eur_per_mwh: Optional[float] = None,
    # Production intensity
    mwh_per_ton: Optional[float] = None,            # Electricity required per ton MeOH (MWh/t)
    # Commercial economics
    methanol_price_eur_per_ton: Optional[float] = None,
    co2_price_eur_per_ton: Optional[float] = None,
    co2_t_per_ton_meoh: Optional[float] = None,     # t CO2 needed per ton MeOH
    maintenance_pct_of_revenue: float = 0.0,        # e.g. 0.03 = 3%
    sga_pct_of_revenue: float = 0.0,                # e.g. 0.02
    insurance_pct_of_revenue: float = 0.0,          # e.g. 0.01
    # Optional metadata for KPIs
    target_margin_fraction: Optional[float] = None,
    margin_method: Optional[str] = None,            # "power-only" or "full-econ"
):
    """
    Optimize quarter-hourly dispatch. Dispatch rule uses:
      - dispatch_threshold_eur_per_mwh (if provided), else
      - break_even_eur_per_mwh

    'break_even_eur_per_mwh' is still used to compute the proxy revenue/profit columns,
    so you can compare the dispatch rule to your BE economics.

    Input CSV must have:
      - timestamp (ISO-8601)
      - price_eur_per_mwh (numeric)
    """
    # ---- Load and prep data ----
    df = pd.read_csv(input_csv)
    if "timestamp" not in df.columns or "price_eur_per_mwh" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'price_eur_per_mwh'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Infer interval length
    df["delta_min"] = df["timestamp"].diff().dt.total_seconds().div(60).fillna(15)
    interval_minutes = df["delta_min"].mode().iloc[0]
    interval_hours = interval_minutes / 60.0

    price = df["price_eur_per_mwh"].values

    # ---- Dispatch threshold (price cap) ----
    threshold = dispatch_threshold_eur_per_mwh if (dispatch_threshold_eur_per_mwh is not None) else break_even_eur_per_mwh

    # ---- Pointwise threshold policy ----
    target_pct = np.where(price <= threshold, max_load_pct, (min_load_pct if always_on else 0.0))

    # ---- Apply ramp constraint (optional) ----
    dispatch_pct = target_pct.copy()
    if ramp_limit_mw_per_step is not None and ramp_limit_mw_per_step >= 0:
        max_step_pct = ramp_limit_mw_per_step / max(plant_capacity_mw, 1e-9)
        # Forward
        for i in range(1, len(dispatch_pct)):
            up = dispatch_pct[i-1] + max_step_pct
            down = dispatch_pct[i-1] - max_step_pct
            dispatch_pct[i] = np.clip(dispatch_pct[i], down, up)
        # Backward
        for i in range(len(dispatch_pct)-2, -1, -1):
            up = dispatch_pct[i+1] + max_step_pct
            down = dispatch_pct[i+1] - max_step_pct
            dispatch_pct[i] = np.clip(dispatch_pct[i], down, up)

    # Enforce bounds and always-on rule
    lower = min_load_pct if always_on else 0.0
    dispatch_pct = np.clip(dispatch_pct, lower, max_load_pct)

    # ---- Core energy & power costs ----
    dispatch_mw = dispatch_pct * plant_capacity_mw
    energy_mwh = dispatch_mw * interval_hours
    power_cost_eur = price * energy_mwh

    # Proxy revenue/profit based on BE (so you can compare against your BE economics)
    profit_per_mwh_input_proxy = (break_even_eur_per_mwh - price)
    profit_proxy_eur = profit_per_mwh_input_proxy * energy_mwh
    revenue_proxy_eur = break_even_eur_per_mwh * energy_mwh

    results = df[["timestamp", "price_eur_per_mwh"]].copy()
    results["dispatch_pct"] = np.round(dispatch_pct, 4)
    results["dispatch_mw"] = np.round(dispatch_mw, 4)
    results["energy_mwh"] = np.round(energy_mwh, 5)
    results["power_cost_eur"] = np.round(power_cost_eur, 2)
    results["profit_per_mwh_input_proxy"] = np.round(profit_per_mwh_input_proxy, 4)
    results["profit_proxy_eur"] = np.round(profit_proxy_eur, 2)
    results["revenue_proxy_eur"] = np.round(revenue_proxy_eur, 2)
    results["cum_profit_proxy_eur"] = np.round(results["profit_proxy_eur"].cumsum(), 2)

    # ---- Optional: production & full economics ----
    total_tons = None
    methanol_revenue_eur = None
    co2_cost_eur = None
    opex_misc_eur = None
    true_profit_eur = None

    if mwh_per_ton and mwh_per_ton > 0:
        tons = energy_mwh / mwh_per_ton
        results["tons"] = np.round(tons, 5)
        total_tons = float(np.nansum(tons))

        if methanol_price_eur_per_ton is not None:
            meth_rev = tons * float(methanol_price_eur_per_ton)
            results["methanol_revenue_eur"] = np.round(meth_rev, 2)
            methanol_revenue_eur = float(np.nansum(meth_rev))

        if (co2_price_eur_per_ton is not None) and (co2_t_per_ton_meoh is not None):
            co2_needed = tons * float(co2_t_per_ton_meoh)
            co2_cost = co2_needed * float(co2_price_eur_per_ton)
            results["co2_cost_eur"] = np.round(co2_cost, 2)
            co2_cost_eur = float(np.nansum(co2_cost))

        pct_total = float(maintenance_pct_of_revenue or 0) + float(sga_pct_of_revenue or 0) + float(insurance_pct_of_revenue or 0)
        if methanol_revenue_eur is not None and pct_total > 0:
            opex_misc = (results.get("methanol_revenue_eur", 0.0).fillna(0.0).values) * pct_total
            results["opex_misc_eur"] = np.round(opex_misc, 2)
            opex_misc_eur = float(np.nansum(opex_misc))

        if methanol_revenue_eur is not None:
            rev_arr = results["methanol_revenue_eur"].fillna(0.0).values
            pwr_arr = results["power_cost_eur"].fillna(0.0).values
            co2_arr = results.get("co2_cost_eur", 0.0)
            if hasattr(co2_arr, "values"):
                co2_arr = co2_arr.fillna(0.0).values
            opex_arr = results.get("opex_misc_eur", 0.0)
            if hasattr(opex_arr, "values"):
                opex_arr = opex_arr.fillna(0.0).values

            true_profit = rev_arr - pwr_arr - co2_arr - opex_arr
            results["true_profit_eur"] = np.round(true_profit, 2)
            results["cum_true_profit_eur"] = np.round(results["true_profit_eur"].cumsum(), 2)
            true_profit_eur = float(np.nansum(true_profit))

    # ---- Totals & KPIs ----
    total_energy = float(results["energy_mwh"].sum())
    weighted_avg_price = (results["power_cost_eur"].sum() / total_energy) if total_energy > 0 else float("nan")

    kpis = {
        "plant_capacity_mw": plant_capacity_mw,
        "min_load_pct": min_load_pct,
        "max_load_pct": max_load_pct,
        "break_even_eur_per_mwh": break_even_eur_per_mwh,
        "dispatch_threshold_eur_per_mwh": threshold,
        "ramp_limit_mw_per_step": ramp_limit_mw_per_step,
        "always_on": always_on,
        "interval_minutes": interval_minutes,
        "mwh_per_ton": mwh_per_ton,
        "methanol_price_eur_per_ton": methanol_price_eur_per_ton,
        "co2_price_eur_per_ton": co2_price_eur_per_ton,
        "co2_t_per_ton_meoh": co2_t_per_ton_meoh,
        "maintenance_pct_of_revenue": maintenance_pct_of_revenue,
        "sga_pct_of_revenue": sga_pct_of_revenue,
        "insurance_pct_of_revenue": insurance_pct_of_revenue,
        "target_margin_fraction": target_margin_fraction,
        "margin_method": margin_method,
        "total_energy_mwh": round(total_energy, 4),
        "weighted_avg_price_eur_per_mwh": round(weighted_avg_price, 4),
        "total_power_cost_eur": round(results["power_cost_eur"].sum(), 2),
        "total_profit_proxy_eur": round(results["profit_proxy_eur"].sum(), 2),
    }
    if total_tons is not None:
        kpis["total_tons"] = round(total_tons, 4)
    if methanol_revenue_eur is not None:
        kpis["total_methanol_revenue_eur"] = round(methanol_revenue_eur, 2)
    if co2_cost_eur is not None:
        kpis["total_co2_cost_eur"] = round(co2_cost_eur, 2)
    if opex_misc_eur is not None:
        kpis["total_opex_misc_eur"] = round(opex_misc_eur, 2)
    if true_profit_eur is not None:
        kpis["total_true_profit_eur"] = round(true_profit_eur, 2)

    # ---- Save Excel ----
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        pd.DataFrame(list(kpis.items()), columns=["parameter", "value"]).to_excel(writer, "Parameters", index=False)
        results.to_excel(writer, "Dispatch", index=False)

    return results, kpis