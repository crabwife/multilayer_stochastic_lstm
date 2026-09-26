import io
import json
import re
import numpy as np
import pandas as pd

CANDIDATES = {
    "year": ["year", "sampleyear", "yr", "date", "sampledate"],
    "field": ["field", "fieldid", "site"],
    "plot": ["plot", "plotid", "plotnumber"],
    "mass": ["mass", "mass.above", "biomass", "biomassg", "drymass", "weight", "livebiomass", "prod"],
    "treatment": ["trt", "treatment", "treat", "nitrogen", "ntrt"],
    "taxon": ["species", "sp", "taxon", "taxa", "speciesname"]}

def column(frame, key, overrides):
    if overrides.get(key):
        return overrides[key]
    norm = lambda s: re.sub("[^a-z0-9]", "", str(s).lower())
    hits = [c for c in frame if norm(c) in {norm(s) for s in CANDIDATES[key]}]
    if len(hits) == 1:
        return hits[0]
    if key == "year" and len(hits) > 1:
        direct = [c for c in hits if norm(c) in ("year", "sampleyear", "yr")]
        if len(direct) == 1:
            return direct[0]
    if not hits and key in ("treatment", "taxon"):
        return None
    raise ValueError(f"Ambiguous or missing {key}: {hits}. Available columns: {list(frame.columns)}. "
                     f"Set data.columns.{key} in config.json")

def years(values, name):
    if re.sub("[^a-z0-9]", "", str(name).lower()) not in ("date", "sampledate"):
        return pd.to_numeric(values, errors="coerce")
    s = values.astype(str).str.strip()
    # E001 dates can be numeric YYMMDD or conventional YYYY-MM-DD.
    start4 = pd.to_numeric(s.str.extract(r"^((?:19|20)\d{2})")[0], errors="coerce")
    start2 = pd.to_numeric(s.str.extract(r"^(\d{2})\d{4}(?:\.0)?$")[0], errors="coerce")
    short = np.where(start2 >= 50, 1900 + start2, 2000 + start2)
    parsed = pd.to_datetime(s, errors="coerce", format="mixed").dt.year
    return start4.fillna(pd.Series(short, index=values.index)).fillna(parsed)

def panelize(raw, settings):
    lines = raw.splitlines()
    skipped = settings.get("skiprows")
    if skipped is None:
        skipped = int(len(lines) > 1 and lines[0].count(b",") < lines[1].count(b","))
    frame = pd.read_csv(io.BytesIO(raw), skiprows=skipped, low_memory=False)
    cols = {key: column(frame, key, settings["columns"]) for key in CANDIDATES}
    d = frame.copy()
    d["year"] = years(d[cols["year"]], cols["year"])
    d["mass"] = pd.to_numeric(d[cols["mass"]], errors="coerce")
    d["field"] = d[cols["field"]].astype(str).str.strip().str.upper().str.replace(
        r"^FIELD\s+", "", regex=True)
    d["plot"] = d[cols["plot"]].astype(str).str.strip()
    d["treatment"] = d[cols["treatment"]].astype(str).str.strip() if cols["treatment"] else "unknown"
    audit = {"mapping": cols, "metadata_lines_skipped": skipped,
             "raw_rows": len(d), "nonnumeric_mass": int(d.mass.isna().sum()),
             "negative_mass": int((d.mass < 0).sum()),
             "fields_raw": d.field.value_counts().to_dict(),
             "years_raw": sorted(d.year.dropna().astype(int).unique().tolist())}
    d = d[d.year.between(*settings["years"]) & d.mass.notna() & (d.mass >= 0)]
    if settings["fields"]:
        d = d[d.field.isin({str(x).upper() for x in settings["fields"]})]
    if settings["measure_mode"] == "species_mass":
        if cols["taxon"]:
            dead = d[cols["taxon"]].astype(str).str.contains(
                r"(?:^|\b)(?:litter|dead|standing dead)(?:\b|$)", case=False, regex=True, na=False)
            audit["excluded_dead_or_litter"] = int(dead.sum())
            d = d[~dead]
        if (d.groupby(["field", "plot", "year"]).treatment.nunique() > 1).any():
            raise ValueError("Multiple treatments within one plot-year")
        group = d.groupby(["field", "plot", "year"], as_index=False)
        panel = group.agg(biomass=("mass", "sum"), treatment=("treatment", "first"))
    elif settings["measure_mode"] == "plot_total":
        if d.duplicated(["field", "plot", "year"]).any():
            raise ValueError("Repeated plot-year in plot_total mode")
        panel = d[["field", "plot", "year", "mass", "treatment"]].rename(columns={"mass": "biomass"})
    else:
        raise ValueError(settings["measure_mode"])
    panel["year"] = panel.year.astype(int)
    panel["plot_id"] = panel.field + ":" + panel["plot"]
    audit["plot_years"] = len(panel)
    audit["plots"] = panel.plot_id.nunique()
    audit["year_counts"] = panel.groupby("year").size().to_dict()
    audit["biomass_quantiles"] = {str(k): v for k, v in panel.biomass.quantile(
        [0, .1, .5, .9, .99, 1]).to_dict().items()}
    return panel.sort_values(["plot_id", "year"]).reset_index(drop=True), audit

def windows(panel, lookback, horizon):
    xs, ys, meta = [], [], []
    for plot_id, group in panel.groupby("plot_id"):
        group = group.sort_values("year")
        for index in range(lookback, len(group)-horizon+1):
            block = group.iloc[index-lookback:index+horizon]
            if not np.all(np.diff(block.year.to_numpy()) == 1):
                continue
            xs.append(block.z.iloc[:lookback].to_numpy(np.float32)[:, None])
            ys.append(block.z.iloc[lookback:].to_numpy(np.float32))
            meta.append({"plot_id": plot_id, "year": int(block.year.iloc[lookback]),
                         "treatment": str(block.treatment.iloc[lookback])})
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)

def prepare(panel, settings, evaluation, output):
    from pathlib import Path
    output = Path(output)
    split = settings["split"]
    train = np.log1p(panel.loc[panel.year <= split["train_end"], "biomass"])
    mu, sd = float(train.mean()), float(train.std())
    panel["z"] = (np.log1p(panel.biomass)-mu)/sd
    panel.to_csv(output/"panel.csv", index=False)
    (output/"scaler.json").write_text(json.dumps({"log1p_mean": mu, "log1p_sd": sd}, indent=2))
    arrays, counts = {}, {}
    for prefix, horizon in [("one", 1), ("path", evaluation["horizon"])]:
        x, y, meta = windows(panel, split["lookback"], horizon)
        years = meta.year.to_numpy()
        masks = {"train": years <= split["train_end"],
                 "val": (years > split["train_end"]) & (years <= split["val_end"]),
                 "test": years > split["val_end"]}
        for label, mask in masks.items():
            if prefix == "path" and label != "test":
                continue
            arrays[f"{prefix}_{label}_x"] = x[mask]
            arrays[f"{prefix}_{label}_y"] = y[mask]
            meta[mask].reset_index(drop=True).to_csv(output/f"{prefix}_{label}_meta.csv", index=False)
            counts[f"{prefix}_{label}"] = {"windows": int(mask.sum()),
                                          "plots": int(meta[mask].plot_id.nunique())}
    if min(counts[f"one_{s}"]["windows"] for s in ["train", "val", "test"]) < 50:
        raise ValueError(f"Insufficient one-year windows: {counts}")
    if counts["one_test"]["plots"] < 20 or counts["path_test"]["windows"] < 30:
        raise ValueError(f"Insufficient independent test plots or path windows: {counts}")
    np.savez_compressed(output/"windows.npz", **arrays)
    return counts
