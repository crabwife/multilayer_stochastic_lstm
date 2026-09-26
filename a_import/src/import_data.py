"""Import the author's archived Cedar Creek E001 biomass table."""

import hashlib
from pathlib import Path

DOI = "10.5061/dryad.dbrv15f5t"
DATASET = "desiervo_2023_e001_1982_2004"
ENTITY = "e001-aboveground-mass-2019-09-13.csv"
DRYAD_FILE = "https://datadryad.org/downloads/file_stream/2244942"
GITHUB_FILE = ("https://raw.githubusercontent.com/melissadesiervo1031/"
               "CedarCreekconvergence/main/data/" + ENTITY)


def _csv(response, url):
    response.raise_for_status()
    raw = response.content
    if not raw or b"," not in raw[:4096] or b"<html" in raw[:4096].lower():
        raise ValueError(f"Expected a CSV from {url}, received a different response")
    return raw


def fetch(raw_input):
    local = sorted(Path(raw_input).glob("*.csv"))
    if local:
        if len(local) != 1:
            raise ValueError("Place exactly one E001 biomass CSV in a_import/input")
        raw = local[0].read_bytes()
        if b"," not in raw[:4096]:
            raise ValueError(f"{local[0]} does not look like a CSV")
        return raw, {"source": str(local[0]), "entity": local[0].name,
                     "dataset": DATASET, "doi": DOI}

    import requests

    errors = []
    try:
        response = requests.get(DRYAD_FILE, timeout=120)
        raw = _csv(response, DRYAD_FILE)
        return raw, {"source": DRYAD_FILE, "entity": ENTITY,
                     "dataset": DATASET, "doi": DOI}
    except (requests.RequestException, ValueError) as exc:
        errors.append(f"Dryad: {exc}")

    # The paper's public code repository also contains this exact CSV.
    try:
        raw = _csv(requests.get(GITHUB_FILE, timeout=120), GITHUB_FILE)
        return raw, {"source": GITHUB_FILE, "entity": ENTITY,
                     "dataset": DATASET, "doi": DOI}
    except (requests.RequestException, ValueError) as exc:
        errors.append(f"GitHub: {exc}")

    raise RuntimeError(
        "Could not download the author-archived E001 CSV. " + "; ".join(errors) +
        ". Download '" + ENTITY + "' from https://doi.org/" + DOI +
        " into a_import/input/ and rerun make all."
    )


def save(raw, info, output):
    output = Path(output)
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["bytes"] = len(raw)
    (output / "e001_raw.csv").write_bytes(raw)
    return info
