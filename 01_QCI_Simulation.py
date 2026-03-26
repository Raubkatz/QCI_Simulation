#!/usr/bin/env python3
"""
QKD backbone planning simulation for Austria and EU countries.

This script implements the country-scale Monte Carlo simulation framework used for the
article's terrestrial QKD backbone sizing and topology analysis. The purpose of the
model is not to design a deployment-ready fiber route with engineering-grade civil
infrastructure detail, but to produce a controlled, reproducible, and geographically
constrained planning model for how many QKD endpoints, backbone edges, and trusted
repeater nodes would be required when terrestrial national segments are built under a
common set of assumptions.

Scientific context
------------------
The associated article studies terrestrial quantum key distribution (QKD) backbone
networks as a practical planning layer for European secure communication
infrastructure. In that article, Austria serves as a detailed reference case and the
same modeling logic is then extended to other European countries through per-country
configuration data. The script therefore has two roles:

1. Austria reference model:
   Austria is handled with an explicit special configuration containing Vienna,
   provincial capitals, and St. Johann im Pongau as named clustering anchors.
   This reflects the article's role of Austria as a calibrated baseline case.

2. Generalized EU model:
   Other countries are loaded from the external module
   ``EU_List_islands_alltogetehr`` (imported as ``EU``), which provides
   ``CountryConfig`` objects describing country-specific endpoint counts,
   center locations, cluster counts, repeater budgets, and detour factors.

What the model represents
-------------------------
The model builds a synthetic terrestrial QKD backbone in several stages:

- A country polygon is loaded from Natural Earth.
- QKD endpoints are sampled inside that geometry.
  These endpoints represent abstract network sites or access nodes in the article's
  planning model, not literal existing telecom facilities.
- A target degree distribution is assigned to endpoints.
  This is a strict graph constraint: the final graph must realize exactly these node
  degrees. The script never "fixes" a graph by appending extra edges afterward.
- Candidate edges are constructed using distance-aware constrained matching.
- Only valid graphs are retained, subject to optional connectivity and bridge rules.
- A robustness score is computed from edge-length statistics.
  The script keeps the best valid candidate per Monte Carlo run.
- Trusted repeater nodes (TRNs) are then placed on the longest links using a greedy
  allocation policy, because the article evaluates how a fixed TRN budget reduces the
  maximum and mean hop length of the terrestrial backbone.

Important modeling semantics
----------------------------
This script uses a deliberately simplified planning abstraction:
- Geographic distances are geodesic distances between sampled points.
- Edges are abstract terrestrial backbone links between sampled QKD endpoints.
- TRNs are not independently optimized as graph vertices during graph generation.
  Instead, they are post-allocated along already selected long edges in order to
  evaluate link splitting and resulting hop lengths.
- The detour factor is applied only in statistical reporting, consistent with the
  article's distinction between straight-line geometric backbone length and adjusted
  route-realistic length.
- The graph generation objective is biased toward shorter, more homogeneous edge
  lengths, because overly long outlier links are undesirable in the article's
  terrestrial QKD interpretation.

Why there is special geometry handling
--------------------------------------
The article includes countries with islands and fragmented geometries. Uniform sampling
inside raw country bounding boxes can become inefficient or unstable for complex
MultiPolygon geometries. This script therefore includes:

- area-weighted polygon component sampling,
- bounded attempts to prevent hangs,
- an EU-focused clipping/stabilization routine,
- and mainland-only overrides for selected countries where the article intentionally
  focuses on the mainland component.

Outputs
-------
For each country the script writes:

- a per-country report text file,
- CSV files for edges and nodes,
- a plot of the selected backbone realization,
- saved configuration JSON files,
- and per-run Monte Carlo artifacts.

In article terms, these outputs provide the reproducible numerical and visual basis
for comparing countries on backbone fiber demand, edge statistics, and post-TRN hop
lengths.

Implementation note
-------------------
The user requested that the executable logic remain exactly unchanged. Accordingly,
this documented version only adds explanatory comments and this module docstring.
No algorithmic behavior, parameter values, control flow, or output semantics have been
modified.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
from pyproj import CRS, Geod, Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.prepared import prep as shapely_prep
from shapely.ops import unary_union, transform as shapely_transform

try:
    # Optional acceleration for nearest-neighbor search during edge construction.
    # If SciPy is missing, the script falls back to a dense NumPy distance matrix.
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

# External per-country configuration source.
# The article's cross-country scaling logic is encoded in this module through
# CountryConfig objects, including country-specific endpoint counts, centers,
# TRN budgets, and reporting factors.
# EU_List.py must be in the same folder or on PYTHONPATH.
import EU_List_islands_alltogetehr as EU


# ============================================================
# Config (Austria overrides must come from THIS config)
# ============================================================

@dataclass(frozen=True)
class Config:
    # Global simulation parameters.
    # Austria is the reference case in the article, so the defaults below are the
    # Austria baseline unless they are explicitly overridden per country later.

    # Reproducibility
    # Base random seed for Monte Carlo runs.
    seed: int = 46

    # QKD endpoints
    # Total number of synthetic endpoint locations for the Austria baseline.
    # In the article these act as abstract terrestrial QKD sites or access nodes.
    n_qkd_endpoints_total: int = 250
    # Number of Austria endpoints clustered around Vienna.
    n_vienna_cluster: int = 25
    # Number of Austria endpoints clustered around each other provincial capital.
    n_per_other_capital: int = 5

    # Trusted Repeater Nodes
    # Placed along the longest edges for hop-length analysis.
    #
    # Policy:
    # - Distribute the fixed TRN budget across the longest edges such that the
    #   maximum resulting hop length is reduced as much as possible.
    # - Each additional TRN on an edge increases its number of segments by 1.
    # - The allocation is greedy: repeatedly place the next TRN on the edge
    #   that currently has the largest resulting segment length L / (m + 1),
    #   where m is the number of TRNs already assigned to that edge.
    # - TRNs on an edge are then placed evenly at fractions j/(m+1), j=1..m.
    #
    # In the article this is a planning-level evaluation step, not a joint graph
    # optimization where TRNs participate in graph construction.
    n_trusted_repeater_nodes: int = 50
    trn_double_fraction: float = 0.20

    # Cluster radii
    # Geographic radius in km for Austria endpoint sampling around capitals.
    vienna_radius_km: float = 30.0
    other_capital_radius_km: float = 15.0

    # Special cluster
    # Additional small Austria cluster retained exactly as in the original code.
    st_johann_name: str = "St. Johann im Pongau"
    n_st_johann: int = 2
    st_johann_radius_km: float = 2.5

    # Rural sampling mixture
    # Rural endpoints are drawn from a mixture of:
    #   (i) uniform country-wide sampling,
    #   (ii) heavy-tailed draws around selected centers.
    # This reflects the article's desire to avoid an unrealistically urban-only model.
    p_uniform_rural: float = 0.65
    heavy_tail_scale_km: float = 25.0
    heavy_tail_alpha: float = 8.0

    # Degree targets
    # The article treats low-degree backbones as a key structural constraint.
    # With the default Austria baseline, 100 nodes must have degree 2 and 150 must
    # have degree 3. This is exact, not approximate.
    use_manual_k_distribution: bool = True
    n_nodes_k2: int = 100
    n_nodes_k3: int = 150
    n_nodes_k4: int = 0
    n_nodes_k5: int = 0
    k_nearest: int = 3  # used only if use_manual_k_distribution=False

    # Constraints (never add edges; reject & rebuild)
    # Connectivity is usually required for a viable national backbone.
    # The no-bridges option is available but disabled by default.
    enforce_connectivity: bool = True
    enforce_no_bridges: bool = False

    # Monte Carlo
    # Number of Monte Carlo runs retained for country-level summary statistics.
    n_monte_carlo_runs: int = 2
    # Article-level adjustment factor used when reporting route-realistic length
    # rather than straight geodesic length.
    detour_factor: float = 1.5

    # Robustness selection: generate many valid graphs and keep best
    # Per Monte Carlo run, several valid candidates are generated and scored.
    n_candidate_graphs_per_run: int = 5

    # Budgets
    # Search budgets controlling retries for graph building and endpoint sampling.
    max_edge_build_attempts: int = 1000
    max_resample_endpoint_attempts: int = 3
    candidate_neighbors_per_node: int = 20

    # Outputs (overridden per-country)
    report_txt: str = "austria_qkd_simulation_report.txt"
    edges_csv: str = "austria_qkd_edges_km.csv"
    nodes_csv: str = "austria_qkd_nodes_lonlat.csv"
    plot_png: str = "austria_qkd_simulation_plot.png"

    # Natural Earth
    # Low-resolution country polygons are used for simulation geometry loading.
    ne_url: str = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    ne_shp_name: str = "ne_110m_admin_0_countries.shp"
    cache_zip: str = "ne_110m_admin_0_countries.zip"

    # Logging
    verbose: bool = True
    print_every_edge_attempt: int = 50  # print progress each N edge attempts
    print_every_candidate: int = 200  # print progress each N valid candidates kept


# ============================================================
# Plot settings (kept)
# ============================================================

# Global visualization styling for exported country figures.
# This affects readability only and does not change any simulation result.
FONT_SCALE = 1.1
FIGSIZE = (14, 12)
FIG_DPI = 220
LABEL_ALL_QKD_ENDPOINTS = False

_BASE_FONT = 10.0
plt.rcParams.update(
    {
        "font.size": _BASE_FONT * FONT_SCALE,
        "axes.titlesize": 1.2 * _BASE_FONT * FONT_SCALE,
        "axes.labelsize": 1.0 * _BASE_FONT * FONT_SCALE,
        "xtick.labelsize": 0.9 * _BASE_FONT * FONT_SCALE,
        "ytick.labelsize": 0.9 * _BASE_FONT * FONT_SCALE,
        "legend.fontsize": 0.9 * _BASE_FONT * FONT_SCALE,
    }
)

# Austria-only centers (kept)
# These are the reference clustering anchors for the Austria case discussed in the
# article. Other countries obtain their centers from EU.CountryConfig.
CAPITALS: Dict[str, Tuple[float, float]] = {
    "Vienna": (16.363449, 48.210033),
    "St. Pölten": (15.633333, 48.200000),
    "Linz": (14.29, 48.31),
    "Salzburg": (13.04, 47.80),
    "Innsbruck": (11.39, 47.26),
    "Bregenz": (9.746, 47.503),
    "Eisenstadt": (16.523, 47.846),
    "Graz": (15.45, 47.07),
    "Klagenfurt": (14.31, 46.62),
    "St. Johann im Pongau": (13.2000, 47.3500),
}


# ============================================================
# Small logging helper (kept)
# ============================================================

def log(cfg: Config, msg: str) -> None:
    # Central verbosity gate. All progress output flows through here.
    if cfg.verbose:
        print(msg, flush=True)


# ============================================================
# Geo helpers (kept)
# ============================================================

# WGS84 geodesic engine used for edge-length and point-on-geodesic calculations.
GEOD = Geod(ellps="WGS84")


def geodesic_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # Great-circle-like geodesic distance on the WGS84 ellipsoid, in km.
    # In article terms this is the base geometric link length before any detour factor.
    lon1, lat1 = a
    lon2, lat2 = b
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist_m) / 1000.0


def geodesic_midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    # Midpoint along the geodesic. Kept as in original code for geodesic utilities.
    lon1, lat1 = a
    lon2, lat2 = b
    az12, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    lonm, latm, _ = GEOD.fwd(lon1, lat1, az12, dist_m / 2.0)
    return float(lonm), float(latm)


def geodesic_point_fraction(a: Tuple[float, float], b: Tuple[float, float], frac: float) -> Tuple[float, float]:
    """
    Point along geodesic from a->b at fraction frac in [0,1].

    This is used when TRNs are placed evenly along a selected long edge. In the
    article's model, that means a long terrestrial link is subdivided into equal
    geodesic segments for hop-length analysis.
    """
    frac = float(max(0.0, min(1.0, frac)))
    lon1, lat1 = a
    lon2, lat2 = b
    az12, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    lonp, latp, _ = GEOD.fwd(lon1, lat1, az12, dist_m * frac)
    return float(lonp), float(latp)


def km_to_deg_lat(km: float) -> float:
    # Small-distance approximation for converting km to latitude degrees.
    return km / 111.32

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    # Small-distance approximation for converting km to longitude degrees,
    # adjusted by latitude-dependent meridian convergence.
    return km / (111.32 * math.cos(math.radians(lat_deg)) + 1e-12)


def random_point_in_radius(center: Tuple[float, float], radius_km: float) -> Tuple[float, float]:
    # Samples a point uniformly over disk area around a center in a local degree-space
    # approximation. This supports clustered endpoint generation around capitals or
    # configured country centers.
    lon0, lat0 = center
    u = random.random()
    r_km = radius_km * math.sqrt(u)
    theta = 2.0 * math.pi * random.random()
    dlat = km_to_deg_lat(r_km) * math.sin(theta)
    dlon = km_to_deg_lon(r_km, lat0) * math.cos(theta)
    return (lon0 + dlon, lat0 + dlat)


def heavy_tailed_radius_km(scale_km: float, alpha: float) -> float:
    # Heavy-tailed radial draw used for rural endpoint placement.
    # This allows occasional long excursions from a center, preventing overly compact
    # synthetic settlement patterns and giving the article's rural component a broader
    # geographic reach.
    u = max(1e-12, random.random())
    return scale_km * (u ** (-1.0 / alpha) - 1.0)


# ============================================================
# Polygon / MultiPolygon sampling (FIXED FOR ISLANDS)
#   - area-weighted component choice
#   - bounded attempts (never hangs)
# ============================================================

# European Lambert Azimuthal Equal Area projection.
# Used here only for geometry area comparison and distance-aware planar operations.
_TRANSFORMER_3035 = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3035), always_xy=True)


def _extract_polygons(geom: Polygon | MultiPolygon) -> List[Polygon]:
    # Normalizes single Polygon and MultiPolygon geometries to a list of polygons.
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in geom.geoms if isinstance(g, Polygon)]


def _project_geom_3035(g):
    # Reprojects geometry from lon/lat into EPSG:3035 for area-based decisions.
    def f(x, y, z=None):
        return _TRANSFORMER_3035.transform(x, y)
    return shapely_transform(f, g)

def build_component_samplers(
    geom: Polygon | MultiPolygon,
) -> Tuple[List[Polygon], List, List[float]]:
    # Precomputes component polygons, prepared geometries, and area weights.
    # This is important for countries with islands: instead of sampling uniformly in
    # the bounding box of the full MultiPolygon, the code first chooses a component
    # using approximate area weights, then samples within that component's bounds.
    polys = _extract_polygons(geom)
    preps = [shapely_prep(p) for p in polys]

    weights: List[float] = []
    for p in polys:
        try:
            a = float(_project_geom_3035(p).area)
        except Exception:
            a = float(p.area)
        weights.append(max(0.0, a))

    s = sum(weights)
    if s <= 0.0:
        weights = [1.0 for _ in polys]

    return polys, preps, weights


def sample_point_in_geometry_uniform(
    polys: List[Polygon],
    preps: List,
    weights: List[float],
    max_component_picks: int = 250,
    max_tries_per_component: int = 4000,
) -> Tuple[float, float]:
    # Uniform-like sampling over possibly fragmented country geometry.
    # Bounded retries prevent pathological infinite loops for narrow islands or
    # awkward bounding boxes.
    if not polys:
        raise RuntimeError("No polygon components available for sampling.")

    for _ in range(int(max_component_picks)):
        idx = random.choices(range(len(polys)), weights=weights, k=1)[0]
        poly = polys[idx]
        prep_poly = preps[idx]
        minx, miny, maxx, maxy = poly.bounds

        for _t in range(int(max_tries_per_component)):
            lon = random.uniform(minx, maxx)
            lat = random.uniform(miny, maxy)
            if prep_poly.contains(Point(lon, lat)):
                return (float(lon), float(lat))

    raise RuntimeError("Geometry sampling exceeded bounded attempts (component acceptance too low).")


def sample_point_in_polygon_uniform(
    prep_poly,
    bounds: Tuple[float, float, float, float],
    max_tries: int = 20000,
) -> Tuple[float, float]:
    # Uniform rejection sampling inside a single polygon bounding box.
    minx, miny, maxx, maxy = bounds
    for _ in range(int(max_tries)):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        if prep_poly.contains(Point(lon, lat)):
            return (lon, lat)
    raise RuntimeError("Uniform polygon sampling exceeded max_tries (geometry too sparse vs bounds).")


# ============================================================
# Geometry stabilization
#   - Default: keep EU-focused geometry incl. nearby islands
#   - Mainland-only override for selected countries (Spain, Portugal, Italy)
# ============================================================

# Countries for which the article's modeling choice is effectively mainland-focused.
MAINLAND_ONLY_ADMINS = {"Spain", "Portugal", "Italy", "Greece", "Croatia", "France"}


def _largest_polygon_component(geom: Polygon | MultiPolygon) -> Polygon:
    # Returns the largest component by projected area.
    # Used for mainland-only countries to prevent remote islands from dominating
    # endpoint sampling or plotting extent.
    polys = _extract_polygons(geom)
    if not polys:
        raise RuntimeError("No polygon components available.")
    return max(polys, key=lambda p: float(_project_geom_3035(p).area) if not p.is_empty else -1.0)


def filter_centers_to_geometry(
    centers_lonlat: Dict[str, Tuple[float, float]],
    geom: Polygon | MultiPolygon,
) -> Dict[str, Tuple[float, float]]:
    # Removes configured centers that no longer lie inside the stabilized geometry.
    # This matters when a country has been reduced to the mainland component.
    if not centers_lonlat:
        return {}

    prep_geom = shapely_prep(geom)
    out: Dict[str, Tuple[float, float]] = {}
    for name, (lon, lat) in centers_lonlat.items():
        pt = Point(float(lon), float(lat))
        try:
            if prep_geom.contains(pt) or prep_geom.intersects(pt):
                out[name] = (float(lon), float(lat))
        except Exception:
            continue
    return out


def stabilize_eu_geometry(
    geom: Polygon | MultiPolygon,
    centers_lonlat: Dict[str, Tuple[float, float]],
    admin_name: str,
) -> Polygon | MultiPolygon:
    # Stabilizes country geometry for EU-wide simulation.
    #
    # For mainland-only countries, only the largest polygon is kept.
    # For others, the code clips to a broad Europe-centered window while also trying
    # to preserve polygon components associated with configured national centers.
    # This keeps the article's focus on European national deployment geography while
    # still allowing relevant nearby islands where intended.
    if admin_name in MAINLAND_ONLY_ADMINS:
        return _largest_polygon_component(geom)

    # Expanded "European focus" window:
    #  - south to 24°N to keep nearby islands (e.g., Canaries are ~27–29°N)
    #  - west to -35° to safely include Azores (-31..-25), Madeira (~-17), etc.
    #  - east to 60° for full eastern EU neighborhood
    eu_focus = box(-35.0, 24.0, 60.0, 73.5)

    try:
        clipped = geom.intersection(eu_focus)
    except Exception:
        clipped = geom

    keep_geoms: List = []
    if not clipped.is_empty:
        keep_geoms.append(clipped)

    if centers_lonlat:
        polys = _extract_polygons(geom)
        for _, (lon, lat) in centers_lonlat.items():
            cpt = Point(float(lon), float(lat))
            kept_any = False
            for p in polys:
                try:
                    if p.contains(cpt) or p.intersects(cpt):
                        keep_geoms.append(p)
                        kept_any = True
                except Exception:
                    continue
            if not kept_any:
                try:
                    dmin = float("inf")
                    nearest = None
                    for p in polys:
                        rp = p.representative_point()
                        d = geodesic_km((float(rp.x), float(rp.y)), (float(lon), float(lat)))
                        if d < dmin:
                            dmin = d
                            nearest = p
                    if nearest is not None:
                        keep_geoms.append(nearest)
                except Exception:
                    pass

    if not keep_geoms:
        return geom

    out = unary_union(keep_geoms)
    if isinstance(out, (Polygon, MultiPolygon)):
        return out
    return geom


# ============================================================
# Natural Earth loading (generalized)
# ============================================================

def _download_bytes(url: str, timeout_s: int = 60) -> bytes:
    # Small helper with requests-first and urllib fallback behavior.
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.content
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.read()


def load_country_polygon(cfg: Config, admin_name: str) -> Polygon | MultiPolygon:
    # Loads a single country's geometry from Natural Earth 110m.
    # This is the baseline geometry source for simulation if higher-resolution 10m
    # geometry is unavailable.
    cache_path = Path(cfg.cache_zip)
    if not cache_path.exists():
        log(cfg, f"[data] Downloading Natural Earth zip -> {cache_path}")
        cache_path.write_bytes(_download_bytes(cfg.ne_url, timeout_s=60))
    else:
        log(cfg, f"[data] Using cached Natural Earth zip -> {cache_path}")

    world = gpd.read_file(f"zip://{str(cache_path)}!{cfg.ne_shp_name}")

    name_col = next((c for c in ("ADMIN", "NAME", "name") if c in world.columns), None)
    if name_col is None:
        raise RuntimeError(f"Cannot find a country name column. Columns: {list(world.columns)}")

    row = world[world[name_col] == admin_name].copy()
    if row.empty:
        raise RuntimeError(f"Country '{admin_name}' not found in Natural Earth dataset.")

    try:
        row = row.to_crs("EPSG:4326")
    except Exception:
        pass

    geom = row.geometry.iloc[0]
    if geom is None or not isinstance(geom, (Polygon, MultiPolygon)):
        raise RuntimeError(f"Unexpected geometry for '{admin_name}': {type(geom)}")

    log(cfg, f"[data] {admin_name} polygon loaded.")
    return geom



# ============================================================
# HIGH-RES MAP DATA (plot-only; improves map presentation)
# ============================================================

# Higher-resolution Natural Earth data used only for presentation-quality plots.
_NE10M_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
_NE10M_SHP = "ne_10m_admin_0_countries.shp"
_NE10M_CACHE = "ne_10m_admin_0_countries.zip"

_world10m_cache: Optional[gpd.GeoDataFrame] = None


def load_world_countries_10m(cfg: Config) -> gpd.GeoDataFrame:
    # Loads and caches the 10m world countries dataset.
    global _world10m_cache
    if _world10m_cache is not None:
        return _world10m_cache

    cache_path = Path(_NE10M_CACHE)
    if not cache_path.exists():
        log(cfg, f"[data] Downloading Natural Earth 10m (plot) zip -> {cache_path}")
        cache_path.write_bytes(_download_bytes(_NE10M_URL, timeout_s=60))
    else:
        log(cfg, f"[data] Using cached Natural Earth 10m (plot) zip -> {cache_path}")

    world = gpd.read_file(f"zip://{str(cache_path)}!{_NE10M_SHP}")
    try:
        world = world.to_crs("EPSG:4326")
    except Exception:
        pass

    _world10m_cache = world
    return world


def _country_row_from_world(world: gpd.GeoDataFrame, admin_name: str) -> gpd.GeoDataFrame:
    # Finds the row corresponding to a specific country name in a world GeoDataFrame.
    name_col = next((c for c in ("ADMIN", "NAME", "name") if c in world.columns), None)
    if name_col is None:
        return world.iloc[0:0].copy()
    return world[world[name_col] == admin_name].copy()


def load_country_polygon_10m(cfg: Config, admin_name: str) -> Optional[Polygon | MultiPolygon]:
    """
    Loads the country geometry from Natural Earth 10m (same source used for plotting).
    Returns None if the country cannot be found.

    The script prefers this geometry for plotting and, where available, also uses it for
    simulation so that the simulated and displayed footprint are aligned.
    """
    world10m = load_world_countries_10m(cfg)
    row = _country_row_from_world(world10m, admin_name)
    if row.empty:
        return None
    g = row.geometry.iloc[0]
    if g is None or (not isinstance(g, (Polygon, MultiPolygon))) or g.is_empty:
        return None
    return g


# ============================================================
# EU_List loading helpers
# ============================================================

def load_all_country_configs_from_module() -> Dict[str, EU.CountryConfig]:
    # Collects all CountryConfig objects from the imported EU module.
    # The code supports both prebuilt dictionaries and builder functions.
    configs: Dict[str, EU.CountryConfig] = {}

    for name in dir(EU):
        if not name.startswith("COUNTRY_CONFIGS"):
            continue
        obj = getattr(EU, name, None)
        if isinstance(obj, dict) and obj:
            for k, v in obj.items():
                if isinstance(v, EU.CountryConfig):
                    configs[k] = v

    for name in dir(EU):
        if not name.startswith("build_country_configs"):
            continue
        fn = getattr(EU, name, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            except Exception:
                continue
            if isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, EU.CountryConfig):
                        configs[k] = v

    if not configs:
        raise RuntimeError("No CountryConfig objects found in EU_List.py (nothing to run).")

    return dict(sorted(configs.items(), key=lambda kv: kv[0]))


def safe_folder_name(name: str) -> str:
    # Sanitizes country names for filesystem-safe output directories.
    s = name.strip()
    s = re.sub(r"[^\w\s\-.()]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:120] if s else "Country"



# ============================================================
# Persistence
# ============================================================

def _write_json(path: Path, obj: Dict) -> None:
    # Writes JSON with UTF-8 and readable indentation.
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_jsonable(obj):
    # Recursive converter for dataclasses / objects into JSON-friendly structures.
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {str(k): _to_jsonable(v) for k, v in vars(obj).items()}
    return str(obj)


def save_country_configuration(cfg: Config, country_name: str, country_cfg: EU.CountryConfig, country_dir: Path) -> None:
    # Saves the effective simulation and country configuration for reproducibility.
    # This is useful for the article because each country's reported results can be
    # traced back to a concrete parameterization.
    cfg_dir = country_dir / "configuration"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    _write_json(cfg_dir / "simulation_config.json", _to_jsonable(cfg))
    _write_json(cfg_dir / "country_config.json", _to_jsonable(country_cfg))
    _write_json(
        cfg_dir / "config_meta.json",
        {
            "country_name": str(country_name),
            "natural_earth_admin": str(getattr(country_cfg, "natural_earth_admin", "")),
            "saved_at_unix_s": float(time.time()),
        },
    )

def save_monte_carlo_run_artifacts(
    country_dir: Path,
    run_idx: int,
    country_name: str,
    natural_earth_admin: str,
    cfg: Config,
    run_result: Dict,
) -> None:
    # Saves per-run outputs, not just the country-level first run.
    # This preserves Monte Carlo detail that may matter for robustness analysis.
    mc_root = country_dir / "monte_carlo_runs"
    run_dir = mc_root / f"run_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    edges_df: pd.DataFrame = run_result["edges_df"]
    nodes_df = pd.DataFrame(run_result["endpoint_meta"] + run_result["trusted_repeater_meta"])

    edges_df.to_csv(run_dir / "qkd_edges_km.csv", index=False)
    nodes_df.to_csv(run_dir / "qkd_nodes_lonlat.csv", index=False)

    meta = {
        "country_name": str(country_name),
        "natural_earth_admin": str(natural_earth_admin),
        "run_idx": int(run_idx),
        "seed": int(run_result.get("seed", -1)),
        "fiber_km": float(run_result.get("fiber_km", float("nan"))),
        "robust_score": float(run_result.get("robust_score", float("nan"))),
        "robust_score_stats": dict(run_result.get("robust_score_stats", {})),
        "hop_stats": dict(run_result.get("hop_stats", {})),
        "deg_hist": dict(run_result.get("deg_hist", {})),
        "candidates_kept": int(run_result.get("candidates_kept", 0)),
        "resample_idx": int(run_result.get("resample_idx", -1)),
        "attempt_idx": int(run_result.get("attempt_idx", -1)),
        "n_endpoints": int(cfg.n_qkd_endpoints_total),
        "n_trusted_repeater_nodes": int(cfg.n_trusted_repeater_nodes),
        "detour_factor": float(cfg.detour_factor),
        "degree_dist_text": f"k2={cfg.n_nodes_k2}, k3={cfg.n_nodes_k3}, k4={cfg.n_nodes_k4}, k5={cfg.n_nodes_k5}"
        if cfg.use_manual_k_distribution
        else f"fixed k={cfg.k_nearest}",
    }
    _write_json(run_dir / "run_meta.json", meta)

# ============================================================
# Endpoint generation safety utilities (kept)
# ============================================================

def _fallback_fill_points(
    prep_poly,
    bounds: Tuple[float, float, float, float],
    n_needed: int,
) -> List[Tuple[float, float]]:
    # Last-resort point filler for difficult geometries.
    # This avoids total failure when country geometry is fragmented or narrow.
    if n_needed <= 0:
        return []

    minx, miny, maxx, maxy = bounds
    out: List[Tuple[float, float]] = []

    side = int(math.ceil(math.sqrt(n_needed * 40)))
    if side < 10:
        side = 10
    dx = (maxx - minx) / side if side > 0 else 0.1
    dy = (maxy - miny) / side if side > 0 else 0.1
    for iy in range(side):
        if len(out) >= n_needed:
            break
        y0 = miny + (iy + 0.5) * dy
        for ix in range(side):
            if len(out) >= n_needed:
                break
            x0 = minx + (ix + 0.5) * dx
            x = x0 + random.uniform(-0.35, 0.35) * dx
            y = y0 + random.uniform(-0.35, 0.35) * dy
            if prep_poly.contains(Point(x, y)):
                out.append((float(x), float(y)))

    if len(out) < n_needed:
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        step_x = max(1e-6, (maxx - minx) / 50.0)
        step_y = max(1e-6, (maxy - miny) / 50.0)
        tries = 0
        while len(out) < n_needed and tries < 500000:
            tries += 1
            x = cx + random.gauss(0.0, 1.0) * step_x * 3.0
            y = cy + random.gauss(0.0, 1.0) * step_y * 3.0
            if prep_poly.contains(Point(x, y)):
                out.append((float(x), float(y)))

    return out


# ============================================================
# QKD endpoint generation
# ============================================================

def make_qkd_endpoints_austria(
    cfg: Config,
    geom: Polygon | MultiPolygon
) -> Tuple[List[Tuple[float, float]], List[Dict], List[str]]:
    # Austria-specific endpoint generator.
    # This is the article's reference-case placement logic with explicit Vienna,
    # provincial-capital, St. Johann, and rural components.
    prep_poly = shapely_prep(geom)
    bounds = geom.bounds

    comp_polys, comp_preps, comp_weights = build_component_samplers(geom)

    vienna_name = "Vienna"
    stj_name = cfg.st_johann_name
    other_capitals = [k for k in CAPITALS.keys() if k not in (vienna_name, stj_name)]
    n_other = len(other_capitals)

    n_rural = cfg.n_qkd_endpoints_total - (cfg.n_vienna_cluster + n_other * cfg.n_per_other_capital + cfg.n_st_johann)
    if n_rural < 0:
        raise ValueError("Counts imply negative rural endpoints; adjust cluster sizes.")

    endpoints: List[Tuple[float, float]] = []
    meta: List[Dict] = []
    macro_labels: List[str] = []

    def add_endpoint(p: Tuple[float, float], cluster: str) -> None:
        # Appends an endpoint together with machine-readable metadata.
        idx = len(endpoints)
        endpoints.append(p)
        meta.append(
            {
                "node_id": f"Q{idx:03d}",
                "type": "QKD_ENDPOINT",
                "cluster": cluster,
                "lon": float(p[0]),
                "lat": float(p[1]),
            }
        )
        macro_labels.append(cluster)

    # Vienna cluster.
    attempts = 0
    while len(endpoints) < cfg.n_vienna_cluster:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Austria Vienna cluster sampling stuck (unexpected).")
        p = random_point_in_radius(CAPITALS[vienna_name], cfg.vienna_radius_km)
        if prep_poly.contains(Point(p[0], p[1])):
            add_endpoint(p, vienna_name)

    # Provincial-capital clusters.
    for cap in other_capitals:
        added = 0
        attempts = 0
        while added < cfg.n_per_other_capital:
            attempts += 1
            if attempts > 200000:
                raise RuntimeError(f"Austria cluster sampling stuck for '{cap}' (unexpected).")
            p = random_point_in_radius(CAPITALS[cap], cfg.other_capital_radius_km)
            if prep_poly.contains(Point(p[0], p[1])):
                add_endpoint(p, cap)
                added += 1

    # Small St. Johann cluster.
    for _ in range(cfg.n_st_johann):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 200000:
                raise RuntimeError("Austria St. Johann sampling stuck (unexpected).")
            p = random_point_in_radius(CAPITALS[stj_name], cfg.st_johann_radius_km)
            if prep_poly.contains(Point(p[0], p[1])):
                add_endpoint(p, stj_name)
                break
