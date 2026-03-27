#!/usr/bin/env python3
"""
Monte Carlo simulation framework for planning terrestrial national QKD backbones.

This module generates synthetic quantum key distribution (QKD) network instances
for Austria and other European countries. It combines country geometries from
Natural Earth with country-specific planning parameters from ``EU_MS_List`` to
construct endpoint sets, degree-constrained backbone graphs, and trusted
repeater node (TRN) placements. The workflow is intended for planning-level
infrastructure sizing rather than deployment design.

Main workflow
-------------
1. Load and optionally stabilize a country polygon or multipolygon.
2. Sample QKD endpoint locations using clustered urban centers and a rural
   mixture model.
3. Assign a target degree sequence to the sampled endpoints.
4. Construct an undirected simple graph that satisfies the degree sequence
   while preferring short edges.
5. Generate multiple valid candidate graphs and retain the most robust
   candidate according to edge-length based scoring.
6. Allocate trusted repeater nodes across long edges in order to reduce the
   maximum resulting hop length.
7. Export node tables, edge tables, plots, reports, and per-run metadata.

Model scope
-----------
The implementation focuses on terrestrial QKD planning. It does not model
traffic demand, application-layer behavior, physical fibre availability,
sector-specific operational requirements, or deployment-specific routing.
Instead, it provides reproducible first-order estimates for endpoint counts,
fibre lengths, hop statistics, and related component counts under explicit and
traceable assumptions.

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
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

# Country-specific planning inputs must be in the current working directory.
import EU_MS_List as EU


# ============================================================
# Global simulation configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    # Reproducibility
    """
    Configuration container for the full QKD network simulation workflow.

    This dataclass collects the global default parameters that control the
    generation, construction, evaluation, and export of synthetic terrestrial
    QKD backbone networks. The values defined here act as the base simulation
    settings. For Austria, these defaults are used directly except where the
    country-specific execution logic applies Austria-specific handling. For
    other countries, selected values are replaced at runtime using the
    corresponding entries from ``EU.CountryConfig`` loaded from ``EU_MS_List``.

    The configuration covers the following groups of parameters:

    1. Reproducibility:
       Controls the base random seed used for Python's random module and
       NumPy-based random sampling. This ensures that endpoint placement,
       degree assignment shuffling, and graph-construction attempts can be
       reproduced.

    2. Endpoint generation:
       Defines how many QKD endpoint sites are generated in total and how
       endpoint clusters are distributed for the Austrian reference case.
       This includes dense placement around Vienna, smaller clusters in other
       state capitals, a special St. Johann im Pongau cluster, and a rural
       sampling component.

    3. Trusted repeater placement:
       Defines the total number of trusted repeater nodes (TRNs) used to split
       long edges into shorter QKD-feasible spans. The implemented allocation
       strategy distributes TRNs across long edges in a way that reduces the
       maximum resulting hop length.

    4. Spatial sampling behavior:
       Controls rural placement through a mixture of uniform geographic
       sampling and heavy-tailed radial sampling around predefined centers.
       This allows the model to combine broad territorial coverage with
       non-uniform concentration around important urban or strategic sites.

    5. Degree constraints:
       Defines the target degree distribution of endpoint nodes in the
       synthetic backbone graph. In the current model, endpoints are assigned
       fixed target degrees, typically degree 2 or 3, in order to represent
       moderate meshing and basic redundancy.

    6. Graph validity constraints:
       Controls whether only connected graphs are accepted and whether bridge-
       free topologies are required. The current default enforces network
       connectivity but does not reject bridge edges.

    7. Monte Carlo execution:
       Defines the number of repeated simulation runs and the detour factor
       used to translate geodesic distances into approximate route lengths.
       The detour factor is meant to represent the difference between straight-
       line geographic distance and actual fiber routing distance.

    8. Robustness-oriented candidate selection:
       Controls how many valid graph realizations are generated per Monte
       Carlo run before the best one is selected according to an edge-length-
       based robustness score.

    9. Computational budgets:
       Defines limits for graph-building attempts, endpoint resampling rounds,
       and nearest-neighbor candidate sets. These values bound runtime and
       prevent unbounded retries when constraints cannot be satisfied easily.

    10. Output paths:
        Provides default filenames for reports, node tables, edge tables, and
        plots. These are later replaced with country-specific output paths.

    11. Map data access:
        Defines the Natural Earth source archive and cached local filenames
        used to load national boundary geometries.

    12. Logging:
        Controls whether progress information is printed and how often progress
        messages are emitted during repeated graph-generation attempts.

    Notes
    -----
    The dataclass is frozen, which means instances are immutable after
    creation. This is useful because country-specific modifications are then
    made explicitly through ``dataclasses.replace(...)``, which makes parameter
    changes transparent and traceable.

    """
    seed: int = 69 #81 good #92 perfect
    # Base random seed for reproducibility.
    # This seed initializes both Python's ``random`` module and NumPy's random
    # generator during each simulation run. Different Monte Carlo runs derive
    # new seeds from this base value by adding the run index.
    # The inline note documents empirically preferred seeds observed during
    # earlier test runs.

    # QKD endpoints (formerly "main nodes")
    n_qkd_endpoints_total: int = 250
    # Total number of endpoint sites in the Austrian reference case.
    # Each endpoint represents a site that requires availability of key
    # material, for example a ministry, authority, critical infrastructure
    # operator, data center, or other security-relevant location.

    n_vienna_cluster: int = 25
    # Number of endpoints placed in the Vienna cluster.
    # Vienna is modeled as the dominant national concentration of relevant
    # institutions and therefore receives a larger endpoint allocation than
    # other centers.

    n_per_other_capital: int = 5
    # Number of endpoints placed around each other Austrian state capital,
    # excluding Vienna and the special St. Johann im Pongau case.
    # This creates smaller but systematic regional clusters.

    # Trusted Repeater Nodes (formerly "connection nodes")
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
    n_trusted_repeater_nodes: int = 50
    # Total budget of trusted repeater nodes available for one simulated
    # national backbone realization.
    # TRNs are not endpoint demand sites. They are auxiliary intermediate nodes
    # introduced to split long QKD links into shorter spans that are more
    # compatible with terrestrial QKD distance limits.

    trn_double_fraction: float = 0.20
    # Fraction associated with double-TRN edge handling in earlier or related
    # configuration logic.
    # In the present implementation shown elsewhere in the code, TRN placement
    # is driven by the full greedy allocation routine over edge lengths. This
    # field remains part of the configuration structure for consistency with
    # prior model variants and country configuration metadata.

    # Cluster radii
    vienna_radius_km: float = 30.0
    # Sampling radius in kilometers for the Vienna endpoint cluster.
    # Random points are generated within this radius around the Vienna center
    # and accepted only if they fall inside the national geometry.

    other_capital_radius_km: float = 15.0
    # Sampling radius in kilometers for the smaller regional capital clusters.
    # This radius is smaller than the Vienna radius to reflect more compact
    # regional concentrations.

    # Special cluster
    st_johann_name: str = "St. Johann im Pongau"
    # Name of the special location treated separately from the general state
    # capital logic. In the model context, this site is included because of
    # the stated relevance of a backup government data-center location.

    n_st_johann: int = 2
    # Number of endpoints assigned to the St. Johann im Pongau cluster.

    st_johann_radius_km: float = 2.5
    # Sampling radius in kilometers for the St. Johann cluster.
    # This is much smaller than the other cluster radii because the model
    # treats this site as a tightly localized special facility.

    # Rural sampling mixture
    p_uniform_rural: float = 0.65
    # Probability that a rural endpoint is sampled uniformly from the country
    # geometry rather than from the heavy-tailed center-based sampling mode.
    # A larger value increases broad territorial coverage.

    heavy_tail_scale_km: float = 25.0
    # Scale parameter for the heavy-tailed radial sampling distribution used in
    # the non-uniform rural placement mode.
    # Larger values allow sampled points to spread farther away from centers.

    heavy_tail_alpha: float = 8.0
    # Shape parameter of the heavy-tailed radial distribution.
    # This controls how quickly the probability decreases with distance from
    # the selected center. Higher values produce a less extreme tail.

    # Degree targets
    use_manual_k_distribution: bool = True
    # If True, the endpoint graph uses an explicitly prescribed degree
    # multiset based on the counts below.
    # If False, all endpoints are assigned the same fixed degree ``k_nearest``.

    n_nodes_k2: int = 100
    # Number of endpoints that must have degree 2 in the final graph.
    # Degree 2 represents a minimal form of redundancy with two incident links.

    n_nodes_k3: int = 150
    # Number of endpoints that must have degree 3 in the final graph.
    # Degree 3 increases connectivity and provides more routing flexibility than
    # degree 2 while still remaining moderate in infrastructure demand.

    n_nodes_k4: int = 0
    # Number of endpoints that must have degree 4.
    # Currently zero in the default Austrian setup.

    n_nodes_k5: int = 0
    # Number of endpoints that must have degree 5.
    # Currently zero in the default Austrian setup.

    k_nearest: int = 3  # used only if use_manual_k_distribution=False
    # Uniform fallback node degree used only when manual degree counts are not
    # applied. In that alternative mode, every endpoint receives this same
    # target degree.

    # Constraints (never add edges; reject & rebuild)
    enforce_connectivity: bool = True
    # If True, only connected graphs are accepted.
    # This ensures that the simulated network forms a single connected
    # component and does not contain isolated subgraphs.

    enforce_no_bridges: bool = False
    # If True, graphs containing bridge edges are rejected.
    # A bridge is an edge whose removal disconnects the graph.
    # The default is False, meaning bridges are allowed as long as the graph is
    # connected overall.

    # Monte Carlo
    n_monte_carlo_runs: int = 1000
    # Number of full Monte Carlo runs performed for a country in this current
    # configuration block.
    # Each run generates endpoints, constructs candidate graphs, selects the
    # best candidate, and records resulting network statistics.

    detour_factor: float = 1.5
    # Multiplicative correction factor used when interpreting geodesic
    # distances as approximate real routing distances.
    # This models the fact that actual fiber routes do not follow straight-line
    # shortest paths in geographic space.

    # Robustness selection: generate many valid graphs and keep best
    n_candidate_graphs_per_run: int = 5
    # Number of valid graph realizations retained or considered per Monte Carlo
    # run before selecting the best one according to the robustness score.
    # The score penalizes long edges, high edge-length variability, and length
    # outliers.

    # Budgets
    max_edge_build_attempts: int = 1000
    # Maximum number of edge-construction attempts allowed per endpoint sample
    # configuration before giving up and moving on or resampling.
    # This limits runtime for difficult degree-constrained graph generation.

    max_resample_endpoint_attempts: int = 3
    # Maximum number of times endpoint locations may be regenerated for a run
    # if no acceptable graph can be built under the imposed constraints.

    candidate_neighbors_per_node: int = 20
    # Number of nearest-neighbor candidates considered per node during the
    # degree-constrained graph construction process.
    # This parameter controls the local search space used when trying to form
    # short edges while satisfying exact degree targets.

    # Outputs (overridden per-country)
    report_txt: str = "austria_qkd_simulation_report.txt"
    # Default path for the plain-text summary report.

    edges_csv: str = "austria_qkd_edges_km.csv"
    # Default path for the CSV file containing edge-level information such as
    # endpoint identifiers, endpoint coordinates, and geodesic edge lengths.

    nodes_csv: str = "austria_qkd_nodes_lonlat.csv"
    # Default path for the CSV file containing node-level information for both
    # QKD endpoints and trusted repeater nodes.

    plot_png: str = "austria_qkd_simulation_plot.png"
    # Default path for the exported network plot.
    # An EPS file with the same base name is also written elsewhere in the code.

    # Natural Earth
    ne_url: str = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    # Download URL for the Natural Earth 110 m country boundary dataset used as
    # the main geographic input source.

    ne_shp_name: str = "ne_110m_admin_0_countries.shp"
    # Shapefile name inside the Natural Earth ZIP archive.

    cache_zip: str = "ne_110m_admin_0_countries.zip"
    # Local cache filename for the downloaded Natural Earth ZIP archive.
    # Reusing the cached archive avoids repeated downloads.

    # Logging
    verbose: bool = True
    # Enables progress output during endpoint generation, graph construction,
    # candidate selection, and file export.

    print_every_edge_attempt: int = 50   # print progress each N edge attempts
    # Frequency of progress messages during repeated edge-construction attempts.

    print_every_candidate: int = 200     # print progress each N valid candidates kept
    # Frequency of progress messages during accumulation of valid candidate
    # graphs across repeated attempts.


# ============================================================
# Plot configuration
# ============================================================

FONT_SCALE = 1.3
# Global font scaling factor applied to the Matplotlib rcParams below.
# This allows proportional adjustment of all main text sizes in the figure.

FIGSIZE = (14, 12)
# Figure size in inches for the exported network plot.

FIG_DPI = 220
# Resolution used when saving raster plot output.

LABEL_ALL_QKD_ENDPOINTS = False
# If True, every QKD endpoint is annotated with its node identifier.
# The default is False to avoid excessive visual clutter in dense networks.

PLOT_EDGE_COLOR = "black"
# Default color used for plotted graph edges.


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

# Austrian reference-case planning centres
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
# Logging helper
# ============================================================

def log(cfg: Config, msg: str) -> None:
    """
    Write a log message when verbose output is enabled in the configuration.
    """
    if cfg.verbose:
        print(msg, flush=True)


# ============================================================
# Geodesic and coordinate helpers
# ============================================================

GEOD = Geod(ellps="WGS84")


def geodesic_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Return the geodesic distance between two longitude/latitude points in kilometres using WGS84.
    """
    lon1, lat1 = a
    lon2, lat2 = b
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist_m) / 1000.0


def geodesic_midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """
    Return the midpoint on the WGS84 geodesic connecting two longitude/latitude points.
    """
    lon1, lat1 = a
    lon2, lat2 = b
    az12, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    lonm, latm, _ = GEOD.fwd(lon1, lat1, az12, dist_m / 2.0)
    return float(lonm), float(latm)


def geodesic_point_fraction(a: Tuple[float, float], b: Tuple[float, float], frac: float) -> Tuple[float, float]:
    """
    Point along geodesic from a->b at fraction frac in [0,1].
    """
    frac = float(max(0.0, min(1.0, frac)))
    lon1, lat1 = a
    lon2, lat2 = b
    az12, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    lonp, latp, _ = GEOD.fwd(lon1, lat1, az12, dist_m * frac)
    return float(lonp), float(latp)


def km_to_deg_lat(km: float) -> float:
    """
    Convert a north-south distance in kilometres to an approximate latitude offset in degrees.
    """
    return km / 111.32


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    """
    Convert an east-west distance in kilometres to an approximate longitude offset in degrees at a given latitude.
    """
    return km / (111.32 * math.cos(math.radians(lat_deg)) + 1e-12)


def random_point_in_radius(center: Tuple[float, float], radius_km: float) -> Tuple[float, float]:
    """
    Sample an areally uniform random point inside a circle around a center given in longitude/latitude coordinates.
    """
    lon0, lat0 = center
    u = random.random()
    r_km = radius_km * math.sqrt(u)
    theta = 2.0 * math.pi * random.random()
    dlat = km_to_deg_lat(r_km) * math.sin(theta)
    dlon = km_to_deg_lon(r_km, lat0) * math.cos(theta)
    return (lon0 + dlon, lat0 + dlat)


def heavy_tailed_radius_km(scale_km: float, alpha: float) -> float:
    """
    Sample a non-negative heavy-tailed radius in kilometres for rural endpoint placement around configured centres.
    """
    u = max(1e-12, random.random())
    return scale_km * (u ** (-1.0 / alpha) - 1.0)


# ============================================================
# Polygon and multipolygon sampling helpers
# ============================================================

_TRANSFORMER_3035 = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3035), always_xy=True)


def _extract_polygons(geom: Polygon | MultiPolygon) -> List[Polygon]:
    """
    Return the polygon components of a polygon or multipolygon geometry as a plain list of ``Polygon`` objects.
    """
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in geom.geoms if isinstance(g, Polygon)]


def _project_geom_3035(g):
    """
    Project a Shapely geometry from WGS84 to EPSG:3035 for area-based weighting and geometry comparison.
    """
    def f(x, y, z=None):
        return _TRANSFORMER_3035.transform(x, y)
    return shapely_transform(f, g)


def build_component_samplers(
    geom: Polygon | MultiPolygon,
) -> Tuple[List[Polygon], List, List[float]]:
    """
    Prepare polygon-component sampling helpers for a country geometry.

    The function extracts polygon components, prepares them for repeated point-in-
    polygon queries, and computes area-based component weights. These weights are
    used to sample points from multipolygons without biasing the draw toward small
    components with large bounding boxes.

    """
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
    """
    Sample a point uniformly from a polygon or multipolygon represented by
    component polygons, prepared geometries, and area-based weights.

    The procedure first selects a component by weight and then performs rejection
    sampling inside that component's bounding box. The number of attempts is
    bounded to avoid infinite loops for thin or fragmented geometries.

    """
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
    """
    Sample a point uniformly from a prepared polygon by rejection sampling within its bounding box.
    """
    minx, miny, maxx, maxy = bounds
    for _ in range(int(max_tries)):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        if prep_poly.contains(Point(lon, lat)):
            return (lon, lat)
    raise RuntimeError("Uniform polygon sampling exceeded max_tries (geometry too sparse vs bounds).")


# ============================================================
# Geometry stabilization for terrestrial planning
# ============================================================

MAINLAND_ONLY_ADMINS = {"Spain", "Portugal", "Italy", "Greece", "Croatia", "France"}


def _largest_polygon_component(geom: Polygon | MultiPolygon) -> Polygon:
    """
    Return the largest polygon component of a possibly multipart geometry using projected area in EPSG:3035.
    """
    polys = _extract_polygons(geom)
    if not polys:
        raise RuntimeError("No polygon components available.")
    return max(polys, key=lambda p: float(_project_geom_3035(p).area) if not p.is_empty else -1.0)


def filter_centers_to_geometry(
    centers_lonlat: Dict[str, Tuple[float, float]],
    geom: Polygon | MultiPolygon,
) -> Dict[str, Tuple[float, float]]:
    """
    Return only those configured centre coordinates that lie on or within the provided geometry.
    """
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
    """
    Stabilize a country geometry for terrestrial European simulation use.

    For selected countries, the function restricts the geometry to the mainland
    component. For the remaining countries, it clips the geometry to a Europe-
    focused window while preserving components that contain configured planning
    centres. This avoids unrealistic inclusion of remote overseas territories in a
    terrestrial backbone model.

    """
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
# Natural Earth data loading
# ============================================================

def _download_bytes(url: str, timeout_s: int = 60) -> bytes:
    """
    Download binary content from a URL using ``requests`` when available and ``urllib`` as a fallback.
    """
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
    """
    Load a country geometry from the cached or downloaded Natural Earth 110m dataset in WGS84 coordinates.
    """
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
# Higher-resolution plotting geometries
# ============================================================

_NE10M_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
_NE10M_SHP = "ne_10m_admin_0_countries.shp"
_NE10M_CACHE = "ne_10m_admin_0_countries.zip"

_world10m_cache: Optional[gpd.GeoDataFrame] = None


def load_world_countries_10m(cfg: Config) -> gpd.GeoDataFrame:
    """
    Load and cache the Natural Earth 10m country dataset used for higher-resolution background plotting.
    """
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
    """
    Return the row subset for a named country from a Natural Earth GeoDataFrame using the available name column.
    """
    name_col = next((c for c in ("ADMIN", "NAME", "name") if c in world.columns), None)
    if name_col is None:
        return world.iloc[0:0].copy()
    return world[world[name_col] == admin_name].copy()


def load_country_polygon_10m(cfg: Config, admin_name: str) -> Optional[Polygon | MultiPolygon]:
    """
    Loads the country geometry from Natural Earth 10m (same source used for plotting).
    Returns None if the country cannot be found.
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
# Country-configuration loading helpers
# ============================================================

def load_all_country_configs_from_module() -> Dict[str, EU.CountryConfig]:
    """
    Load all available ``EU.CountryConfig`` objects from the imported country configuration module.
    """
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
    """
    Convert an arbitrary country name to a filesystem-safe directory name while preserving readability.
    """
    s = name.strip()
    s = re.sub(r"[^\w\s\-.()]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:120] if s else "Country"


# ============================================================
# Persistence helpers
# ============================================================

def _write_json(path: Path, obj: Dict) -> None:
    """
    Write a dictionary to disk as UTF-8 encoded, indented JSON followed by a trailing newline.
    """
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_jsonable(obj):
    """
    Recursively convert nested Python objects, dataclasses, and tuples into structures that can be serialized to JSON.
    """
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
    """
    Persist the effective simulation configuration and country configuration for a single country run.
    """
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
    """
    Save per-run node tables, edge tables, and summary metadata for one Monte Carlo realization.
    """
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
# Endpoint generation fallback utilities
# ============================================================

def _fallback_fill_points(
    prep_poly,
    bounds: Tuple[float, float, float, float],
    n_needed: int,
) -> List[Tuple[float, float]]:
    """
    Generate additional in-geometry points when the main endpoint sampling
    procedures do not produce enough valid samples.

    The fallback uses a jittered grid first and a Gaussian proposal around the
    geometry centre second. It is intended as a bounded recovery mechanism for
    difficult or fragmented geometries.

    """
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
    """
    Generate Austrian QKD endpoints according to the fixed national reference case.

    The Austrian model uses explicit clusters around Vienna, the other provincial
    capitals, and St. Johann im Pongau, followed by a rural component drawn from a
    mixture of uniform country-wide samples and heavy-tailed draws around selected
    centres.

    """
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

    attempts = 0
    while len(endpoints) < cfg.n_vienna_cluster:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Austria Vienna cluster sampling stuck (unexpected).")
        p = random_point_in_radius(CAPITALS[vienna_name], cfg.vienna_radius_km)
        if prep_poly.contains(Point(p[0], p[1])):
            add_endpoint(p, vienna_name)

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

    rural_label = "Rural"
    capitals_for_rural = [k for k in CAPITALS.keys() if k != stj_name]
    while len(endpoints) < cfg.n_qkd_endpoints_total:
        if random.random() < cfg.p_uniform_rural:
            p = sample_point_in_geometry_uniform(comp_polys, comp_preps, comp_weights)
            add_endpoint(p, rural_label)
        else:
            cap = random.choice(capitals_for_rural)
            r_km = heavy_tailed_radius_km(cfg.heavy_tail_scale_km, cfg.heavy_tail_alpha)
            p = random_point_in_radius(CAPITALS[cap], r_km)
            if prep_poly.contains(Point(p[0], p[1])):
                add_endpoint(p, rural_label)

    return endpoints, meta, macro_labels


def make_qkd_endpoints_eu_country(
    cfg: Config,
    geom: Polygon | MultiPolygon,
    country_cfg: EU.CountryConfig,
) -> Tuple[List[Tuple[float, float]], List[Dict], List[str]]:
    """
    Generate QKD endpoints for a non-Austrian country from its ``CountryConfig``.

    Configured centres receive fixed cluster counts within defined radii. The
    remaining endpoints are drawn from a rural mixture that combines uniform
    sampling over the full country geometry with heavy-tailed radial sampling
    around existing centres.

    """
    prep_poly = shapely_prep(geom)
    bounds = geom.bounds

    comp_polys, comp_preps, comp_weights = build_component_samplers(geom)

    centers = dict(getattr(country_cfg, "centers_lonlat", {}) or {})
    center_counts = dict(getattr(country_cfg, "center_counts", {}) or {})
    radii = dict(getattr(country_cfg, "center_radius_km", {}) or {})
    default_radius = float(radii.get("__default__", 10.0))

    n_total = int(getattr(country_cfg, "n_endpoints", cfg.n_qkd_endpoints_total))
    n_cluster = int(sum(center_counts.get(k, 0) for k in centers.keys()))
    n_rural = n_total - n_cluster
    if n_rural < 0:
        raise ValueError(
            f"{getattr(country_cfg, 'natural_earth_admin', 'UNKNOWN')}: center_counts sum to {n_cluster} > n_endpoints={n_total}"
        )

    endpoints: List[Tuple[float, float]] = []
    meta: List[Dict] = []
    macro_labels: List[str] = []

    def add_endpoint(p: Tuple[float, float], cluster: str) -> None:
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

    for name, lonlat in centers.items():
        target = int(center_counts.get(name, 0))
        if target <= 0:
            continue
        r_km = float(radii.get(name, default_radius))

        added = 0
        attempts = 0
        max_attempts = 50000 + 5000 * target

        while added < target:
            attempts += 1
            if attempts > max_attempts:
                remaining = target - added
                filled = 0

                for _ in range(remaining):
                    try:
                        p = sample_point_in_geometry_uniform(comp_polys, comp_preps, comp_weights)
                        add_endpoint(p, str(name))
                        filled += 1
                    except Exception:
                        break

                if filled < remaining:
                    for _ in range(remaining - filled):
                        try:
                            p = sample_point_in_polygon_uniform(prep_poly, bounds, max_tries=20000)
                            add_endpoint(p, str(name))
                            filled += 1
                        except Exception:
                            break

                if filled < remaining:
                    fallback_pts = _fallback_fill_points(prep_poly, bounds, remaining - filled)
                    for p in fallback_pts:
                        add_endpoint(p, str(name))
                        filled += 1

                if filled < remaining:
                    raise RuntimeError(
                        f"Center sampling stuck for {getattr(country_cfg, 'natural_earth_admin', 'UNKNOWN')}:{name} "
                        f"({added}/{target})."
                    )
                added = target
                break

            p = random_point_in_radius((float(lonlat[0]), float(lonlat[1])), r_km)
            if prep_poly.contains(Point(p[0], p[1])):
                add_endpoint(p, str(name))
                added += 1

    rural_label = "Rural"
    center_list = list(centers.values())
    fallback_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)

    max_attempts = 250000 + 2000 * max(0, n_rural)
    attempts = 0

    while len(endpoints) < n_total:
        attempts += 1
        if attempts > max_attempts:
            remaining = n_total - len(endpoints)

            filled = 0
            for _ in range(remaining):
                try:
                    p = sample_point_in_geometry_uniform(comp_polys, comp_preps, comp_weights)
                    add_endpoint(p, rural_label)
                    filled += 1
                except Exception:
                    break

            if filled < remaining:
                fallback_pts = _fallback_fill_points(prep_poly, bounds, remaining - filled)
                for p in fallback_pts:
                    add_endpoint(p, rural_label)

            if len(endpoints) < n_total:
                raise RuntimeError(
                    f"Endpoint sampling stuck: generated {len(endpoints)}/{n_total} "
                    f"after {max_attempts} attempts; fallback still insufficient (geometry likely broken)."
                )
            break

        if random.random() < float(cfg.p_uniform_rural):
            try:
                p = sample_point_in_geometry_uniform(comp_polys, comp_preps, comp_weights)
                add_endpoint(p, rural_label)
            except Exception:
                c = random.choice(center_list) if center_list else fallback_center
                p = random_point_in_radius((float(c[0]), float(c[1])), 150.0)
                if prep_poly.contains(Point(p[0], p[1])):
                    add_endpoint(p, rural_label)
        else:
            c = random.choice(center_list) if center_list else fallback_center
            r_km = heavy_tailed_radius_km(float(cfg.heavy_tail_scale_km), float(cfg.heavy_tail_alpha))
            p = random_point_in_radius((float(c[0]), float(c[1])), r_km)
            if prep_poly.contains(Point(p[0], p[1])):
                add_endpoint(p, rural_label)

    return endpoints, meta, macro_labels


# ============================================================
# Degree target construction
# ============================================================

def _validate_manual_k_distribution(cfg: Config, n_nodes: int) -> None:
    """
    Validate that the configured manual degree counts match the number of endpoints and define an even degree sum.
    """
    total = cfg.n_nodes_k2 + cfg.n_nodes_k3 + cfg.n_nodes_k4 + cfg.n_nodes_k5
    if total != n_nodes:
        raise ValueError(f"Manual k counts sum to {total}, but n_qkd_endpoints_total={n_nodes}.")
    s = 2 * cfg.n_nodes_k2 + 3 * cfg.n_nodes_k3 + 4 * cfg.n_nodes_k4 + 5 * cfg.n_nodes_k5
    if s % 2 != 0:
        raise ValueError(f"Sum of target degrees must be even for an undirected graph. Got sum={s}.")


def build_k_vector(cfg: Config, n_nodes: int, seed: int) -> np.ndarray:
    """
    Construct the endpoint target-degree vector either from the manual degree distribution or from a fixed degree value.
    """
    if not cfg.use_manual_k_distribution:
        k = int(cfg.k_nearest)
        if (n_nodes * k) % 2 != 0:
            raise ValueError("n_nodes * k must be even for an undirected graph.")
        return np.full(n_nodes, k, dtype=int)

    _validate_manual_k_distribution(cfg, n_nodes)

    ks: List[int] = []
    ks += [2] * cfg.n_nodes_k2
    ks += [3] * cfg.n_nodes_k3
    ks += [4] * cfg.n_nodes_k4
    ks += [5] * cfg.n_nodes_k5

    rng = random.Random(seed)
    rng.shuffle(ks)
    return np.asarray(ks, dtype=int)


# ============================================================
# Degree-constrained edge construction
# ============================================================

def project_lonlat(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    Project longitude/latitude point coordinates to EPSG:3035 and return the resulting planar coordinate array.
    """
    xs, ys = _TRANSFORMER_3035.transform([p[0] for p in points], [p[1] for p in points])
    return np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])


def neighbor_lists_by_distance(xy: np.ndarray, k: int) -> np.ndarray:
    """
    Return, for each projected point, the indices of its nearest neighbours in projected space.
    """
    n = xy.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=int)

    if cKDTree is None:
        d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return np.argsort(d2, axis=1)[:, : min(k, n - 1)]

    tree = cKDTree(xy)
    kq = min(n, k + 1)  # +1 for self
    _, nbrs = tree.query(xy, k=kq)
    if nbrs.ndim == 1:
        nbrs = nbrs[:, None]
    return nbrs[:, 1:]  # drop self


def _edge_cost_sq(xy: np.ndarray, u: int, v: int) -> float:
    """
    Return the squared Euclidean distance between two projected points for internal edge-cost comparisons.
    """
    dx = float(xy[u, 0] - xy[v, 0])
    dy = float(xy[u, 1] - xy[v, 1])
    return dx * dx + dy * dy


def _try_add_edge_simple(
    i: int,
    j: int,
    rem: np.ndarray,
    adj: List[set[int]],
    edges: set[Tuple[int, int]],
) -> bool:
    """
    Attempt to add an undirected edge while respecting remaining degree budget and graph simplicity constraints.
    """
    if i == j:
        return False
    if rem[i] <= 0 or rem[j] <= 0:
        return False
    if j in adj[i]:
        return False
    a, b = (i, j) if i < j else (j, i)
    edges.add((a, b))
    adj[i].add(j)
    adj[j].add(i)
    rem[i] -= 1
    rem[j] -= 1
    return True


def _build_edges_stub_matching_distance_aware(
    cfg: Config,
    xy: np.ndarray,
    target_deg: np.ndarray,
    nbrs: np.ndarray,
    rng: random.Random,
) -> Optional[List[Tuple[int, int]]]:
    """
    Construct a simple undirected graph that matches a prescribed degree sequence.

    The heuristic repeatedly connects endpoints with remaining degree demand while
    preferentially selecting short candidate edges. The procedure operates in
    projected coordinate space and returns ``None`` if it cannot satisfy the exact
    degree sequence within the internal step budget.

    """
    n = int(target_deg.size)
    rem = target_deg.astype(int).copy()
    adj: List[set[int]] = [set() for _ in range(n)]
    edges: set[Tuple[int, int]] = set()

    nbr_costs: List[List[Tuple[float, int]]] = []
    for i in range(n):
        lst: List[Tuple[float, int]] = []
        for j in nbrs[i].tolist() if nbrs.shape[1] else []:
            if j == i:
                continue
            lst.append((_edge_cost_sq(xy, i, int(j)), int(j)))
        lst.sort(key=lambda t: t[0])
        nbr_costs.append(lst)

    max_steps = int(10 * max(1, int(rem.sum())))
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            return None

        active = np.flatnonzero(rem > 0)
        if active.size == 0:
            break

        active_list = active.tolist()
        rng.shuffle(active_list)
        i = max(active_list, key=lambda x: (int(rem[x]), -len(adj[x])))

        cand = []
        for cost, j in nbr_costs[i]:
            if rem[j] <= 0:
                continue
            if j in adj[i]:
                continue
            cand.append((cost, j))
            if len(cand) >= 24:
                break

        if not cand:
            feasible = [j for j in range(n) if (j != i and rem[j] > 0 and j not in adj[i])]
            if not feasible:
                return None
            j_best = min(feasible, key=lambda j: _edge_cost_sq(xy, i, int(j)))
            if not _try_add_edge_simple(i, int(j_best), rem, adj, edges):
                return None
            continue

        top = cand[: min(8, len(cand))]
        weights = []
        for cost, _j in top:
            weights.append(1.0 / (1e-12 + float(cost)))
        j = rng.choices([j for _, j in top], weights=weights, k=1)[0]

        if not _try_add_edge_simple(i, int(j), rem, adj, edges):
            return None

    degs = np.zeros(n, dtype=int)
    for u, v in edges:
        degs[u] += 1
        degs[v] += 1
    if not np.array_equal(degs, target_deg.astype(int)):
        return None

    return sorted(edges)


def _local_optimize_edges_by_swaps(
    xy: np.ndarray,
    edges: List[Tuple[int, int]],
    seed: int,
    n_iters: int = 2500,
) -> List[Tuple[int, int]]:
    """
    Improve an exact-degree graph by performing distance-reducing double-edge swaps.

    The optimization preserves node degrees and graph simplicity. Candidate swaps
    are accepted only when the total squared edge length decreases.

    """
    rng = random.Random(seed)
    if len(edges) < 2:
        return edges

    edge_set: set[Tuple[int, int]] = set((min(u, v), max(u, v)) for u, v in edges)
    edges_list = list(edge_set)

    n = int(xy.shape[0])
    adj: List[set[int]] = [set() for _ in range(n)]
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u: int, v: int) -> None:
        a, b = (u, v) if u < v else (v, u)
        edge_set.discard((a, b))
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u: int, v: int) -> None:
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))
        adj[u].add(v)
        adj[v].add(u)

    def has_edge(u: int, v: int) -> bool:
        a, b = (u, v) if u < v else (v, u)
        return (a, b) in edge_set

    def cost(u: int, v: int) -> float:
        return _edge_cost_sq(xy, u, v)

    m = len(edges_list)
    for _ in range(int(n_iters)):
        e1 = edges_list[rng.randrange(m)]
        e2 = edges_list[rng.randrange(m)]
        if e1 == e2:
            continue
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) < 4:
            continue

        patterns = [
            (a, d, c, b),
            (a, c, b, d),
        ]
        rng.shuffle(patterns)

        old = cost(a, b) + cost(c, d)

        for x1, y1, x2, y2 in patterns:
            if x1 == y1 or x2 == y2:
                continue
            if has_edge(x1, y1) or has_edge(x2, y2):
                continue
            if (y1 in adj[x1]) or (y2 in adj[x2]):
                continue

            new = cost(x1, y1) + cost(x2, y2)
            if new + 1e-9 < old:
                remove_edge(a, b)
                remove_edge(c, d)
                add_edge(x1, y1)
                add_edge(x2, y2)
                edges_list = list(edge_set)
                m = len(edges_list)
                break

    return sorted(edge_set)


def build_degree_constrained_edges(
    cfg: Config,
    endpoints: List[Tuple[float, float]],
    target_deg: np.ndarray,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Build a degree-constrained endpoint graph under exact degree requirements.

    The function first uses a distance-aware constructive heuristic with local
    edge-swap improvement. If that fails, it falls back to a simpler restart-based
    construction procedure. In both cases the result must match the target degree
    sequence exactly.

    """
    n = len(endpoints)
    if n < 2:
        return []

    if int(target_deg.sum()) % 2 != 0:
        raise ValueError("Target degree sum must be even.")

    xy = project_lonlat(endpoints)
    cand_k = int(min(max(cfg.candidate_neighbors_per_node, int(target_deg.max()) + 8), max(1, n - 1)))
    nbrs = neighbor_lists_by_distance(xy, k=cand_k)

    rng = random.Random(seed)

    for _restart in range(int(cfg.max_edge_build_attempts)):
        edges = _build_edges_stub_matching_distance_aware(
            cfg=cfg,
            xy=xy,
            target_deg=target_deg,
            nbrs=nbrs,
            rng=random.Random(seed + 10007 * _restart),
        )
        if edges is None:
            continue

        edges = _local_optimize_edges_by_swaps(
            xy=xy,
            edges=edges,
            seed=seed + 900001 * _restart,
            n_iters=2500,
        )

        degs = np.zeros(n, dtype=int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        if not np.array_equal(degs, target_deg.astype(int)):
            continue

        return edges

    for _restart in range(int(cfg.max_edge_build_attempts)):
        rem = target_deg.astype(int).copy()
        adj: List[set[int]] = [set() for _ in range(n)]
        edges: set[Tuple[int, int]] = set()

        ok = True
        while True:
            active = np.flatnonzero(rem > 0)
            if active.size == 0:
                break

            active_list = active.tolist()
            rng.shuffle(active_list)
            i = max(active_list, key=lambda x: rem[x])

            placed = False
            candidates = nbrs[i].tolist() if nbrs.shape[1] else []
            if len(candidates) > 6:
                head = candidates[:8]
                tail = candidates[8:]
                rng.shuffle(head)
                candidates = head + tail

            for j in candidates:
                if j == i:
                    continue
                if rem[j] <= 0:
                    continue
                if j in adj[i]:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
                adj[i].add(j)
                adj[j].add(i)
                rem[i] -= 1
                rem[j] -= 1
                placed = True
                break

            if placed:
                continue

            feasible = [j for j in range(n) if (j != i and rem[j] > 0 and j not in adj[i])]
            if not feasible:
                ok = False
                break

            xi = xy[i]
            best_j = min(feasible, key=lambda j: float((xi[0] - xy[j, 0]) ** 2 + (xi[1] - xy[j, 1]) ** 2))
            a, b = (i, best_j) if i < best_j else (best_j, i)
            edges.add((a, b))
            adj[i].add(best_j)
            adj[best_j].add(i)
            rem[i] -= 1
            rem[best_j] -= 1

        if not ok:
            continue

        degs = np.zeros(n, dtype=int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        if not np.array_equal(degs, target_deg.astype(int)):
            continue

        return sorted(edges)

    raise RuntimeError("Edge construction failed under exact degree constraints. Increase budgets or candidates.")


# ============================================================
# Robustness scoring
# ============================================================

def edge_length_robustness_score(edges_km: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Compute a robustness-oriented score from the distribution of edge lengths.

    Lower scores are preferred. The score penalizes edge-length outliers most
    strongly, then large maximum edges, and finally high edge-length coefficient of
    variation.

    """
    x = np.asarray(edges_km, dtype=float)
    if x.size == 0:
        return float("inf"), {"max": float("nan"), "mean": float("nan"), "std": float("nan"), "cv": float("nan"), "outliers_madz": float("nan")}

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    mx = float(x.max())

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-12:
        outliers = 0.0
    else:
        mod_z = 0.6745 * (x - med) / mad
        outliers = float(np.sum(np.abs(mod_z) > 3.5))

    cv = float(std / mean) if mean > 1e-12 else float("inf")

    score = (1e6 * outliers) + (1e3 * mx) + (1e2 * cv)
    stats = {"max": mx, "mean": mean, "std": float(std), "cv": cv, "outliers_madz": outliers}
    return float(score), stats


# ============================================================
# Graph construction, export tables, and hop statistics
# ============================================================

def build_graph(endpoints: List[Tuple[float, float]], edges: List[Tuple[int, int]]) -> nx.Graph:
    """
    Build a NetworkX graph from endpoint coordinates and undirected edge pairs, storing geodesic edge lengths in kilometres.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(endpoints)))
    for u, v in edges:
        G.add_edge(u, v, distance_km=geodesic_km(endpoints[u], endpoints[v]))
    return G


def degree_histogram(G: nx.Graph) -> Dict[int, int]:
    """
    Return the node-degree histogram of a NetworkX graph as a degree-to-count dictionary.
    """
    hist: Dict[int, int] = {}
    for _, d in G.degree():
        hist[int(d)] = hist.get(int(d), 0) + 1
    return dict(sorted(hist.items()))


def total_fiber_km(G: nx.Graph) -> float:
    """
    Return the sum of all unique undirected edge lengths stored in the graph in kilometres.
    """
    return float(sum(data["distance_km"] for _, _, data in G.edges(data=True)))


def edges_to_dataframe(endpoints: List[Tuple[float, float]], G: nx.Graph) -> pd.DataFrame:
    """
    Convert graph edges and endpoint coordinates to a sorted edge table suitable for export and plotting.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        a = endpoints[u]
        b = endpoints[v]
        rows.append(
            {
                "u": f"Q{u:03d}",
                "v": f"Q{v:03d}",
                "u_lon": float(a[0]),
                "u_lat": float(a[1]),
                "v_lon": float(b[0]),
                "v_lat": float(b[1]),
                "distance_km": float(data["distance_km"]),
                "edge_type": "QKD_ENDPOINT-QKD_ENDPOINT",
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("distance_km", ascending=False).reset_index(drop=True)


def allocate_trns_across_edges(edge_lengths_km: Sequence[float], n_trns: int) -> List[int]:
    """
    Allocate a fixed trusted repeater budget across edges.

    Allocation is greedy with respect to the current maximum resulting segment
    length ``L / (m + 1)`` on each edge, where ``m`` is the number of repeaters
    already assigned to that edge.

    """
    x = np.asarray(edge_lengths_km, dtype=float)
    m = int(x.size)
    total = int(max(0, n_trns))

    if m == 0 or total <= 0:
        return [0] * m

    alloc = [0] * m

    for _ in range(total):
        best_i = 0
        best_val = -float("inf")
        for i in range(m):
            current_max_segment = float(x[i]) / float(alloc[i] + 1)
            if current_max_segment > best_val:
                best_val = current_max_segment
                best_i = i
        alloc[best_i] += 1

    return alloc


def build_trusted_repeater_nodes(cfg: Config, edges_df: pd.DataFrame) -> Tuple[List[Tuple[float, float]], List[Dict]]:
    """
    Create trusted repeater node coordinates and metadata by splitting selected edges according to the greedy TRN allocation.
    """
    trn_points: List[Tuple[float, float]] = []
    trn_meta: List[Dict] = []

    total = int(max(0, cfg.n_trusted_repeater_nodes))
    if total <= 0 or edges_df.empty:
        return trn_points, trn_meta

    lengths = edges_df["distance_km"].to_numpy(dtype=float)
    alloc = allocate_trns_across_edges(lengths, total)

    t_idx = 0

    for edge_idx, row in enumerate(edges_df.itertuples(index=False)):
        n_on_edge = int(alloc[edge_idx])
        if n_on_edge <= 0:
            continue

        a = (float(row.u_lon), float(row.u_lat))
        b = (float(row.v_lon), float(row.v_lat))
        L = float(row.distance_km)

        n_segments = n_on_edge + 1
        for j in range(1, n_on_edge + 1):
            frac = float(j) / float(n_segments)
            p = geodesic_point_fraction(a, b, frac)

            trn_points.append(p)
            trn_meta.append(
                {
                    "node_id": f"T{t_idx:03d}",
                    "type": "TRUSTED_REPEATER_NODE",
                    "lon": float(p[0]),
                    "lat": float(p[1]),
                    "from_edge_u": row.u,
                    "from_edge_v": row.v,
                    "from_edge_distance_km": L,
                    "split_mode": f"EVEN_{n_segments}_SEGMENTS",
                    "split_fraction": float(frac),
                    "n_trns_on_edge": int(n_on_edge),
                    "n_segments_on_edge": int(n_segments),
                    "resulting_segment_km": float(L) / float(n_segments),
                }
            )
            t_idx += 1

    return trn_points, trn_meta


def hop_length_stats_after_splitting(cfg: Config, edges_df: pd.DataFrame) -> Dict[str, float]:
    """
    Return minimum, mean, and maximum hop lengths after applying the trusted repeater allocation to the edge set.
    """
    x = edges_df["distance_km"].to_numpy(dtype=float)
    if x.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}

    total = int(max(0, cfg.n_trusted_repeater_nodes))
    if total <= 0:
        hop_arr = np.asarray(x, dtype=float)
        return {"min": float(hop_arr.min()), "mean": float(hop_arr.mean()), "max": float(hop_arr.max())}

    alloc = allocate_trns_across_edges(x, total)

    hops: List[float] = []
    for i, L in enumerate(x):
        n_on_edge = int(alloc[i])
        n_segments = n_on_edge + 1
        seg_len = float(L) / float(n_segments)
        hops.extend([seg_len] * n_segments)

    hop_arr = np.asarray(hops, dtype=float)
    return {"min": float(hop_arr.min()), "mean": float(hop_arr.mean()), "max": float(hop_arr.max())}


# ============================================================
# Plotting
# ============================================================

def _k_text(cfg: Config) -> str:
    """
    Return a compact textual representation of the active endpoint degree specification for plotting and reporting.
    """
    if cfg.use_manual_k_distribution:
        return f"k2={cfg.n_nodes_k2}, k3={cfg.n_nodes_k3}, k4={cfg.n_nodes_k4}, k5={cfg.n_nodes_k5}"
    return f"fixed k={cfg.k_nearest}"


def plot_simulation(
    cfg: Config,
    country_name: str,
    admin_name: str,
    geom: Polygon | MultiPolygon,
    endpoints: List[Tuple[float, float]],
    trusted_repeater_nodes: List[Tuple[float, float]],
    edges_df: pd.DataFrame,
    fiber_km: float,
    centers_lonlat: Dict[str, Tuple[float, float]],
) -> None:
    """
    Create and save the geographic plot for a selected network realization.

    The plot shows the simulated country geometry, nearby countries for context,
    endpoint-to-endpoint fibre edges, trusted repeater nodes, and configured
    planning centres. The figure is written to both PNG and EPS formats.

    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)

    ax.set_facecolor("#f4f7fb")

    plot_geom = geom
    neighbors_gdf = None

    try:
        world10m = load_world_countries_10m(cfg)

        minx, miny, maxx, maxy = gpd.GeoSeries([plot_geom], crs="EPSG:4326").total_bounds
        pad_x = max(0.4, 0.10 * (maxx - minx + 1e-9))
        pad_y = max(0.4, 0.10 * (maxy - miny + 1e-9))
        view_box = box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
        neighbors_gdf = world10m[world10m.geometry.intersects(view_box)].copy()
    except Exception:
        neighbors_gdf = None

    if neighbors_gdf is not None and (not neighbors_gdf.empty):
        try:
            neighbors_gdf.plot(ax=ax, color="#eef2f6", edgecolor="#bfc7d1", linewidth=0.5, zorder=0)
        except Exception:
            pass

    gpd.GeoSeries([plot_geom], crs="EPSG:4326").plot(
        ax=ax,
        color="none",
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )

    for _, row in edges_df.iterrows():
        ax.plot(
            [row["u_lon"], row["v_lon"]],
            [row["u_lat"], row["v_lat"]],
            color=PLOT_EDGE_COLOR,
            linewidth=0.6,
            alpha=0.5,
            zorder=3,
            linestyle="--",
        )
    ax.scatter([p[0] for p in endpoints], [p[1] for p in endpoints], s=18, c="blue", label="QKD Endpoints (blue)", zorder=4)
    ax.scatter(
        [p[0] for p in trusted_repeater_nodes],
        [p[1] for p in trusted_repeater_nodes],
        s=20,
        c="red",
        label="Trusted Repeater Nodes (red)",
        zorder=5,
    )

    text_effects = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]
    for name, (lon, lat) in centers_lonlat.items():
        ax.scatter([lon], [lat], s=35, c="black", marker="x", zorder=6)
        ax.text(
            lon + 0.03,
            lat + 0.02,
            str(name),
            fontsize=9 * FONT_SCALE,
            weight="bold",
            zorder=7,
            path_effects=text_effects,
        )

    if LABEL_ALL_QKD_ENDPOINTS:
        for i, (lon, lat) in enumerate(endpoints):
            ax.text(lon + 0.01, lat + 0.01, f"Q{i:03d}", fontsize=6 * FONT_SCALE, zorder=8)

    try:
        minx, miny, maxx, maxy = gpd.GeoSeries([plot_geom], crs="EPSG:4326").total_bounds
        pad_x = max(0.4, 0.08 * (maxx - minx + 1e-9))
        pad_y = max(0.4, 0.08 * (maxy - miny + 1e-9))
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    except Exception:
        pass

    ax.grid(True, linewidth=0.5, alpha=0.25)

    ax.set_title(
        f"{country_name} QKD network simulation: {cfg.n_qkd_endpoints_total} endpoints, {len(trusted_repeater_nodes)} TRNs, {_k_text(cfg)}\n"
        f"Total fiber length: {1.5*fiber_km:.3f} ({fiber_km:.3f}; raw) km"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    png_path = Path(cfg.plot_png)
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Single Monte Carlo realization
# ============================================================

def simulate_once(
    cfg: Config,
    geom: Polygon | MultiPolygon,
    seed: int,
    run_idx: int,
    endpoint_mode: str,
    country_cfg: Optional[EU.CountryConfig] = None,
) -> Dict:
    """
    Run one Monte Carlo realization for a country geometry.

    A realization consists of endpoint generation, exact-degree graph
    construction, constraint checks, robustness scoring, trusted repeater
    placement, and selection of the best candidate graph among multiple valid
    candidates.

    """
    random.seed(seed)
    np.random.seed(seed)

    log(cfg, f"\n[mc {run_idx:03d}] seed={seed} target_candidates={cfg.n_candidate_graphs_per_run}")

    best: Optional[Dict] = None
    best_score = float("inf")
    candidates_kept = 0
    last_err: Optional[Exception] = None

    t0 = time.time()

    for resample_idx in range(int(cfg.max_resample_endpoint_attempts)):
        log(cfg, f"[mc {run_idx:03d}] endpoint resample {resample_idx+1}/{cfg.max_resample_endpoint_attempts}")

        if endpoint_mode == "AUSTRIA":
            endpoints, endpoint_meta, macro_labels = make_qkd_endpoints_austria(cfg, geom)
        else:
            if country_cfg is None:
                raise RuntimeError("simulate_once(endpoint_mode=EU) needs country_cfg")
            endpoints, endpoint_meta, macro_labels = make_qkd_endpoints_eu_country(cfg, geom, country_cfg)

        target_deg = build_k_vector(cfg, n_nodes=len(endpoints), seed=seed + 1000 * resample_idx)

        if cfg.verbose:
            vals, counts = np.unique(target_deg, return_counts=True)
            deg_summary = ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)])
            log(cfg, f"[mc {run_idx:03d}] target degree multiset -> {deg_summary} (sum={int(target_deg.sum())})")

        for attempt in range(int(cfg.max_edge_build_attempts)):
            if cfg.verbose and (attempt % max(1, cfg.print_every_edge_attempt) == 0):
                log(cfg, f"[mc {run_idx:03d}] edge-build attempt {attempt}/{cfg.max_edge_build_attempts} (kept={candidates_kept})")

            try:
                edges = build_degree_constrained_edges(
                    cfg,
                    endpoints=endpoints,
                    target_deg=target_deg,
                    seed=seed + 17 * attempt + 99991 * resample_idx,
                )
                G = build_graph(endpoints, edges)

                if cfg.enforce_connectivity and (not nx.is_connected(G)):
                    continue
                if cfg.enforce_no_bridges and list(nx.bridges(G)):
                    continue

                edges_df = edges_to_dataframe(endpoints, G)
                edge_lengths = edges_df["distance_km"].to_numpy(dtype=float)
                score, score_stats = edge_length_robustness_score(edge_lengths)

                candidates_kept += 1
                if cfg.verbose and (candidates_kept % max(1, cfg.print_every_candidate) == 0):
                    log(cfg, f"[mc {run_idx:03d}] kept={candidates_kept} | best={best_score:.3f} | last={score:.3f}")

                if score < best_score:
                    fiber_km = total_fiber_km(G)
                    trn_points, trn_meta = build_trusted_repeater_nodes(cfg, edges_df)
                    hop_stats = hop_length_stats_after_splitting(cfg, edges_df)
                    deg_hist = degree_histogram(G)

                    best = {
                        "seed": seed,
                        "endpoints": endpoints,
                        "endpoint_meta": endpoint_meta,
                        "macro_labels": macro_labels,
                        "G": G,
                        "ks": target_deg,
                        "edges_df": edges_df,
                        "fiber_km": float(fiber_km),
                        "trusted_repeater_nodes": trn_points,
                        "trusted_repeater_meta": trn_meta,
                        "hop_stats": hop_stats,
                        "deg_hist": deg_hist,
                        "robust_score": float(score),
                        "robust_score_stats": score_stats,
                        "candidates_kept": int(candidates_kept),
                        "resample_idx": int(resample_idx),
                        "attempt_idx": int(attempt),
                    }
                    best_score = float(score)

                    log(
                        cfg,
                        f"[mc {run_idx:03d}] NEW BEST score={best_score:.3f} | max_edge={score_stats['max']:.3f} km | cv={score_stats['cv']:.4f} | outliers={score_stats['outliers_madz']:.0f}"
                        f" | fiber={fiber_km:.1f} km | kept={candidates_kept}",
                    )

                if candidates_kept >= int(max(1, cfg.n_candidate_graphs_per_run)) and best is not None:
                    dt = time.time() - t0
                    log(cfg, f"[mc {run_idx:03d}] done (met target) in {dt:.1f}s; best_score={best_score:.3f}")
                    return best

            except Exception as e:
                last_err = e
                continue

    if best is not None:
        dt = time.time() - t0
        log(cfg, f"[mc {run_idx:03d}] done (budget exhausted) in {dt:.1f}s; best_score={best_score:.3f}")
        return best

    raise RuntimeError(f"Failed to generate a valid graph under constraints. Last error: {last_err!r}")


# ============================================================
# Monte Carlo aggregation and reporting
# ============================================================

def _mean_std_min_max(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return mean, sample standard deviation, minimum, and maximum for a numeric array.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if x.size == 1:
        v = float(x[0])
        return v, 0.0, v, v
    return float(x.mean()), float(x.std(ddof=1)), float(x.min()), float(x.max())


def _stderr(std: float, n: int) -> float:
    """
    Return the standard error of the mean from a standard deviation and sample count.
    """
    return float("nan") if n <= 0 else float(std / math.sqrt(n))


def format_summary(title: str, arr: np.ndarray, factor: float = 1.0) -> str:
    """
    Format a textual summary block containing raw and detour-adjusted descriptive statistics for a numeric array.
    """
    n = int(np.asarray(arr).size)
    mean, std, mn, mx = _mean_std_min_max(arr)
    se = _stderr(std, n)
    return (
        f"{title}\n"
        f"  raw:      mean ± std = {mean:.6f} ± {std:.6f}   (stderr {se:.6f})\n"
        f"            min / max  = {mn:.6f} / {mx:.6f}\n"
        f"  adjusted: mean ± std = {(mean*factor):.6f} ± {(std*factor):.6f}   (stderr {(se*factor):.6f})   [x{factor:.3f}]\n"
        f"            min / max  = {(mn*factor):.6f} / {(mx*factor):.6f}      [x{factor:.3f}]\n"
    )


def run_monte_carlo(
    cfg: Config,
    geom: Polygon | MultiPolygon,
    endpoint_mode: str,
    country_cfg: Optional[EU.CountryConfig] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the configured number of Monte Carlo realizations and return the first realization together with the full result list.
    """
    first = simulate_once(cfg, geom, seed=cfg.seed, run_idx=0, endpoint_mode=endpoint_mode, country_cfg=country_cfg)
    rows: List[Dict] = [first]
    for i in range(1, int(cfg.n_monte_carlo_runs)):
        rows.append(simulate_once(cfg, geom, seed=cfg.seed + i, run_idx=i, endpoint_mode=endpoint_mode, country_cfg=country_cfg))
    return first, rows


def write_report(cfg: Config, country_name: str, first: Dict, rows: List[Dict]) -> None:
    """
    Write the text report summarizing configuration assumptions, first-run details, and Monte Carlo statistics for one country.
    """
    report_path = Path(cfg.report_txt)

    fiber_arr = np.asarray([r["fiber_km"] for r in rows], dtype=float)
    hop_mean_arr = np.asarray([r["hop_stats"]["mean"] for r in rows], dtype=float)
    hop_max_arr = np.asarray([r["hop_stats"]["max"] for r in rows], dtype=float)

    score_arr = np.asarray([float(r.get("robust_score", float("nan"))) for r in rows], dtype=float)
    max_edge_arr = np.asarray([float(r.get("robust_score_stats", {}).get("max", float("nan"))) for r in rows], dtype=float)
    cv_arr = np.asarray([float(r.get("robust_score_stats", {}).get("cv", float("nan"))) for r in rows], dtype=float)
    outliers_arr = np.asarray([float(r.get("robust_score_stats", {}).get("outliers_madz", float("nan"))) for r in rows], dtype=float)

    lines: List[str] = []
    lines.append(f"{country_name} QKD Network Simulation Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Model semantics")
    lines.append("-" * 40)
    lines.append("QKD Endpoints:")
    lines.append("  Randomly sampled points inside the country polygon, clustered (if configured) + rural mixture.")
    lines.append("Trusted Repeater Nodes (TRNs):")
    lines.append(
        "  Policy: distribute the fixed TRN budget greedily across the longest edges to reduce the maximum resulting hop length, "
        "placing each edge's assigned TRNs evenly along that edge."
    )
    lines.append("")
    lines.append("Degree semantics (critical)")
    lines.append("-" * 40)
    lines.append("Target degrees (k vector) define the REQUIRED number of incident edges per endpoint.")
    lines.append("No extra edges are ever added. Connectivity/bridge constraints are enforced by rejection & rebuild.")
    lines.append("")
    lines.append("Robustness selection (critical)")
    lines.append("-" * 40)
    lines.append("Per Monte Carlo run: generate many valid graphs and keep the one minimizing long edges/outliers.")
    lines.append(f"Candidate graphs per run: {cfg.n_candidate_graphs_per_run}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 40)
    lines.append(f"QKD endpoints total:      {cfg.n_qkd_endpoints_total}")
    lines.append(f"TRNs (placed on edges):   {cfg.n_trusted_repeater_nodes}")
    lines.append(f"Target degree dist:       {_k_text(cfg)}")
    lines.append(f"Connectivity enforced:    {cfg.enforce_connectivity}")
    lines.append(f"No-bridges enforced:      {cfg.enforce_no_bridges}")
    lines.append(f"Monte Carlo runs:         {cfg.n_monte_carlo_runs}")
    lines.append(f"Detour factor:            {cfg.detour_factor}")
    lines.append("")
    lines.append("First run (selected best candidate)")
    lines.append("-" * 40)
    lines.append(f"Seed:                     {first['seed']}")
    lines.append(f"Edges (undirected):       {int(first['edges_df'].shape[0])}")
    lines.append(f"Total fiber (km):         {first['fiber_km']:.6f}")
    lines.append(
        f"Hop stats after TRN split (km): min/mean/max = "
        f"{first['hop_stats']['min']:.6f} / {first['hop_stats']['mean']:.6f} / {first['hop_stats']['max']:.6f}"
    )
    lines.append(f"Degree histogram:         {first['deg_hist']}")
    rs = first.get("robust_score_stats", {})
    lines.append(f"Robustness score (lower better): {float(first['robust_score']):.6f}")
    lines.append(f"  max edge (km):          {float(rs.get('max', float('nan'))):.6f}")
    lines.append(f"  CV (std/mean):          {float(rs.get('cv', float('nan'))):.6f}")
    lines.append(f"  outliers (MAD z>3.5):   {float(rs.get('outliers_madz', float('nan'))):.0f}")
    lines.append("")
    lines.append("Monte Carlo summary")
    lines.append("-" * 40)
    lines.append(format_summary("Total fiber length (km)", fiber_arr, cfg.detour_factor))
    lines.append(format_summary("Mean hop length after TRN split (km)", hop_mean_arr, cfg.detour_factor))
    lines.append(format_summary("Max hop length after TRN split (km)", hop_max_arr, cfg.detour_factor))
    lines.append(format_summary("Robustness score (lower better)", score_arr, 1.0))
    lines.append(format_summary("Max edge length (km)", max_edge_arr, 1.0))
    lines.append(format_summary("Edge-length CV (std/mean)", cv_arr, 1.0))
    lines.append(format_summary("Outlier count (MAD z>3.5)", outliers_arr, 1.0))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(cfg, f"[out] Wrote report -> {report_path}")


# ============================================================
# Per-country execution
# ============================================================

def run_one_country(base_cfg: Config, country_name: str, country_cfg: EU.CountryConfig, out_root: Path) -> None:
    """
    Execute the full simulation workflow for a single country.

    This includes configuration preparation, geometry loading and stabilization,
    Monte Carlo evaluation, export of CSV and JSON artefacts, plot generation, text
    report generation, and console summary output.

    """
    admin = str(getattr(country_cfg, "natural_earth_admin", "")).strip() or str(country_name).strip()
    is_austria = (admin == "Austria") or (country_name == "Austria")

    country_dir = out_root / safe_folder_name(country_name)
    country_dir.mkdir(parents=True, exist_ok=True)

    if is_austria:
        cfg = replace(
            base_cfg,
            report_txt=str(country_dir / "qkd_simulation_report.txt"),
            edges_csv=str(country_dir / "qkd_edges_km.csv"),
            nodes_csv=str(country_dir / "qkd_nodes_lonlat.csv"),
            plot_png=str(country_dir / "qkd_simulation_plot.png"),
        )
        endpoint_mode = "AUSTRIA"
        centers_for_plot = dict(CAPITALS)
        save_country_configuration(cfg, country_name, country_cfg, country_dir)
    else:
        cfg = replace(
            base_cfg,
            n_qkd_endpoints_total=int(getattr(country_cfg, "n_endpoints", base_cfg.n_qkd_endpoints_total)),
            n_trusted_repeater_nodes=int(getattr(country_cfg, "n_trusted_repeater_nodes", base_cfg.n_trusted_repeater_nodes)),
            detour_factor=float(getattr(country_cfg, "detour_factor", base_cfg.detour_factor)),
            p_uniform_rural=float(getattr(country_cfg, "p_uniform_rural", base_cfg.p_uniform_rural)),
            heavy_tail_scale_km=float(getattr(country_cfg, "heavy_tail_scale_km", base_cfg.heavy_tail_scale_km)),
            heavy_tail_alpha=float(getattr(country_cfg, "heavy_tail_alpha", base_cfg.heavy_tail_alpha)),
            use_manual_k_distribution=True,
            n_nodes_k2=int(getattr(country_cfg, "k2_count", base_cfg.n_nodes_k2)),
            n_nodes_k3=int(getattr(country_cfg, "k3_count", base_cfg.n_nodes_k3)),
            n_nodes_k4=0,
            n_nodes_k5=0,
            report_txt=str(country_dir / "qkd_simulation_report.txt"),
            edges_csv=str(country_dir / "qkd_edges_km.csv"),
            nodes_csv=str(country_dir / "qkd_nodes_lonlat.csv"),
            plot_png=str(country_dir / "qkd_simulation_plot.png"),
        )
        endpoint_mode = "EU"
        centers_for_plot = dict(getattr(country_cfg, "centers_lonlat", {}) or {})
        save_country_configuration(cfg, country_name, country_cfg, country_dir)

    log(cfg, f"\n[start] {country_name} ({admin}) -> {country_dir}")

    # Use the SAME geometry for simulation and plotting:
    # Prefer 10m (plot-quality) if available; fallback to 110m.
    geom_110m = load_country_polygon(cfg, admin)

    geom_10m = None
    try:
        geom_10m = load_country_polygon_10m(cfg, admin)
    except Exception:
        geom_10m = None

    geom = geom_10m if geom_10m is not None else geom_110m

    country_cfg_for_run = country_cfg

    if not is_austria:
        geom = stabilize_eu_geometry(geom, centers_for_plot, admin)

        if admin in MAINLAND_ONLY_ADMINS:
            filtered_centers = filter_centers_to_geometry(centers_for_plot, geom)
            filtered_center_counts = {
                k: v for k, v in dict(getattr(country_cfg, "center_counts", {}) or {}).items()
                if k in filtered_centers
            }
            filtered_center_radii = dict(getattr(country_cfg, "center_radius_km", {}) or {})
            filtered_center_radii = {
                k: v for k, v in filtered_center_radii.items()
                if (k in filtered_centers) or (k == "__default__")
            }

            centers_for_plot = filtered_centers
            country_cfg_for_run = replace(
                country_cfg,
                centers_lonlat=filtered_centers,
                center_counts=filtered_center_counts,
                center_radius_km=filtered_center_radii,
            )

    first, rows = run_monte_carlo(
        cfg,
        geom,
        endpoint_mode=endpoint_mode,
        country_cfg=(None if is_austria else country_cfg_for_run),
    )

    for i, r in enumerate(rows):
        save_monte_carlo_run_artifacts(
            country_dir=country_dir,
            run_idx=i,
            country_name=country_name,
            natural_earth_admin=admin,
            cfg=cfg,
            run_result=r,
        )

    edges_df: pd.DataFrame = first["edges_df"]
    nodes_df = pd.DataFrame(first["endpoint_meta"] + first["trusted_repeater_meta"])
    edges_df.to_csv(cfg.edges_csv, index=False)
    nodes_df.to_csv(cfg.nodes_csv, index=False)
    log(cfg, f"[out] Wrote edges CSV -> {cfg.edges_csv}")
    log(cfg, f"[out] Wrote nodes CSV -> {cfg.nodes_csv}")

    plot_simulation(
        cfg=cfg,
        country_name=country_name,
        admin_name=admin,
        geom=geom,
        endpoints=first["endpoints"],
        trusted_repeater_nodes=first["trusted_repeater_nodes"],
        edges_df=edges_df,
        fiber_km=float(first["fiber_km"]),
        centers_lonlat=centers_for_plot,
    )
    log(cfg, f"[out] Wrote plot -> {cfg.plot_png}")

    write_report(cfg, country_name, first, rows)

    rs = first.get("robust_score_stats", {})
    print(f"\n=== Summary ({country_name}) ===")
    print(f"Seed: {first['seed']}")
    print(f"Edges (unique undirected): {len(edges_df)}")
    print(f"Total fiber length (km): {float(first['fiber_km']):,.3f}")
    print(f"Hop mean/max after TRN split (km): {first['hop_stats']['mean']:.3f} / {first['hop_stats']['max']:.3f}")
    print(f"Degree histogram: {first['deg_hist']}")
    print(f"Robustness score (lower better): {float(first['robust_score']):.3f}")
    print(
        f"  max edge (km): {float(rs.get('max', float('nan'))):.3f} | "
        f"CV: {float(rs.get('cv', float('nan'))):.4f} | "
        f"outliers: {float(rs.get('outliers_madz', float('nan'))):.0f}"
    )
    print(f"Wrote: {cfg.edges_csv}")
    print(f"Wrote: {cfg.nodes_csv}")
    print(f"Wrote: {cfg.plot_png}")
    print(f"Wrote: {cfg.report_txt}")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """
    Run the simulation workflow for all country configurations and write results to the output root directory.
    """
    base_cfg = Config()

    out_root = Path("EUMS_QKD_Network_Results")
    out_root.mkdir(parents=True, exist_ok=True)

    configs = load_all_country_configs_from_module()
    print(f"[info] Loaded {len(configs)} country configs from EU_List.py")

    for country_name, ccfg in configs.items():
        try:
            run_one_country(base_cfg, country_name, ccfg, out_root)
        except Exception as e:
            print(f"[WARN] Skipping {country_name} due to error: {e!r}", flush=True)
            continue

    print(f"\n[done] All results written under: {out_root.resolve()}")


if __name__ == "__main__":
    main()