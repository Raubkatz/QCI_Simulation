#!/usr/bin/env python3
"""
Recreate national QKD network plots from previously saved simulation outputs.

This script reads edge and node tables that were produced by an earlier
simulation workflow and regenerates visualization files for each available
country result directory. It does not rebuild the simulated networks. Instead,
it loads the already exported CSV files, reconstructs the plotting geometry,
and writes new plot files in PNG and EPS format.

Workflow overview
-----------------
1. Load country-specific simulation outputs from an existing results root.
2. Load national boundary geometries from Natural Earth datasets.
3. Use 10 m geometry where available for plotting and 110 m geometry as a
   fallback.
4. Apply country-specific geometry stabilization rules, especially for
   countries with island or overseas components.
5. Recover plotting centers from the country configuration module.
6. Plot:
   - background neighboring countries,
   - the focal country boundary,
   - saved network edges,
   - saved QKD endpoints,
   - saved trusted repeater nodes,
   - configured center markers and labels.
7. Save recreated plots under a separate output directory.

The script is intended for post-processing and visualization only. It assumes
that the simulation output structure already exists and that the corresponding
country configuration module can be imported.

Input requirements
------------------
- A directory tree containing one subdirectory per country.
- Each country directory must contain:
  - ``qkd_edges_km.csv``
  - ``qkd_nodes_lonlat.csv``
- Optionally, the directory may contain:
  - ``configuration/config_meta.json``

External data
-------------
The script downloads Natural Earth country boundary datasets if they are not
already cached locally. These datasets are used only for plotting and geometry
selection.

Notes
-----
The code is kept functionally identical to the original version. Only
documentation and explanatory comments are added here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import unary_union, transform as shapely_transform
from shapely.prepared import prep as shapely_prep

# EU_List.py must be available in the same folder or on PYTHONPATH
import EU_List_islands_alltogetehr as EU


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class PlotConfig:
    """
    Static configuration for plot recreation from saved simulation results.

    This dataclass defines directory paths, Natural Earth download and cache
    locations, and basic runtime verbosity for the plotting-only workflow.

    The configuration is separate from the simulation configuration used to
    generate the original network data. Its purpose is limited to locating
    previously saved outputs, loading background map data, and writing the
    recreated plot files.

    Attributes
    ----------
    results_root:
        Root directory containing the existing saved country result folders.
        Each country folder is expected to contain edge and node CSV exports.

    plots_root:
        Output directory into which the recreated plots are written.

    ne_url_110m, ne_shp_110m, ne_cache_110m:
        Source URL, shapefile name inside the archive, and local cache filename
        for the 110 m Natural Earth dataset. This lower-resolution dataset is
        used as a robust fallback when higher-resolution geometry is not
        available.

    ne_url_10m, ne_shp_10m, ne_cache_10m:
        Source URL, shapefile name inside the archive, and local cache filename
        for the 10 m Natural Earth dataset. This higher-resolution dataset is
        preferred for visual output.

    verbose:
        Controls whether progress messages are printed during execution.
    """
    # Input root with existing saved simulation results
    results_root: str = "EUMS_QKD_Network_Results"

    # Output root for recreated plots
    plots_root: str = "EUMS_QKD_Network_Plots"

    # Natural Earth
    ne_url_110m: str = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    ne_shp_110m: str = "ne_110m_admin_0_countries.shp"
    ne_cache_110m: str = "ne_110m_admin_0_countries.zip"

    ne_url_10m: str = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    ne_shp_10m: str = "ne_10m_admin_0_countries.shp"
    ne_cache_10m: str = "ne_10m_admin_0_countries.zip"

    verbose: bool = True


# ============================================================
# Plot settings (kept as close as possible)
# ============================================================

# Global plot text scaling factor applied to Matplotlib font parameters.
FONT_SCALE = 1.3

# Output figure size in inches.
FIGSIZE = (14, 12)

# Raster output resolution.
FIG_DPI = 220

# If enabled, all QKD endpoints are labeled individually by node identifier.
LABEL_ALL_QKD_ENDPOINTS = False

# Default color used for plotted graph edges.
PLOT_EDGE_COLOR = "black"

# Base font size used to derive all other plot font sizes.
_BASE_FONT = 10.0

# Global Matplotlib plotting style parameters used for all recreated figures.
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


# ============================================================
# Helpers
# ============================================================

# Coordinate transformer from geographic coordinates (EPSG:4326) to the
# projected CRS EPSG:3035, used for area-based comparisons and component
# selection.
_TRANSFORMER_3035 = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3035), always_xy=True)

# Countries for which only the mainland polygon component is retained during
# geometry stabilization.
MAINLAND_ONLY_ADMINS = {"Spain", "Portugal", "Italy", "Greece", "Croatia", "France"}

# Explicit Austria center definitions used for plotting. Austria is handled as
# a special case in the broader simulation framework and therefore retains its
# own fixed center set here.
_AUSTRIA_CAPITALS: Dict[str, Tuple[float, float]] = {
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

# Cache for the high-resolution world country dataset to avoid repeated file
# loads during multi-country plotting.
_world10m_cache: Optional[gpd.GeoDataFrame] = None


def log(cfg: PlotConfig, msg: str) -> None:
    """
    Print a log message if verbose output is enabled.

    Parameters
    ----------
    cfg:
        Plot configuration object controlling verbosity.
    msg:
        Text message to print.
    """
    if cfg.verbose:
        print(msg, flush=True)


def safe_folder_name(name: str) -> str:
    """
    Convert an arbitrary country name into a filesystem-safe folder name.

    The function preserves letters, digits, whitespace, underscores, hyphens,
    dots, and parentheses, replaces other characters with underscores, and
    collapses repeated whitespace into single underscores.

    Parameters
    ----------
    name:
        Original display name.

    Returns
    -------
    str
        Sanitized folder-compatible name.
    """
    s = name.strip()
    s = re.sub(r"[^\w\s\-.()]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:120] if s else "Country"


def _download_bytes(url: str, timeout_s: int = 60) -> bytes:
    """
    Download binary content from a URL.

    The function first tries to use the ``requests`` package. If that fails,
    it falls back to Python's standard ``urllib.request``.

    Parameters
    ----------
    url:
        Resource URL.
    timeout_s:
        Timeout in seconds.

    Returns
    -------
    bytes
        Downloaded binary content.
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


def _extract_polygons(geom: Polygon | MultiPolygon) -> List[Polygon]:
    """
    Convert a polygon or multipolygon geometry into a list of polygon parts.

    Parameters
    ----------
    geom:
        Input geometry.

    Returns
    -------
    List[Polygon]
        List of polygon components.
    """
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in geom.geoms if isinstance(g, Polygon)]


def _project_geom_3035(g):
    """
    Reproject a geometry from geographic coordinates to EPSG:3035.

    This helper is used primarily for area-based comparisons, since area
    calculations in a projected CRS are more meaningful than in geographic
    coordinates.

    Parameters
    ----------
    g:
        Input geometry.

    Returns
    -------
    Geometry
        Reprojected geometry.
    """
    def f(x, y, z=None):
        return _TRANSFORMER_3035.transform(x, y)
    return shapely_transform(f, g)


def _largest_polygon_component(geom: Polygon | MultiPolygon) -> Polygon:
    """
    Return the largest polygon component of a geometry.

    Area is evaluated after projection to EPSG:3035 in order to compare
    components in a projected metric space.

    Parameters
    ----------
    geom:
        Polygon or multipolygon input geometry.

    Returns
    -------
    Polygon
        Largest polygon component.

    Raises
    ------
    RuntimeError
        If no polygon components are available.
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
    Keep only center points that lie inside or intersect the given geometry.

    This function is used when the geometry is reduced, for example to mainland
    components only. It prevents plotting centers that are no longer part of
    the retained geometry.

    Parameters
    ----------
    centers_lonlat:
        Mapping from center names to longitude/latitude tuples.
    geom:
        Target geometry.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Filtered center dictionary.
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
    Stabilize a country geometry for plotting.

    For selected countries, only the mainland component is kept. For others,
    the geometry is clipped to a Europe-focused window and optionally extended
    by retaining polygon components that contain configured centers.

    Parameters
    ----------
    geom:
        Original country geometry.
    centers_lonlat:
        Plot centers potentially used to retain relevant polygon components.
    admin_name:
        Natural Earth administrative country name.

    Returns
    -------
    Polygon | MultiPolygon
        Stabilized geometry used for plotting.
    """
    if admin_name in MAINLAND_ONLY_ADMINS:
        return _largest_polygon_component(geom)

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
            for p in polys:
                try:
                    if p.contains(cpt) or p.intersects(cpt):
                        keep_geoms.append(p)
                except Exception:
                    continue

    if not keep_geoms:
        return geom

    out = unary_union(keep_geoms)
    if isinstance(out, (Polygon, MultiPolygon)):
        return out
    return geom


# ============================================================
# Natural Earth loading
# ============================================================

def load_country_polygon(cfg: PlotConfig, admin_name: str) -> Polygon | MultiPolygon:
    """
    Load a country geometry from the Natural Earth 110 m dataset.

    This function ensures that the low-resolution archive exists locally,
    reads the shapefile, identifies the requested country by administrative
    name, and returns its geometry in EPSG:4326 coordinates.

    Parameters
    ----------
    cfg:
        Plot configuration containing cache paths and download URLs.
    admin_name:
        Country name used in the Natural Earth dataset.

    Returns
    -------
    Polygon | MultiPolygon
        Country geometry.

    Raises
    ------
    RuntimeError
        If the dataset is malformed or the country cannot be found.
    """
    cache_path = Path(cfg.ne_cache_110m)
    if not cache_path.exists():
        log(cfg, f"[data] Downloading Natural Earth 110m -> {cache_path}")
        cache_path.write_bytes(_download_bytes(cfg.ne_url_110m, timeout_s=60))

    world = gpd.read_file(f"zip://{str(cache_path)}!{cfg.ne_shp_110m}")
    name_col = next((c for c in ("ADMIN", "NAME", "name") if c in world.columns), None)
    if name_col is None:
        raise RuntimeError(f"Cannot find a country name column in 110m dataset.")

    row = world[world[name_col] == admin_name].copy()
    if row.empty:
        raise RuntimeError(f"Country '{admin_name}' not found in 110m dataset.")

    try:
        row = row.to_crs("EPSG:4326")
    except Exception:
        pass

    geom = row.geometry.iloc[0]
    if geom is None or not isinstance(geom, (Polygon, MultiPolygon)):
        raise RuntimeError(f"Unexpected geometry for '{admin_name}': {type(geom)}")
    return geom


def load_world_countries_10m(cfg: PlotConfig) -> gpd.GeoDataFrame:
    """
    Load the high-resolution Natural Earth world country dataset.

    The dataset is cached globally after the first successful load.

    Parameters
    ----------
    cfg:
        Plot configuration containing cache paths and download URLs.

    Returns
    -------
    geopandas.GeoDataFrame
        World country dataset in EPSG:4326 if conversion succeeds.
    """
    global _world10m_cache
    if _world10m_cache is not None:
        return _world10m_cache

    cache_path = Path(cfg.ne_cache_10m)
    if not cache_path.exists():
        log(cfg, f"[data] Downloading Natural Earth 10m -> {cache_path}")
        cache_path.write_bytes(_download_bytes(cfg.ne_url_10m, timeout_s=60))

    world = gpd.read_file(f"zip://{str(cache_path)}!{cfg.ne_shp_10m}")
    try:
        world = world.to_crs("EPSG:4326")
    except Exception:
        pass

    _world10m_cache = world
    return world


def load_country_polygon_10m(cfg: PlotConfig, admin_name: str) -> Optional[Polygon | MultiPolygon]:
    """
    Load a country geometry from the Natural Earth 10 m dataset if available.

    Parameters
    ----------
    cfg:
        Plot configuration.
    admin_name:
        Country name used in the Natural Earth dataset.

    Returns
    -------
    Optional[Polygon | MultiPolygon]
        High-resolution country geometry, or ``None`` if loading fails or the
        country is missing.
    """
    world10m = load_world_countries_10m(cfg)
    name_col = next((c for c in ("ADMIN", "NAME", "name") if c in world10m.columns), None)
    if name_col is None:
        return None

    row = world10m[world10m[name_col] == admin_name].copy()
    if row.empty:
        return None

    geom = row.geometry.iloc[0]
    if geom is None or not isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty:
        return None
    return geom


# ============================================================
# Config loading from EU module
# ============================================================

def load_all_country_configs_from_module() -> Dict[str, EU.CountryConfig]:
    """
    Load all available ``CountryConfig`` objects from the imported EU module.

    The loader searches both dictionary-style module variables with names
    starting with ``COUNTRY_CONFIGS`` and callable builders with names starting
    with ``build_country_configs``.

    Returns
    -------
    Dict[str, EU.CountryConfig]
        Sorted mapping from country names to configuration objects.

    Raises
    ------
    RuntimeError
        If no valid country configurations can be found.
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
            except Exception:
                continue
            if isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, EU.CountryConfig):
                        configs[k] = v

    if not configs:
        raise RuntimeError("No CountryConfig objects found in EU module.")
    return dict(sorted(configs.items(), key=lambda kv: kv[0]))


# ============================================================
# Saved-data loading
# ============================================================

def load_saved_country_data(country_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load saved edge and node CSV files for a country.

    Parameters
    ----------
    country_dir:
        Directory containing saved simulation outputs for one country.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        Edge table and node table.

    Raises
    ------
    FileNotFoundError
        If one of the required CSV files is missing.
    ValueError
        If one of the tables does not contain the required columns.
    """
    edges_path = country_dir / "qkd_edges_km.csv"
    nodes_path = country_dir / "qkd_nodes_lonlat.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges CSV: {edges_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {nodes_path}")

    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(nodes_path)

    required_edge_cols = {"u_lon", "u_lat", "v_lon", "v_lat", "distance_km"}
    required_node_cols = {"type", "lon", "lat"}

    if not required_edge_cols.issubset(edges_df.columns):
        raise ValueError(f"Edges CSV missing required columns: {required_edge_cols - set(edges_df.columns)}")
    if not required_node_cols.issubset(nodes_df.columns):
        raise ValueError(f"Nodes CSV missing required columns: {required_node_cols - set(nodes_df.columns)}")

    return edges_df, nodes_df


def load_admin_name_from_saved_config(country_dir: Path, fallback_country_name: str) -> str:
    """
    Recover the Natural Earth administrative country name from saved metadata.

    The function reads ``configuration/config_meta.json`` if present and falls
    back to the provided country name otherwise.

    Parameters
    ----------
    country_dir:
        Country result directory.
    fallback_country_name:
        Country name used if no saved administrative name is available.

    Returns
    -------
    str
        Administrative country name.
    """
    meta_path = country_dir / "configuration" / "config_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            admin = str(meta.get("natural_earth_admin", "")).strip()
            if admin:
                return admin
        except Exception:
            pass
    return fallback_country_name


def get_plot_centers(country_name: str, country_cfg: Optional[EU.CountryConfig], admin_name: str, geom) -> Dict[str, Tuple[float, float]]:
    """
    Determine which center coordinates should be plotted for a country.

    Austria uses a fixed explicit center dictionary. Other countries draw
    centers from the loaded country configuration. For countries whose geometry
    is restricted to the mainland, centers are filtered so that only retained
    centers remain visible.

    Parameters
    ----------
    country_name:
        Display country name.
    country_cfg:
        Optional loaded country configuration.
    admin_name:
        Natural Earth administrative name.
    geom:
        Final geometry used for plotting.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping from center names to longitude/latitude tuples.
    """
    if admin_name == "Austria" or country_name == "Austria":
        return dict(_AUSTRIA_CAPITALS)

    centers = {}
    if country_cfg is not None:
        centers = dict(getattr(country_cfg, "centers_lonlat", {}) or {})

    if admin_name in MAINLAND_ONLY_ADMINS:
        centers = filter_centers_to_geometry(centers, geom)

    return centers


# ============================================================
# Plotting
# ============================================================

def plot_loaded_simulation(
    cfg: PlotConfig,
    out_png: Path,
    country_name: str,
    geom: Polygon | MultiPolygon,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    centers_lonlat: Dict[str, Tuple[float, float]],
) -> None:
    """
    Recreate a saved network plot from already exported node and edge tables.

    The function separates endpoints and trusted repeater nodes, reconstructs
    the visual layers, adds optional background neighboring countries, and
    saves the plot in PNG and EPS format.

    Parameters
    ----------
    cfg:
        Plot configuration.
    out_png:
        Output PNG path. An EPS file with the same basename is written as well.
    country_name:
        Display name used in the plot title.
    geom:
        Country geometry used for plotting.
    edges_df:
        Edge table loaded from saved results.
    nodes_df:
        Node table loaded from saved results.
    centers_lonlat:
        Mapping of plotting center names to coordinates.
    """
    endpoints_df = nodes_df[nodes_df["type"] == "QKD_ENDPOINT"].copy()
    trn_df = nodes_df[nodes_df["type"] == "TRUSTED_REPEATER_NODE"].copy()

    fiber_km = float(edges_df["distance_km"].sum())
    n_endpoints = int(endpoints_df.shape[0])
    n_trns = int(trn_df.shape[0])

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
    ax.set_facecolor("#f4f7fb")

    neighbors_gdf = None
    try:
        world10m = load_world_countries_10m(cfg)
        minx, miny, maxx, maxy = gpd.GeoSeries([geom], crs="EPSG:4326").total_bounds
        pad_x = max(0.4, 0.10 * (maxx - minx + 1e-9))
        pad_y = max(0.4, 0.10 * (maxy - miny + 1e-9))
        view_box = box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
        neighbors_gdf = world10m[world10m.geometry.intersects(view_box)].copy()
    except Exception:
        neighbors_gdf = None

    if neighbors_gdf is not None and not neighbors_gdf.empty:
        try:
            neighbors_gdf.plot(ax=ax, color="#eef2f6", edgecolor="#bfc7d1", linewidth=0.5, zorder=0)
        except Exception:
            pass

    gpd.GeoSeries([geom], crs="EPSG:4326").plot(
        ax=ax,
        color="none",
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )

    # Plot all previously saved network edges directly from the edge table.
    for _, row in edges_df.iterrows():
        ax.plot(
            [row["u_lon"], row["v_lon"]],
            [row["u_lat"], row["v_lat"]],
            color=PLOT_EDGE_COLOR,
            linewidth=0.6,
            linestyle="--",
            alpha=0.5,
            zorder=3,
        )

    # Plot QKD endpoint nodes.
    if not endpoints_df.empty:
        ax.scatter(
            endpoints_df["lon"].to_numpy(),
            endpoints_df["lat"].to_numpy(),
            s=18,
            c="blue",
            label="QKD Endpoints (blue)",
            zorder=4,
        )

    # Plot trusted repeater nodes.
    if not trn_df.empty:
        ax.scatter(
            trn_df["lon"].to_numpy(),
            trn_df["lat"].to_numpy(),
            s=20,
            c="red",
            label="Trusted Repeater Nodes (red)",
            zorder=5,
        )

    # Draw configured centers with labels and white outline effects to improve
    # readability against the background map.
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

    # Optionally annotate all endpoint identifiers. This is disabled by default
    # because dense national networks become visually cluttered quickly.
    if LABEL_ALL_QKD_ENDPOINTS and not endpoints_df.empty and "node_id" in endpoints_df.columns:
        for _, row in endpoints_df.iterrows():
            ax.text(row["lon"] + 0.01, row["lat"] + 0.01, str(row["node_id"]), fontsize=6 * FONT_SCALE, zorder=8)

    # Adjust the visible coordinate range while preserving the target figure
    # aspect ratio.
    try:
        minx, miny, maxx, maxy = gpd.GeoSeries([geom], crs="EPSG:4326").total_bounds
        pad_x = max(0.4, 0.08 * (maxx - minx + 1e-9))
        pad_y = max(0.4, 0.08 * (maxy - miny + 1e-9))

        x0 = minx - pad_x
        x1 = maxx + pad_x
        y0 = miny - pad_y
        y1 = maxy + pad_y

        width = x1 - x0
        height = y1 - y0

        target_ratio = FIGSIZE[0] / FIGSIZE[1]

        if height <= 0:
            height = 1e-9
        if width <= 0:
            width = 1e-9

        current_ratio = width / height

        if current_ratio < target_ratio:
            target_width = target_ratio * height
            extra = 0.5 * (target_width - width)
            x0 -= extra
            x1 += extra
        else:
            target_height = width / target_ratio
            extra = 0.5 * (target_height - height)
            y0 -= extra
            y1 += extra

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    except Exception:
        pass

    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_title(
        f"{country_name} QKD network simulation: {n_endpoints} endpoints, {n_trns} TRNs\n"
        f"Total fiber length: {1.5*fiber_km:.3f} ({fiber_km:.3f}; raw) km"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# One-country recreation
# ============================================================

def recreate_one_country_plot(
    cfg: PlotConfig,
    country_name: str,
    country_dir: Path,
    all_configs: Dict[str, EU.CountryConfig],
    plots_root: Path,
) -> None:
    """
    Recreate the saved plot for one country.

    The function loads the saved edge and node data, determines the appropriate
    country geometry, applies geometry stabilization when required, determines
    which center labels should be shown, and then writes the recreated plot.

    Parameters
    ----------
    cfg:
        Plot configuration.
    country_name:
        Display country name.
    country_dir:
        Directory containing the country's saved simulation outputs.
    all_configs:
        All loaded country configurations.
    plots_root:
        Root directory for recreated plots.
    """
    log(cfg, f"[plot] Recreating plot for {country_name}")

    edges_df, nodes_df = load_saved_country_data(country_dir)
    admin_name = load_admin_name_from_saved_config(country_dir, fallback_country_name=country_name)

    geom_110m = load_country_polygon(cfg, admin_name)
    geom_10m = load_country_polygon_10m(cfg, admin_name)
    geom = geom_10m if geom_10m is not None else geom_110m

    country_cfg = all_configs.get(country_name)

    if admin_name != "Austria" and country_name != "Austria":
        centers_for_geom = dict(getattr(country_cfg, "centers_lonlat", {}) or {}) if country_cfg is not None else {}
        geom = stabilize_eu_geometry(geom, centers_for_geom, admin_name)

    centers_lonlat = get_plot_centers(country_name, country_cfg, admin_name, geom)

    out_png = plots_root / f"qkd_simulation_plot_{safe_folder_name(country_name).lower()}.png"
    plot_loaded_simulation(
        cfg=cfg,
        out_png=out_png,
        country_name=country_name,
        geom=geom,
        edges_df=edges_df,
        nodes_df=nodes_df,
        centers_lonlat=centers_lonlat,
    )
    log(cfg, f"[out] Wrote {out_png}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Entry point for recreating all available saved plots.

    The function validates the results root, loads all country configurations,
    discovers available country directories, maps directory names back to
    configuration keys where possible, and recreates one plot per country.
    """
    cfg = PlotConfig()

    results_root = Path(cfg.results_root)
    plots_root = Path(cfg.plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    all_configs = load_all_country_configs_from_module()

    country_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir()]
    if not country_dirs:
        raise RuntimeError(f"No country directories found under {results_root}")

    for country_dir in country_dirs:
        country_name = country_dir.name.replace("_", " ")
        # Prefer exact folder name mapping back to config keys where possible
        matched_name = None
        for k in all_configs.keys():
            if safe_folder_name(k) == country_dir.name:
                matched_name = k
                break
        if matched_name is not None:
            country_name = matched_name

        try:
            recreate_one_country_plot(
                cfg=cfg,
                country_name=country_name,
                country_dir=country_dir,
                all_configs=all_configs,
                plots_root=plots_root,
            )
        except Exception as e:
            print(f"[WARN] Skipping {country_name}: {e!r}", flush=True)

    print(f"\n[done] Recreated plots written to: {plots_root.resolve()}")


if __name__ == "__main__":
    main()