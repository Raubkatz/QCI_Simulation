# EU QCI Terrestrial QKD Network Simulation — Experimental Code Repository

Author: Dr. Sebastian Raubitzek

## Overview

This repository implements a full Monte Carlo simulation and post-processing pipeline for  
**planning and analyzing terrestrial QKD backbone networks across EU member states (QCI context)**.

The code base provides a reproducible framework to:

- generate synthetic national QKD network topologies,
- allocate trusted repeater nodes,
- evaluate network statistics via Monte Carlo simulation,
- reconstruct visualizations from saved outputs,
- and aggregate results into publication-ready LaTeX tables.

The implementation is structured around a modular workflow:

- **Simulation core**: generation of QKD backbone networks based on country-specific configurations  
- **Configuration layer**: reproducible national parameter sets for all EU member states  
- **Post-processing**: plotting and statistical aggregation of simulation outputs  

All components are designed to work together without manual intervention once configured.

---

## Conceptual Structure

1. **Country-level configuration**
   - Population- and area-based scaling
   - Endpoint counts and trusted repeater counts
   - Center definitions and spatial sampling parameters
   - Degree distribution constraints

2. **Synthetic network generation**
   - Sampling of QKD endpoint locations (clustered + rural mixture)
   - Degree-constrained graph construction
   - Candidate graph generation and robustness-based selection

3. **Trusted repeater allocation**
   - Greedy allocation across longest edges
   - Reduction of maximum hop length
   - Segment-based hop statistics

4. **Monte Carlo evaluation**
   - Repeated simulation runs per country
   - Aggregation of:
     - fiber length,
     - hop length statistics,
     - node and edge counts

5. **Post-processing and visualization**
   - Reconstruction of network plots from saved outputs
   - Geometry stabilization using Natural Earth datasets

6. **Result aggregation**
   - Parsing simulation reports
   - Extraction of summary statistics
   - Generation of LaTeX tables for publication

---

## Repository Structure

├── 01_QCI_Simulation.py

├── 02_create_EUMS_plots.py

├── 03_LateX_Table_EUMS_Results.py

├── EU_MS_List.py

│

├── EUMS_QKD_Network_Results/ # simulation outputs per country, scripts create the folder

├── EUMS_QKD_Network_Plots/ # regenerated plots, scripts create the folder

└── README.md

---

## Core Scripts

### 1. QKD Network Simulation

python 01_QCI_Simulation.py

- Monte Carlo simulation framework for terrestrial QKD backbone planning :contentReference[oaicite:0]{index=0}
- Uses:
  - Natural Earth country geometries
  - country-specific configurations from `EU_MS_List.py`
- Generates:
  - endpoint sets,
  - degree-constrained graphs,
  - trusted repeater placements
- Outputs:
  - node tables,
  - edge tables,
  - plots,
  - simulation reports

**Key features:**
- Degree-constrained graph construction
- Robustness-based candidate selection
- Geodesic distance computation (WGS84)
- Detour-factor-based fiber estimation

---

### 2. Country Configuration Module

python EU_MS_List.py

- Defines structured country-specific simulation parameters
- Provides:
  - endpoint counts,
  - trusted repeater counts,
  - geographic centers,
  - spatial sampling parameters,
  - degree distributions

**Core functionality:**
- Scaling rules (relative to Austria reference case)
- Center-weight to absolute count conversion
- Unified configuration interface via `CountryConfig`

---

### 3. Plot Reconstruction

python 02_create_EUMS_plots.py

- Recreates plots from previously saved simulation outputs
- Does **not** rerun simulations
- Loads:
  - node CSVs,
  - edge CSVs,
  - country geometries

**Plot contents:**
- country boundaries,
- QKD endpoints,
- trusted repeater nodes,
- network edges,
- configured centers

**Output:**
- PNG and EPS plots per country

---

### 4. LaTeX Table Generation

python 03_LateX_Table_EUMS_Results.py

- Parses simulation report files and aggregates results
- Extracts:
  - endpoint counts,
  - TRN counts,
  - degree distributions,
  - fiber length statistics,
  - hop statistics

**Outputs:**
- `eums_summary_table.tex`
- `eums_summary_table_longtable.tex`

**Table structure:**
- One row per country (estimate)
- One row per country (standard deviation)
- Metrics:
  - mean hop length (raw / adjusted),
  - max hop length (raw / adjusted),
  - total fiber length (raw / adjusted)

---

## Simulation Workflow

### Step 1 — Run Simulation

python 01_QCI_Simulation.py

- Generates full Monte Carlo results for all configured countries
- Writes outputs

### Step 2 — Recreate Plots

python 02_create_EUMS_plots.py

- Reads saved results
- Generates plots

---

### Step 3 — Generate Summary Tables

python 03_LateX_Table_EUMS_Results.py

- Parses all country reports
- Writes LaTeX tables for publication

---

## Requirements

Dependencies:

- python==3.11.14
numpy==2.4.2
pandas==3.0.0
scipy==1.17.0
networkx==3.6.1
geopandas==1.1.2
shapely==2.1.2
pyproj==3.7.2
matplotlib==3.10.8
requests==2.32.5

---

## Notes

- The simulation is designed for **planning-level analysis**, not deployment design.
- All results depend on explicit, reproducible country configurations.
- Monte Carlo outputs include both:
  - raw geodesic metrics,
  - detour-adjusted estimates.
- Plot reconstruction and LaTeX generation operate entirely on saved outputs and do not require rerunning simulations.
