from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CountryConfig:
    iso2: str
    natural_earth_admin: str
    population_2025: int
    area_km2_2025: int
    n_endpoints: int
    k2_count: int
    k3_count: int
    n_trusted_repeater_nodes: int
    n_double_trn_edges: int
    detour_factor: float
    centers_lonlat: Dict[str, Tuple[float, float]]
    center_counts: Dict[str, int]
    center_radius_km: Dict[str, float]
    p_uniform_rural: float
    heavy_tail_scale_km: float
    heavy_tail_alpha: float
    min_rural_point_distance_km: float
    min_rural_distance_to_centers_km: float


def _scaled_endpoints(population: int, pop_at: int = 9_197_213, n_ep_at: int = 250) -> int:
    return max(50, int(round(n_ep_at * population / pop_at)))


def _scaled_trns(area_km2: int, area_at: int = 83_882, n_tr_at: int = 50) -> int:
    return max(5, int(round(n_tr_at * area_km2 / area_at)))


def _k2_k3_split(n_endpoints: int, frac_k2: float = 0.4) -> Tuple[int, int]:
    k2 = int(round(frac_k2 * n_endpoints))
    k2 = max(0, min(n_endpoints, k2))
    k3 = n_endpoints - k2
    return k2, k3


def _center_counts_from_weights(n_endpoints: int, weights: Dict[str, float]) -> Dict[str, int]:
    raw = {k: max(0, int(round(v * n_endpoints))) for k, v in weights.items()}
    s = sum(raw.values())
    if s > n_endpoints:
        factor = n_endpoints / s
        raw = {k: int(round(v * factor)) for k, v in raw.items()}
        s = sum(raw.values())
    return raw


def load_all_country_configs_from_module() -> Dict[str, CountryConfig]:
    detour = 1.5

    AT_POP = 9_197_213
    AT_AREA = 83_882
    AT_N_EP = 250
    AT_N_TR = 50
    AT_K2, AT_K3 = 100, 150

    austria_centers = {
        "Vienna": (16.363449, 48.210033),
        "St. Pölten": (15.633333, 48.200000),
        "Linz": (14.290000, 48.310000),
        "Salzburg": (13.040000, 47.800000),
        "Innsbruck": (11.390000, 47.260000),
        "Bregenz": (9.746000, 47.503000),
        "Eisenstadt": (16.523000, 47.846000),
        "Graz": (15.450000, 47.070000),
        "Klagenfurt": (14.310000, 46.620000),
        "St. Johann im Pongau": (13.200000, 47.350000),
    }
    austria_center_counts = {
        "Vienna": 25,
        "St. Pölten": 5,
        "Linz": 5,
        "Salzburg": 5,
        "Innsbruck": 5,
        "Bregenz": 5,
        "Eisenstadt": 5,
        "Graz": 5,
        "Klagenfurt": 5,
        "St. Johann im Pongau": 2,
    }
    austria_center_radii = {
        "Vienna": 30.0,
        "St. Johann im Pongau": 2.5,
        "__default__": 5.0,
    }

    configs: Dict[str, CountryConfig] = {}

    configs["Austria"] = CountryConfig(
        iso2="AT",
        natural_earth_admin="Austria",
        population_2025=AT_POP,
        area_km2_2025=AT_AREA,
        n_endpoints=AT_N_EP,
        k2_count=AT_K2,
        k3_count=AT_K3,
        n_trusted_repeater_nodes=AT_N_TR,
        n_double_trn_edges=round(0.2 * AT_N_TR),
        detour_factor=detour,
        centers_lonlat=austria_centers,
        center_counts=austria_center_counts,
        center_radius_km=austria_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=25.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    fr_pop, fr_area = 68_635_943, 638_475
    fr_n_ep = 800
    fr_n_tr = 200
    fr_k2, fr_k3 = _k2_k3_split(fr_n_ep)

    fr_centers = {
        "Paris": (2.3522, 48.8566),
        "Marseille": (5.3698, 43.2965),
        "Lyon": (4.8357, 45.7640),
        "Toulouse": (1.4442, 43.6047),
        "Nice": (7.2619, 43.7102),
        "Nantes": (-1.5536, 47.2184),
        "Strasbourg": (7.7521, 48.5734),
        "Montpellier": (3.8767, 43.6108),
        "Bordeaux": (-0.5792, 44.8378),
        "Lille": (3.0573, 50.6292),
        "Taverny (BA 921)": (2.2167, 49.0333),
    }

    fr_weights = {
        "Paris": 0.12,
        "Marseille": 0.03,
        "Lyon": 0.03,
        "Toulouse": 0.03,
        "Nice": 0.02,
        "Nantes": 0.02,
        "Strasbourg": 0.02,
        "Montpellier": 0.015,
        "Bordeaux": 0.02,
        "Lille": 0.02,
        "Taverny (BA 921)": 0.010,
    }
    fr_center_counts = _center_counts_from_weights(fr_n_ep, fr_weights)
    fr_center_radii = {"Paris": 35.0, "Taverny (BA 921)": 5.0, "__default__": 12.0}

    configs["France"] = CountryConfig(
        iso2="FR",
        natural_earth_admin="France",
        population_2025=fr_pop,
        area_km2_2025=fr_area,
        n_endpoints=fr_n_ep,
        k2_count=fr_k2,
        k3_count=fr_k3,
        n_trusted_repeater_nodes=fr_n_tr,
        n_double_trn_edges=round(0.2 * fr_n_tr),
        detour_factor=detour,
        centers_lonlat=fr_centers,
        center_counts=fr_center_counts,
        center_radius_km=fr_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=30.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    es_pop, es_area = 49_077_984, 505_983
    es_n_ep = 600
    es_n_tr = 300
    es_k2, es_k3 = _k2_k3_split(es_n_ep)

    es_centers = {
        "Madrid": (-3.7038, 40.4168),
        "Barcelona": (2.1734, 41.3851),
        "Valencia": (-0.3763, 39.4699),
        "Seville": (-5.9845, 37.3891),
        "Zaragoza": (-0.8891, 41.6488),
        "Málaga": (-4.4214, 36.7213),
        "Murcia": (-1.1307, 37.9922),
        "Palma": (2.6502, 39.5696),
        "Bilbao": (-2.9350, 43.2630),
        "Valladolid": (-4.7245, 41.6523),
        "Torrejón Air Base": (-3.4408, 40.4913),
    }

    es_weights = {
        "Madrid": 0.12,
        "Barcelona": 0.04,
        "Valencia": 0.03,
        "Seville": 0.03,
        "Zaragoza": 0.02,
        "Málaga": 0.02,
        "Murcia": 0.02,
        "Palma": 0.015,
        "Bilbao": 0.02,
        "Valladolid": 0.015,
        "Torrejón Air Base": 0.010,
    }
    es_center_counts = _center_counts_from_weights(es_n_ep, es_weights)
    es_center_radii = {"Madrid": 35.0, "Torrejón Air Base": 5.0, "__default__": 12.0}

    configs["Spain"] = CountryConfig(
        iso2="ES",
        natural_earth_admin="Spain",
        population_2025=es_pop,
        area_km2_2025=es_area,
        n_endpoints=es_n_ep,
        k2_count=es_k2,
        k3_count=es_k3,
        n_trusted_repeater_nodes=es_n_tr,
        n_double_trn_edges=round(0.2 * es_n_tr),
        detour_factor=detour,
        centers_lonlat=es_centers,
        center_counts=es_center_counts,
        center_radius_km=es_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=35.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )

    it_pop, it_area = 58_934_177, 302_073
    it_n_ep = 600
    it_n_tr = 150
    it_k2, it_k3 = _k2_k3_split(it_n_ep)

    it_centers = {
        "Rome": (12.4964, 41.9028),
        "Milan": (9.1900, 45.4642),
        "Naples": (14.2681, 40.8518),
        "Turin": (7.6869, 45.0703),
        "Palermo": (13.3615, 38.1157),
        "Genoa": (8.9463, 44.4056),
        "Bologna": (11.3426, 44.4949),
        "Florence": (11.2558, 43.7696),
        "Bari": (16.8719, 41.1171),
        "Catania": (15.0873, 37.5079),
        "NAS Sigonella": (14.9200, 37.4010),
    }

    it_weights = {
        "Rome": 0.12,
        "Milan": 0.04,
        "Naples": 0.03,
        "Turin": 0.02,
        "Palermo": 0.02,
        "Genoa": 0.02,
        "Bologna": 0.02,
        "Florence": 0.02,
        "Bari": 0.02,
        "Catania": 0.015,
        "NAS Sigonella": 0.010,
    }
    it_center_counts = _center_counts_from_weights(it_n_ep, it_weights)
    it_center_radii = {"Rome": 35.0, "NAS Sigonella": 6.0, "__default__": 12.0}

    configs["Italy"] = CountryConfig(
        iso2="IT",
        natural_earth_admin="Italy",
        population_2025=it_pop,
        area_km2_2025=it_area,
        n_endpoints=it_n_ep,
        k2_count=it_k2,
        k3_count=it_k3,
        n_trusted_repeater_nodes=it_n_tr,
        n_double_trn_edges=round(0.2 * it_n_tr),
        detour_factor=detour,
        centers_lonlat=it_centers,
        center_counts=it_center_counts,
        center_radius_km=it_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=30.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )

    hr_pop, hr_area, hr_ep, hr_tr = 3_874_000, 56_594, 120, 30
    hr_k2, hr_k3 = _k2_k3_split(hr_ep)
    hr_centers = {
        "Zagreb": (15.9819, 45.8150),
        "Split": (16.4402, 43.5081),
        "Rijeka": (14.4422, 45.3271),
        "Osijek": (18.6955, 45.5540),
        "Zadar": (15.2314, 44.1194),
        "Pula": (13.8496, 44.8666),
        "Slavonski Brod": (18.0186, 45.1603),
        "Dubrovnik": (18.0944, 42.6507),
        "Zagreb/Pleso Air Base": (16.0679, 45.7392),
    }
    hr_weights = {
        "Zagreb": 0.16,
        "Split": 0.04,
        "Rijeka": 0.025,
        "Osijek": 0.025,
        "Zadar": 0.02,
        "Pula": 0.015,
        "Slavonski Brod": 0.015,
        "Dubrovnik": 0.015,
        "Zagreb/Pleso Air Base": 0.01,
    }
    configs["Croatia"] = CountryConfig(
        iso2="HR",
        natural_earth_admin="Croatia",
        population_2025=hr_pop,
        area_km2_2025=hr_area,
        n_endpoints=hr_ep,
        k2_count=hr_k2,
        k3_count=hr_k3,
        n_trusted_repeater_nodes=hr_tr,
        n_double_trn_edges=round(0.2 * hr_tr),
        detour_factor=detour,
        centers_lonlat=hr_centers,
        center_counts=_center_counts_from_weights(hr_ep, hr_weights),
        center_radius_km={"Zagreb": 18.0, "Zagreb/Pleso Air Base": 5.0, "__default__": 10.0},
        p_uniform_rural=0.70,
        heavy_tail_scale_km=25.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    gr_pop, gr_area = 10_410_000, 131_694
    gr_n_ep, gr_n_tr = 280, 80
    gr_k2, gr_k3 = _k2_k3_split(gr_n_ep)
    gr_centers = {
        "Athens": (23.7275, 37.9838),
        "Thessaloniki": (22.9444, 40.6401),
        "Patras": (21.7346, 38.2466),
        "Larissa": (22.4200, 39.6390),
        "Heraklion": (25.1442, 35.3387),
        "Volos": (22.9430, 39.3610),
        "Ioannina": (20.8520, 39.6650),
        "Alexandroupoli": (25.8730, 40.8480),
        "Rhodes": (28.2270, 36.4340),
        "Souda Bay Naval Base": (24.0734, 35.4872),
    }
    gr_weights = {
        "Athens": 0.12,
        "Thessaloniki": 0.04,
        "Patras": 0.03,
        "Larissa": 0.02,
        "Heraklion": 0.02,
        "Volos": 0.015,
        "Ioannina": 0.015,
        "Alexandroupoli": 0.015,
        "Rhodes": 0.015,
        "Souda Bay Naval Base": 0.010,
    }
    gr_center_counts = _center_counts_from_weights(gr_n_ep, gr_weights)
    gr_center_radii = {"Athens": 30.0, "Souda Bay Naval Base": 6.0, "__default__": 12.0}
    configs["Greece"] = CountryConfig(
        iso2="GR",
        natural_earth_admin="Greece",
        population_2025=gr_pop,
        area_km2_2025=gr_area,
        n_endpoints=gr_n_ep,
        k2_count=gr_k2,
        k3_count=gr_k3,
        n_trusted_repeater_nodes=gr_n_tr,
        n_double_trn_edges=round(0.2 * gr_n_tr),
        detour_factor=detour,
        centers_lonlat=gr_centers,
        center_counts=gr_center_counts,
        center_radius_km=gr_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=35.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )

    pt_pop, pt_area = 10_750_000, 92_227
    pt_n_ep, pt_n_tr = 260, 60
    pt_k2, pt_k3 = _k2_k3_split(pt_n_ep)
    pt_centers = {
        "Lisbon": (-9.1393, 38.7223),
        "Porto": (-8.6291, 41.1579),
        "Braga": (-8.4265, 41.5454),
        "Coimbra": (-8.4292, 40.2033),
        "Aveiro": (-8.6455, 40.6405),
        "Faro": (-7.9304, 37.0194),
        "Funchal (Madeira)": (-16.9255, 32.6669),
        "Ponta Delgada (Azores)": (-25.6666, 37.7412),
        "Évora": (-7.9137, 38.5714),
        "Monte Real Air Base": (-8.8667, 39.8333),
    }
    pt_weights = {
        "Lisbon": 0.14,
        "Porto": 0.05,
        "Braga": 0.02,
        "Coimbra": 0.03,
        "Aveiro": 0.02,
        "Faro": 0.02,
        "Funchal (Madeira)": 0.02,
        "Ponta Delgada (Azores)": 0.01,
        "Évora": 0.01,
        "Monte Real Air Base": 0.01,
    }
    pt_center_counts = _center_counts_from_weights(pt_n_ep, pt_weights)
    pt_center_radii = {"Lisbon": 25.0, "Porto": 18.0, "Monte Real Air Base": 6.0, "__default__": 10.0}
    configs["Portugal"] = CountryConfig(
        iso2="PT",
        natural_earth_admin="Portugal",
        population_2025=pt_pop,
        area_km2_2025=pt_area,
        n_endpoints=pt_n_ep,
        k2_count=pt_k2,
        k3_count=pt_k3,
        n_trusted_repeater_nodes=pt_n_tr,
        n_double_trn_edges=round(0.2 * pt_n_tr),
        detour_factor=detour,
        centers_lonlat=pt_centers,
        center_counts=pt_center_counts,
        center_radius_km=pt_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=28.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )


    dk_pop, dk_area = 5_993_000, 42_925
    dk_n_ep, dk_n_tr = 200, 40
    dk_k2, dk_k3 = _k2_k3_split(dk_n_ep)
    dk_centers = {
        "Copenhagen": (12.5683, 55.6761),
        "Aarhus": (10.2039, 56.1629),
        "Odense": (10.3883, 55.4038),
        "Aalborg": (9.9217, 57.0488),
        "Esbjerg": (8.4519, 55.4765),
        "Skrydstrup Air Base": (9.2670, 55.2210),
    }

    dk_weights = {
        "Copenhagen": 0.14,
        "Aarhus": 0.04,
        "Odense": 0.03,
        "Aalborg": 0.03,
        "Esbjerg": 0.02,
        "Skrydstrup Air Base": 0.01,
    }
    dk_center_counts = _center_counts_from_weights(dk_n_ep, dk_weights)
    dk_center_radii = {"Copenhagen": 20.0, "Skrydstrup Air Base": 2.5, "__default__": 10.0}
    configs["Denmark"] = CountryConfig(
        iso2="DK",
        natural_earth_admin="Denmark",
        population_2025=dk_pop,
        area_km2_2025=dk_area,
        n_endpoints=dk_n_ep,
        k2_count=dk_k2,
        k3_count=dk_k3,
        n_trusted_repeater_nodes=dk_n_tr,
        n_double_trn_edges=round(0.2 * dk_n_tr),
        detour_factor=detour,
        centers_lonlat=dk_centers,
        center_counts=dk_center_counts,
        center_radius_km=dk_center_radii,
        p_uniform_rural=0.60,
        heavy_tail_scale_km=20.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )

    de_pop, de_area = 83_577_140, 357_569
    de_n_ep = 1000
    de_n_tr = 300
    de_k2, de_k3 = _k2_k3_split(de_n_ep)

    de_centers = {
        "Berlin": (13.4050, 52.5200),
        "Hamburg": (9.9937, 53.5511),
        "Munich": (11.5820, 48.1351),
        "Cologne": (6.9603, 50.9375),
        "Frankfurt am Main": (8.6821, 50.1109),
        "Stuttgart": (9.1829, 48.7758),
        "Düsseldorf": (6.7735, 51.2277),
        "Dresden": (13.7373, 51.0504),
        "Hannover": (9.7320, 52.3759),
        "Kiel": (10.1228, 54.3233),
        "Mainz": (8.2473, 49.9929),
        "Potsdam": (13.0645, 52.3906),
        "Magdeburg": (11.6350, 52.1205),
        "Erfurt": (11.0299, 50.9848),
        "Wiesbaden": (8.2398, 50.0782),
        "Saarbrücken": (6.9969, 49.2402),
        "Bonn (KdoCIR)": (7.0982, 50.7374),
    }

    de_weights = {
        "Berlin": 0.12,
        "Hamburg": 0.03,
        "Munich": 0.03,
        "Cologne": 0.03,
        "Frankfurt am Main": 0.03,
        "Stuttgart": 0.02,
        "Düsseldorf": 0.02,
        "Dresden": 0.015,
        "Hannover": 0.015,
        "Kiel": 0.010,
        "Mainz": 0.010,
        "Potsdam": 0.008,
        "Magdeburg": 0.008,
        "Erfurt": 0.008,
        "Wiesbaden": 0.008,
        "Saarbrücken": 0.008,
        "Bonn (KdoCIR)": 0.010,
    }
    de_center_counts = _center_counts_from_weights(de_n_ep, de_weights)
    de_center_radii = {"Berlin": 35.0, "Bonn (KdoCIR)": 5.0, "__default__": 10.0}

    configs["Germany"] = CountryConfig(
        iso2="DE",
        natural_earth_admin="Germany",
        population_2025=de_pop,
        area_km2_2025=de_area,
        n_endpoints=de_n_ep,
        k2_count=de_k2,
        k3_count=de_k3,
        n_trusted_repeater_nodes=de_n_tr,
        n_double_trn_edges=round(0.2 * de_n_tr),
        detour_factor=detour,
        centers_lonlat=de_centers,
        center_counts=de_center_counts,
        center_radius_km=de_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=25.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    fi_pop, fi_area = 5_636_000, 338_411
    fi_n_ep, fi_n_tr = 250, 150
    fi_k2, fi_k3 = _k2_k3_split(fi_n_ep)

    fi_centers = {
        "Helsinki": (24.9458, 60.1921),
        "Espoo": (24.6559, 60.2055),
        "Tampere": (23.7610, 61.4978),
        "Turku": (22.2666, 60.4518),
        "Oulu": (25.4717, 65.0121),
        "Kuopio": (27.6770, 62.8924),
        "Jyväskylä": (25.7473, 62.2415),
        "Rovaniemi": (25.7294, 66.5039),
        "Rovaniemi Air Base": (25.8308, 66.5617),
    }
    fi_weights = {
        "Helsinki": 0.12,
        "Espoo": 0.03,
        "Tampere": 0.03,
        "Turku": 0.03,
        "Oulu": 0.02,
        "Kuopio": 0.02,
        "Jyväskylä": 0.02,
        "Rovaniemi": 0.015,
        "Rovaniemi Air Base": 0.010,
    }
    fi_center_counts = _center_counts_from_weights(fi_n_ep, fi_weights)
    fi_center_radii = {"Helsinki": 25.0, "Rovaniemi Air Base": 6.0, "__default__": 12.0}
    configs["Finland"] = CountryConfig(
        iso2="FI",
        natural_earth_admin="Finland",
        population_2025=fi_pop,
        area_km2_2025=fi_area,
        n_endpoints=fi_n_ep,
        k2_count=fi_k2,
        k3_count=fi_k3,
        n_trusted_repeater_nodes=fi_n_tr,
        n_double_trn_edges=round(0.2 * fi_n_tr),
        detour_factor=detour,
        centers_lonlat=fi_centers,
        center_counts=fi_center_counts,
        center_radius_km=fi_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=45.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )


    pl_pop, pl_area = 36_497_000, 311_928
    pl_n_ep, pl_n_tr = 650, 200
    pl_k2, pl_k3 = _k2_k3_split(pl_n_ep)

    pl_centers = {
        "Warsaw": (21.0122, 52.2297),
        "Kraków": (19.9445, 50.0647),
        "Łódź": (19.4550, 51.7592),
        "Wrocław": (17.0385, 51.1079),
        "Poznań": (16.9252, 52.4064),
        "Gdańsk": (18.6466, 54.3520),
        "Szczecin": (14.5528, 53.4285),
        "Lublin": (22.5684, 51.2465),
        "Katowice": (19.0238, 50.2649),
        "Białystok": (23.1688, 53.1325),
        "Rzeszów": (21.9991, 50.0413),
        "Powidz Air Base": (17.8544, 52.3831),
    }
    pl_weights = {
        "Warsaw": 0.08,
        "Kraków": 0.03,
        "Łódź": 0.025,
        "Wrocław": 0.025,
        "Poznań": 0.025,
        "Gdańsk": 0.02,
        "Szczecin": 0.015,
        "Lublin": 0.015,
        "Katowice": 0.02,
        "Białystok": 0.012,
        "Rzeszów": 0.012,
        "Powidz Air Base": 0.008,
    }
    pl_center_counts = _center_counts_from_weights(pl_n_ep, pl_weights)
    pl_center_radii = {"Warsaw": 35.0, "Powidz Air Base": 6.0, "__default__": 14.0}

    configs["Poland"] = CountryConfig(
        iso2="PL",
        natural_earth_admin="Poland",
        population_2025=pl_pop,
        area_km2_2025=pl_area,
        n_endpoints=pl_n_ep,
        k2_count=pl_k2,
        k3_count=pl_k3,
        n_trusted_repeater_nodes=pl_n_tr,
        n_double_trn_edges=round(0.2 * pl_n_tr),
        detour_factor=detour,
        centers_lonlat=pl_centers,
        center_counts=pl_center_counts,
        center_radius_km=pl_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=35.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )

    ro_pop, ro_area = 19_036_000, 238_398
    ro_n_ep, ro_n_tr = 550, 180
    ro_k2, ro_k3 = _k2_k3_split(ro_n_ep)
    ro_centers = {
        "Bucharest": (26.1025, 44.4268),
        "Cluj-Napoca": (23.6236, 46.7712),
        "Timișoara": (21.2087, 45.7489),
        "Iași": (27.5879, 47.1585),
        "Constanța": (28.6348, 44.1598),
        "Craiova": (23.7949, 44.3302),
        "Brașov": (25.6012, 45.6579),
        "Galați": (28.0360, 45.4353),
        "Oradea": (21.9364, 47.0465),
        "Sibiu": (24.1434, 45.7936),
        "RoAF 57th Air Base (Mihail Kogălniceanu)": (28.4870, 44.3630),
    }
    ro_weights = {
        "Bucharest": 0.14,
        "Cluj-Napoca": 0.03,
        "Timișoara": 0.03,
        "Iași": 0.03,
        "Constanța": 0.03,
        "Craiova": 0.02,
        "Brașov": 0.02,
        "Galați": 0.02,
        "Oradea": 0.02,
        "Sibiu": 0.015,
        "RoAF 57th Air Base (Mihail Kogălniceanu)": 0.01,
    }
    ro_center_counts = _center_counts_from_weights(ro_n_ep, ro_weights)
    ro_center_radii = {"Bucharest": 30.0, "RoAF 57th Air Base (Mihail Kogălniceanu)": 6.0, "__default__": 12.0}
    configs["Romania"] = CountryConfig(
        iso2="RO",
        natural_earth_admin="Romania",
        population_2025=ro_pop,
        area_km2_2025=ro_area,
        n_endpoints=ro_n_ep,
        k2_count=ro_k2,
        k3_count=ro_k3,
        n_trusted_repeater_nodes=ro_n_tr,
        n_double_trn_edges=round(0.2 * ro_n_tr),
        detour_factor=detour,
        centers_lonlat=ro_centers,
        center_counts=ro_center_counts,
        center_radius_km=ro_center_radii,
        p_uniform_rural=0.70,
        heavy_tail_scale_km=30.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=12.0,
        min_rural_distance_to_centers_km=12.0,
    )

    se_pop, se_area = 10_588_000, 447_424
    se_n_ep, se_n_tr = 280, 180
    se_k2, se_k3 = _k2_k3_split(se_n_ep)
    se_centers = {
        "Stockholm": (18.0686, 59.3293),
        "Gothenburg": (11.9746, 57.7089),
        "Malmö": (13.0038, 55.6050),
        "Uppsala": (17.6389, 59.8586),
        "Linköping": (15.6214, 58.4108),
        "Örebro": (15.2134, 59.2753),
        "Västerås": (16.5448, 59.6099),
        "Helsingborg": (12.6945, 56.0465),
        "Luleå": (22.1547, 65.5848),
        "Umeå": (20.2630, 63.8258),
        "Muskö naval base": (18.1000, 58.9500),
    }
    se_weights = {
        "Stockholm": 0.14,
        "Gothenburg": 0.05,
        "Malmö": 0.04,
        "Uppsala": 0.02,
        "Linköping": 0.02,
        "Örebro": 0.02,
        "Västerås": 0.02,
        "Helsingborg": 0.02,
        "Luleå": 0.02,
        "Umeå": 0.02,
        "Muskö naval base": 0.01,
    }
    se_center_counts = _center_counts_from_weights(se_n_ep, se_weights)
    se_center_radii = {"Stockholm": 28.0, "Muskö naval base": 6.0, "__default__": 14.0}
    configs["Sweden"] = CountryConfig(
        iso2="SE",
        natural_earth_admin="Sweden",
        population_2025=se_pop,
        area_km2_2025=se_area,
        n_endpoints=se_n_ep,
        k2_count=se_k2,
        k3_count=se_k3,
        n_trusted_repeater_nodes=se_n_tr,
        n_double_trn_edges=round(0.2 * se_n_tr),
        detour_factor=detour,
        centers_lonlat=se_centers,
        center_counts=se_center_counts,
        center_radius_km=se_center_radii,
        p_uniform_rural=0.75,
        heavy_tail_scale_km=40.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=14.0,
        min_rural_distance_to_centers_km=14.0,
    )


    nl_pop, nl_area = 18_044_027, 37_391
    nl_n_ep = _scaled_endpoints(nl_pop)
    nl_n_tr = _scaled_trns(nl_area)
    nl_k2, nl_k3 = _k2_k3_split(nl_n_ep)

    nl_centers = {
        "Amsterdam": (4.9041, 52.3676),
        "The Hague": (4.3007, 52.0705),
        "Rotterdam": (4.4792, 51.9244),
        "Utrecht": (5.1214, 52.0907),
        "Eindhoven": (5.4790, 51.4416),
        "Groningen": (6.5665, 53.2194),
        "Maastricht": (5.6900, 50.8514),
        "Arnhem": (5.8987, 51.9851),
        "Zwolle": (6.0945, 52.5168),
        "Leeuwarden": (5.7999, 53.2012),
        "The Hague (HSD Campus)": (4.3007, 52.0705),
    }

    nl_weights = {
        "Amsterdam": 0.12,
        "The Hague": 0.06,
        "Rotterdam": 0.06,
        "Utrecht": 0.04,
        "Eindhoven": 0.03,
        "Groningen": 0.02,
        "Maastricht": 0.02,
        "Arnhem": 0.02,
        "Zwolle": 0.02,
        "Leeuwarden": 0.02,
        "The Hague (HSD Campus)": 0.01,
    }
    nl_center_counts = _center_counts_from_weights(nl_n_ep, nl_weights)
    nl_center_radii = {"Amsterdam": 20.0, "The Hague": 15.0, "__default__": 8.0}

    configs["Netherlands"] = CountryConfig(
        iso2="NL",
        natural_earth_admin="Netherlands",
        population_2025=nl_pop,
        area_km2_2025=nl_area,
        n_endpoints=nl_n_ep,
        k2_count=nl_k2,
        k3_count=nl_k3,
        n_trusted_repeater_nodes=nl_n_tr,
        n_double_trn_edges=round(0.2 * nl_n_tr),
        detour_factor=detour,
        centers_lonlat=nl_centers,
        center_counts=nl_center_counts,
        center_radius_km=nl_center_radii,
        p_uniform_rural=0.60,
        heavy_tail_scale_km=20.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )


    hu_pop, hu_area = 9_540_000, 93_012
    hu_n_ep, hu_n_tr = 260, 60
    hu_k2, hu_k3 = _k2_k3_split(hu_n_ep)
    hu_centers = {
        "Budapest": (19.0402, 47.4979),
        "Debrecen": (21.6273, 47.5316),
        "Szeged": (20.1414, 46.2530),
        "Miskolc": (20.7900, 48.1035),
        "Pécs": (18.2323, 46.0727),
        "Győr": (17.6504, 47.6875),
        "Kecskemét": (19.6913, 46.9062),
        "Nyíregyháza": (21.7244, 47.9554),
        "Székesfehérvár": (18.4080, 47.1860),
        "Pápa Air Base": (17.5008, 47.3636),
    }
    hu_weights = {
        "Budapest": 0.12,
        "Debrecen": 0.03,
        "Szeged": 0.03,
        "Miskolc": 0.03,
        "Pécs": 0.02,
        "Győr": 0.02,
        "Kecskemét": 0.02,
        "Nyíregyháza": 0.02,
        "Székesfehérvár": 0.02,
        "Pápa Air Base": 0.010,
    }
    hu_center_counts = _center_counts_from_weights(hu_n_ep, hu_weights)
    hu_center_radii = {"Budapest": 25.0, "Pápa Air Base": 6.0, "__default__": 12.0}
    configs["Hungary"] = CountryConfig(
        iso2="HU",
        natural_earth_admin="Hungary",
        population_2025=hu_pop,
        area_km2_2025=hu_area,
        n_endpoints=hu_n_ep,
        k2_count=hu_k2,
        k3_count=hu_k3,
        n_trusted_repeater_nodes=hu_n_tr,
        n_double_trn_edges=round(0.2 * hu_n_tr),
        detour_factor=detour,
        centers_lonlat=hu_centers,
        center_counts=hu_center_counts,
        center_radius_km=hu_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=25.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    lv_pop, lv_area = 1_857_000, 64_586
    lv_n_ep, lv_n_tr = 50, 40
    lv_k2, lv_k3 = _k2_k3_split(lv_n_ep)

    lv_centers = {
        "Riga": (24.1052, 56.9496),
        "Daugavpils": (26.5350, 55.8750),
        "Liepāja": (21.0119, 56.5047),
        "Jelgava": (23.7221, 56.6511),
        "Ventspils": (21.5606, 57.3937),
        "Rēzekne": (27.3274, 56.5099),
        "Ādaži Military Base": (24.4111, 57.1200),
    }
    lv_weights = {
        "Riga": 0.16,
        "Daugavpils": 0.03,
        "Liepāja": 0.03,
        "Jelgava": 0.02,
        "Ventspils": 0.02,
        "Rēzekne": 0.02,
        "Ādaži Military Base": 0.02,
    }
    lv_center_counts = _center_counts_from_weights(lv_n_ep, lv_weights)
    lv_center_radii = {"Riga": 18.0, "Ādaži Military Base": 4.0, "__default__": 8.0}

    configs["Latvia"] = CountryConfig(
        iso2="LV",
        natural_earth_admin="Latvia",
        population_2025=lv_pop,
        area_km2_2025=lv_area,
        n_endpoints=lv_n_ep,
        k2_count=lv_k2,
        k3_count=lv_k3,
        n_trusted_repeater_nodes=lv_n_tr,
        n_double_trn_edges=round(0.2 * lv_n_tr),
        detour_factor=detour,
        centers_lonlat=lv_centers,
        center_counts=lv_center_counts,
        center_radius_km=lv_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=22.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )

    lt_pop, lt_area = 2_891_000, 65_284
    lt_n_ep, lt_n_tr = 80, 40
    lt_k2, lt_k3 = _k2_k3_split(lt_n_ep)

    lt_centers = {
        "Vilnius": (25.2797, 54.6872),
        "Kaunas": (23.9036, 54.8985),
        "Klaipėda": (21.1175, 55.7033),
        "Šiauliai": (23.3167, 55.9333),
        "Panevėžys": (24.3573, 55.7348),
        "Alytus": (24.0492, 54.3964),
        "Rukla": (24.3969, 55.0531),
    }
    lt_weights = {
        "Vilnius": 0.14,
        "Kaunas": 0.05,
        "Klaipėda": 0.03,
        "Šiauliai": 0.02,
        "Panevėžys": 0.02,
        "Alytus": 0.015,
        "Rukla": 0.02,
    }
    lt_center_counts = _center_counts_from_weights(lt_n_ep, lt_weights)
    lt_center_radii = {"Vilnius": 18.0, "Rukla": 5.0, "__default__": 9.0}

    configs["Lithuania"] = CountryConfig(
        iso2="LT",
        natural_earth_admin="Lithuania",
        population_2025=lt_pop,
        area_km2_2025=lt_area,
        n_endpoints=lt_n_ep,
        k2_count=lt_k2,
        k3_count=lt_k3,
        n_trusted_repeater_nodes=lt_n_tr,
        n_double_trn_edges=round(0.2 * lt_n_tr),
        detour_factor=detour,
        centers_lonlat=lt_centers,
        center_counts=lt_center_counts,
        center_radius_km=lt_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=22.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )

    lu_pop, lu_area = 682_000, 2_595
    lu_n_ep, lu_n_tr = 20, 4
    lu_k2, lu_k3 = _k2_k3_split(lu_n_ep)

    lu_centers = {
        "Luxembourg": (6.1319, 49.6116),
        "Esch-sur-Alzette": (5.9806, 49.4958),
        "Differdange": (5.8914, 49.5242),
        "Dudelange": (6.0875, 49.4794),
        "Ettelbruck": (6.1056, 49.8475),
        "Diekirch": (6.1558, 49.8672),
        "Betzdorf (SES)": (6.3497, 49.6873),
    }
    lu_weights = {
        "Luxembourg": 0.20,
        "Esch-sur-Alzette": 0.05,
        "Differdange": 0.04,
        "Dudelange": 0.03,
        "Ettelbruck": 0.02,
        "Diekirch": 0.02,
        "Betzdorf (SES)": 0.03,
    }
    lu_center_counts = _center_counts_from_weights(lu_n_ep, lu_weights)
    lu_center_radii = {"Luxembourg": 10.0, "Betzdorf (SES)": 3.0, "__default__": 6.0}

    configs["Luxembourg"] = CountryConfig(
        iso2="LU",
        natural_earth_admin="Luxembourg",
        population_2025=lu_pop,
        area_km2_2025=lu_area,
        n_endpoints=lu_n_ep,
        k2_count=lu_k2,
        k3_count=lu_k3,
        n_trusted_repeater_nodes=lu_n_tr,
        n_double_trn_edges=round(0.2 * lu_n_tr),
        detour_factor=detour,
        centers_lonlat=lu_centers,
        center_counts=lu_center_counts,
        center_radius_km=lu_center_radii,
        p_uniform_rural=0.55,
        heavy_tail_scale_km=10.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=5.0,
        min_rural_distance_to_centers_km=5.0,
    )




    mt_pop, mt_area = 574_000, 316
    mt_n_ep, mt_n_tr = 30, 0
    mt_k2, mt_k3 = _k2_k3_split(mt_n_ep)

    mt_centers = {
        "Valletta": (14.5147, 35.8989),
        "Sliema": (14.5069, 35.9122),
        "Birkirkara": (14.4611, 35.8956),
        "St Julian's": (14.4890, 35.9189),
        "Mosta": (14.4256, 35.9097),
        "Rabat": (14.4000, 35.8833),
        "Luqa / Malta International Airport": (14.4775, 35.8575),
        "Victoria (Gozo)": (14.2397, 36.0443),
    }

    mt_weights = {
        "Valletta": 0.10,
        "Sliema": 0.05,
        "Birkirkara": 0.06,
        "St Julian's": 0.04,
        "Mosta": 0.04,
        "Rabat": 0.03,
        "Luqa / Malta International Airport": 0.04,
        "Victoria (Gozo)": 0.05,
    }

    mt_center_counts = _center_counts_from_weights(mt_n_ep, mt_weights)

    mt_center_radii = {
        "Valletta": 4.0,
        "Luqa / Malta International Airport": 2.5,
        "Victoria (Gozo)": 3.0,
        "__default__": 3.5,
    }

    configs["Malta"] = CountryConfig(
        iso2="MT",
        natural_earth_admin="Malta",
        population_2025=mt_pop,
        area_km2_2025=mt_area,
        n_endpoints=mt_n_ep,
        k2_count=mt_k2,
        k3_count=mt_k3,
        n_trusted_repeater_nodes=mt_n_tr,
        n_double_trn_edges=0,
        detour_factor=detour,
        centers_lonlat=mt_centers,
        center_counts=mt_center_counts,
        center_radius_km=mt_center_radii,
        p_uniform_rural=0.45,
        heavy_tail_scale_km=3.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=2.0,
        min_rural_distance_to_centers_km=2.0,
    )

    sk_pop, sk_area = 5_419_000, 49_035
    sk_n_ep, sk_n_tr = 160, 30
    sk_k2, sk_k3 = _k2_k3_split(sk_n_ep)
    sk_centers = {
        "Bratislava": (17.1077, 48.1486),
        "Košice": (21.2611, 48.7164),
        "Prešov": (21.2399, 48.9984),
        "Žilina": (18.7394, 49.2232),
        "Nitra": (18.0870, 48.3064),
        "Banská Bystrica": (19.1462, 48.7363),
        "Trnava": (17.5872, 48.3774),
        "Trenčín": (18.0444, 48.8945),
        "Martin": (18.9211, 49.0665),
        "Poprad": (20.2986, 49.0564),
        "Sliač Airport": (19.1333, 48.6333),
    }
    sk_weights = {
        "Bratislava": 0.14,
        "Košice": 0.05,
        "Prešov": 0.02,
        "Žilina": 0.02,
        "Nitra": 0.02,
        "Banská Bystrica": 0.02,
        "Trnava": 0.02,
        "Trenčín": 0.02,
        "Martin": 0.01,
        "Poprad": 0.01,
        "Sliač Airport": 0.01,
    }
    sk_center_counts = _center_counts_from_weights(sk_n_ep, sk_weights)
    sk_center_radii = {"Bratislava": 20.0, "Sliač Airport": 6.0, "__default__": 9.0}
    configs["Slovakia"] = CountryConfig(
        iso2="SK",
        natural_earth_admin="Slovakia",
        population_2025=sk_pop,
        area_km2_2025=sk_area,
        n_endpoints=sk_n_ep,
        k2_count=sk_k2,
        k3_count=sk_k3,
        n_trusted_repeater_nodes=sk_n_tr,
        n_double_trn_edges=round(0.2 * sk_n_tr),
        detour_factor=detour,
        centers_lonlat=sk_centers,
        center_counts=sk_center_counts,
        center_radius_km=sk_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=24.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )

    si_pop, si_area = 2_131_000, 20_273
    si_n_ep, si_n_tr = 60, 14
    si_k2, si_k3 = _k2_k3_split(si_n_ep)
    si_centers = {
        "Ljubljana": (14.5058, 46.0569),
        "Maribor": (15.6459, 46.5547),
        "Celje": (15.2675, 46.2397),
        "Kranj": (14.3556, 46.2389),
        "Koper": (13.7294, 45.5481),
        "Novo Mesto": (15.1689, 45.8030),
        "Nova Gorica": (13.6436, 45.9560),
        "Murska Sobota": (16.1664, 46.6625),
        "Ptuj": (15.8700, 46.4200),
        "Cerklje ob Krki Airport": (15.5417, 45.9017),
    }
    si_weights = {
        "Ljubljana": 0.18,
        "Maribor": 0.06,
        "Celje": 0.03,
        "Kranj": 0.03,
        "Koper": 0.03,
        "Novo Mesto": 0.03,
        "Nova Gorica": 0.02,
        "Murska Sobota": 0.02,
        "Ptuj": 0.02,
        "Cerklje ob Krki Airport": 0.02,
    }
    si_center_counts = _center_counts_from_weights(si_n_ep, si_weights)
    si_center_radii = {"Ljubljana": 12.0, "Cerklje ob Krki Airport": 5.0, "__default__": 7.0}
    configs["Slovenia"] = CountryConfig(
        iso2="SI",
        natural_earth_admin="Slovenia",
        population_2025=si_pop,
        area_km2_2025=si_area,
        n_endpoints=si_n_ep,
        k2_count=si_k2,
        k3_count=si_k3,
        n_trusted_repeater_nodes=si_n_tr,
        n_double_trn_edges=round(0.2 * si_n_tr),
        detour_factor=detour,
        centers_lonlat=si_centers,
        center_counts=si_center_counts,
        center_radius_km=si_center_radii,
        p_uniform_rural=0.60,
        heavy_tail_scale_km=18.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )


    # Ireland
    ie_pop, ie_area = 5_440_000, 69_947
    ie_n_ep, ie_n_tr = 160, 40
    ie_k2, ie_k3 = _k2_k3_split(ie_n_ep)

    ie_centers = {
        "Dublin": (-6.2603, 53.3498),
        "Cork": (-8.4715, 51.8978),
        "Galway": (-9.0568, 53.2707),
        "Limerick": (-8.6267, 52.6638),
        "Waterford": (-7.1190, 52.2583),
        "Sligo": (-8.4694, 54.2697),
        "Athlone": (-7.940689, 53.424880),
        "Kilkenny": (-7.2450, 52.6542),
        "Drogheda": (-6.3470, 53.7179),
        "Casement Aerodrome (Baldonnel)": (-6.451111, 53.301667),
    }

    ie_weights = {
        "Dublin": 0.16,
        "Cork": 0.05,
        "Galway": 0.03,
        "Limerick": 0.03,
        "Waterford": 0.02,
        "Sligo": 0.02,
        "Athlone": 0.015,
        "Kilkenny": 0.015,
        "Drogheda": 0.015,
        "Casement Aerodrome (Baldonnel)": 0.010,
    }

    ie_center_counts = _center_counts_from_weights(ie_n_ep, ie_weights)
    ie_center_radii = {"Dublin": 22.0, "Casement Aerodrome (Baldonnel)": 5.0, "__default__": 10.0}

    configs["Ireland"] = CountryConfig(
        iso2="IE",
        natural_earth_admin="Ireland",
        population_2025=ie_pop,
        area_km2_2025=ie_area,
        n_endpoints=ie_n_ep,
        k2_count=ie_k2,
        k3_count=ie_k3,
        n_trusted_repeater_nodes=ie_n_tr,
        n_double_trn_edges=round(0.2 * ie_n_tr),
        detour_factor=detour,
        centers_lonlat=ie_centers,
        center_counts=ie_center_counts,
        center_radius_km=ie_center_radii,
        p_uniform_rural=0.65,
        heavy_tail_scale_km=25.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )



    be_pop, be_area, be_ep, be_tr = 11_900_000, 30_667, 250, 20
    be_k2, be_k3 = _k2_k3_split(be_ep)
    be_centers = {
        "Brussels": (4.3517, 50.8503),
        "Antwerp": (4.4025, 51.2194),
        "Ghent": (3.7174, 51.0543),
        "Charleroi": (4.4446, 50.4108),
        "Liège": (5.5701, 50.6337),
        "Bruges": (3.2247, 51.2093),
        "Leuven": (4.7009, 50.8798),
        "Namur": (4.8675, 50.4674),
        "NATO HQ (Haren, Brussels)": (4.4199, 50.8724),
    }
    be_weights = {
        "Brussels": 0.16,
        "Antwerp": 0.06,
        "Ghent": 0.04,
        "Charleroi": 0.03,
        "Liège": 0.03,
        "Bruges": 0.02,
        "Leuven": 0.02,
        "Namur": 0.02,
        "NATO HQ (Haren, Brussels)": 0.01,
    }
    configs["Belgium"] = CountryConfig(
        iso2="BE",
        natural_earth_admin="Belgium",
        population_2025=be_pop,
        area_km2_2025=be_area,
        n_endpoints=be_ep,
        k2_count=be_k2,
        k3_count=be_k3,
        n_trusted_repeater_nodes=be_tr,
        n_double_trn_edges=round(0.2 * be_tr),
        detour_factor=detour,
        centers_lonlat=be_centers,
        center_counts=_center_counts_from_weights(be_ep, be_weights),
        center_radius_km={"Brussels": 18.0, "NATO HQ (Haren, Brussels)": 4.0, "__default__": 8.0},
        p_uniform_rural=0.60,
        heavy_tail_scale_km=18.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )

    bg_pop, bg_area, bg_ep, bg_tr = 6_437_000, 110_996, 180, 60
    bg_k2, bg_k3 = _k2_k3_split(bg_ep)
    bg_centers = {
        "Sofia": (23.3219, 42.6977),
        "Plovdiv": (24.7453, 42.1354),
        "Varna": (27.9147, 43.2141),
        "Burgas": (27.4678, 42.5048),
        "Ruse": (25.9552, 43.8356),
        "Stara Zagora": (25.6345, 42.4258),
        "Pleven": (24.6180, 43.4170),
        "Blagoevgrad": (23.0950, 42.0209),
        "Bezmer Air Base": (26.3513, 42.4528),
    }
    bg_weights = {
        "Sofia": 0.16,
        "Plovdiv": 0.04,
        "Varna": 0.03,
        "Burgas": 0.03,
        "Ruse": 0.02,
        "Stara Zagora": 0.02,
        "Pleven": 0.02,
        "Blagoevgrad": 0.015,
        "Bezmer Air Base": 0.01,
    }
    configs["Bulgaria"] = CountryConfig(
        iso2="BG",
        natural_earth_admin="Bulgaria",
        population_2025=bg_pop,
        area_km2_2025=bg_area,
        n_endpoints=bg_ep,
        k2_count=bg_k2,
        k3_count=bg_k3,
        n_trusted_repeater_nodes=bg_tr,
        n_double_trn_edges=round(0.2 * bg_tr),
        detour_factor=detour,
        centers_lonlat=bg_centers,
        center_counts=_center_counts_from_weights(bg_ep, bg_weights),
        center_radius_km={"Sofia": 22.0, "Bezmer Air Base": 5.0, "__default__": 10.0},
        p_uniform_rural=0.70,
        heavy_tail_scale_km=30.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )


    cy_pop, cy_area, cy_ep, cy_tr = 980_000, 9_253, 50, 4
    cy_k2, cy_k3 = _k2_k3_split(cy_ep)
    cy_centers = {
        "Nicosia": (33.3823, 35.1856),
        "Limassol": (33.0440, 34.7071),
        "Larnaca": (33.6232, 34.9180),
        "Paphos": (32.4218, 34.7754),
        "Famagusta": (33.9500, 35.1167),
        "RAF Akrotiri": (32.9879, 34.5904),
    }
    cy_weights = {
        "Nicosia": 0.20,
        "Limassol": 0.06,
        "Larnaca": 0.05,
        "Paphos": 0.04,
        "Famagusta": 0.04,
        "RAF Akrotiri": 0.02,
    }
    configs["Cyprus"] = CountryConfig(
        iso2="CY",
        natural_earth_admin="Cyprus",
        population_2025=cy_pop,
        area_km2_2025=cy_area,
        n_endpoints=cy_ep,
        k2_count=cy_k2,
        k3_count=cy_k3,
        n_trusted_repeater_nodes=cy_tr,
        n_double_trn_edges=round(0.2 * cy_tr),
        detour_factor=detour,
        centers_lonlat=cy_centers,
        center_counts=_center_counts_from_weights(cy_ep, cy_weights),
        center_radius_km={"Nicosia": 12.0, "RAF Akrotiri": 4.0, "__default__": 6.0},
        p_uniform_rural=0.55,
        heavy_tail_scale_km=12.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=5.0,
        min_rural_distance_to_centers_km=5.0,
    )

    cz_pop, cz_area, cz_ep, cz_tr = 10_910_000, 78_871, 250, 50
    cz_k2, cz_k3 = _k2_k3_split(cz_ep)
    cz_centers = {
        "Prague": (14.4378, 50.0755),
        "Brno": (16.6068, 49.1951),
        "Ostrava": (18.2625, 49.8209),
        "Plzeň": (13.3776, 49.7384),
        "Liberec": (15.0562, 50.7671),
        "Olomouc": (17.2509, 49.5938),
        "České Budějovice": (14.4747, 48.9747),
        "Hradec Králové": (15.8328, 50.2092),
        "Ústí nad Labem": (14.0400, 50.6606),
        "Náměšť Air Base": (16.1240, 49.1663),
    }
    cz_weights = {
        "Prague": 0.16,
        "Brno": 0.04,
        "Ostrava": 0.03,
        "Plzeň": 0.02,
        "Liberec": 0.015,
        "Olomouc": 0.015,
        "České Budějovice": 0.015,
        "Hradec Králové": 0.015,
        "Ústí nad Labem": 0.015,
        "Náměšť Air Base": 0.01,
    }
    configs["Czechia"] = CountryConfig(
        iso2="CZ",
        natural_earth_admin="Czechia",
        population_2025=cz_pop,
        area_km2_2025=cz_area,
        n_endpoints=cz_ep,
        k2_count=cz_k2,
        k3_count=cz_k3,
        n_trusted_repeater_nodes=cz_tr,
        n_double_trn_edges=round(0.2 * cz_tr),
        detour_factor=detour,
        centers_lonlat=cz_centers,
        center_counts=_center_counts_from_weights(cz_ep, cz_weights),
        center_radius_km={"Prague": 20.0, "Náměšť Air Base": 5.0, "__default__": 10.0},
        p_uniform_rural=0.65,
        heavy_tail_scale_km=22.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=10.0,
        min_rural_distance_to_centers_km=10.0,
    )


    ee_pop, ee_area = 1_370_000, 45_336
    ee_n_ep, ee_n_tr = 50, 20
    ee_k2, ee_k3 = _k2_k3_split(ee_n_ep)
    ee_centers = {
        "Tallinn": (24.7536, 59.4370),
        "Tartu": (26.7290, 58.3776),
        "Narva": (28.1903, 59.3772),
        "Pärnu": (24.4971, 58.3859),
        "Viljandi": (25.5903, 58.3639),
        "Tapa Army Base": (25.9510, 59.2343),
    }
    ee_weights = {
        "Tallinn": 0.20,
        "Tartu": 0.06,
        "Narva": 0.04,
        "Pärnu": 0.04,
        "Viljandi": 0.03,
        "Tapa Army Base": 0.02,
    }
    ee_center_counts = _center_counts_from_weights(ee_n_ep, ee_weights)
    ee_center_radii = {"Tallinn": 15.0, "Tapa Army Base": 5.0, "__default__": 8.0}
    configs["Estonia"] = CountryConfig(
        iso2="EE",
        natural_earth_admin="Estonia",
        population_2025=ee_pop,
        area_km2_2025=ee_area,
        n_endpoints=ee_n_ep,
        k2_count=ee_k2,
        k3_count=ee_k3,
        n_trusted_repeater_nodes=ee_n_tr,
        n_double_trn_edges=round(0.2 * ee_n_tr),
        detour_factor=detour,
        centers_lonlat=ee_centers,
        center_counts=ee_center_counts,
        center_radius_km=ee_center_radii,
        p_uniform_rural=0.60,
        heavy_tail_scale_km=22.0,
        heavy_tail_alpha=8.0,
        min_rural_point_distance_km=8.0,
        min_rural_distance_to_centers_km=8.0,
    )



    return configs


COUNTRY_CONFIGS_EU = load_all_country_configs_from_module()
