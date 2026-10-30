# Dataset and Feature Engineering Audit

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** Data Architecture, Feature Engineering, and Forensic Data Quality Audit (`docs/report/data_and_features.md`)

---

## 1. Primary Data Sources and Acquisition Modalities

GeoAgri AI integrates multimodal agronomic, meteorological, pedological, and geomorphic data sources. The table below details each verified source, its spatial/temporal resolution, access modality, and technical role.

| Data Source | Providing Institution / Endpoint | Spatial / Temporal Resolution | Ingestion Modality | Physical Parameters Ingested | Known Limitations & Usage Constraints |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Area, Production and Yield (APY)** | Ministry of Agriculture & Farmers Welfare, GoI (`data.gov.in`) | District-level; annual / seasonal (1997–2020) | Preprocessed offline tabular dataset | Historical crop occurrence, harvested area (ha), production (tonnes), realized yield (t/ha) | District aggregates mask intra-district micro-climatic variance; historical records contain reporting lag. |
| **ICRISAT District-Level Database (DLD)** | International Crops Research Institute for the Semi-Arid Tropics | 311 Indian districts; longitudinal annual | Preprocessed offline tabular dataset | Long-term biophysical baselines, fertilizer consumption trends, irrigation percentages | Discontinued post-2017 in some districts; requires cross-referencing with DES data. |
| **National Soil Health Card Scheme & ICAR-IISS** | Dept. of Agriculture & Farmers Welfare, GoI (`soilhealth.dac.gov.in`) | District & sub-district grid benchmarks | Preprocessed offline dataset & fallback priors | Soil macronutrients: Nitrogen (N), Phosphorus (P), Potassium (K), Organic Carbon (OC), pH, Electrical Conductivity (EC) | Laboratory test variability across state testing labs; static spatial interpolation. |
| **SoilGrids REST API** | ISRIC — World Soil Information (`rest.isric.org`) | 250m global spatial resolution; standard depth intervals | Dynamic live HTTP GET request (`src/geospatial/fetch_soil.py`) | Sand (g/kg), Silt (g/kg), Clay (g/kg), pH ($\times 10$), Bulk density, CEC | Web service subject to network latency and occasional downtime; fallback to district priors required. |
| **Open-Meteo Historical & Forecast Weather** | Open-Meteo GmbH (`api.open-meteo.com`) | 0.1° (~11 km) spatial grid; hourly / daily intervals | Dynamic live HTTP GET request (`src/geospatial/fetch_weather.py`) | Temperature (mean, min, max), Relative Humidity, Precipitation series (7d, 30d, 90d, annual), Soil moisture, $ET_0$ | Free tier limited to non-commercial rate limits; reanalysis models have smoothing bias in mountainous regions. |
| **Open-Elevation REST API** | Open-Elevation Project (`api.open-elevation.com`) | SRTM 90m DEM global elevation grid | Dynamic live HTTP GET request (`src/geospatial/fetch_terrain.py`) | Ground elevation (m MSL), 4-point cardinal slope derivation (degrees) | Public server exhibits high request latency (~1.5s per query); cached SRTM elevation fallback used. |
| **FAO EcoCrop Ecological Database** | Food and Agriculture Organization of the UN (`fao.org`) | Global agronomic species profiles | Structured JSON rule matrix (`crop_knowledge_base.json`) | Temperature optimality, pH limits, rainfall minimum/maximum, slope tolerance, drought resilience | Static rule baselines; does not reflect newly developed hybrid cultivars with expanded tolerance bounds. |

---

## 2. Dataset Construction, Target Definition, and Splits

### Dataset Statistics
The primary training dataset is stored at `data/agriculture/v1.0/processed/crop_suitability_dataset.csv`.
- **Total Record Count ($N$):** 10,091 observation records.
- **Total Raw Columns:** 31 columns.
- **Unique Crop Classes ($K$):** 16 major Indian agricultural crops.
- **Cropping Seasons:** 4 discrete seasons (`Kharif`, `Rabi`, `Summer`, `Whole Year`).
- **Temporal Span:** 1997 through 2020 (24 continuous agricultural years).
- **Spatial Coverage:** 30 Indian states and union territories encompassing all 15 major ICAR Planning Commission Agro-Climatic Zones.

### Class Distribution and Imbalance Audit
The 16 crop classes exhibit natural real-world agricultural imbalance:
- **Dominant Classes:** Rice ($n=1,197$, 11.86%), Maize ($n=975$, 9.66%), Moong ($n=740$, 7.33%), Urad ($n=733$, 7.26%), Groundnut ($n=725$, 7.18%).
- **Moderate Classes:** Sesamum ($n=685$), Sugarcane ($n=605$), Wheat ($n=545$), Mustard ($n=528$), Pearl Millet ($n=524$), Sorghum ($n=513$), Pigeonpea ($n=508$).
- **Minority Classes:** Finger Millet ($n=498$, 4.93%), Chickpea ($n=490$, 4.86%), Cotton ($n=476$, 4.72%), Soybean ($n=349$, 3.46%).
- **Imbalance Ratio:** Maximum-to-minimum ratio is $1,197 : 349 \approx 3.43 : 1$. While not severe enough to cause degenerate majority-class collapse, minority classes suffer significantly lower precision in pure multiclass classification.

---

## 3. Data Dictionary: Raw Attributes vs Engineered Features

### Raw Input Schema (31 Columns)
```
latitude, longitude, state, year, season, crop, soil_ph, nitrogen, phosphorus,
potassium, organic_carbon, electrical_conductivity, clay, sand, silt,
soil_texture_class, temperature_mean, temperature_min, temperature_max,
humidity_mean, rainfall_annual, rainfall_season, rainfall_7d, rainfall_30d,
rainfall_90d, soil_moisture, et0, elevation, slope, erosion_risk_score, yield
```

### Engineered Model Feature Vector (26 Dimensions)
The 26 features ingested by the XGBoost models (`models/crop/v5/crop_model.pkl`) are strictly ordered as follows:

| Index | Feature Code Name | Data Type | Physical Unit | Agronomic Role & Description | Source Modality |
| :---: | :--- | :---: | :---: | :--- | :--- |
| `0` | `soil_ph` | Float | $-\log[H^+]$ | Soil acidity/alkalinity measure (4.5 to 8.6) | Soil Health Card / SoilGrids |
| `1` | `nitrogen` | Float | kg/ha | Available soil mineral nitrogen rating | Soil Health Card |
| `2` | `phosphorus` | Float | kg/ha | Available soil phosphorus rating | Soil Health Card |
| `3` | `potassium` | Float | kg/ha | Available soil potassium rating | Soil Health Card |
| `4` | `organic_carbon` | Float | % | Soil organic carbon content (fertility proxy) | Soil Health Card |
| `5` | `electrical_conductivity`| Float | dS/m | Soil salinity / total dissolved salt indicator | Soil Health Card |
| `6` | `clay` | Float | % | Mass fraction of clay particles ($< 0.002$ mm) | SoilGrids / Physical texture |
| `7` | `sand` | Float | % | Mass fraction of sand particles ($0.05 - 2.0$ mm) | SoilGrids / Physical texture |
| `8` | `silt` | Float | % | Mass fraction of silt particles ($0.002 - 0.05$ mm)| SoilGrids / Physical texture |
| `9` | `temperature_mean` | Float | °C | Seasonal average ambient temperature | Open-Meteo Climatology |
| `10` | `temperature_range` | Float | °C | Diurnal / seasonal thermal swing ($T_{\max} - T_{\min}$) | Engineered ($T_{\max} - T_{\min}$) |
| `11` | `humidity_mean` | Float | % | Seasonal average ambient relative humidity | Open-Meteo Climatology |
| `12` | `rainfall_season` | Float | mm | Cumulative precipitation during crop growing season | Open-Meteo Climatology |
| `13` | `rainfall_30d` | Float | mm | Critical short-term vegetative moisture window | Open-Meteo Climatology |
| `14` | `rainfall_90d` | Float | mm | Extended mid-season reproductive moisture window | Open-Meteo Climatology |
| `15` | `soil_moisture` | Float | $\text{m}^3/\text{m}^3$ | Volumetric root-zone water content (0–7 cm depth)| Open-Meteo Soil Profile |
| `16` | `et0` | Float | mm/day | Reference Penman-Monteith crop evapotranspiration| Open-Meteo Climatology |
| `17` | `elevation` | Float | m MSL | Surface topography above mean sea level | Open-Elevation SRTM DEM |
| `18` | `slope` | Float | degrees | Local topographic inclination / grade angle | Derived 4-point elevation gradient |
| `19` | `erosion_risk_score` | Float | Probability [0, 1] | Geomorphic soil loss vulnerability score | Inference from `erosion_model.pkl` |
| `20` | `soil_fertility_index` | Float | Score [5, 100] | Composite multi-nutrient agronomic fertility index| Engineered ICAR rating formula |
| `21` | `water_stress_index` | Float | Ratio [0.1, 5.0] | Atmospheric demand to root moisture availability | Engineered FAO-56 ratio formula |
| `22` | `rainfall_deviation` | Float | Ratio [-1.0, 2.0]| Precipitation anomaly relative to annual baseline | Engineered partition deviation |
| `23` | `soil_moisture_index` | Float | Ratio [0.05, 1.25]| Moisture content relative to field capacity (0.40) | Engineered normalization |
| `24` | `season_code` | Float | Discrete {0,1,2,3}| Integer encoding: Kharif=0, Rabi=1, Summer=2, Year=3| Categorical mapping |
| `25` | `texture_code` | Float | Discrete {0..6} | Integer encoding of USDA soil texture class | Categorical mapping |

---

## 4. Mathematical Formulations of Composite Agronomic Indicators

The four composite agronomic indicators implemented in `src/preprocessing/agri_features.py` are mathematically defined as follows:

### 1. Soil Fertility Index (SFI)
The SFI aggregates five fundamental soil parameters into a standardized 100-point scale based on ICAR agronomic thresholds:
$$\text{SFI} = \min\left(25, \frac{\text{N}}{300} \cdot 25\right) + \min\left(20, \frac{\text{P}}{25} \cdot 20\right) + \min\left(20, \frac{\text{K}}{260} \cdot 20\right) + \min\left(20, \frac{\text{OC}}{0.75} \cdot 20\right) + \max\left(0, 15 - 5 \cdot |\text{pH} - 7.0|\right)$$
$$\text{SFI} \in [5.0, 100.0]$$
- **Agronomic Rationale:** Benchmarks nitrogen against 300 kg/ha, phosphorus against 25 kg/ha, potassium against 260 kg/ha, and organic carbon against 0.75%. Soil pH deviations from neutral (7.0) are penalized at 5 points per pH unit.

### 2. Water Stress Index (WSI)
The WSI quantifies the imbalance between atmospheric evaporative demand and available root-zone moisture:
$$\text{WSI} = \frac{\max(10.0, \, \text{ET}_0 \cdot 30)}{\max(5.0, \, \text{rainfall}_{30\text{d}} + (\text{soil\_moisture} \cdot 150))}$$
$$\text{WSI} \in [0.1, 5.0]$$
- **Agronomic Rationale:** A high WSI ($> 1.5$) indicates acute vegetative drought stress where monthly crop water demand ($\text{ET}_0 \cdot 30$) severely outstrips short-term precipitation and soil reservoir storage.

### 3. Seasonal Rainfall Deviation ($\Delta R$)
Captures precipitation anomalies relative to the expected seasonal partition:
$$\Delta R = \frac{\text{rainfall}_{\text{season}} - \max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}{\max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}$$
$$\Delta R \in [-1.0, 2.0]$$
- **Agronomic Rationale:** In monsoonal climates, ~65% of annual rainfall is expected during the primary cropping season. Values near $-1.0$ indicate severe monsoon failure, while values $> 1.0$ indicate flash flooding or waterlogging risk.

### 4. Soil Moisture Index (SMI)
Normalizes topsoil volumetric water content against nominal field capacity:
$$\text{SMI} = \text{clip}\left(\frac{\text{soil\_moisture}}{0.40}, \, 0.05, \, 1.25\right)$$
- **Agronomic Rationale:** A volumetric moisture content of $0.40 \, \text{m}^3/\text{m}^3$ represents optimal field capacity for loam soils. SMI values $< 0.3$ signal water deficit near the permanent wilting point.

---

## 5. Forensic Dataset Validation Audit & Texture Violations

An audit of `data/agriculture/v1.0/dataset_validation_report.json` and direct dataframe verification revealed a critical discrepancy between documented sub-check results and the summary status:

```json
"checks": {
  "missing_values": { "status": "PASS", "missing_counts": {} },
  "duplicates": { "status": "PASS", "duplicate_rows": 0 },
  "coordinates": { "status": "PASS", "latitude_range": [8.59, 36.05], "longitude_range": [68.82, 97.11] },
  "soil_ranges": {
    "status": "FAIL",
    "ph_range": [4.55, 8.64],
    "texture_sum_violations": 4
  },
  "weather_ranges": { "status": "PASS" },
  "target_distribution": { "status": "PASS", "unique_crops": 16 }
},
"status": "PASS"
```

### Forensic Disclosure of the 4 Texture Violations
In physical pedology, soil mineral particles must strictly conserve mass:
$$\text{Sand} (\%) + \text{Silt} (\%) + \text{Clay} (\%) = 100.0\%$$
In the processed dataset, exactly **4 rows** violate this physical conservation law ($\text{sum} \ne 100\%$, with sums ranging from 98.8% to 101.4% due to rounding artifacts in source soil table digitization). 

While the summary report recorded `"status": "PASS"`, this audit formally documents the failure of the `soil_ranges` constraint check. Because these 4 rows represent only $0.039\%$ of the dataset ($4 / 10,091$), their impact on tree split thresholds is negligible, but transparent academic disclosure is mandatory.

---

## 6. Data Leakage Assessment and Splitting Invariants

### 1. Pre-Harvest Information Invariant (Zero Target Leakage)
A common pitfall in agricultural ML is including realized harvest statistics (e.g., total harvested production or realized yield) in the feature matrix used for crop selection. In GeoAgri AI V5:
- **Realized Yield (`yield`)** is strictly excluded from the classification and ranking feature matrices.
- The 26 input features contain exclusively pre-sowing environmental variables (soil chemistry, topography, historical climatology, and seasonal weather). Realized yield is reserved exclusively as the target variable for the downstream yield regressor.

### 2. Grouped Query Atomicity in Candidate Ranking
In the candidate-level Learning-to-Rank formulation (161,456 candidate rows), each decision context $q = (\text{lat}, \text{lon}, \text{year}, \text{season})$ generates exactly 16 candidate item pairs. 
- **Leakage Control:** All 16 candidate rows sharing the same query identifier $q$ are grouped atomically. Under no split strategy are candidates from the same decision context split across training and evaluation partitions.

### 3. Chronological Forward Splitting
To guard against temporal data leakage (where future weather patterns leak into historical models):
- **Training Partition:** Historical observations strictly $\le 2017$ ($n = 7,603$ queries).
- **Validation Partition:** Intermediate transition year $2018$ ($n = 1,942$ queries).
- **Test Partition:** Held-out future years $2019–2020$ ($n = 546$ queries).
This guarantees that evaluation reflects true operational deployment into future agricultural seasons.
