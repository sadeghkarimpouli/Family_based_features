# Per-family Features Categorization for Earthquake Analysis

This repository provides a workflow for constructing **Per-family seismic features** from earthquake catalogs using **Event-based feature extraction** within moving spatio-temporal windows.

The extracted feature families are subsequently categorized using unsupervised approaches to identify patterns associated with the **preparatory phase of large earthquakes**.

---

# Overview

The workflow consists of two main steps:

1. **Event-based Feature Extraction**  
   Seismicity features are computed from earthquake catalogs using moving spatial and temporal windows.

2. **Per-family Categorization**  
   Extracted features are grouped into families and categorized to reveal evolving seismic behavior preceding large earthquakes.

The framework is designed for earthquake catalog analysis and can be adapted to different tectonic regions and catalog datasets.

---

# Implemented Case Studies

This workflow is currently implemented for the following earthquake sequences:

1. **Türkiye:** 2023 Mw 7.8 Kahramanmaraş  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadeghkarimpouli/family_based_features/blob/main/Kahramanmaras_Family_clustering.ipynb)

2. **Central Italy:** 2009 Mw 6.1 L'Aquila  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadeghkarimpouli/family_based_features/blob/main/LAquila_Family_clustering.ipynb)

3. **Central Italy:** 2016 Mw 6.2 Amatrice  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadeghkarimpouli/family_based_features/blob/main/Amatrice_Family_clustering.ipynb)

---

# Citation

If you use this code or methodology in your research, please cite:

> Karimpouli, S., Martínez-Garzón, P., Núñez-Jara, S. et al.  
> *Preparatory phase of large earthquakes illuminated by unsupervised categorization of earthquake catalog features.*  
> **Nature Communications**, 17, 4024 (2026).  
> https://doi.org/10.1038/s41467-026-72279-x

---

# Event-Based Feature Extraction

Use:

```python
Event_based_Features_functions.py
```

to compute seismicity features from an earthquake catalog.

Features are calculated within moving **spatio-temporal windows** and may require specific parameter selections depending on the target feature.

Please review the corresponding function documentation and parameter definitions carefully before computing features.

---

# Input Catalog Format

The input seismic catalog must contain the following columns:

| Column | Description |
|---|---|
| `Event_ID` | Event identifier |
| `UTM_Easting[m]` | Easting coordinate in meters |
| `UTM_Northing[m]` | Northing coordinate in meters |
| `Depth[m]` | Hypocentral depth in meters |
| `Time[d.s]` | Event time in decimal days |
| `Magnitude` | Earthquake magnitude |

Example:

```text
Event_ID    UTM_Easting[m]   UTM_Northing[m]   Depth[m]   Time[d.s]     Magnitude
15961290    256156.51        4858519.47        15130      732070.3964   1.57
15961300    256240.22        4858601.10        14980      732070.5123   1.83
...
```

---

# Coordinate and Time Requirements

- Spatial coordinates must be provided in **meters** (UTM projection recommended).
- Depth must be provided in **meters**.
- Time must be represented in **decimal days**.
- Magnitudes should be consistently defined throughout the catalog.

---

# Available Event-Based Features

The framework currently supports extraction of several seismicity features, including:

- Event rate
- Moment rate
- b-value
- Correlation integral
- Inter-event time
- Inter-event distance
- Clustering features
- Convex hull volume
- Kostrov strain

Additional features can be incorporated through the modular feature extraction framework.

---

# Example Workflow

```python
import pandas as pd
import Event_based_Features_functions as eff

df = pd.read_csv("catalog.csv")

df_b = eff.b_value_event_based(
    df,
    time_window=[15, 30],
    space_window=[17, 33],
    ev_lim=100,
    Mc=1.5
)
```

---

# License

This project includes portions of code derived from external open-source projects licensed under the Apache License 2.0.

Please see the `LICENSE` file for details.

---

# Acknowledgments

Parts of the clustering workflow are derived from:

- Mark Williams, Nevada Seismological Laboratory, University of Nevada, Reno
- `eqclustering` (Apache License 2.0)

---

