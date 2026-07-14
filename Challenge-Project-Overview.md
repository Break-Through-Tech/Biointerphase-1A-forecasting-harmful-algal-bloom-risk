# Forecasting Harmful Algal Bloom (HAB) Risk Using Multi-Source Water Quality and Weather Data

**Company / Org:** Biointerphase  
**Challenge Advisor:** Alyssa Long, along@biointerphase.com; Mia Maksin, mmaksin@biointerphase.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Biointerphase

Biointerphase specializes in the development of advanced biomaterials and biosignal intelligence platforms designed to restore, reinforce, and better understand complex biological systems. Within this framework, our work in the water quality sector applies these capabilities through data‑driven approaches that enable more precise monitoring, prediction, and mitigation of biological risks in aquatic environments.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use publicly available water quality, meteorological, and satellite-derived environmental data and machine learning techniques such as classification techniques (ex. bloom, at risk for bloom, no bloom, etc.) to build a model that predicts the risk or occurrence of harmful algal blooms, with optional extensions into time-series forecasting that predict the risk, occurrence, or growth of harmful algal blooms. This will help our company address the challenge of developing practical early-warning and monitoring tools for harmful algal blooms to support targeted environmental management decisions.

### Success Criteria
Model performance (Classification: accuracy, precision/recall, F1, ROC-AUC; Regression: RMSE, MAE, R2); integration of multi-source environmental data; clear interpretation of key environmental drivers; and a reproducible, well-documented Python workflow.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Understanding | Explore dataset, handle missing values, document findings |
| **October** | Model Development | Train baseline model, experiment with approaches, iterate |
| **November** | Evaluation & Presentation | Finalize model, prepare presentation, document results |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Publicly available multi-source datasets focused on Florida waters around Mote Marine Laboratory  
**Format:** CSV, TSV, JSON  
**Size:** 1gb to 5gb  
**Location:** All data is accessible via the data folder on GitHub. Links to additional data found in table below, if necessary

### Key Details
- Publicly available multi-source datasets including water quality data (temperature, nutrients, chlorophyll), meteorological data (wind), and satellite-derived environmental data.

| Filename                     | Website/Source                                                                 | Dates                 | Data Included |
|------------------------------|--------------------------------------------------------------------------------|-----------------------|---------------|
| Habsos_cellcount (All FL)    | https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system       | 1953 – April 2026     | Latitude, Longitude, Date, Category, Cell Count, Salinity, Water Temp |
| Florida_cyanotoxins (All FL) | https://storymaps.arcgis.com/stories/ccc4c98425194605924ae474fb8002ba          | 2015 – 2026           | Latitude, Longitude, Location Type, Date, Depth of Measurement, Cyanotoxin Type, Cyanotoxin Concentration, Detection Limit |
| BradentonFL_tempData         | https://climatecenter.fsu.edu/climate-data-access-tools/downloadable-data      | 2015 – 2025           | Date, Precipitation, Max Air Temp, Min Air Temp |
| Sarasota_Wind&Temp           | https://erddap.secoora.org/erddap/tabledap/gov_noaa_awc_ksrq.html              | July 2022 – Current   | Air pressure, air temp, dew point temp, visibility, wind gust speed, wind speed, wind direction |
| MoteMarineBottomTemp         | https://erddap.secoora.org/erddap/tabledap/org_secoora_scan_rng_sw.html        | 2020 – 2025           | Latitude, Longitude, Date, Bottom Ocean Temp |
| NutrientsFL_2006_2025 (Sarasota) | https://sarasota.wateratlas.usf.edu/water-quality-trends/               | 2006 – 2025           | Date, Time, Depth; BOD, Chlorophyll-a (pheophytin corrected), DO, DO saturation, Fecal Coliform, Total Nitrogen, Ammonia (N), Kjeldahl Nitrogen, NO₂+NO₃ (N), pH, Total Phosphorus, Phosphate (PO₄), Specific Conductance, Water Temperature, TSS, Turbidity |
| Chlorophyll_subset_sarasota  | https://coastwatch.noaa.gov/cwn/products/noaa-msl12-multi-sensor-dineof-global-2-km-gap-filled-products-chlorophyll-diffuse.html | 2020 – 2026 | Latitude, Longitude, Date, Chlorophyll-a (subset: lat 26.5–28, lon -83.5 to -81.8) |

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification (primary), Time-Series Analysis (secondary)

**Recommended Libraries:**
- Classification
- Regression
- Time-Series Analysis
- Deep Learning / Neural Networks
- Transfer Learning / Pre-trained Models
- Random Forest
- Gradient Boosting
- Python

**Evaluation Metrics:**
- Accuracy, Precision/Recall, RMSE
- Success will be measured by both model performance and practical usability:
  - **Classification performance** in identifying at-risk and bloom conditions 
  - **Ability to correctly identify "at-risk" conditions**, which are crucial for early intervention 
  - **Interpretability of key environmental drivers** that influence the transition between non-blook, at-risk, or bloom states 
  - **Clarity of outputs**, such that results could be used to support real-world monitoring decisions (ex. additional sampling or mitigation techniques) 

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [What are Harmful Algae, Cyanobacteria, and Cyanotoxins? (EPA)](https://www.epa.gov/habs/learn-about-harmful-algae-cyanobacteria-and-cyanotoxins)
- [HAB Contributing Factors & Impacts (CDC)](https://www.cdc.gov/harmful-algal-blooms/about/harmful-algal-blooms-contributing-factors-and-impacts.html)
- [Existing HAB Forecasting (EPA)](https://www.epa.gov/water-research/cyanobacterial-harmful-algal-blooms-forecasting-research)

**Technical Tutorials:**
- [Classification Algorithms](https://developer.ibm.com/tutorials/learn-classification-algorithms-using-python-and-scikit-learn/)

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).

---
