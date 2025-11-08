# Data Processing, Feature Encoding, and CVAE Model Training

This document summarizes the complete workflow for preparing the EV specifications dataset, engineering conditioning variables, scaling continuous targets, and training the Conditional Variational Autoencoder (CVAE) with β-annealing. It is meant to be a clear, self-contained reference suitable for inclusion in a research repository or production model pipeline.

---

## 1. Dataset Preparation

### 1.1 Source and Filtering

The dataset includes electric vehicle specifications such as manufacturing region, pricing, range, battery capacity, charging formats, and chemistry types. Only **pure battery electric vehicles (BEVs)** are retained to ensure consistency in energy systems.

### 1.2 Required Fields

| Column Name            | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| CO2_Emissions_g_per_km | Ensures only BEVs are included             |
| Charging_Type          | Supports charging ecosystem classification |
| Price_USD              | Used for market segmentation               |
| Battery_Type           | Determines chemistry family                |
| Country_of_Manufacture | Enables regional design conditioning       |

Rows missing any of these fields are removed.

---

## 2. Feature Engineering

### 2.1 Region Grouping

Countries are grouped into broader regions for generalization robustness.

| Region Group  | Example Countries                    |
| ------------- | ------------------------------------ |
| Europe        | Germany, UK, France, Sweden          |
| Asia          | China, India, Japan, South Korea     |
| North America | United States, Canada, Mexico        |
| Rest of World | Brazil, South Africa, UAE, Australia |

Resulting Feature: **C_Region**

### 2.2 Battery Chemistry Grouping

| Battery Type Examples                        | Group Name        |
| -------------------------------------------- | ----------------- |
| NMC, NCA, standard Li-ion                    | Standard Li-ion   |
| LFP                                          | LFP               |
| Solid-state, Sodium-ion, Lithium-Sulfur, LTO | Future / Advanced |
| Lead-acid, NiMH, Flow batteries, Zinc-air    | Legacy / Other    |

Resulting Feature: **C_Battery_Chem**

### 2.3 Charging System Encodings

| Derived Feature | Description                                        |
| --------------- | -------------------------------------------------- |
| C_Connector     | Charging connector type (e.g., CCS, NACS, CHAdeMO) |
| C_Charge_Speed  | Charging tier classification                       |
| C_Has_V2X       | Indicates V2H / V2L / V2G capability               |

### 2.4 Budget Class

Vehicles are grouped into market segments using price percentiles:

* Budget
* Mid-Range
* Luxury

Resulting Feature: **C_Budget**

---

## 3. Model Target Variables

| Target Variable      | Units | Typical Range       |
| -------------------- | ----- | ------------------- |
| Range_km             | km    | ~120 to 700         |
| Price_USD            | USD   | ~18,000 to 120,000+ |
| Battery_Capacity_kWh | kWh   | ~30 to 120          |

These form the **continuous output vector** modeled by the CVAE.

---

## 4. Encoding and Scaling

Continuous targets are standardized. Categorical conditions are one-hot encoded and combined with binary-engineered indicators.

| Processing Step                         | Resulting Output Form    |
| --------------------------------------- | ------------------------ |
| Standardizing continuous targets        | Y_scaled (N × 3 matrix)  |
| One-hot encoding regional and chemistry | C_encoded (N × ~20 dims) |
| Appending binary condition flags        | C_final (complete C set) |

Final shapes used in the model:

* Continuous outputs **Y:** (N × 3)
* Condition vectors **C:** (N × ~20)

---

## 5. Conditional VAE Architecture

The CVAE learns to reconstruct target vehicle specifications conditioned on design and contextual factors.

### Encoder

Takes the combined vehicle specification and condition vector to produce a latent representation.

### Latent Dimension

A compact 6-dimensional latent space is used to balance expressive capacity and generalization.

### Decoder

Reconstructs the continuous target specifications conditioned on the same categorical feature vector.

---

## 6. β-Annealing Training Strategy

A **β-annealing schedule** is used to gradually introduce the KL-divergence regularization to avoid posterior collapse.

| Phase          | β (KL Weight)      | Training Effect                                 |
| -------------- | ------------------ | ----------------------------------------------- |
| Early Training | Low → Gradual Rise | Prioritizes reconstruction stability            |
| Mid Training   | Increasing to ~1.0 | Encourages formation of structured latent space |
| Final          | Maintained ~1.0    | Ensures smooth and generalizable representation |

This approach produces stable, disentangled latent structure while preserving generative variability.

---

## 7. Model Performance

### Standardized Scale Metrics

| Feature          | MAE  | RMSE |
| ---------------- | ---- | ---- |
| Range_km         | 0.51 | 0.64 |
| Price_USD        | 0.32 | 0.37 |
| Battery_Capacity | 0.56 | 0.70 |

### Converted Back to Real Units

| Feature          | MAE Approx. | Interpretation                                     |
| ---------------- | ----------- | -------------------------------------------------- |
| Range_km         | ~74 km      | Comparable to trim or tire configuration effects   |
| Price_USD        | ~$10.9k     | Matches typical MSRP variance bands                |
| Battery_Capacity | ~21 kWh     | Reflects distinct chemistry and platform groupings |

This confirms the model learned practical market and engineering tradeoffs.

---

## 8. Model Usage

The trained decoder can synthesize new EV specifications conditioned on targeted design constraints such as:

* Region of manufacture
* Battery chemistry type
* Market price segment
* Charging ecosystem compatibility

This enables structured **EV concept generation and scenario modeling**.

---

## 9. Summary

| Component                        | Status |
| -------------------------------- | ------ |
| Clean, filtered dataset          | ✅      |
| Consistent engineered conditions | ✅      |
| Scaled continuous targets        | ✅      |
| CVAE with β-warmup               | ✅      |
| Stable latent representation     | ✅      |
| Generative synthesis ready       | ✅      |

---

## Next Potential Enhancements

* Incorporate text-based brand embeddings
* Add normalizing-flow layers to improve output sharpness
* Deploy interactive model UI for design exploration
