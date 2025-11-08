# 6


import os
import json
import numpy as np
import pandas as pd
import joblib
from packaging import version
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

print("sklearn:", sklearn.__version__)

ART_DIR = 'artifacts'
os.makedirs(ART_DIR, exist_ok=True)

OHE_KW = {}
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    OHE_KW["sparse_output"] = False
else:
    OHE_KW["sparse"] = False


# 7


CSV_PATH = '/kaggle/input/ev-records/electric_vehicles_dataset.csv'
df_raw = pd.read_csv(CSV_PATH)
print("Raw Data Shape:", df_raw.shape)

critical_cols = ['CO2_Emissions_g_per_km', 'Charging_Type', 'Price_USD', 'Battery_Type', 'Country_of_Manufacture',
                 'Range_km', 'Battery_Capacity_kWh']
df = df_raw.dropna(subset=critical_cols).copy()

# EV only
df = df[df['CO2_Emissions_g_per_km'] <= 0].copy()
print(f"Pure EVs selected for training: {len(df)}")


# 8


for col in ['Charging_Type', 'Battery_Type', 'Country_of_Manufacture']:
    df[col] = df[col].astype(str)

region_map = {
    # Europe
    'UK':'Europe','Netherlands':'Europe','Sweden':'Europe','Denmark':'Europe','Belgium':'Europe','Norway':'Europe',
    'Hungary':'Europe','France':'Europe','Poland':'Europe','Finland':'Europe','Switzerland':'Europe','Portugal':'Europe',
    'Germany':'Europe','Ireland':'Europe','Italy':'Europe','Austria':'Europe','Spain':'Europe','Czech Republic':'Europe',
    'Russia':'Europe',
    # Asia
    'China':'Asia','Japan':'Asia','South Korea':'Asia','India':'Asia','Thailand':'Asia','Indonesia':'Asia',
    'Singapore':'Asia','Vietnam':'Asia','Malaysia':'Asia',
    # North America
    'USA':'North America','Canada':'North America','Mexico':'North America',
    # RoW
    'Argentina':'Rest of World','Brazil':'Rest of World','United Arab Emirates':'Rest of World',
    'Saudi Arabia':'Rest of World','South Africa':'Rest of World','Australia':'Rest of World',
    'New Zealand':'Rest of World','Israel':'Rest of World','Turkey':'Rest of World'
}
df['C_Region'] = df['Country_of_Manufacture'].map(region_map).fillna('Other')

# Battery map
battery_map = {
    'Lithium-ion': 'Standard Li-ion',
    'Nickel-manganese-cobalt': 'Standard Li-ion', 'Nickel–manganese–cobalt': 'Standard Li-ion',
    'Nickel-cobalt-aluminum': 'Standard Li-ion', 'NCA': 'Standard Li-ion',
    'Lithium-iron phosphate': 'LFP', 'LFP': 'LFP',
    'Solid-state': 'Future/Advanced', 'Lithium-sulfur': 'Future/Advanced', 'Lithium-titanate': 'Future/Advanced',
    'Sodium-ion': 'Future/Advanced', 'Aluminum-ion': 'Future/Advanced', 'Magnesium-ion': 'Future/Advanced',
    'Lead-acid': 'Legacy/Other', 'Nickel-metal hydride': 'Legacy/Other',
    'Calcium-ion': 'Legacy/Other', 'Ca-ion': 'Legacy/Other', 'Zinc-air': 'Legacy/Other', 'Flow batteries': 'Legacy/Other'
}
df['C_Battery_Chem'] = df['Battery_Type'].map(battery_map).fillna('Legacy/Other')

# Connector & speed
def get_connector(x: str) -> str:
    x_low = x.lower()
    if 'nacs' in x_low: return 'NACS'
    if 'ccs' in x_low: return 'CCS'
    if 'chademo' in x_low: return 'CHAdeMO'
    return 'Other/Plug-in'

def get_speed_tier(x: str) -> str:
    x_low = x.lower()
    if 'ultra-fast' in x_low or '350' in x_low: return 'Ultra-Fast (350kW+)'
    if 'dc fast' in x_low or 'dcfc' in x_low:   return 'DC Fast'
    if 'level 2' in x_low:                      return 'Level 2'
    return 'Level 1 / Slow'

df['C_Connector'] = df['Charging_Type'].apply(get_connector)
df['C_Charge_Speed'] = df['Charging_Type'].apply(get_speed_tier)

# V2X flag
df['C_Has_V2X'] = df['Charging_Type'].str.contains('V2G|V2L|V2H', case=False, regex=True).astype(int)

# Targets & conditions
Y_features = ['Range_km', 'Price_USD', 'Battery_Capacity_kWh']
C_features_cat = ['C_Region', 'C_Battery_Chem', 'C_Connector', 'C_Charge_Speed', 'C_Budget']
C_features_num = ['C_Has_V2X']

print("Y_features:", Y_features)
print("Categorical C (with budget to be added later):", C_features_cat[:-1])
print("Numeric C:", C_features_num)


# 9


train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, shuffle=True
)
df_train = df.iloc[train_idx].copy()
df_val   = df.iloc[val_idx].copy()


q = [0, 1/3, 2/3, 1]
edges = np.quantile(df_train['Price_USD'].values, q)

edges = np.unique(edges)

if len(edges) < 4:
    p = df_train['Price_USD'].median()
    edges = np.array([p-1, p, p+1, p+2])

labels = ['Budget', 'Mid-Range', 'Luxury']

df_train['C_Budget'] = pd.cut(df_train['Price_USD'], bins=edges, labels=labels, include_lowest=True)
df_val['C_Budget']   = pd.cut(df_val['Price_USD'],   bins=edges, labels=labels, include_lowest=True)

C_features_cat = ['C_Region', 'C_Battery_Chem', 'C_Connector', 'C_Charge_Speed', 'C_Budget']


# 10


encoder_c = OneHotEncoder(handle_unknown='ignore', **OHE_KW)
C_cat_train = encoder_c.fit_transform(df_train[C_features_cat])
C_cat_val   = encoder_c.transform(df_val[C_features_cat])

C_num_train = df_train[C_features_num].to_numpy(dtype=np.float32)
C_num_val   = df_val[C_features_num].to_numpy(dtype=np.float32)

C_train = np.hstack([C_cat_train, C_num_train.reshape(-1, len(C_features_num))]).astype(np.float32)
C_val   = np.hstack([C_cat_val,   C_num_val.reshape(-1, len(C_features_num))]).astype(np.float32)

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(df_train[Y_features]).astype(np.float32)
Y_val   = scaler_y.transform(df_val[Y_features]).astype(np.float32)

print("\nData shapes ready for training:")
print("Y_train:", Y_train.shape, "| C_train:", C_train.shape)
print("Y_val:  ", Y_val.shape,   "| C_val:  ", C_val.shape)

c_cat_names = list(encoder_c.get_feature_names_out(C_features_cat))
c_feature_names = c_cat_names + C_features_num

print("C categorical one-hot features:", len(c_cat_names))
print("C total features:", len(c_feature_names))


# 11


np.savez(os.path.join(ART_DIR, 'training_data.npz'),
         Y_train=Y_train, C_train=C_train, Y_val=Y_val, C_val=C_val)

joblib.dump(scaler_y, os.path.join(ART_DIR, 'scaler_y.pkl'))
joblib.dump(encoder_c, os.path.join(ART_DIR, 'encoder_c.pkl'))
joblib.dump(Y_features, os.path.join(ART_DIR, 'y_features.pkl'))
joblib.dump(c_feature_names, os.path.join(ART_DIR, 'c_features_names.pkl'))

with open(os.path.join(ART_DIR, 'budget_bins.json'), 'w') as f:
    json.dump({"edges": edges.tolist(), "labels": labels}, f)

print("\nPreprocessing complete. Artifacts saved to 'artifacts/' folder.")


# 12


get_ipython().system('zip -r artifacts.zip /kaggle/working/artifacts')

from IPython.display import FileLink
FileLink('artifacts.zip')

