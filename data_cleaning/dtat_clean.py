import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================
import os
print("Loading dataset...")
# Construct the full path to the CSV file based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'global_air_quality_data_10000.csv')
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

# Define numeric columns explicitly
numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']

# ==========================================
# 2. DATA CLEANING & INTERPOLATION (a)
# ==========================================
print("Step A: Cleaning & Interpolation (Rounding to 2 decimals)...")
# A. Aggregation
df_daily = df.groupby(['City', 'Date'])[numeric_cols].mean().reset_index()

# B. Resampling
cities = df_daily['City'].unique()
all_dates = pd.date_range(start=df_daily['Date'].min(), end=df_daily['Date'].max(), freq='D')
idx = pd.MultiIndex.from_product([cities, all_dates], names=['City', 'Date'])
df_full = df_daily.set_index(['City', 'Date']).reindex(idx)

# C. Interpolation (Round to 2 decimals for readability)
df_clean = df_full.groupby(level='City')[numeric_cols].transform(lambda x: x.interpolate(method='linear'))
df_clean = df_clean.bfill().ffill().round(2).reset_index()

# ==========================================
# 3. OUTLIER HANDLING (b)
# ==========================================
print("Step B: Handling Outliers...")
def cap_outliers(df, cols, factor=1.5):
    df_capped = df.copy()
    for col in cols:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        # Round bounds to 2 decimals
        lower_bound = round(Q1 - (factor * IQR), 2)
        upper_bound = round(Q3 + (factor * IQR), 2)
        
        df_capped[col] = np.where(df_capped[col] < lower_bound, lower_bound, df_capped[col])
        df_capped[col] = np.where(df_capped[col] > upper_bound, upper_bound, df_capped[col])
    return df_capped

df_processed = cap_outliers(df_clean, numeric_cols)

# ==========================================
# 4. ENCODE AQI CATEGORY (c)
# ==========================================
print("Step C: Encoding AQI Categories...")
def get_aqi_category_custom(pm25):
    if pm25 <= 66.0: return 'Good'
    elif pm25 <= 99.0: return 'Moderate'
    elif pm25 <= 149.0: return 'Unhealthy'
    elif pm25 <= 199.0: return 'Very Unhealthy'
    else: return 'Hazardous'

df_processed['AQI_Category'] = df_processed['PM2.5'].apply(get_aqi_category_custom)

# SAVE FILE 1: Pre-Scaled Data (For your viewing/EDA)
df_processed.to_csv('cleaned_data_before_scaling.csv', index=False)
print("-> Saved 'cleaned_data_before_scaling.csv'")

# ==========================================
# 5. SCALE & SPLIT (d & e)
# ==========================================
print("Step D & E: Scaling and Splitting...")

# Separate Columns
# We KEEP 'City', 'Date', and 'PM2.5' as metadata so you can see them in the file,
# but we DO NOT scale them or feed them to the model.
meta_cols = ['City', 'Date', 'PM2.5']
features_to_scale = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
target_col = ['AQI_Category']

# Scale Features
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(df_processed[features_to_scale])

# Convert back to DataFrame and round to 4 decimals (Efficient!)
X_scaled = pd.DataFrame(X_scaled_array, columns=features_to_scale).round(4)

# Re-assemble everything into one big table before splitting
# Structure: [City, Date, PM2.5] + [Scaled Numbers] + [AQI Category]
df_final_assembled = pd.concat([
    df_processed[meta_cols], 
    X_scaled, 
    df_processed[target_col]
], axis=1)

# Split
train_df, test_df = train_test_split(df_final_assembled, test_size=0.2, random_state=42)

# Save Files
train_df.to_csv('final_train_data.csv', index=False)
test_df.to_csv('final_test_data.csv', index=False)

print("-> Saved 'final_train_data.csv'")
print("-> Saved 'final_test_data.csv'")
print("\nSuccess! Check the CSV files.")