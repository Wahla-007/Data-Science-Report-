import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD AND SETUP
# ==========================================
# Load the dataset
file_path = 'global_air_quality_data_10000.csv'
df = pd.read_csv(file_path)

# Convert Date column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Create a mapping of City -> Country so we don't lose this info during aggregation
city_country_map = df[['City', 'Country']].drop_duplicates().set_index('City')['Country'].to_dict()

# ==========================================
# STEP 2: AGGREGATION (Handling Duplicates)
# ==========================================
# Explanation: The raw data has multiple rows per day (e.g., morning/evening).
# We group by City and Date, taking the MEAN to get a daily average.
numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
df_daily = df.groupby(['City', 'Date'])[numeric_cols].mean().reset_index()

print(f"Rows after aggregation: {len(df_daily)} (Duplicates merged)")

# ==========================================
# STEP 3: RESAMPLING (Revealing Hidden Gaps)
# ==========================================
# Explanation: Some cities skip days. We create a full calendar for 2023 
# and force the data to conform to it, creating NaN rows for missing days.

# Get unique cities and the full date range found in the data
cities = df_daily['City'].unique()
all_dates = pd.date_range(start=df_daily['Date'].min(), end=df_daily['Date'].max(), freq='D')

# Create a MultiIndex (Every City x Every Date)
idx = pd.MultiIndex.from_product([cities, all_dates], names=['City', 'Date'])

# Reindex the dataframe. This creates the missing rows (NaNs)
df_full = df_daily.set_index(['City', 'Date']).reindex(idx)

# ==========================================
# STEP 4: INTERPOLATION (Filling Missing Values)
# ==========================================
# Explanation: We fill the missing days using Linear Interpolation.
# CRITICAL: We group by 'City' so we don't interpolate between London and Paris.

df_clean = df_full.groupby(level='City').transform(lambda x: x.interpolate(method='linear'))

# Handle edge cases (e.g., if the first day of the year is missing, linear interp can't fill it)
# We use Backfill (bfill) for starts and Forward fill (ffill) for ends
df_clean = df_clean.bfill().ffill()

# ==========================================
# STEP 5: FINAL CLEANUP
# ==========================================
# Reset index to make 'City' and 'Date' normal columns again
df_clean = df_clean.reset_index()

# Restore the 'Country' column using our map
df_clean['Country'] = df_clean['City'].map(city_country_map)

# Reorder columns to look like the original
cols = ['City', 'Country', 'Date'] + numeric_cols
df_final = df_clean[cols]

# ==========================================
# VERIFICATION
# ==========================================
print("\nCleaning Complete.")
print(f"Final Shape: {df_final.shape}")
print(f"Missing Values:\n{df_final.isnull().sum()}")

# Show the first few rows
print("\nFirst 5 rows of clean data:")
print(df_final.head())

# OPTIONAL: Save to CSV for the next step (EDA/Modeling)
df_final.to_csv('cleaned_air_quality_data.csv', index=False)