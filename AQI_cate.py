import pandas as pd

# ==========================================
# 1. DEFINE CATEGORIZATION LOGIC
# ==========================================
def get_aqi_category(pm25):
    """
    Maps PM2.5 values to AQI Categories based on simplified EPA standards.
    """
    if pm25 <= 12.0:
        return 'Good'
    elif pm25 <= 35.4:
        return 'Moderate'
    elif pm25 <= 150.4:
        return 'Unhealthy'
    else:
        return 'Hazardous'

# ==========================================
# 2. APPLY TO DATASET
# ==========================================
# We assume 'df_final' is your clean dataframe from the previous step.
# If your variable is named differently (e.g., df_no_outliers), change it here.
df_categorized = pd.read_csv('cleaned_air_quality_data.csv')

# Create the new column
df_categorized['AQI_Category'] = df_categorized['PM2.5'].apply(get_aqi_category)

# Check the distribution (How many rows per category?)
print("Category Counts:")
print(df_categorized['AQI_Category'].value_counts())

# ==========================================
# 3. SAVE TO CSV
# ==========================================
output_filename = 'AQI_categorized.csv'
df_categorized.to_csv(output_filename, index=False)

print(f"\nSuccess! File saved as: {output_filename}")
print(f"Total Columns: {len(df_categorized.columns)}")
print(f"New Column Added: 'AQI_Category'")

# Preview the new file data
print("\nFirst 5 rows of the new dataset:")
print(df_categorized[['City', 'Date', 'PM2.5', 'AQI_Category']].head())