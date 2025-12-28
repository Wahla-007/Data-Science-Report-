# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Load the Readable Data (Unscaled)
# # Use the file we created before scaling
# df_eda = pd.read_csv('cleaned_data_before_scaling.csv')
# df_eda['Date'] = pd.to_datetime(df_eda['Date'])

# # Set plot style
# sns.set(style="whitegrid")

# # ==========================================
# # A. UNIVARIATE ANALYSIS
# # (Analyzing one variable at a time)
# # ==========================================
# print("Generating Univariate Plots...")
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# # Plot 1: Distribution of PM2.5 (The main pollutant)
# sns.histplot(df_eda['PM2.5'], kde=True, ax=axes[0], color='skyblue')
# axes[0].set_title('Univariate: Distribution of PM2.5 Levels')
# axes[0].set_xlabel('PM2.5 Concentration')

# # Plot 2: Count of AQI Categories (The target)
# sns.countplot(x='AQI_Category', data=df_eda, ax=axes[1], 
#               order=['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
#               palette='viridis')
# axes[1].set_title('Univariate: Class Imbalance Check')
# plt.tight_layout()
# plt.show()

# # ==========================================
# # B. BIVARIATE ANALYSIS
# # (Analyzing relationships between two variables)
# # ==========================================
# print("Generating Bivariate Plots...")
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# # Plot 1: Does Temperature affect PM2.5?
# sns.scatterplot(x='Temperature', y='PM2.5', hue='AQI_Category', data=df_eda, ax=axes[0], alpha=0.6)
# axes[0].set_title('Bivariate: Temperature vs. PM2.5')

# # Plot 2: How does Wind Speed vary across AQI Categories?
# sns.boxplot(x='AQI_Category', y='Wind Speed', data=df_eda, ax=axes[1], 
#             order=['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
# axes[1].set_title('Bivariate: Wind Speed by Air Quality')
# plt.tight_layout()
# plt.show()

# # ==========================================
# # C. CORRELATION ANALYSIS
# # (Heatmap of all numerical features)
# # ==========================================
# print("Generating Correlation Heatmap...")
# plt.figure(figsize=(10, 8))

# # Select only numeric columns for correlation
# numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
# corr_matrix = df_eda[numeric_cols].corr()

# # Plot Heatmap

# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Matrix: Feature Relationships')
# plt.show()

# # ==========================================
# # D. COMPARATIVE ANALYSIS
# # (Comparing different Cities)
# # ==========================================
# print("Generating Comparative Plots...")
# plt.figure(figsize=(12, 6))

# # Calculate average PM2.5 per city
# city_avg = df_eda.groupby('City')['PM2.5'].mean().sort_values(ascending=False).head(10)

# # Bar Chart
# sns.barplot(x=city_avg.values, y=city_avg.index, palette='magma')
# plt.title('Comparative: Top 10 Most Polluted Cities (Avg PM2.5)')
# plt.xlabel('Average PM2.5 Concentration')
# plt.show()

# # ==========================================
# # E. IDENTIFY CYCLES (Time Series)
# # (Visualizing trends over time)
# # ==========================================
# print("Generating Time Series Cycles...")
# plt.figure(figsize=(14, 6))

# # Select one city to visualize cycles (e.g., the first one in the list)
# sample_city = df_eda['City'].unique()[0]
# city_data = df_eda[df_eda['City'] == sample_city]

# # Line Chart
# sns.lineplot(x='Date', y='PM2.5', data=city_data, label='PM2.5 Level')
# # Add a rolling average to see the "Cycle" clearly
# sns.lineplot(x='Date', y=city_data['PM2.5'].rolling(window=30).mean(), 
#              data=city_data, label='30-Day Trend (Cycle)', color='red', linewidth=2)

# plt.title(f'Time Series Cycles: Air Quality Trends in {sample_city}')
# plt.ylabel('PM2.5 Concentration')
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load Data
# Ensure you have 'cleaned_data_before_scaling.csv' in the same folder
try:
    df_eda = pd.read_csv('cleaned_data_before_scaling.csv')
    df_eda['Date'] = pd.to_datetime(df_eda['Date'])
except FileNotFoundError:
    print("Error: 'cleaned_data_before_scaling.csv' not found. Run the cleaning step first.")
    exit()

# Set visual style
sns.set(style="whitegrid")

# ==========================================
# A. UNIVARIATE ANALYSIS
# ==========================================
print("Generating Univariate Plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Histogram
sns.histplot(df_eda['PM2.5'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Univariate: Distribution of PM2.5')

# Plot 2: Count Plot (FIXED WARNING)
# Added hue='AQI_Category' and legend=False
sns.countplot(x='AQI_Category', data=df_eda, ax=axes[1], 
              hue='AQI_Category', legend=False,
              order=['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
              palette='viridis')
axes[1].set_title('Univariate: Class Imbalance')

plt.tight_layout()
plt.savefig('EDA_1_Univariate.png')  # <--- SAVES FILE INSTEAD OF POPUP
print("-> Saved 'EDA_1_Univariate.png'")
plt.close()

# ==========================================
# B. BIVARIATE ANALYSIS
# ==========================================
print("Generating Bivariate Plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Scatter
sns.scatterplot(x='Temperature', y='PM2.5', hue='AQI_Category', data=df_eda, ax=axes[0], alpha=0.6)
axes[0].set_title('Temperature vs. PM2.5')

# Plot 2: Boxplot (FIXED WARNING)
# Added hue='AQI_Category' and legend=False
sns.boxplot(x='AQI_Category', y='Wind Speed', data=df_eda, ax=axes[1],
            hue='AQI_Category', legend=False,
            order=['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
            palette='coolwarm')
axes[1].set_title('Wind Speed by Category')

plt.tight_layout()
plt.savefig('EDA_2_Bivariate.png')
print("-> Saved 'EDA_2_Bivariate.png'")
plt.close()

# ==========================================
# C. CORRELATION ANALYSIS
# ==========================================
print("Generating Correlation Matrix...")
plt.figure(figsize=(10, 8))

numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
corr_matrix = df_eda[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')

plt.savefig('EDA_3_Correlation.png')
print("-> Saved 'EDA_3_Correlation.png'")
plt.close()

# ==========================================
# D. COMPARATIVE ANALYSIS
# ==========================================
print("Generating Comparative Plots...")
plt.figure(figsize=(12, 6))

city_pollution = df_eda.groupby('City')['PM2.5'].mean().sort_values(ascending=False).head(10)

# FIXED WARNING: Added hue=city_pollution.index and legend=False
sns.barplot(x=city_pollution.values, y=city_pollution.index, 
            hue=city_pollution.index, legend=False,
            palette='magma')

plt.title('Top 10 Most Polluted Cities')
plt.xlabel('Avg PM2.5')

plt.savefig('EDA_4_Comparative.png')
print("-> Saved 'EDA_4_Comparative.png'")
plt.close()

# ==========================================
# E. IDENTIFY CYCLES (Global)
# ==========================================
print("Generating Global Cycles...")

# Aggregate globally to prevent errors with specific cities
global_daily_avg = df_eda.groupby('Date')['PM2.5'].mean()
decomposition = seasonal_decompose(global_daily_avg, model='additive', period=30)

plt.figure(figsize=(12, 10))

plt.subplot(411)
plt.plot(global_daily_avg, label='Global Avg', color='navy')
plt.legend(loc='upper left')
plt.title('Global Pollution Cycles')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality (30-Day)', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='grey')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('EDA_5_Cycles.png')
print("-> Saved 'EDA_5_Cycles.png'")
plt.close()

print("\nAll EDA charts have been saved successfully!")