import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("PHASE 2: FEATURE ENGINEERING")
print("="*60)

# Load merged dataset
df = pd.read_csv('vizag_complete_dataset.csv')
print(f"\nLoaded {len(df)} records")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp (important for time-series features)
df = df.sort_values('timestamp').reset_index(drop=True)

print("\n1. Creating Temporal Features...")
# Extract temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Season (India-specific)
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Summer
    elif month in [6, 7, 8, 9]:
        return 2  # Monsoon
    else:
        return 3  # Post-monsoon

df['season'] = df['month'].apply(get_season)

print("  ✓ Created: hour, day_of_week, month, season, is_weekend")

print("\n2. Creating Lag Features for PM2.5...")
# Previous PM2.5 values
df['pm25_lag_1h'] = df['pm2_5'].shift(1)
df['pm25_lag_3h'] = df['pm2_5'].shift(3)
df['pm25_lag_6h'] = df['pm2_5'].shift(6)
df['pm25_lag_12h'] = df['pm2_5'].shift(12)
df['pm25_lag_24h'] = df['pm2_5'].shift(24)

print("  ✓ Created: lag features (1h, 3h, 6h, 12h, 24h)")

print("\n3. Creating Rolling Averages...")
# Rolling averages
df['pm25_rolling_3h'] = df['pm2_5'].rolling(window=3, min_periods=1).mean()
df['pm25_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
df['pm25_rolling_12h'] = df['pm2_5'].rolling(window=12, min_periods=1).mean()
df['pm25_rolling_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()

print("  ✓ Created: rolling averages (3h, 6h, 12h, 24h)")

print("\n4. Creating Trend Features...")
# PM2.5 trend (change from previous hour)
df['pm25_change_1h'] = df['pm2_5'] - df['pm25_lag_1h']
df['pm25_change_24h'] = df['pm2_5'] - df['pm25_lag_24h']

# Temperature change
df['temp_change_1h'] = df['temperature'] - df['temperature'].shift(1)

# Pressure change
df['pressure_change_1h'] = df['pressure'] - df['pressure'].shift(1)

print("  ✓ Created: change/trend features")

print("\n5. Creating Interaction Features...")
# Temperature-Humidity interaction
df['temp_humidity'] = df['temperature'] * df['humidity']

# Wind vector components
df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))

# Pressure × Humidity
df['pressure_humidity'] = df['pressure'] * df['humidity']

print("  ✓ Created: interaction features")

print("\n6. Creating Cyclical Features...")
# Cyclical encoding for hour (0-23)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Cyclical encoding for month (1-12)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("  ✓ Created: cyclical features for hour and month")

# Drop rows with NaN values (due to lag/rolling features)
print(f"\n7. Handling Missing Values...")
print(f"   Rows before: {len(df)}")
df_clean = df.dropna().reset_index(drop=True)
print(f"   Rows after: {len(df_clean)}")
print(f"   Dropped: {len(df) - len(df_clean)} rows")

# Save engineered dataset
output_file = 'vizag_featured_dataset.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n✓ Featured dataset saved to: {output_file}")

print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)
print(f"\nTotal features: {len(df_clean.columns)}")
print("\nFeature categories:")
print(f"  - Original features: 16")
print(f"  - Temporal features: 7 (hour, day, month, season, etc.)")
print(f"  - Lag features: 5 (1h, 3h, 6h, 12h, 24h)")
print(f"  - Rolling features: 4 (3h, 6h, 12h, 24h)")
print(f"  - Trend features: 4 (PM2.5, temp, pressure changes)")
print(f"  - Interaction features: 5 (temp×humidity, wind vectors, etc.)")
print(f"  - Cyclical features: 4 (hour and month sin/cos)")

print("\nAll columns:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "="*60)
print("✓ PHASE 2: FEATURE ENGINEERING COMPLETE!")
print("="*60)
print("\nNext: Phase 3 - Model Development")