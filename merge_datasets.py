import pandas as pd

print("="*60)
print("MERGING WEATHER + AIR QUALITY DATA")
print("="*60)

# Load both datasets
print("\nLoading datasets...")
weather_df = pd.read_csv('vizag_weather_historical_openmeteo.csv')
airquality_df = pd.read_csv('vizag_air_quality_historical_openmeteo.csv')

print(f"✓ Weather records: {len(weather_df)}")
print(f"✓ Air quality records: {len(airquality_df)}")

# Merge on timestamp
print("\nMerging on timestamp...")
merged_df = pd.merge(
    weather_df, 
    airquality_df, 
    on='timestamp', 
    how='inner'
)

print(f"✓ Merged records: {len(merged_df)}")

# Display columns
print(f"\nTotal columns: {len(merged_df.columns)}")
print("Columns:", merged_df.columns.tolist())

# Save merged dataset
output_file = 'vizag_complete_dataset.csv'
merged_df.to_csv(output_file, index=False)
print(f"\n✓ Complete dataset saved to: {output_file}")

# Show summary statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

print(f"\nDate range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
print(f"Total hours of data: {len(merged_df)}")
print(f"Approximate days: {len(merged_df) / 24:.1f}")

print("\nPM2.5 Statistics:")
print(f"  Mean: {merged_df['pm2_5'].mean():.2f} µg/m³")
print(f"  Median: {merged_df['pm2_5'].median():.2f} µg/m³")
print(f"  Min: {merged_df['pm2_5'].min():.2f} µg/m³")
print(f"  Max: {merged_df['pm2_5'].max():.2f} µg/m³")
print(f"  Std Dev: {merged_df['pm2_5'].std():.2f} µg/m³")

print("\nTemperature Statistics:")
print(f"  Mean: {merged_df['temperature'].mean():.2f} °C")
print(f"  Min: {merged_df['temperature'].min():.2f} °C")
print(f"  Max: {merged_df['temperature'].max():.2f} °C")

print("\nMissing Values per Column:")
missing = merged_df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(missing[missing > 0])

print("\nFirst 3 rows:")
print(merged_df.head(3))

print("\n" + "="*60)
print("✓ PHASE 1: DATA COLLECTION COMPLETE!")
print("="*60)