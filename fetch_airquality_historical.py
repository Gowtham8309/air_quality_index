import requests
import csv

# Visakhapatnam coordinates
LAT = 17.6868
LON = 83.2185
START_DATE = '2023-01-01'
END_DATE = '2026-02-22'

print("Fetching historical air quality from Open-Meteo...")

url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    'latitude': LAT,
    'longitude': LON,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'hourly': ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust', 'aerosol_optical_depth'],
    'timezone': 'Asia/Kolkata'
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    hourly = data['hourly']
    times = hourly['time']
    
    records = []
    for i in range(len(times)):
        record = {
            'timestamp': times[i],
            'pm2_5': hourly['pm2_5'][i],
            'pm10': hourly['pm10'][i],
            'co': hourly['carbon_monoxide'][i],
            'no2': hourly['nitrogen_dioxide'][i],
            'so2': hourly['sulphur_dioxide'][i],
            'o3': hourly['ozone'][i],
            'dust': hourly['dust'][i],
            'aod': hourly['aerosol_optical_depth'][i]
        }
        records.append(record)
    
    # Save to CSV
    filename = 'vizag_air_quality_historical_openmeteo.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    
    print(f"✓ Success! Downloaded {len(records)} air quality records")
    print(f"✓ Data saved to: {filename}")
    print(f"✓ Date range: {times[0]} to {times[-1]}")
    
else:
    print(f"Error: {response.status_code}")
    print(response.text)