import requests
import csv
from datetime import datetime

# Configuration for Visakhapatnam
LAT = 17.6868
LON = 83.2185
START_DATE = '2023-01-01'
END_DATE = '2026-02-22'
print("Fetching historical weather data from Open-Meteo...")

# API endpoint - NO API KEY NEEDED!
url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters - request all weather variables you need
params = {
    'latitude': LAT,
    'longitude': LON,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'hourly': [
        'temperature_2m',           # Temperature at 2m height
        'relative_humidity_2m',     # Humidity
        'precipitation',            # Rainfall
        'surface_pressure',         # Atmospheric pressure
        'windspeed_10m',           # Wind speed
        'winddirection_10m',       # Wind direction
        'cloudcover'               # Cloud coverage
    ],
    'timezone': 'Asia/Kolkata'     # IST timezone
}

# Make the request
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # Extract hourly data
    hourly = data['hourly']
    times = hourly['time']
    
    # Prepare data for CSV
    weather_records = []
    
    for i in range(len(times)):
        record = {
            'timestamp': times[i],
            'temperature': hourly['temperature_2m'][i],
            'humidity': hourly['relative_humidity_2m'][i],
            'precipitation': hourly['precipitation'][i],
            'pressure': hourly['surface_pressure'][i],
            'wind_speed': hourly['windspeed_10m'][i],
            'wind_direction': hourly['winddirection_10m'][i],
            'cloud_cover': hourly['cloudcover'][i]
        }
        weather_records.append(record)
    
    # Save to CSV
    filename = 'vizag_weather_historical_openmeteo.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=weather_records[0].keys())
        writer.writeheader()
        writer.writerows(weather_records)
    
    print(f"✓ Success! Downloaded {len(weather_records)} hourly records")
    print(f"✓ Data saved to: {filename}")
    print(f"✓ Date range: {times[0]} to {times[-1]}")
    
else:
    print(f"Error: {response.status_code}")
    print(response.text)