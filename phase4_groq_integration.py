import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from datetime import datetime, timedelta
from groq import Groq
import json

print("="*70)
print("PHASE 4: GENAI INTEGRATION WITH GROQ")
print("="*70)

# Initialize Groq client
GROQ_API_KEY = "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"  # Replace with your actual key
client = Groq(api_key=GROQ_API_KEY)

# Load trained models
print("\n📥 Loading trained models...")

def build_lstm_model(time_steps, n_features):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

# Load models and scaler
lstm_model = build_lstm_model(24, 42)
lstm_model.load_weights('lstm_model_best.h5')

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ensemble_weights.pkl', 'rb') as f:
    ensemble_weights = pickle.load(f)

print("✓ Models loaded successfully")

# AQI calculation and categorization
def calculate_aqi_from_pm25(pm25):
    """Convert PM2.5 to AQI using EPA breakpoints"""
    if pm25 <= 12.0:
        return int(((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0), "Good"
    elif pm25 <= 35.4:
        return int(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51), "Moderate"
    elif pm25 <= 55.4:
        return int(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101), "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return int(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151), "Unhealthy"
    elif pm25 <= 250.4:
        return int(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201), "Very Unhealthy"
    else:
        return int(((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301), "Hazardous"

def get_health_recommendations(aqi_category):
    """Get health recommendations based on AQI category"""
    recommendations = {
        "Good": {
            "general": "Air quality is satisfactory. Enjoy outdoor activities!",
            "sensitive": "No precautions needed.",
            "activities": "All outdoor activities are safe."
        },
        "Moderate": {
            "general": "Air quality is acceptable for most people.",
            "sensitive": "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
            "activities": "Most outdoor activities are fine."
        },
        "Unhealthy for Sensitive Groups": {
            "general": "General public is not likely to be affected.",
            "sensitive": "People with respiratory or heart conditions, children, and older adults should limit prolonged outdoor exertion.",
            "activities": "Reduce intense outdoor activities if you're sensitive."
        },
        "Unhealthy": {
            "general": "Everyone may begin to experience health effects.",
            "sensitive": "Sensitive groups should avoid outdoor exertion.",
            "activities": "Limit outdoor activities, especially intense exercise."
        },
        "Very Unhealthy": {
            "general": "Health alert: everyone may experience serious health effects.",
            "sensitive": "Sensitive groups should avoid all outdoor activities.",
            "activities": "Stay indoors. Avoid all outdoor physical activities."
        },
        "Hazardous": {
            "general": "Health warning: everyone should avoid outdoor activities.",
            "sensitive": "Remain indoors and keep activity levels low.",
            "activities": "Emergency conditions. Stay indoors with windows closed."
        }
    }
    return recommendations.get(aqi_category, recommendations["Moderate"])

def predict_pm25_future(current_features, hours_ahead=24):
    """Predict PM2.5 for next N hours"""
    # This is a simplified version - use your actual feature engineering
    predictions = []
    
    # For demo, we'll use the LSTM model
    # In production, you'd need to properly engineer features for future timestamps
    current_seq = current_features[-24:].reshape(1, 24, 42)
    
    for h in range(hours_ahead):
        # Predict next hour
        pred = lstm_model.predict(current_seq, verbose=0)[0][0]
        predictions.append(pred)
        
        # Note: This is simplified. In production, you'd update all features
        # For now, we'll just append and shift
        
    return predictions

def generate_explanation_with_groq(current_pm25, forecast_24h, forecast_48h, weather_conditions):
    """Generate natural language explanation using Groq"""
    
    current_aqi, current_category = calculate_aqi_from_pm25(current_pm25)
    forecast_24h_aqi, forecast_24h_category = calculate_aqi_from_pm25(forecast_24h)
    forecast_48h_aqi, forecast_48h_category = calculate_aqi_from_pm25(forecast_48h)
    
    # Get health recommendations
    current_health = get_health_recommendations(current_category)
    
    # Create context for Groq
    context = f"""You are an air quality expert assistant for Visakhapatnam, India. 

Current Air Quality:
- PM2.5: {current_pm25:.1f} µg/m³
- AQI: {current_aqi} ({current_category})

24-Hour Forecast:
- PM2.5: {forecast_24h:.1f} µg/m³
- AQI: {forecast_24h_aqi} ({forecast_24h_category})

48-Hour Forecast:
- PM2.5: {forecast_48h:.1f} µg/m³
- AQI: {forecast_48h_aqi} ({forecast_48h_category})

Weather Conditions:
{weather_conditions}

Health Recommendations:
- General Public: {current_health['general']}
- Sensitive Groups: {current_health['sensitive']}
- Activities: {current_health['activities']}

Provide a concise, friendly explanation of the air quality situation. Include:
1. Current air quality status
2. What to expect in the next 24-48 hours
3. Practical health advice
4. Reasons for any changes (based on weather)

Keep it under 150 words and conversational."""

    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful air quality expert who explains PM2.5 levels and health impacts in simple, friendly language for the general public in India."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            model="llama-3.3-70b-versatile",  # Fast and accurate
            temperature=0.7,
            max_tokens=300
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Current air quality is {current_category} with PM2.5 at {current_pm25:.1f} µg/m³. {current_health['general']}"

def answer_question_with_groq(question, context_data):
    """Answer user questions about air quality using Groq"""
    
    system_prompt = """You are an air quality expert assistant for Visakhapatnam, India. 
    You help people understand PM2.5 levels, AQI, health impacts, and provide practical advice.
    Be friendly, concise, and scientifically accurate. Use simple language that anyone can understand."""
    
    user_prompt = f"""Context:
{json.dumps(context_data, indent=2)}

User Question: {question}

Provide a helpful, concise answer (under 100 words)."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=200
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "I'm having trouble connecting right now. Please try again."

# Demo function
def demo_air_quality_chatbot():
    """Demonstrate the air quality forecasting chatbot"""
    
    print("\n" + "="*70)
    print("🌍 VISAKHAPATNAM AIR QUALITY CHATBOT - DEMO")
    print("="*70)
    
    # Simulate current conditions (in production, fetch from API)
    current_pm25 = 37.5  # Example value
    forecast_24h = 42.3
    forecast_48h = 35.8
    
    weather_conditions = """
    Temperature: 27°C
    Humidity: 74%
    Wind Speed: 4.8 km/h
    Cloud Cover: 42%
    """
    
    # Generate main explanation
    print("\n📊 Generating air quality report...")
    explanation = generate_explanation_with_groq(
        current_pm25, 
        forecast_24h, 
        forecast_48h, 
        weather_conditions
    )
    
    print("\n" + "-"*70)
    print(explanation)
    print("-"*70)
    
    # Demo Q&A
    print("\n💬 Q&A Demo:\n")
    
    context = {
        "current_pm25": current_pm25,
        "current_aqi": calculate_aqi_from_pm25(current_pm25)[0],
        "forecast_24h": forecast_24h,
        "forecast_48h": forecast_48h,
        "location": "Visakhapatnam",
        "weather": weather_conditions
    }
    
    questions = [
        "Should I go for a morning jog tomorrow?",
        "Is it safe for my kids to play outside?",
        "What causes PM2.5 pollution in Visakhapatnam?"
    ]
    
    for q in questions:
        print(f"❓ User: {q}")
        answer = answer_question_with_groq(q, context)
        print(f"🤖 Assistant: {answer}\n")
    
    print("="*70)
    print("✅ PHASE 4: GENAI INTEGRATION COMPLETE!")
    print("="*70)

# Run demo
if __name__ == "__main__":
    demo_air_quality_chatbot()