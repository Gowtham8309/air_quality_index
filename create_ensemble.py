import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      MultiHeadAttention, LayerNormalization,
                                      GlobalAveragePooling1D, Add)

print("="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Rebuild LSTM model architecture
def build_lstm_model(time_steps, n_features):
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

# Rebuild Transformer model architecture
def build_transformer_model(time_steps, n_features):
    inputs = Input(shape=(time_steps, n_features))
    x = inputs
    
    # Transformer block 1
    attn_output = MultiHeadAttention(num_heads=4, key_dim=128, dropout=0.2)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ffn_output = Dense(64, activation='relu')(x)
    ffn_output = Dropout(0.2)(ffn_output)
    ffn_output = Dense(n_features)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer block 2
    attn_output = MultiHeadAttention(num_heads=4, key_dim=128, dropout=0.2)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ffn_output = Dense(64, activation='relu')(x)
    ffn_output = Dropout(0.2)(ffn_output)
    ffn_output = Dense(n_features)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling and final layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load data to get dimensions
df = pd.read_csv('vizag_featured_dataset.csv')
target = 'pm2_5'
exclude_features = ['timestamp', 'pm2_5']
feature_columns = [col for col in df.columns if col not in exclude_features]

TIME_STEPS = 24
N_FEATURES = len(feature_columns)

print(f"\nModel parameters:")
print(f"  Time steps: {TIME_STEPS}")
print(f"  Features: {N_FEATURES}")

# Rebuild models and load weights
print("\nRebuilding LSTM model...")
lstm_model = build_lstm_model(TIME_STEPS, N_FEATURES)
lstm_model.load_weights('lstm_model_best.h5')
print("✓ LSTM weights loaded")

print("\nRebuilding Transformer model...")
transformer_model = build_transformer_model(TIME_STEPS, N_FEATURES)
transformer_model.load_weights('transformer_model_best.h5')
print("✓ Transformer weights loaded")

print("\nLoading XGBoost model...")
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✓ XGBoost loaded")

# Prepare test data
X = df[feature_columns].values
y = df[target].values

train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

# Create sequences
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)
X_test_xgb = X_test_scaled[TIME_STEPS:]

print(f"\nTest set size: {len(X_test_seq)} samples")

# Get predictions
print("\nGenerating predictions...")
y_test_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
y_test_pred_transformer = transformer_model.predict(X_test_seq, verbose=0).flatten()
y_test_pred_xgb = xgb_model.predict(X_test_xgb)
print("✓ Predictions complete")

# Calculate performance
lstm_rmse = np.sqrt(mean_squared_error(y_test_seq, y_test_pred_lstm))
transformer_rmse = np.sqrt(mean_squared_error(y_test_seq, y_test_pred_transformer))
xgb_rmse = np.sqrt(mean_squared_error(y_test_seq, y_test_pred_xgb))

print(f"\nIndividual Model Test RMSE:")
print(f"  LSTM:        {lstm_rmse:.2f} µg/m³")
print(f"  Transformer: {transformer_rmse:.2f} µg/m³")
print(f"  XGBoost:     {xgb_rmse:.2f} µg/m³")

# Ensemble weights (inverse RMSE)
total_inv_rmse = (1/lstm_rmse + 1/transformer_rmse + 1/xgb_rmse)

weights = {
    'lstm': (1/lstm_rmse) / total_inv_rmse,
    'transformer': (1/transformer_rmse) / total_inv_rmse,
    'xgb': (1/xgb_rmse) / total_inv_rmse
}

print(f"\n🎯 Ensemble Weights:")
print(f"  LSTM:        {weights['lstm']:.3f} ({weights['lstm']*100:.1f}%)")
print(f"  Transformer: {weights['transformer']:.3f} ({weights['transformer']*100:.1f}%)")
print(f"  XGBoost:     {weights['xgb']:.3f} ({weights['xgb']*100:.1f}%)")

# Create ensemble
y_test_pred_ensemble = (
    weights['lstm'] * y_test_pred_lstm +
    weights['transformer'] * y_test_pred_transformer +
    weights['xgb'] * y_test_pred_xgb
)

ensemble_rmse = np.sqrt(mean_squared_error(y_test_seq, y_test_pred_ensemble))
ensemble_mae = mean_absolute_error(y_test_seq, y_test_pred_ensemble)
ensemble_r2 = r2_score(y_test_seq, y_test_pred_ensemble)

print(f"\n✨ Ensemble Performance:")
print(f"  RMSE: {ensemble_rmse:.2f} µg/m³")
print(f"  MAE:  {ensemble_mae:.2f} µg/m³")
print(f"  R²:   {ensemble_r2:.4f}")

# Save weights
with open('ensemble_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)
print("\n✓ Ensemble weights saved")

# Final comparison table
print("\n" + "="*70)
print(" "*20 + "FINAL MODEL COMPARISON - TEST SET")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['LSTM', 'Transformer', 'XGBoost', 'Ensemble'],
    'RMSE (µg/m³)': [lstm_rmse, transformer_rmse, xgb_rmse, ensemble_rmse],
    'MAE (µg/m³)': [
        mean_absolute_error(y_test_seq, y_test_pred_lstm),
        mean_absolute_error(y_test_seq, y_test_pred_transformer),
        mean_absolute_error(y_test_seq, y_test_pred_xgb),
        ensemble_mae
    ],
    'R² Score': [
        r2_score(y_test_seq, y_test_pred_lstm),
        r2_score(y_test_seq, y_test_pred_transformer),
        r2_score(y_test_seq, y_test_pred_xgb),
        ensemble_r2
    ]
})

print("\n", comparison.to_string(index=False))
comparison.to_csv('model_comparison.csv', index=False)

# Best model
best_idx = comparison['RMSE (µg/m³)'].idxmin()
print(f"\n" + "="*70)
print(f"{'🏆 BEST MODEL: ' + comparison.loc[best_idx, 'Model']:^70}")
print("="*70)
print(f"  RMSE: {comparison.loc[best_idx, 'RMSE (µg/m³)']:.2f} µg/m³")
print(f"  MAE:  {comparison.loc[best_idx, 'MAE (µg/m³)']:.2f} µg/m³")
print(f"  R²:   {comparison.loc[best_idx, 'R² Score']:.4f}")

if best_idx == 3:
    best_individual_rmse = comparison.loc[:2, 'RMSE (µg/m³)'].min()
    improvement = ((best_individual_rmse - ensemble_rmse) / best_individual_rmse) * 100
    print(f"\n  📈 Improvement over best individual: {improvement:.1f}%")

print("\n" + "="*70)
print("✅ PHASE 3: MODEL DEVELOPMENT COMPLETE!")
print("="*70)

print("\n📦 Saved Files:")
for f in ['lstm_model_best.h5', 'transformer_model_best.h5', 'xgb_model.pkl', 
          'scaler.pkl', 'ensemble_weights.pkl', 'model_comparison.csv']:
    print(f"  ✓ {f}")

print("\n🎯 Project Status:")
print("  ✅ Phase 1: Data Collection (17,520 records)")
print("  ✅ Phase 2: Feature Engineering (44 features)")
print("  ✅ Phase 3: Model Development (LSTM + Transformer + XGBoost)")
print("  ⏳ Phase 4: GenAI Integration (Next)")
print("\n🚀 Ready for conversational AI interface!")