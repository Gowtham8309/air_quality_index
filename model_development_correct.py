import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                      MultiHeadAttention, LayerNormalization,
                                      GlobalAveragePooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("="*60)
print("PHASE 3: MODEL DEVELOPMENT (LSTM + Transformer + XGBoost)")
print("="*60)

# Load featured dataset
df = pd.read_csv('vizag_featured_dataset.csv')
print(f"\n✓ Loaded {len(df)} records with {len(df.columns)} features")

# Define target and features
target = 'pm2_5'
exclude_features = ['timestamp', 'pm2_5']

feature_columns = [col for col in df.columns if col not in exclude_features]
print(f"\n✓ Using {len(feature_columns)} input features to predict PM2.5")

# Prepare X and y
X = df[feature_columns].values
y = df[target].values

print("\n" + "="*60)
print("STEP 1: TRAIN/VALIDATION/TEST SPLIT")
print("="*60)

# Chronological split: 70% train, 15% validation, 15% test
train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print("\n" + "="*60)
print("STEP 2: FEATURE SCALING")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to: scaler.pkl")

print("\n" + "="*60)
print("STEP 3: PREPARE SEQUENCE DATA FOR LSTM & TRANSFORMER")
print("="*60)

# Create sequences for time-series models
def create_sequences(X, y, time_steps=24):
    """Create sequences for LSTM/Transformer models"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24  # Use past 24 hours to predict next hour

# Create sequences
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

print(f"\nSequence shape for LSTM/Transformer:")
print(f"  X_train: {X_train_seq.shape} (samples, time_steps, features)")
print(f"  y_train: {y_train_seq.shape}")

print("\n" + "="*60)
print("STEP 4: BUILD AND TRAIN LSTM MODEL")
print("="*60)

# Build LSTM model
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nLSTM Model Architecture:")
lstm_model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('lstm_model_best.h5', monitor='val_loss', save_best_only=True)

print("\nTraining LSTM model...")
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Save final LSTM model
lstm_model.save('lstm_model_final.h5')
print("\n✓ LSTM model saved to: lstm_model_final.h5")

# LSTM predictions
y_train_pred_lstm = lstm_model.predict(X_train_seq, verbose=0).flatten()
y_val_pred_lstm = lstm_model.predict(X_val_seq, verbose=0).flatten()
y_test_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()

# Evaluate LSTM
def evaluate_model(y_true, y_pred, dataset_name, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - {dataset_name} Metrics:")
    print(f"  RMSE: {rmse:.2f} µg/m³")
    print(f"  MAE:  {mae:.2f} µg/m³")
    print(f"  R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

lstm_results = {
    'train': evaluate_model(y_train_seq, y_train_pred_lstm, "Train", "LSTM"),
    'val': evaluate_model(y_val_seq, y_val_pred_lstm, "Validation", "LSTM"),
    'test': evaluate_model(y_test_seq, y_test_pred_lstm, "Test", "LSTM")
}

print("\n" + "="*60)
print("STEP 5: BUILD AND TRAIN TRANSFORMER MODEL")
print("="*60)

# Build Transformer model
def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=[128], dropout=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout
        )(x, x)
        
        # Skip connection and normalization
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(x)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        
        # Skip connection and normalization
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Global pooling and final layers
    x = GlobalAveragePooling1D()(x)
    
    for dim in mlp_units:
        x = Dense(dim, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

transformer_model = build_transformer_model(
    input_shape=(TIME_STEPS, X_train_scaled.shape[1]),
    head_size=128,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    mlp_units=[64, 32],
    dropout=0.2
)

transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nTransformer Model Architecture:")
transformer_model.summary()

# Callbacks
early_stop_transformer = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_transformer = ModelCheckpoint('transformer_model_best.h5', monitor='val_loss', save_best_only=True)

print("\nTraining Transformer model...")
transformer_history = transformer_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop_transformer, checkpoint_transformer],
    verbose=1
)

# Save final Transformer model
transformer_model.save('transformer_model_final.h5')
print("\n✓ Transformer model saved to: transformer_model_final.h5")

# Transformer predictions
y_train_pred_transformer = transformer_model.predict(X_train_seq, verbose=0).flatten()
y_val_pred_transformer = transformer_model.predict(X_val_seq, verbose=0).flatten()
y_test_pred_transformer = transformer_model.predict(X_test_seq, verbose=0).flatten()

transformer_results = {
    'train': evaluate_model(y_train_seq, y_train_pred_transformer, "Train", "Transformer"),
    'val': evaluate_model(y_val_seq, y_val_pred_transformer, "Validation", "Transformer"),
    'test': evaluate_model(y_test_seq, y_test_pred_transformer, "Test", "Transformer")
}

print("\n" + "="*60)
print("STEP 6: BUILD AND TRAIN XGBOOST MODEL")
print("="*60)

# For XGBoost, we'll use the same data points as LSTM/Transformer for fair comparison
# We'll use the last observation from each sequence
print("\n" + "="*60)
print("STEP 6: BUILD AND TRAIN XGBOOST MODEL")
print("="*60)

# For XGBoost, we'll use the same data points as LSTM/Transformer for fair comparison
X_train_xgb = X_train_scaled[TIME_STEPS:]
y_train_xgb = y_train[TIME_STEPS:]

X_val_xgb = X_val_scaled[TIME_STEPS:]
y_val_xgb = y_val[TIME_STEPS:]

X_test_xgb = X_test_scaled[TIME_STEPS:]
y_test_xgb = y_test[TIME_STEPS:]

print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
    early_stopping_rounds=20  # Move this here for newer XGBoost versions
)

xgb_model.fit(
    X_train_xgb, y_train_xgb,
    eval_set=[(X_val_xgb, y_val_xgb)],
    verbose=True
)

# Save XGBoost model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("\n✓ XGBoost model saved to: xgb_model.pkl")

# XGBoost predictions
y_train_pred_xgb = xgb_model.predict(X_train_xgb)
y_val_pred_xgb = xgb_model.predict(X_val_xgb)
y_test_pred_xgb = xgb_model.predict(X_test_xgb)

xgb_results = {
    'train': evaluate_model(y_train_xgb, y_train_pred_xgb, "Train", "XGBoost"),
    'val': evaluate_model(y_val_xgb, y_val_pred_xgb, "Validation", "XGBoost"),
    'test': evaluate_model(y_test_xgb, y_test_pred_xgb, "Test", "XGBoost")
}