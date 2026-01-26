import mlflow
import numpy as np

def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    x = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

# MLflow Tracking
mlflow.set_experiment("3D_Volumetric_Analysis")
with mlflow.start_run(run_name="Conv3D_Baseline"):
    model_3d = simple_conv3d_block()
    model_config = model_3d.to_json()
    mlflow.log_dict({"model_config": model_config}, "artifacts/model_architecture.json")
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("filters_start", 16)
    mlflow.log_metric("final_val_loss", 0.12)  # Exemple
