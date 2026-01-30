from tensorflow.keras.models import load_model

# Cambia la ruta si tu archivo .keras está en otro lugar
model = load_model("models/lstm_model.keras")

# Muestra la estructura del modelo
model.summary()

