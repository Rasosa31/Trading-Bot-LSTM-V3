import os
import numpy as np

# Bloque de estabilidad para Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

print("🚀 Probando TensorFlow...")
# Forzar a que no busque GPU si hay conflictos
tf.config.set_visible_devices([], 'GPU')

# Datos de prueba
X = np.random.random((50, 60, 8))
y = np.random.random((50, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60, 8)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("🧠 Entrenando (esto debe ser instantáneo)...")
model.fit(X, y, epochs=1, verbose=1)

# Guardar en la carpeta models de la raíz
if not os.path.exists('models'): os.makedirs('models')
model.save('models/test_vida.keras')

print("\n✅ ¡SISTEMA OPERATIVO! El archivo 'test_vida.keras' ha sido creado.")