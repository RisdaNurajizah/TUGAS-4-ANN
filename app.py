from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ========== 1. Persiapan Awal ==========

# Inisialisasi Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("jml pengunjung gedung sate.csv")
df.dropna(inplace=True)

# Normalisasi Data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["tahun", "jumlah_pengunjung"]])

# Pisahkan input (X) dan output (Y)
X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

# Bagi data training dan testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Membangun model ANN
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

# Kompilasi dan latih model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), verbose=0)

# Evaluasi model
loss, mae = model.evaluate(X_test, Y_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Buat folder static/ kalau belum ada
if not os.path.exists('static'):
    os.makedirs('static')

# ========== 2. Web Route ==========

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            tahun_input = int(request.form['tahun'])

            # Normalisasi tahun input
            tahun_scaled = scaler.transform(np.column_stack((np.array([tahun_input]), np.zeros(1))))[:, 0].reshape(-1, 1)

            # Prediksi jumlah pengunjung
            prediksi_scaled = model.predict(tahun_scaled)
            hasil_prediksi = scaler.inverse_transform(np.column_stack((tahun_scaled[:, 0], prediksi_scaled)))[:, 1][0]

            prediction = f"Prediksi jumlah pengunjung tahun {tahun_input}: {int(hasil_prediksi)} orang"

            # === Buat Grafik ===
            tahun_asli = df['tahun'].values
            jumlah_asli = df['jumlah_pengunjung'].values

            plt.figure(figsize=(8,5))
            plt.scatter(tahun_asli, jumlah_asli, color='blue', label='Data Aktual')  # Semua data aktual
            plt.scatter(tahun_input, hasil_prediksi, color='red', s=100, label=f'Prediksi Tahun {tahun_input}')  # Prediksi user
            plt.xlabel('Tahun')
            plt.ylabel('Jumlah Pengunjung')
            plt.title('Jumlah Pengunjung Museum Gedung Sate')
            plt.legend()
            plt.tight_layout()

            plt.savefig('static/plot.png')
            plt.close()

        except Exception as e:
            prediction = f"Terjadi kesalahan: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
