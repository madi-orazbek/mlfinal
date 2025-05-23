import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# Инициализация Flask
app = Flask(__name__)

# Надёжные пути к моделям
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path  = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Загрузка модели и скейлера
model  = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Список 30 признаков (должен совпадать с обучением)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Собираем и валидируем признаки
            features = []
            for fname in feature_names:
                raw = request.form.get(fname, '').strip()
                if raw == '':
                    raise ValueError(f"Поле '{fname}' не заполнено.")
                try:
                    val = float(raw)
                except ValueError:
                    raise ValueError(f"Поле '{fname}' должно быть числом, получили '{raw}'")
                features.append(val)

            # Преобразуем и предсказываем
            X = np.array([features])  # shape (1, 30)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
            label = 'Злокачественная опухоль' if pred == 1 else 'Доброкачественная опухоль'
            result = f"Диагноз: {label} (вероятность: {prob:.2%})"

        except Exception as e:
            # Показываем пользователю детальную ошибку
            result = f"Ошибка во вводе данных: {e}"

        return render_template('result.html', result=result)

    # GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
