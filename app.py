from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
# Load the trained pipeline
model = joblib.load('model.pkl')
# Load feature names
df = pd.read_csv('cleaned_selected_data.csv')
features = df.drop('HighestSeverity', axis=1)
cat_choices = {col: sorted(df[col].dropna().unique().tolist()) for col in features.select_dtypes(include=['object']).columns}
num_fields = features.select_dtypes(include=['int64','float64']).columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = {}
        # Categorical
        for col, choices in cat_choices.items():
            val = request.form.get(col)
            data[col] = val
        # Numerical
        for col in num_fields:
            val = request.form.get(col)
            data[col] = float(val) if val is not None and val != '' else 0.0
        # Create DataFrame
        input_df = pd.DataFrame([data])
        # Predict
        prediction = model.predict(input_df)[0]
    return render_template('index.html', cat_choices=cat_choices, num_fields=num_fields, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
