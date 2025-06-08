# from flask import Flask, render_template, request, redirect, url_for
# import joblib
# import pandas as pd

# app = Flask(__name__)
# # Load the trained pipeline
# model = joblib.load('model.pkl')
# # Load feature names
# # df = pd.read_csv('cleaned_selected_data.csv')
# df = pd.read_csv('unique_values.csv')
# features = df.drop('HighestSeverity', axis=1)
# cat_choices = {col: sorted(df[col].dropna().unique().tolist()) for col in features.select_dtypes(include=['object']).columns}
# num_fields = features.select_dtypes(include=['int64','float64']).columns.tolist()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     if request.method == 'POST':
#         data = {}
#         # Categorical
#         for col, choices in cat_choices.items():
#             val = request.form.get(col)
#             data[col] = val
#         # Numerical
#         for col in num_fields:
#             val = request.form.get(col)
#             data[col] = float(val) if val is not None and val != '' else 0.0
#         # Create DataFrame
#         input_df = pd.DataFrame([data])
#         # Predict
#         prediction = model.predict(input_df)[0]
#     return render_template('index.html', cat_choices=cat_choices, num_fields=num_fields, prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')
df = pd.read_csv('unique_values.csv')
features = df.drop('HighestSeverity', axis=1)

# Define binary columns
binary_cols = [
    'PedestrianInvolved', 'BicycleInvolved', 'Intoxication', 'Speeding',
    'DisregardControl', 'NightTime', 'HeavyTruck', 'MotorCycle',
    'AnimalInvolved', 'ElectronicDistraction'
]

# Prepare choices
cat_choices = {}
for col in features.select_dtypes(include=['object']).columns:
    cat_choices[col] = sorted(df[col].dropna().unique().tolist())
    
# Add binary columns to categorical choices with "Yes"/"No"
for col in binary_cols:
    cat_choices[col] = ["No", "Yes"]  # Frontend will show Yes/No

# Numerical fields (excluding binary columns)
num_fields = [col for col in features.select_dtypes(include=['int64','float64']).columns.tolist() 
              if col not in binary_cols]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = {}
        # Handle categorical fields
        for col in cat_choices.keys():
            val = request.form.get(col)
            # Convert binary fields to 0/1
            if col in binary_cols:
                data[col] = 1 if val == "Yes" else 0
            else:
                data[col] = val
                
        # Handle numerical fields
        for col in num_fields:
            val = request.form.get(col)
            data[col] = float(val) if val and val != '' else 0.0
            
        # Create DataFrame and predict
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        
    return render_template(
        'index.html',
        cat_choices=cat_choices,
        num_fields=num_fields,
        binary_cols=binary_cols,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)