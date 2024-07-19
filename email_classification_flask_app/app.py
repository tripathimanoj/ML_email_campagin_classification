from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models
models = {
    'xgb_smote': pickle.load(open('models/model_xgb_smote.pkl', 'rb')),
    'rf_tuned_smote_fsel': pickle.load(open('models/model_randomF_tuned_smote_FSel.pkl', 'rb')),
    'rf_tuned_smote': pickle.load(open('models/model_randomF_tuned_smote.pkl', 'rb'))
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model')

    # Get input values from form
    data = {
        'Subject_Hotness_Score': float(request.form.get('Subject_Hotness_Score')),
        'Total_Past_Communications': float(request.form.get('Total_Past_Communications')),
        'Word_Count': float(request.form.get('Word_Count')),
        'Total_Links': float(request.form.get('Total_Links')),
        'Total_Images': float(request.form.get('Total_Images')),
        'Email_Type_1': request.form.get('Email_Type_1') == 'True',
        'Email_Source_Type_1': request.form.get('Email_Source_Type_1') == 'True',
        'Email_Campaign_Type_1': request.form.get('Email_Campaign_Type_1') == 'True',
        'Email_Campaign_Type_2': request.form.get('Email_Campaign_Type_2') == 'True',
        'Email_Campaign_Type_3': request.form.get('Email_Campaign_Type_3') == 'True',
        'Time_Email_sent_Category_1': request.form.get('Time_Email_sent_Category_1') == 'True',
        'Time_Email_sent_Category_2': request.form.get('Time_Email_sent_Category_2') == 'True',
        'Time_Email_sent_Category_3': request.form.get('Time_Email_sent_Category_3') == 'True'
    }

    # Create a DataFrame
    df = pd.DataFrame([data])

    # Derive Total_Img_links col
    df['Total_Img_links'] = df['Total_Links'] + df['Total_Images']
    df.drop(['Total_Links', 'Total_Images'], axis=1, inplace=True)

    # Preprocess the input data
    cont_var = ['Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count', 'Total_Img_links']
    for elem in cont_var:
        df[elem] = (df[elem] - df[elem].mean()) / df[elem].std()

    # Select columns based on the selected model
    if model_name == 'xgb_smote':
        cols = [
            'Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count', 'Total_Img_links',
            'Email_Type_1', 'Email_Source_Type_1', 'Email_Campaign_Type_1', 'Email_Campaign_Type_2', 
            'Email_Campaign_Type_3', 'Time_Email_sent_Category_1', 'Time_Email_sent_Category_2', 
            'Time_Email_sent_Category_3'
        ]
    elif model_name == 'rf_tuned_smote_fsel':
        cols = [
            'Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count', 'Total_Img_links',
            'Email_Type_1', 'Email_Source_Type_1', 'Email_Campaign_Type_1', 'Email_Campaign_Type_2', 
            'Email_Campaign_Type_3'
        ]
    elif model_name == 'rf_tuned_smote':
        cols = [
            'Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count', 'Total_Img_links',
            'Email_Type_1', 'Email_Source_Type_1', 'Email_Campaign_Type_1', 'Email_Campaign_Type_2', 
            'Email_Campaign_Type_3', 'Time_Email_sent_Category_1', 'Time_Email_sent_Category_2', 
            'Time_Email_sent_Category_3'
        ]

    # Filter the DataFrame columns based on the selected model
    df = df[cols]

    # Get the selected model
    model = models[model_name]

    # Predict
    prediction = model.predict(df)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
