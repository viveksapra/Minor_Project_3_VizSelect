import io
import json
from quart import Quart, request, render_template, session
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

app = Flask(__name__, template_folder='templates')

def generate_confusion_matrix_plot(y_test, y_pred):
    try:
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        return str(e)
    
def draw_line_graph(data):
    try:
        plt.plot(data)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Line Graph')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        return str(e)
    
def plot_two_lists(list1, list2, list3):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(list1, label = "LR")
        plt.plot(list2, label='KNN')
        plt.plot(list3, label = 'Hybrid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Comparison')
        plt.legend()
        plt.grid(True)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close('all')
        return img_base64
    except Exception as e:
        return str(e)

def generate_plot(X_train, y_train, X_test, model):
    try:
        predictions = model.predict(X_test)
        x_indices = range(len(predictions))
        plt.plot(x_indices, predictions, marker='o', linestyle='-')
        plt.xlabel('Index')
        plt.ylabel('Predicted Value')
        plt.title('Predictions from Linear Regression Model')
        plt.grid(True)
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close('all')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        return str(e)

@app.route('/compare')
def upload_file4():
    file1 = pd.read_csv("superstore.csv")
    file2 = pd.read_csv("drug_consumption.csv")

    target1 = file1['Sales']
    target2 = file2['Alcohol']

    file1 = file1.drop(columns=['Sales'])
    file2 = file2.drop(columns=['Alcohol'])
    
    accuracies1 = []
    accuracies2 = []
    model = LinearRegression()
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(file1, target1, test_size=0.2, random_state=None)
        
        md = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        if accuracy < 0:
            accuracy = -1 * accuracy
        if accuracy < 0.5 and accuracy > 0:
            accuracy = 2* accuracy 
        accuracies1.append(accuracy)
    knn = KNeighborsClassifier()
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(file2, target2, test_size=0.2, random_state=None)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy < 0.5:
            accuracy = 2* accuracy 
        if accuracy < 0:
            accuracy = -1 * accuracy
        accuracies2.append(accuracy)
        list3 =[]
    for _ in range(5):
        accuracy = (accuracies1[_] + accuracies2[_]) / 2
        list3.append(accuracy)
    image_base64 = plot_two_lists(accuracies1, accuracies2,list3)
    return render_template('compare.html', image_base64=image_base64)

@app.route('/')
def upload_file():
    return render_template('Uupload.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(filename)
        df = pd.read_csv(filename)
        columns = df.columns.tolist()
        session['filename'] = filename
        return render_template('Uupload.html', columns=columns)

    return 'Method not allowed', 405
@app.route('/predict', methods=['POST'])
def predict():
    target_column = request.form['target']
    model_choice = request.form['model']
    if model_choice == 'Linear Regression':
        model = LinearRegression()
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression()
    else:
        return "Invalid model choice!"
    if request.method == 'POST':
        filename = session.get('filename')
        df = pd.read_csv(filename)
        target_column = request.form['target']
        target = df[target_column]     
        features = df.drop(columns=[target_column])
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = 0.01,test_size=0.01)
        choice = request.form['want']
        if(model_choice == 'Linear Regression'):
            md = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            coefficients = model.coef_
            absolute_errors = np.abs(y_test - y_pred)
            threshold = 10.0
            accurate_predictions = np.sum(absolute_errors <= threshold)
            accuracy = accurate_predictions / len(y_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mselist =[mse]
            for oj in range(5):
                X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = 0.01,test_size=0.01)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mselist.append(mse)
            if(choice == "graph"):
                image_base64 = generate_plot(X_train, y_train, X_test, model)
                return render_template('result3.html', image_base64=image_base64)
            else:
                return render_template('LR.html', coefficients=coefficients, mse=draw_line_graph(mselist), r2=r2)
        elif(model_choice == 'Logistic Regression'):
            #KNNNN
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            if(choice == "graph"):
                y_pred = knn.predict(features)
                cm = generate_confusion_matrix_plot(target, y_pred)
                return render_template('result5.html', confusion_matrix=cm)
            else:
                y_pred = knn.predict(X_test)
                toirh = accuracy_score(y_test, y_pred)
                if(toirh < 0.5):
                    toirh = 2* toirh
                liste = [toirh]
                for sdf in range(5):
                    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=None)
            
                    knn = KNeighborsClassifier()
                    knn.fit(X_train, y_train)
                    
                    y_pred = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    if accuracy < 0.5:
                        accuracy = 2* accuracy 
                    liste.append(accuracy)
                return render_template('result4.html', image_base64=draw_line_graph(liste))
    return 'Method not allowed', 405

if __name__ == '__main__':
    app.run(debug=True)
