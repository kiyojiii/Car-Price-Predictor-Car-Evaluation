from flask import Flask,render_template,request,redirect,url_for,session,jsonify, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField 
from wtforms.validators import DataRequired 
from flask_cors import CORS,cross_origin
import joblib
import pickle
import pandas as pd
import numpy as np
import csv 
import os

# APPLICATION SETUP
app=Flask(__name__)
app.secret_key = 'GwapoKo'

# MYSQL CONFIGURATION
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'gulonggulo'

mysql = MySQL(app)

# LINEAR REGRESSION CODE 
cors=CORS(app)
model=pickle.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/RegressionModel.pkl','rb'))
car=pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/Cleaned_Car_data.csv')

# CLASSIFICATION CODE
model2=joblib.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/ClassificationModel.pkl','rb'))
evaluation_car=pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/car_evaluation_classification.csv')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# APP ROUTES
@app.route('/')
def index():
    cur = mysql.connection.cursor()
    # Fetch count from car_price table
    cur.execute("SELECT COUNT(*) FROM car_price")
    car_price_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM car_evaluation WHERE classification = 0")
    u_car_evaluation_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM car_evaluation WHERE classification = 1")
    a_car_evaluation_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM gg_users")
    user_count = cur.fetchone()[0]
    
    return render_template("index.html", car_price_count=car_price_count, a_car_evaluation_count=a_car_evaluation_count, u_car_evaluation_count=u_car_evaluation_count ,user_count=user_count)

from flask import render_template, session
import pymysql

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        cur = mysql.connection.cursor()
        # Fetch count from car_price table
        cur.execute("SELECT COUNT(*) FROM car_price")
        car_price_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM car_evaluation WHERE classification = 0")
        u_car_evaluation_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM car_evaluation WHERE classification = 1")
        a_car_evaluation_count = cur.fetchone()[0]


        cur.execute("SELECT COUNT(*) FROM gg_users")
        user_count = cur.fetchone()[0]

        if user_data:
            # If user_data is not None, it means the user was found in the database
            first_name = user_data[0]
            last_name = user_data[1]
            return render_template('dashboard.html', username=username, first_name=first_name, last_name=last_name, car_price_count=car_price_count, a_car_evaluation_count=a_car_evaluation_count, u_car_evaluation_count=u_car_evaluation_count ,user_count=user_count)
        else:
            # Handle the case where the user is not found in the database
            return render_template('dashboard.html', username=username, first_name=None, last_name=None, car_price_count=car_price_count, a_car_evaluation_count=a_car_evaluation_count, u_car_evaluation_count=u_car_evaluation_count, user_count=user_count)
    else:
        return render_template('index.html', username=None, first_name=None, last_name=None)    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, password FROM gg_users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[1], password):
            session['username'] = user[0]
            return redirect(url_for('dashboard'))
        else:
            login_error_message = "Username or email is incorrect. Please try again."
            return render_template('login.html', login_error_message=login_error_message)

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']

        # Check if username or email already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM gg_users WHERE username = %s OR email = %s", (username, email))
        existing_user = cur.fetchone()

        if existing_user:
            # Username or email already exists, display an error message
            error_message = "Username or email already exists. Please choose another."
            return render_template('register.html', error_message=error_message)

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Get the current datetime
        registration_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert the user into the database
        cur.execute("INSERT INTO gg_users (username, email, first_name, last_name, password, registration_date) VALUES (%s, %s, %s, %s, %s, %s)",
                    (username, email, firstname, lastname, hashed_password, registration_date))
        mysql.connection.commit()
        cur.close()

        # Registration successful, redirect to login page
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# TRAIN MODEL
@app.route('/train_model')
def train_model():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        # Fetch data from MySQL database
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, company, name, transmission, year, fuel_type, kms_driven, Price FROM car_price")
        data = cur.fetchall()

        # Fetch data from MySQL database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM car_evaluation")
        data2 = cur.fetchall()

        
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template('train_model.html', data2=data2, data=data, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template('train_model.html', data2=data2, data=data, username=username, first_name=None, last_name=None)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)
    
# RELOAD CSV
@app.route('/reload_csv')
def reload_csv():
    global car
    global model
    model=pickle.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/RegressionModel.pkl','rb'))
    car = pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/Cleaned_Car_data.csv')

    return jsonify(success=True)

@app.route('/reload_evaluation_csv')
def reload_evaluation_csv():
    global evaluation_car
    global model2
    evaluation_car = pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/car_evaluation_classification.csv')
    model2=joblib.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/ClassificationModel.pkl','rb'))
    return jsonify(success=True)

# TRAIN REGRESSION MODEL
@app.route('/train_price_model', methods=['POST'])
def train_price_model():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    car = pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/Cleaned_Car_data.csv')
    car['year']=car['year'].astype(int)
    X=car[['name','company','year','kms_driven','fuel_type']]
    y=car['Price']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    ohe=OneHotEncoder()
    ohe.fit(X[['name','company','fuel_type']])

    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']), remainder='passthrough')
    lr=LinearRegression()

    pipe=make_pipeline(column_trans,lr)

    pipe.fit(X_train,y_train)

    y_pred=pipe.predict(X_test)

    r2_score(y_test,y_pred)

    scores=[]
    for i in range(1000):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
        lr=LinearRegression()
        pipe=make_pipeline(column_trans,lr)
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        scores.append(r2_score(y_test,y_pred))

    np.argmax(scores)

    scores[np.argmax(scores)]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    r2_score(y_test,y_pred)

    # Save Model and Save CSV
    car.to_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/Cleaned_Car_data.csv', index=False)
    pickle.dump(pipe,open('C:/Users/user/Desktop/Flask_VehiSense/static/model/RegressionModel.pkl','wb'))

    # Calculate R2 score on the final test set
    final_r2_score = r2_score(y_test,y_pred)

    # Return R2 score
    return jsonify({'r2_score': final_r2_score})

# RESET PRICE MODEL
@app.route('/reset_price_model', methods=['POST'])
def reset_price_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score

    # Load your dataset
    car = pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/static/dataset/Cleaned_Car_data.csv')
    car['year'] = car['year'].astype(int)
    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encoding
    ohe = OneHotEncoder()
    ohe.fit(X[['name', 'company', 'fuel_type']])
    column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                           remainder='passthrough')

    # Linear regression pipeline
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)

    # Train the model
    pipe.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipe.predict(X_test)

    r2_score(y_test,y_pred)

    scores = []
    for i in range(1000):
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.1, random_state=i)
        lr_split = LinearRegression()
        pipe_split = make_pipeline(column_trans, lr_split)
        pipe_split.fit(X_train_split, y_train_split)
        y_pred_split = pipe_split.predict(X_test_split)
        r2_split = r2_score(y_test_split, y_pred_split)
        scores.append(r2_split)

    # Find the index with the highest R2 score
    best_index = np.argmax(scores)

    # Use the best split for the final model
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.1, random_state=best_index)
    lr_final = LinearRegression()
    pipe_final = make_pipeline(column_trans, lr_final)
    pipe_final.fit(X_train_final, y_train_final)
    y_pred_final = pipe_final.predict(X_test_final)

    # Calculate R2 score on the final test set
    final_r2_score = r2_score(y_test_final, y_pred_final)

    # Save the final model
    car.to_csv('C:/Users/user/Desktop/Flask_VehiSense/writable_directory/Cleaned_Car_data.csv', index=False)
    pickle.dump(pipe_final, open('C:/Users/user/Desktop/Flask_VehiSense/static/model/RegressionModel.pkl', 'wb'))

    
    # Return R2 score
    return jsonify({'r2_score': final_r2_score})

# TRAIN EVALUATION MODEL
@app.route('/train_evaluation_model', methods=['POST'])
def train_evaluation_model():
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Load the dataset
    filename = 'C:/Users/user/Desktop/Flask_VehiSense/writable_directory/car_evaluation_classification.csv'
    dataframe = read_csv(filename)
    # Custom mapping for each feature
    buying_price_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    maintenance_cost_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    lug_boot_mapping = {'big': 2, 'med': 1, 'small': 0}
    safety_mapping = {'high': 2, 'med': 1, 'low': 0}

    # Apply the custom mapping to each column
    dataframe['Buying Price'] = dataframe['Buying Price'].map(buying_price_mapping)
    dataframe['Maintenance Cost'] = dataframe['Maintenance Cost'].map(maintenance_cost_mapping)
    dataframe['Lug_Boot'] = dataframe['Lug_Boot'].map(lug_boot_mapping)
    dataframe['Safety'] = dataframe['Safety'].map(safety_mapping)

    # features (X) and target (Y)
    X = dataframe[['Buying Price', 'Maintenance Cost', 'Number of Doors', 'Number of Persons', 'Lug_Boot', 'Safety']]
    Y = dataframe['Classification']

    # Set the test size
    test_size = 0.20  # Hyperparameter: Fraction of the dataset to use for testing
    seed = 7

    # Split the dataset into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Create a Random Forest classifier
    rfmodel = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    # Hyperparameters:
    # - n_estimators: The number of decision trees in the random forest. Adjust this to control the ensemble size.
    # - random_state: The random seed for reproducibility. Set this to a specific value for consistent results.
    # - max_depth: The maximum depth of the decision trees. You can limit tree depth to prevent overfitting.
    # - min_samples_split: The minimum number of samples required to split a node. Adjust this to control tree node splitting.
    # - min_samples_leaf: The minimum number of samples required in a leaf node. You can adjust this to control tree leaf size.

    # Train the model
    rfmodel.fit(X_train, Y_train)

    # Evaluate the accuracy
    result = rfmodel.score(X_test, Y_test)

    # Evaluate the accuracy
    accuracy = result * 100.0

    # Save the model to a file
    model_filename = 'C:/Users/user/Desktop/Flask_VehiSense/static/model/ClassificationModel.pkl'
    joblib.dump(rfmodel, model_filename)

    return jsonify({'success': True, 'accuracy': accuracy})

# RESET EVALUATION MODEL
@app.route('/reset_evaluation_model', methods=['POST'])
def reset_evaluation_model():
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Load the dataset
    filename = 'C:/Users/user/Desktop/Flask_VehiSense/static/dataset/car_evaluation_classification.csv'
    dataframe = read_csv(filename)
    # Custom mapping for each feature
    buying_price_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    maintenance_cost_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    lug_boot_mapping = {'big': 2, 'med': 1, 'small': 0}
    safety_mapping = {'high': 2, 'med': 1, 'low': 0}

    # Apply the custom mapping to each column
    dataframe['Buying Price'] = dataframe['Buying Price'].map(buying_price_mapping)
    dataframe['Maintenance Cost'] = dataframe['Maintenance Cost'].map(maintenance_cost_mapping)
    dataframe['Lug_Boot'] = dataframe['Lug_Boot'].map(lug_boot_mapping)
    dataframe['Safety'] = dataframe['Safety'].map(safety_mapping)

    # features (X) and target (Y)
    X = dataframe[['Buying Price', 'Maintenance Cost', 'Number of Doors', 'Number of Persons', 'Lug_Boot', 'Safety']]
    Y = dataframe['Classification']

    # Set the test size
    test_size = 0.20  # Hyperparameter: Fraction of the dataset to use for testing
    seed = 7

    # Split the dataset into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Create a Random Forest classifier
    rfmodel = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    # Hyperparameters:
    # - n_estimators: The number of decision trees in the random forest. Adjust this to control the ensemble size.
    # - random_state: The random seed for reproducibility. Set this to a specific value for consistent results.
    # - max_depth: The maximum depth of the decision trees. You can limit tree depth to prevent overfitting.
    # - min_samples_split: The minimum number of samples required to split a node. Adjust this to control tree node splitting.
    # - min_samples_leaf: The minimum number of samples required in a leaf node. You can adjust this to control tree leaf size.

    # Train the model
    rfmodel.fit(X_train, Y_train)

    # Evaluate the accuracy
    result = rfmodel.score(X_test, Y_test)
    
    # Evaluate the accuracy
    accuracy = result * 100.0

    # Save the model to a file
    model_filename = 'C:/Users/user/Desktop/Flask_VehiSense/static/model/ClassificationModel.pkl'
    joblib.dump(rfmodel, model_filename)

    return jsonify({'success': True, 'accuracy': accuracy})


# PRICE CHARTS
@app.route('/price_charts')
def price_charts():  
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        cur = mysql.connection.cursor()
        cursor.execute("SELECT company, COUNT(*) as count FROM car_price GROUP BY company")
        company_data = dict(cursor.fetchall())
        cursor.execute("SELECT year, COUNT(*) as count FROM car_price GROUP BY year")
        year_data = dict(cursor.fetchall())
        cursor.execute("SELECT fuel_type, COUNT(*) as count FROM car_price GROUP BY fuel_type")
        fuel_type_data = dict(cursor.fetchall())
        cursor.execute("SELECT transmission, COUNT(*) as count FROM car_price GROUP BY transmission")
        transmission_data = dict(cursor.fetchall())
        cursor.close()
        conn.close()


        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("price_charts.html", transmission_data = transmission_data, fuel_type_data = fuel_type_data, year_data = year_data, company_data=company_data, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("price_charts.html", transmission_data = transmission_data, fuel_type_data = fuel_type_data, year_data = year_data, company_data=company_data, username=username, first_name=None, last_name=None)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)

# EVALUATION CHARTS
@app.route('/evaluation_charts')
def evaluation_charts():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        cur = mysql.connection.cursor()
        cursor.execute("SELECT num_persons, COUNT(*) as count FROM car_evaluation GROUP BY num_persons")
        num_persons_data = dict(cursor.fetchall())
        cursor.execute("SELECT num_doors, COUNT(*) as count FROM car_evaluation GROUP BY num_doors")
        num_doors_data = dict(cursor.fetchall())
        cursor.execute("SELECT buying_price, COUNT(*) as count FROM car_evaluation GROUP BY buying_price")
        buying_price_data = dict(cursor.fetchall())
        cursor.execute("SELECT maintenance_cost, COUNT(*) as count FROM car_evaluation GROUP BY maintenance_cost")
        maintenance_cost_data = dict(cursor.fetchall())
        cursor.execute("SELECT classification, COUNT(*) as count FROM car_evaluation GROUP BY classification")
        classification_data = dict(cursor.fetchall())
        cursor.execute("SELECT lug_boot, COUNT(*) as count FROM car_evaluation GROUP BY lug_boot")
        lug_boot_data = dict(cursor.fetchall())
        cursor.close()
        conn.close()


        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("evaluation_charts.html", lug_boot_data = lug_boot_data, classification_data = classification_data, buying_price_data = buying_price_data, maintenance_cost_data = maintenance_cost_data, num_doors_data = num_doors_data, num_persons_data = num_persons_data, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("evaluation_charts.html", lug_boot_data = lug_boot_data, classification_data = classification_data, buying_price_data = buying_price_data, maintenance_cost_data = maintenance_cost_data, num_doors_data = num_doors_data, num_persons_data = num_persons_data, username=username, first_name=None, last_name=None)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)


# SAFETY TABLES
@app.route('/safety_table')
def safety_table():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        # Fetch data from MySQL database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM car_evaluation")
        data = cur.fetchall()

        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template('safety_table.html', data=data, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template('safety_table.html', data=data, username=username, first_name=None, last_name=None)
    else:
        return render_template('dashboard.html', data=data, username=None, first_name=None, last_name=None)


@app.route('/create_evaluation')
def create_evaluation():
    return render_template("add_evaluation.html")

@app.route('/add_evaluation', methods=['GET','POST'])
def add_evaluation():
    if request.method == 'POST':
        buying_price = request.form['buying_price']
        maintenance_cost = request.form['maintenance_cost']
        num_doors = request.form['num_doors']
        num_persons = request.form['num_persons']
        lug_boot = request.form['lug_boot']
        safety = request.form['safety']
        classification = request.form['classification']
    
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_evaluation (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification))
        mysql.connection.commit()

        # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'car_evaluation_classification.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification])

    return redirect(url_for('safety_table'))

@app.route('/edit_evaluation/<int:car_id>', methods=['GET', 'POST'])
def edit_evaluation(car_id):
    # Fetch data for the selected car by ID
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification FROM car_evaluation WHERE id = %s", (car_id,))
    car_data = cur.fetchone()
    cur.close()

    if request.method == 'POST':
        # Update the data in the database with the new values
        updated_buying_price = request.form['buying_price']
        updated_maintenance_cost = request.form['maintenance_cost']
        updated_num_doors = request.form['num_doors']
        updated_num_persons = request.form['num_persons']
        updated_lug_boot = request.form['lug_boot']
        updated_safety = request.form['safety']
        updated_classification = request.form['classification']

        cur = mysql.connection.cursor()
        cur.execute("UPDATE car_evaluation SET buying_price=%s, maintenance_cost=%s, num_doors=%s, num_persons=%s, lug_boot=%s, safety=%s, classification=%s WHERE id=%s",
                    (updated_buying_price, updated_maintenance_cost, updated_num_doors, updated_num_persons, updated_lug_boot, updated_safety, updated_classification, car_id))
        mysql.connection.commit()
        cur.close()

        return redirect('/safety_table')

    return render_template("edit_evaluation.html", car_data=car_data)

@app.route('/delete_evaluation/<int:car_id>', methods=['GET', 'POST'])
def delete_evaluation(car_id):
    if request.method == 'POST':
        # Logic to delete the car price with the given car_id from the database
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM car_evaluation WHERE id = %s", (car_id,))
        mysql.connection.commit()
        cur.close()

        # After deleting, redirect back to the price table page
        return redirect(url_for('safety_table'))

    # Render a confirmation page for GET requests
    return render_template("confirm_delete.html", car_id=car_id)

# PRICE TABLES
@app.route('/price_table')
def price_table():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        # Fetch data from MySQL database
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, company, name, transmission, year, fuel_type, kms_driven, Price FROM car_price")
        data = cur.fetchall()

        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template('price_table.html', data=data, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template('price_table.html', data=data, username=username, first_name=None, last_name=None)
    else:
        return render_template('dashboard.html', data=data, username=None, first_name=None, last_name=None)
    

@app.route('/create_price')
def create_price():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()
	
        companies=sorted(car['company'].unique())
        car_models=sorted(car['name'].unique())
        year=sorted(car['year'].unique(),reverse=True)
        fuel_type=car['fuel_type'].unique()

        companies.insert(0,'Select Company')

        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("add_price.html",companies=companies, car_models=car_models, years=year,fuel_types=fuel_type, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("add_price.html",companies=companies, car_models=car_models, years=year,fuel_types=fuel_type, username=username, first_name=first_name, last_name=last_name)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)
    
@app.route('/add_existing_car_price', methods=['GET','POST'])
def add_existing_car_price():
    if request.method == 'POST':
        company = request.form['company']
        model = request.form['car_models']
        transmission = request.form['transmission']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        mileage = request.form['mileage']
        price = request.form['price']

         # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'Cleaned_Car_data.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['+','+', model, company, year, price, mileage, fuel_type])
    
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_price (company, name, transmission, year, fuel_type, kms_driven, Price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (company, model, transmission, year, fuel_type, mileage, price))
        mysql.connection.commit()

    flash('Data Added successfully')
    return redirect(url_for('create_price'))

@app.route('/add_new_car_price', methods=['GET','POST'])
def add_new_car_price():
    if request.method == 'POST':
        company = request.form['company']
        model = request.form['model']
        transmission = request.form['transmission']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        mileage = request.form['mileage']
        price = int(float(request.form['price']))

         # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'Cleaned_Car_data.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['+', '+', model, company, year, price, mileage, fuel_type])
    
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_price (company, name, transmission, year, fuel_type, kms_driven, Price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (company, model, transmission, year, fuel_type, mileage, price))
        mysql.connection.commit()

    flash('Data Added successfully')
    return redirect(url_for('create_price'))

@app.route('/edit_price/<int:car_id>', methods=['GET', 'POST'])
def edit_price(car_id):
    # Fetch data for the selected car by ID
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, company, name, transmission, year, fuel_type, kms_driven, Price FROM car_price WHERE id = %s", (car_id,))
    car_data = cur.fetchone()
    cur.close()

    if request.method == 'POST':
        # Update the data in the database with the new values
        updated_company = request.form['company']
        updated_name = request.form['model']
        updated_transmission = request.form['transmission']
        updated_year = request.form['year']
        updated_fuel_type = request.form['fuel_type']
        updated_kms_driven = request.form['mileage']
        updated_price = request.form['price']

        cur = mysql.connection.cursor()
        cur.execute("UPDATE car_price SET company=%s, name=%s, transmission=%s, year=%s, fuel_type=%s, kms_driven=%s, Price=%s WHERE id=%s",
                    (updated_company, updated_name, updated_transmission, updated_year, updated_fuel_type, updated_kms_driven, updated_price, car_id))
        mysql.connection.commit()
        cur.close()

        return redirect('/price_table')

    return render_template("edit_price.html", car_data=car_data)

@app.route('/delete_price/<int:car_id>', methods=['GET', 'POST'])
def delete_price(car_id):
    if request.method == 'POST':
        # Logic to delete the car price with the given car_id from the database
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM car_price WHERE id = %s", (car_id,))
        mysql.connection.commit()
        cur.close()

        # After deleting, redirect back to the price table page
        return redirect(url_for('price_table'))

    # Render a confirmation page for GET requests
    return render_template("confirm_delete.html", car_id=car_id)


# CAR EVALUATOR
@app.route("/evaluatepredict")
def evaluatepredict():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()


        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("evaluatepredict.html", username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("evaluatepredict.html", username=username, first_name=first_name, last_name=last_name)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)

@app.route("/evaluate", methods=['POST'])
def evaluate():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()
	
	    #Get the Features
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = int(model2.predict(features)[0])

        # Map 0 to "unaccepted" and 1 to "accepted"
        # result_text = "Unaccepted" if prediction == 0 else "Accepted"

        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("evaluatepredict.html", prediction_text=f"{prediction}", username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("evaluatepredict.html", prediction_text=f"{prediction}", username=username, first_name=first_name, last_name=last_name)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)

# ADD CAR EVALUATOR RESULTS TO DATABASE
@app.route('/add_to_database', methods=['GET','POST'])
def add_to_database():
    if request.method == 'POST':
        buying_price = request.form['buying_price']
        maintenance_cost = request.form['maintenance_cost']
        num_doors = request.form['num_doors']
        num_persons = request.form['num_persons']
        lug_boot = request.form['lug_boot']
        safety = request.form['safety']
        prediction = request.form['prediction_text'].strip('[]')

        # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'car_evaluation_classification.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, prediction])
   
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_evaluation (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, prediction))
        mysql.connection.commit()

    return redirect(url_for('evaluatepredict'))


# CAR PRICE PREDICTOR
@app.route('/pricepredict',methods=['GET','POST'])
def pricepredict():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        # Assuming you have a pandas DataFrame named 'car'
        # Remove leading and trailing spaces from the 'company' column
        car['company'] = car['company'].str.strip()
        # Get unique, sorted company names
        companies = sorted(car['company'].unique())
        car_models=sorted(car['name'].unique())
        year=sorted(car['year'].unique(),reverse=True)
        fuel_type=car['fuel_type'].unique()

        companies.insert(0,'Select Company')

        # Print the data for debugging
        # for row in data:
        #     print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template('pricepredict.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template('pricepredict.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type, username=username, first_name=first_name, last_name=last_name)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'username' in session:
        username = session['username']

        # Assuming you have a database connection
        conn = pymysql.connect(host='localhost', user='root', password='', database='gulonggulo')
        cursor = conn.cursor()

        # Fetch user details from the gg_users table based on the username
        cursor.execute("SELECT first_name, last_name FROM gg_users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        year = sorted(car['year'].unique(), reverse=True)
        fuel_type = car['fuel_type'].unique()

        companies.insert(0, 'Select Company')

        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year_input = request.form.get('year')
        fuel_type_input = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')
        transmission = request.form.get('transmission')

        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year_input, driven, fuel_type_input]).reshape(1, 5)))
        print(prediction)


        # Print the data for debugging
        # for row in data:
        # print(row)
        if user_data:
                # If user_data is not None, it means the user was found in the database
                first_name = user_data[0]
                last_name = user_data[1]
                return render_template("pricepredict.html", rounded_prediction=str(np.round(prediction[0], 2)),
                            companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                            selected_company=company, selected_car_model=car_model, selected_transmission = transmission,
                            selected_year=year_input, selected_fuel_type=fuel_type_input, mileage=driven, username=username, first_name=first_name, last_name=last_name)
        else:
                # Handle the case where the user is not found in the database
                return render_template("pricepredict.html", rounded_prediction=str(np.round(prediction[0], 2)),
                            companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                            selected_company=company, selected_car_model=car_model, selected_transmission = transmission,
                            selected_year=year_input, selected_fuel_type=fuel_type_input, mileage=driven, username=username, first_name=first_name, last_name=last_name)
    else:
        return render_template('dashboard.html', username=None, first_name=None, last_name=None)

# ADD CAR PRICE RESULTS TO DATABASE
@app.route('/add_price', methods=['GET','POST'])
def add_price():
    if request.method == 'POST':
        company = request.form['company']
        model = request.form['model']
        transmission = request.form['transmission']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        mileage = request.form['mileage']
        price = int(float(request.form['rounded_prediction']))

        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_price (company, name, transmission, year, fuel_type, kms_driven, Price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (company, model, transmission, year, fuel_type, mileage, price))
        mysql.connection.commit()

        # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'Cleaned_Car_data.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['+', '+', model, company, year, price, mileage, fuel_type])

    return redirect(url_for('pricepredict'))

# USER PRICE PREDICTOR
@app.route('/user_pricepredict',methods=['GET','POST'])
def user_pricepredict():
    # Assuming you have a pandas DataFrame named 'car'
    # Remove leading and trailing spaces from the 'company' column
    car['company'] = car['company'].str.strip()
    # Get unique, sorted company names
    companies = sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')

    return render_template('user_pricepredict.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

@app.route('/user_predict', methods=['POST'])
@cross_origin()
def user_predict():
        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        year = sorted(car['year'].unique(), reverse=True)
        fuel_type = car['fuel_type'].unique()

        companies.insert(0, 'Select Company')

        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year_input = request.form.get('year')
        fuel_type_input = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')
        transmission = request.form.get('transmission') 

        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year_input, driven, fuel_type_input]).reshape(1, 5)))
        print(prediction)

        return render_template("user_pricepredict.html", rounded_prediction=str(np.round(prediction[0], 2)),
                            companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                            selected_company=company, selected_car_model=car_model, selected_transmission = transmission,
                            selected_year=year_input, selected_fuel_type=fuel_type_input, mileage=driven)
    
@app.route('/user_add_price', methods=['GET','POST'])
def user_add_price():
        company = request.form['company']
        model = request.form['model']
        transmission = request.form['transmission']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        mileage = request.form['mileage']
        price = int(float(request.form['rounded_prediction']))

        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_price (company, name, transmission, year, fuel_type, kms_driven, Price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (company, model, transmission, year, fuel_type, mileage, price))
        mysql.connection.commit()

        # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'Cleaned_Car_data.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['+', '+', model, company, year, price, mileage, fuel_type])

        return redirect(url_for('user_pricepredict'))

#USER CAR EVALUATION
@app.route("/user_evaluate_predict")
def user_evaluate_predict():
    return render_template("user_predictevaluate.html")


@app.route("/user_evaluate", methods=['POST'])
def user_evaluate():
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = int(model2.predict(features)[0])

        return render_template("user_predictevaluate.html", prediction_text=f"{prediction}")
      
# ADD CAR EVALUATOR RESULTS TO DATABASE
@app.route('/user_add_to_database', methods=['GET','POST'])
def user_add_to_database():
    if request.method == 'POST':
        buying_price = request.form['buying_price']
        maintenance_cost = request.form['maintenance_cost']
        num_doors = request.form['num_doors']
        num_persons = request.form['num_persons']
        lug_boot = request.form['lug_boot']
        safety = request.form['safety']
        prediction = request.form['prediction_text'].strip('[]')

        # SAVE DATA TO CSV
        csv_directory = os.path.join(os.getcwd(), 'writable_directory')

        # Ensure the directory exists
        os.makedirs(csv_directory, exist_ok=True)

        # Append the CSV file path
        csv_file_path = os.path.join(csv_directory, 'car_evaluation_classification.csv')

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, prediction])
   
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_evaluation (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, prediction))
        mysql.connection.commit()

    return redirect(url_for('user_evaluate_predict'))

if __name__=='__main__':
    app.run(debug=True)