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
model=pickle.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/NewLinearRegressionModel.pkl','rb'))
car=pd.read_csv('C:/Users/user/Desktop/Flask_VehiSense/static/dataset/Cleaned_Car_data.csv')

# CLASSIFICATION CODE
model2=joblib.load(open('C:/Users/user/Desktop/Flask_VehiSense/static/model/ClassificationModel.pkl','rb'))

# APP ROUTES
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    else:
        return render_template('dashboard.html')

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
            login_error_message = "Username or email already exists. Please choose another."
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
    return redirect(url_for('dashboard'))

# SAFETY TABLES
@app.route('/safety_table')
def safety_table():
    # Fetch data from MySQL database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM car_evaluation")
    data = cur.fetchall()

    # Print the data for debugging
    # for row in data:
    #     print(row)

    return render_template("safety_table.html", data=data)

# PRICE TABLES
@app.route('/price_table')
def price_table():
    # Fetch data from MySQL database
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, company, name, transmission, year, fuel_type, kms_driven, Price FROM car_price")
    data = cur.fetchall()

    # Print the data for debugging
    # for row in data:
    #     print(row)

    # Pass data to the template
    return render_template("price_table.html", data=data)



# CAR EVALUATOR
@app.route("/evaluatepredict")
def evaluatepredict():
    return render_template("evaluatepredict.html")

@app.route("/evaluate", methods=['POST'])
def evaluate():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model2.predict(features)

    # Map 0 to "unaccepted" and 1 to "accepted"
    result_text = "Unaccepted" if prediction == 0 else "Accepted"

    return render_template("evaluatepredict.html", prediction_text=f"{result_text}")

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
        prediction = request.form['prediction_text']
   
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_evaluation (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, classification) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (buying_price, maintenance_cost, num_doors, num_persons, lug_boot, safety, prediction))
        mysql.connection.commit()

    return redirect(url_for('evaluatepredict'))


# CAR PRICE PREDICTOR
@app.route('/pricepredict',methods=['GET','POST'])
def pricepredict():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('pricepredict.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
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

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year_input, driven, fuel_type_input]).reshape(1, 5)))
    print(prediction)

    return render_template("pricepredict.html", rounded_prediction=str(np.round(prediction[0], 2)),
                           companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                           selected_company=company, selected_car_model=car_model,
                           selected_year=year_input, selected_fuel_type=fuel_type_input, mileage=driven)

@app.route('/add_price', methods=['GET','POST'])
def add_price():
    if request.method == 'POST':
        company = request.form['company']
        model = request.form['model']
        transmission = request.form['transmission']
        year = request.form['year']
        fuel_type = request.form['fuel_type']
        mileage = request.form['mileage']
        price = request.form['rounded_prediction']
   
        cur = mysql.connection.cursor()
        # Create a new CarData instance
        cur.execute("INSERT INTO car_price (company, name, transmission, year, fuel_type, kms_driven, Price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (company, model, transmission, year, fuel_type, mileage, price))
        mysql.connection.commit()

    return redirect(url_for('pricepredict'))



if __name__=='__main__':
    app.run(debug=True)