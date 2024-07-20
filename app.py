from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, session, redirect, url_for,jsonify
from flask_pymongo import PyMongo
import bcrypt
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
from sklearn.metrics import mean_squared_error
app = Flask(__name__)
file_path = 'final_financial_data.xlsx'
retirement_plan_path = 'Retirement_Plan.xlsx'
Corpus_Invest = 'Corpus_Investment.xlsx'
# MongoDB configuration
app.config['MONGO_DBNAME'] = 'Employees'
app.config['MONGO_URI'] = "mongodb://localhost:27017/Employees"
THRESHOLD_PRICE =30  # Example input price for stocks and bonds
MAX_SYMBOLS = 100  # Limit to the top 100 companies for the example
CRYPTO_THRESHOLD_PRICE = 50000  # Example input price for cryptocurrencies
ALPHA_VANTAGE_API_KEY = 'NQYUJ42SRPHBNYUO'  # Replace with your Alpha Vantage API key

mongo = PyMongo(app)

# Load processed data
data = pd.read_excel('final_financial_data.xlsx')

@app.route('/')
def index():
    if 'username' in session:
        username = session['username']
        return render_template('home.html')
    else:
        return render_template('cover_page.html')
    # return render_template('cover_page.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'name': request.form['username']})
        if login_user:
            if bcrypt.checkpw(request.form['pass'].encode('utf-8'), login_user['password']):
                session['username'] = request.form['username']
                return render_template('home.html')
                # Check if the username exists in the data DataFrame
                # if session['username'] in data['Name'].values:
                #     user = data[data['Name'] == session['username']].iloc[0]
                #     return redirect(url_for('user_profile', username=session['username']))
                # else:
                # return f"User {session['username']} not found in database."
        return 'Invalid username and password combination'
    return render_template('login.html')

@app.route('/home')
def home():
    if 'username' in session: 
        return render_template('home.html')
    else:
        return render_template('cover_page.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    users=mongo.db.users
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['pass']
        email = request.form['email']

        # Check if username already exists
        if users.find_one({'name': username}):
            msg = 'Username already exists in the database'
            return render_template('register.html', msg=msg)

        # Check if passwords match
        if password != request.form['pass1']:
            msg = "The passwords are not Matching"
            return render_template('register.html', msg=msg)

        # Hash the password and save the user
        hashpass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users.insert_one({'name': username, 'password': hashpass, 'email': email})
        session['username'] = username

        # Check if user exists in DataFrame
        user_data = data[data['Name'] == session['username']]
        if not user_data.empty:
            user = user_data.iloc[0]
            return redirect(url_for('user_profile', username=session['username']))
        else:
            msg = 'User does not exist in the DataFrame'
            return render_template('cover_page.html', msg=msg)
    return render_template('register.html', msg='')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/user/<username>')
def user_profile(username):
    try:
        user = data[data['user_name'] == username].iloc[0]
        fig, ax = plt.subplots()
        labels = ['Monthly Income', 'Salary', 'Bonuses', 'Investment Income', 'Miscellaneous Expenses']
        sizes = [user[label] for label in labels]
        
        # Validate sizes list
        if any(size < 0 for size in sizes):
            return "Error: Negative values found in pie chart sizes."
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return render_template('user_profile.html', user=user, plot_url=plot_url)
    
    except IndexError:
        return f"User '{username}' not found in database."
    
    except ValueError as e:
        return f"Error generating pie chart: {str(e)}"
    

@app.route('/add_userdetails', methods=['GET', 'POST'])
def add_userdetails():
    if request.method == 'POST':
        # Load existing data
        data = pd.read_excel(file_path)
        
        user_name=session['username']
        
        # Generate new User ID
        new_user_id = data['User Id'].max() + 1 if not data.empty else 1
        
        # Extract data from form
        user_data={
            'User Id': new_user_id,
            'Name': request.form['name'],
            'Current Age': int(request.form['age']),
            'Income': float(request.form['monthly_income']),
            'Monthly_Income_Amount': float(request.form['investment_income']),
            'Risk Tolerance': request.form['risk_tolerance'],
            'Investment preferences': request.form['investment_preferences'],
            'user_name':user_name,
        }
        
        
        # Append new data
        new_data = pd.DataFrame([user_data])
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Save back to Excel
        data.to_excel(file_path, index=False)
        print(data)
        
        return redirect(url_for('predict', user_name=session['username']))
    return render_template('add_userdetails.html')


@app.route('/update_userdetails', methods=['GET', 'POST'])
def update_userdetails():
    if request.method == 'POST':
        data = pd.read_excel(file_path)
        
        user_name = session['username']
        
        # Find the existing user data by username
        user_index = data[data['Name'] == user_name].index
        if not user_index.empty:
            user_index = user_index[0]
            
            data.at[user_index, 'Age'] = int(request.form['age'])
            data.at[user_index, 'Gender'] = request.form['gender']
            data.at[user_index, 'Monthly Income'] = float(request.form['monthly_income'])
            data.at[user_index, 'Salary'] = float(request.form['salary'])
            data.at[user_index, 'Bonuses'] = float(request.form['bonuses'])
            data.at[user_index, 'Investment Income'] = float(request.form['investment_income'])
            data.at[user_index, 'Miscellaneous Expenses'] = float(request.form['miscellaneous_expenses'])
            data.at[user_index, 'Risk Tolerance'] = request.form['risk_tolerance']
            data.at[user_index, 'Investment preferences'] = request.form['investment_preferences']
            
            data.to_excel(file_path, index=False)
            
            return redirect(url_for('user_profile', username=user_name))
        else:
            return "User not found in database."
    return render_template('update_userdetails.html')



@app.route('/retirement_plan', methods=['GET', 'POST'])
def retirement_plan():
    if request.method == 'POST':
        # Load existing data
        data = pd.read_excel(retirement_plan_path)
        
        # Generate new User ID
        new_user_id = data['User Id'].max() + 1 if not data.empty else 1
        
        # Extract data from form
        preferred_values = request.form.get('preferred_values', 'no')
        
        if preferred_values == 'yes':
            stocks = float(request.form.get('stocks', 0))
            bonds = float(request.form.get('bonds', 0))
            gold = float(request.form.get('gold', 0))
            cryptocurrency = float(request.form.get('cryptocurrency', 0))
        else:
            stocks = bonds = gold = cryptocurrency = 0.0
        
        user_data = {
            'User Id': new_user_id,
            'Name': request.form['name'],
            'Current Age': int(request.form['current_age']),
            'Retirement Age': int(request.form['retirement_age']),
            'Income': float(request.form['income']),
            'Expected Amount': float(request.form['expected_amount']),
            'preferred_values': preferred_values,
            'stocks': stocks,
            'bonds': bonds,
            'gold': gold,
            'cryptocurrency': cryptocurrency,
        }
        
        # Append new data
        new_data = pd.DataFrame([user_data])
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Save back to Excel
        data.to_excel(retirement_plan_path, index=False)
        
        return redirect(url_for('home'))
    return render_template('retirement_plan.html')

data1 = {
    'investment_budget': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    'age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 70],
    'risk_tolerance': ['High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium'],
    'FD': [0.1, 0.2, 0.15, 0.1, 0.05, 0.2, 0.1, 0.1, 0.15, 0.2],  # Example optimal allocation percentages
    'Bonds': [0.2, 0.15, 0.1, 0.2, 0.25, 0.15, 0.1, 0.2, 0.1, 0.15],
    'Stocks': [0.3, 0.4, 0.5, 0.3, 0.2, 0.4, 0.3, 0.3, 0.4, 0.5],
    'SIP': [0.1, 0.1, 0.1, 0.15, 0.2, 0.1, 0.15, 0.1, 0.1, 0.1],
    'Mutual_Funds': [0.3, 0.15, 0.15, 0.25, 0.3, 0.15, 0.35, 0.3, 0.25, 0.1]
}
df = pd.DataFrame(data1)

# Encode categorical risk tolerance to numeric for modeling
df['risk_tolerance_encoded'] = df['risk_tolerance'].map({'High': 2, 'Medium': 1, 'Low': 0})

# Define feature matrix X and target matrix y
X = df[['investment_budget', 'age', 'risk_tolerance_encoded']]
y = df[['FD', 'Bonds', 'Stocks', 'SIP', 'Mutual_Funds']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model using Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse:.2f}')

def suggest_optimal_allocation(investment_budget, age, risk_tolerance):
    risk_tolerance_encoded = {'High': 2, 'Medium': 1, 'Low': 0}[risk_tolerance]
    predicted_allocation = regressor.predict([[investment_budget, age, risk_tolerance_encoded]])
    return predicted_allocation[0]

@app.route('/predict/<user_name>', methods=['POST', 'GET'])
def predict(user_name):
    file_path1 = 'final_financial_data.xlsx'  # Replace with your actual file path

    try:
        data2 = pd.read_excel(file_path1)
        user = data2[data2['user_name'] == user_name].iloc[0]
        investment_budget = float(user['Income'])
        age = int(user['Current Age'])
        risk_tolerance = user['Risk Tolerance']  # Assuming this column contains 'Low', 'Medium', 'High'
    except IndexError:
        return f"User '{user_name}' not found in database."
    except ValueError as e:
        return f"Error reading data: {str(e)}"

    predicted_allocation = suggest_optimal_allocation(investment_budget, age, risk_tolerance)

    # # Create pie chart
    # labels = ['FD', 'Bonds', 'Stocks', 'SIP', 'Mutual Funds']
    # fig, ax = plt.subplots()
    # ax.pie(predicted_allocation, labels=labels, autopct='%1.1f%%', startangle=140)
    # ax.axis('equal')

    # # Save plot as a PNG image in memory
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # image_png = buffer.getvalue()
    # buffer.close()

    # # Convert image to base64 for HTML rendering
    # plot_data = base64.b64encode(image_png).decode('utf8')

    # return jsonify({'allocation': predicted_allocation.tolist(), 'plot_data': plot_data})
    #  predicted_allocation = data1['allocation']

    # Create pie chart
    labels = ['FD', 'Bonds', 'Stocks', 'SIP', 'Mutual Funds']
    fig, ax = plt.subplots()
    ax.pie(predicted_allocation, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')

    # Save plot as PNG in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode PNG image to base64 for HTML rendering
    plot_url = base64.b64encode(image_png).decode('utf8')

    # Render HTML template with prediction results and pie chart
    return render_template('predict.html', allocation=predicted_allocation, plot_url=plot_url)
from flask import session

def corpus_sum():
    print("hello")
    file_path2 = 'Corpus_Investment.xlsx'  # Replace with your actual file path

    try:
        data2 = pd.read_excel(file_path2)
        user = data2[data2['user_name'] == session.get('username')].iloc[0]
        investment_budget = float(user['Amount'])
        age = int(user['Current Age'])
        risk_tolerance = user['Risk Tolerance']  # Assuming this column contains 'Low', 'Medium', 'High'
    except IndexError:
        return f"User '{session.get('username')}' not found in database."
    except ValueError as e:
        return f"Error reading data: {str(e)}"

    predicted_allocation = suggest_optimal_allocation(investment_budget, age, risk_tolerance)

    # Create pie chart
    labels = ['FD', 'Bonds', 'Stocks', 'SIP', 'Mutual Funds']
    fig, ax = plt.subplots()
    ax.pie(predicted_allocation, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')

    # Save plot as PNG in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode PNG image to base64 for HTML rendering
    plot_url = base64.b64encode(image_png).decode('utf8')

    # Render HTML template with prediction results and pie chart
    return render_template('corpussum.html', allocation=predicted_allocation, plot_url=plot_url)



@app.route('/corpus_invest', methods=['GET', 'POST'])
def corpus_invest():
    if request.method == 'POST':
        # Load existing data
        data = pd.read_excel(Corpus_Invest)
        
        # Generate new User ID
        new_user_id = data['User Id'].max() + 1 if not data.empty else 1
        
        # Extract data from form        
        user_data = {
            'User Id': new_user_id,
            'Name': request.form['name'],
            'Current Age': int(request.form['current_age']),
            'Amount':float(request.form['Amount']),
            'Investment Duration': int(request.form['Invest_time']),
            'Risk Tolerance': (request.form['risk_tolerance']),
            'user_name':session['username']
        }
        
        # Append new data
        new_data = pd.DataFrame([user_data])
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Save back to Excel
        data.to_excel(Corpus_Invest, index=False)
    
    # corpus_sum()
        # return redirect(url_for('home'))
   
# Simulated dataset (you can replace this with your actual dataset)

        file_path2 = 'Corpus_Investment.xlsx'  # Replace with your actual file path

        try:
            data2 = pd.read_excel(file_path2)
            user = data2[data2['user_name'] == session.get('username')].iloc[0]
            investment_budget = float(user['Amount'])
            age = int(user['Current Age'])
            risk_tolerance = user['Risk Tolerance']  # Assuming this column contains 'Low', 'Medium', 'High'
        except IndexError:
            return f"User '{session.get('username')}' not found in database."
        except ValueError as e:
            return f"Error reading data: {str(e)}"

        predicted_allocation = suggest_optimal_allocation(investment_budget, age, risk_tolerance)

        # Create pie chart
        labels = ['FD', 'Bonds', 'Stocks', 'SIP', 'Mutual Funds']
        fig, ax = plt.subplots()
        ax.pie(predicted_allocation, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')

    # Save plot as PNG in memory
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

    # Encode PNG image to base64 for HTML rendering
        plot_url = base64.b64encode(image_png).decode('utf8')

    # Render HTML template with prediction results and pie chart
        return render_template('corpussum.html', allocation=predicted_allocation, plot_url=plot_url)
    else:
         return render_template('Corpus_Investment.html')   
def get_sp500_symbols():
    """Fetch the list of S&P 500 stock symbols from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    symbols = table['Symbol'].tolist()
    return symbols[:MAX_SYMBOLS]


def get_stock_price(symbol):
    """Fetch the current stock price for a given symbol using yfinance."""
    stock = yf.Ticker(symbol)
    price = stock.history(period='1d')['Close'].iloc[-1]
    return price


def fetch_stocks_below_threshold(symbols, threshold_price):
    """Fetch stocks with prices below the given threshold."""
    below_threshold = []
    for symbol in symbols:
        try:
            price = get_stock_price(symbol)
            if price is not None and price < threshold_price:
                below_threshold.append((symbol, price))
        except Exception as e:
            print(f"Failed to get ticker '{symbol}' reason: {e}")
    return below_threshold


def fetch_bond_symbols_and_prices():
    """Fetch bond symbols and their current prices."""
    # Manually adding bond symbols and prices
    bond_prices = [
        ('USTB10Y', 1.58),  # Example 10-year US Treasury bond yield
        ('USTB30Y', 1.97)   # Example 30-year US Treasury bond yield
    ]
    return bond_prices


def fetch_crypto_data():
    """Fetch top 5 trending cryptocurrencies from CoinGecko."""
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'trending_desc',
        'per_page': 5,
        'page': 1,
        'sparkline': 'false'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data


def fetch_cryptos_below_threshold(cryptos, threshold_price):
    """Fetch cryptocurrencies with prices below the given threshold."""
    below_threshold = []
    for crypto in cryptos:
        try:
            price = crypto['current_price']
            if price is not None and price < threshold_price:
                below_threshold.append((crypto['id'], price))
        except Exception as e:
            print(f"Failed to get crypto data for {crypto['id']} reason: {e}")
    return below_threshold




@app.route('/financials')
def get_financials():
    # Fetch symbols dynamically from the S&P 500 list
    symbols = get_sp500_symbols()

    # Fetch stocks below the threshold price
    stocks_below_threshold = fetch_stocks_below_threshold(symbols, THRESHOLD_PRICE)

    # Fetch bond symbols and their prices
    bonds_below_threshold = fetch_bond_symbols_and_prices()

    # Fetch cryptocurrency data and filter below threshold price
    cryptos = fetch_crypto_data()
    cryptos_below_threshold = fetch_cryptos_below_threshold(cryptos, CRYPTO_THRESHOLD_PRICE)

    # Sort the results by price in ascending order and get the top 20
    stocks_below_threshold_sorted = sorted(stocks_below_threshold, key=lambda x: x[1])[:20]
    bonds_below_threshold_sorted = sorted(bonds_below_threshold, key=lambda x: x[1])[:20]
    cryptos_below_threshold_sorted = sorted(cryptos_below_threshold, key=lambda x: x[1])[:20]

    # Select the best combination based on lowest price
    best_stock = stocks_below_threshold_sorted[0] if stocks_below_threshold_sorted else None
    best_bond = bonds_below_threshold_sorted[0] if bonds_below_threshold_sorted else None
    best_crypto = cryptos_below_threshold_sorted[0] if cryptos_below_threshold_sorted else None

    # Example data preparation for FD interest rate prediction
    data = {
        'Age': [25, 35, 45, 55, 65],
        'FD_1_year': [5.0, 5.5, 6.0, 6.2, 6.5],
        'FD_3_years': [5.5, 6.0, 6.5, 6.8, 7.0],
        'FD_5_years': [6.0, 6.5, 7.0, 7.2, 7.5]
    }

    df = pd.DataFrame(data)

    # Define feature matrix X and target vectors y for different FD terms
    X = df[['Age']]
    y_1_year = df['FD_1_year']
    y_3_years = df['FD_3_years']
    y_5_years = df['FD_5_years']

    # Initialize linear regression models for each FD term
    model_1_year = LinearRegression()
    model_3_years = LinearRegression()
    model_5_years = LinearRegression()

    # Fit models for each FD term
    model_1_year.fit(X, y_1_year)
    model_3_years.fit(X, y_3_years)
    model_5_years.fit(X, y_5_years)

    # Predict FD returns for different age groups
    age_to_predict = 40  # Example age to predict

    predicted_fd_1_year = model_1_year.predict([[age_to_predict]])[0]
    predicted_fd_3_years = model_3_years.predict([[age_to_predict]])[0]
    predicted_fd_5_years = model_5_years.predict([[age_to_predict]])[0]

    # Calculate total threshold for demonstration
    total_threshold = (best_stock[1] if best_stock else 0) + (best_bond[1] if best_bond else 0) + (best_crypto[1] if best_crypto else 0)

    # Output the best combination
    best_financials = {
        'stocks': stocks_below_threshold_sorted,
        'bonds': bonds_below_threshold_sorted,
        'cryptos': cryptos_below_threshold_sorted,
        'fd_1_year': predicted_fd_1_year,
        'fd_3_years': predicted_fd_3_years,
        'fd_5_years': predicted_fd_5_years,
        'best_stock': best_stock,
        'best_bond': best_bond,
        'best_crypto': best_crypto,
        'total_threshold': total_threshold
    }

    return jsonify(best_financials)


@app.route('/gold')
def get_gold_price():
    """Fetch the current price of gold from CoinGecko API and convert to INR."""
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'gold',
        'vs_currencies': 'usd'
    }
    response = requests.get(url, params=params)
    data = response.json()
    gold_price_usd = data['gold']['usd']

    # Convert gold price from USD to INR (1 USD = 75 INR for example)
    gold_price_inr = '7380/g'  # Example fixed rate for demonstration

    return jsonify({'gold_price_inr': gold_price_inr})

@app.route('/assets1')
def assets1():
    # get_financials()
    return render_template('assets1.html')

if __name__ == '__main__':
    app.secret_key = 'secretkey'
    app.run(debug=True)
