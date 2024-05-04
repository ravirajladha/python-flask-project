from flask import Flask, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def create_app():
    app = Flask(__name__)

    # Load your CSV file
    df = pd.read_csv('bus_crowd_data_holidays_OG_copy(changed)_1.csv')

    X = df[['time_of_day', 'weather_condition', 'holidays']]
    y = df['crowded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    @app.route('/')
    def index():
        return 'Hello, world!'

    @app.route('/predict', methods=['POST'])
    def predict():
        time_of_day = int(request.form['time_of_day'])
        weather_condition = int(request.form['weather_condition'])
        holidays = int(request.form['holidays'])

        new_data = [[time_of_day, weather_condition, holidays]]
        prediction = model.predict(new_data)
        result = "The bus is crowded." if prediction[0] else "The bus is not crowded."
        return result

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
