import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import time

def create_dictionaries(df):
    # Create dictionaries for encoding
    season_dict = {val: i + 1 for i, val in enumerate(df['Season'].unique())}
    location_dict = {val: i + 1 for i, val in enumerate(df['Location'].unique())}
    weather_type_dict = {val: i + 1 for i, val in enumerate(df['Weather Type'].unique())}

    return season_dict, location_dict, weather_type_dict

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Rename columns to standardize names
    df.columns = [col.strip() for col in df.columns]

    # Check if 'Cloud Cover' is in the DataFrame and drop it
    if 'Cloud Cover' in df.columns:
        df = df.drop(columns=['Cloud Cover'])

    # Define numerical features, excluding 'Cloud Cover'
    numerical_features = [
        'Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Atmospheric Pressure',
        'UV Index', 'Visibility (km)'
    ]

    # Generate encoding dictionaries
    season_dict, location_dict, weather_type_dict = create_dictionaries(df)

    # Apply encoding dictionaries to the categorical columns
    df['Season'] = df['Season'].map(season_dict)
    df['Location'] = df['Location'].map(location_dict)
    df['Weather Type'] = df['Weather Type'].map(weather_type_dict)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Ensure that all categorical columns are mapped correctly
    for col in ['Season', 'Location', 'Weather Type']:
        if df[col].isnull().any():
            raise ValueError(f"Missing values detected in column '{col}' after mapping.")

    # Separate features and target
    X = df.drop(columns='Weather Type')
    y = df['Weather Type']

    # Check if all numerical features exist in the DataFrame
    for feature in numerical_features:
        if feature not in X.columns:
            raise ValueError(f"Numerical feature '{feature}' not found in the data.")

    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y, scaler, df, season_dict, location_dict, weather_type_dict, numerical_features

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(1, len(y.unique()) + 1)]))

    return model

def predict_weather(model, user_input, season_dict, location_dict, weather_type_dict, scaler, numerical_features):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Map categorical inputs to numerical values using the dictionaries
    user_df['Season'] = season_dict.get(user_input['Season'], -1)
    user_df['Location'] = location_dict.get(user_input['Location'], -1)

    # Check if any values are not in the dictionary
    if user_df[['Season', 'Location']].eq(-1).any().any():
        raise ValueError("One or more categorical values are not recognized.")

    # Ensure all required columns are present
    for feature in numerical_features:
        if feature not in user_df.columns:
            user_df[feature] = 0  # Default value if column is missing

    # Ensure correct order of columns
    #user_df = user_df[numerical_features]

    # Apply the same scaling to the user input as was applied to the training data
    #user_df[numerical_features] = scaler.transform(user_df[numerical_features])

    # Predict the weather type
    prediction = model.predict(user_df)

    # Map numerical predictions back to weather types
    weather_type_dict_reverse = {v: k for k, v in weather_type_dict.items()}
    return weather_type_dict_reverse.get(prediction[0], "Unknown")

def plot_data(df, season_dict, location_dict, weather_type_dict):
    # Histogram of average humidity by weather type
    plt.figure(figsize=(14, 7))

    # Calculate average humidity for each weather type
    weathavg = {}
    weather_counts = {}
    humidness = df['Humidity']
    weathtype = df['Weather Type']

    for i in range(len(df)):
        weather = df['Weather Type'].values[i]
        humidity = int(df['Humidity'].values[i])

        if weather not in weathavg:
            weathavg[weather] = [0, 0]
            weather_counts[weather] = 0

        weathavg[weather][0] += humidity
        weathavg[weather][1] += 1
        weather_counts[weather] += 1

    weather_types = list(weathavg.keys())
    average_humidities = [weathavg[weather][0] / weathavg[weather][1] for weather in weather_types]

    sorted_indices = np.argsort(average_humidities)
    sorted_weather_types = [weather_types[i] for i in sorted_indices]
    sorted_average_humidities = [average_humidities[i] for i in sorted_indices]

    # Define a color set
    colors = ['skyblue', 'lightgreen', 'coral', 'gold', 'purple']

    # Create the histogram (bar chart)
    plt.subplot(1, 2, 1)
    plt.bar(sorted_weather_types, sorted_average_humidities, color=colors[:len(sorted_weather_types)], width=0.5)
    plt.xlabel('Weather Type', rotation=45)  # Rotate x-axis label
    plt.ylabel('Average Humidity', rotation=90)  # Rotate y-axis label
    plt.title('Average Humidity by Weather Type')
    for i, temp in enumerate(sorted_average_humidities):
        plt.text(i, temp, f'{temp:.2f}', ha='center', va='bottom')
    plt.grid(False)  # Removes the grid

    # Pie chart (donut chart) of weather type distribution
    plt.subplot(1, 2, 2)
    weather_types_pie = list(weather_counts.keys())
    counts = list(weather_counts.values())

    # Donut chart parameters
    thickness = 0.4
    plt.pie(counts, labels=None, colors=colors[:len(weather_types_pie)], wedgeprops=dict(width=thickness))

    # Add legend to list weather types with colors
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') for
                 color in colors[:len(weather_types_pie)]],
        labels=weather_types_pie,
        title="Weather Types",
        bbox_to_anchor=(1, 1),
        loc="upper left"
    )
    plt.title('Weather Type Distribution')

    plt.tight_layout()
    plt.show()

    # Boxplots for numerical features by season
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Season', y='Temperature')
    plt.title('Temperature by Season')
    plt.xlabel('Season')
    plt.ylabel('Temperature (°C)')
    plt.grid(False)

    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Season', y='Humidity')
    plt.title('Humidity by Season')
    plt.xlabel('Season')
    plt.ylabel('Humidity (%)')
    plt.grid(False)

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Season', y='Wind Speed')
    plt.title('Wind Speed by Season')
    plt.xlabel('Season')
    plt.ylabel('Wind Speed (km/h)')
    plt.grid(False)

    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Season', y='Precipitation')
    plt.title('Precipitation by Season')
    plt.xlabel('Season')
    plt.ylabel('Precipitation (%)')
    plt.grid(False)

    plt.tight_layout()
    plt.show()

    # Boxplots for numerical features by location
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Location', y='Temperature')
    plt.title('Temperature by Location')
    plt.xlabel('Location')
    plt.ylabel('Temperature (°C)')
    plt.grid(False)

    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Location', y='Humidity')
    plt.title('Humidity by Location')
    plt.xlabel('Location')
    plt.ylabel('Humidity (%)')
    plt.grid(False)

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Location', y='Wind Speed')
    plt.title('Wind Speed by Location')
    plt.xlabel('Location')
    plt.ylabel('Wind Speed (km/h)')
    plt.grid(False)

    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Location', y='Precipitation')
    plt.title('Precipitation by Location')
    plt.xlabel('Location')
    plt.ylabel('Precipitation (%)')
    plt.grid(False)

    plt.tight_layout()
    plt.show()

def main():
    file_path = 'weather_classification_data.csv'

    # Load and preprocess data
    X, y, scaler, df, season_dict, location_dict, weather_type_dict, numerical_features = load_and_preprocess_data(file_path)

    # Train the model
    model = train_model(X, y)

    # User input
    print("Please enter the weather information:")
    user_input = {
        'Temperature': float(40),
        'Humidity': float(83),
        'Wind Speed': float(1.5),
        'Precipitation': float(82),
        'Atmospheric Pressure': float(1000),
        'UV Index': float(8),
        'Season': str("Summer"),
        'Visibility (km)': float(1),
        'Location': str("mountain")
    }

    # Predict weather type based on user input
    try:
        prediction = predict_weather(model, user_input, season_dict, location_dict, weather_type_dict, scaler, numerical_features)
        print(f"Predicted Weather Type: {prediction}")
        print(time.time()-stime)
    except ValueError as e:
        print(f"Error: {e}")

    # Generate plots
    plot_data(df, season_dict, location_dict, weather_type_dict)

if __name__ == "__main__":
    stime = time.time()
    main()
