from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-seismic-events/")
async def predict_seismic_events(file: UploadFile = File(...)):
    # Load the dataset from the uploaded CSV file
    data = pd.read_csv(file.file)

    # Limit data to the first 1000 rows for faster processing
    data_limited = data.head(1000).copy()  # Explicitly create a copy of the DataFrame

    # Now modify the copied DataFrame
    data_limited['velocity_diff'] = data_limited['velocity'].diff()
    window_size = 5
    data_limited['rolling_mean'] = data_limited['velocity'].rolling(window=window_size).mean()
    data_limited['rolling_std'] = data_limited['velocity'].rolling(window=window_size).std()

    # Drop NaN values created by rolling functions
    data_limited.dropna(inplace=True)

    # Label seismic events based on velocity changes
    threshold = data_limited['velocity_diff'].quantile(0.95)  # 95th percentile as threshold
    data_limited['seismic_event'] = (data_limited['velocity_diff'] > threshold).astype(int)


    # Train the Random Forest model
    features = ['velocity', 'velocity_diff', 'rolling_mean', 'rolling_std']
    X = data_limited[features]
    y = data_limited['seismic_event']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and fit the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    predicted_seismic_events = clf.predict(X)

    # Plotting
    csv_times = np.array(data_limited['abs_time'].tolist())
    csv_data = np.array(data_limited['velocity'].tolist())

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(csv_times, csv_data, label='Velocity', color='blue')

    # Highlight predicted seismic events
    for i, pred in enumerate(predicted_seismic_events):
        if pred == 1:  # Seismic event detected
            plt.axvline(x=csv_times[i], color='red', linestyle='--', linewidth=1)

    # Make the plot pretty
    plt.xlim([min(csv_times), max(csv_times)])
    plt.ylim([min(csv_data) - 0.1, max(csv_data) + 0.1])  # Adjust limits for better visibility
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Seismic Event Detection', fontweight='bold')

    # Add legend and grid
    plt.legend(['Velocity', 'Predicted Seismic Events'], loc='upper right')
    plt.grid()

    # Save the plot to a BytesIO object and encode as base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Return JSON response with the base64-encoded plot and predicted seismic events
    return {
        "predicted_seismic_events":  "\n".join(map(str, predicted_seismic_events)),
        "plot": plot_base64
    }