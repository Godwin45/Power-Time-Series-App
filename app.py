from powerTimeSeries.pipeline.predict import PredictionPipeline
from flask import Flask, render_template, request, send_file
import tkinter as tk
import pandas as pd
import plotly.graph_objs as go
from io import BytesIO
import os

app = Flask(__name__)

# Create the Tkinter root and suppress its main window
root = tk.Tk()
root.withdraw()

def run_tkinter():
    # Start the Tkinter main loop
    root.mainloop()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        prediction_pipeline = PredictionPipeline("artifacts/transformed_data/data.csv",
                                                  "artifacts/training/model.json")
        future_w_features = prediction_pipeline.predict_future(start_date, end_date)

        # Create a Plotly scatter plot for the predicted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_w_features.index, y=future_w_features['pred'],
                                 mode='markers+lines', marker=dict(size=4),
                                 line=dict(width=1)))

        fig.update_layout(title='Future Predictions', xaxis_title='Date', yaxis_title='Predicted Value')

        # Convert the Plotly figure to JSON to pass to the HTML template
        plot_json = fig.to_json()

        return render_template('result.html', future_w_features=future_w_features, plot_json=plot_json)

    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download_csv():
    csv_data = request.args.get('csv_data', default='', type=str)  # Get the CSV data from the URL query string
    csv_filename = "future_predictions.csv"  # Hardcoded filename

    # Save the predicted data to a CSV file in the 'static' folder
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    csv_file_path = os.path.join(static_folder, csv_filename)

    with open(csv_file_path, 'w') as file:
        file.write(csv_data)

    return send_file(csv_file_path, as_attachment=True, download_name=csv_filename)

if __name__ == '__main__':
    # Start the Tkinter main loop in a separate thread
    import threading
    tkinter_thread = threading.Thread(target=run_tkinter)
    tkinter_thread.start()

    # Run Flask in a separate thread
    app.run(debug=True, threaded=True)
