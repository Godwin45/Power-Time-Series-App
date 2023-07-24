from powerTimeSeries.pipeline.predict import PredictionPipeline
from flask import Flask, render_template, request, send_file
import pandas as pd
import plotly.graph_objs as go
from io import BytesIO
import os
import threading

app = Flask(__name__)


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

        fig.update_layout(title='Future Power Predictions', xaxis_title='Date', yaxis_title='Predicted Value', plot_bgcolor='#F2F2F2')

        # Convert the Plotly figure to JSON to pass to the HTML template
        plot_json = fig.to_json()

        return render_template('result.html', future_w_features=future_w_features, plot_json=plot_json)

    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download_csv():
    csv_data = request.args.get('csv_data', default='', type=str)  # Get the CSV data from the URL query string
    excel_filename = "future_predictions.xlsx"  # Hardcoded filename for Excel

    # Parse the CSV data into a DataFrame
    df = pd.read_csv(BytesIO(csv_data.encode()))

    # Save the DataFrame to an Excel file in the 'static' folder
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    excel_file_path = os.path.join(static_folder, excel_filename)

    df.to_excel(excel_file_path, index=False)

    return send_file(excel_file_path, as_attachment=True, download_name=excel_filename)

if __name__ == '__main__':
  
    # Run Flask in the main thread with host='0.0.0.0' and port=8080
    app.run()
