from powerTimeSeries.pipeline.predict import PredictionPipeline
from flask import Flask, render_template, request, send_file
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

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

        # Plot the graph for the predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(future_w_features.index, future_w_features['pred'], color='b', marker='o', markersize=2, linestyle='-')
        plt.title('Future Predictions')
        plt.xlabel('Date')
        plt.ylabel('Predicted Value')
        plt.tight_layout()

        # Save the plot as an image and return the image path
        plot_image_path = 'static/plot.png'
        plt.savefig(plot_image_path)
        plt.close()

        return render_template('result.html', future_w_features=future_w_features, plot_image_path=plot_image_path)

    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download_csv():
    csv_data = request.args.get('csv_data')
    csv_filename = "future_predictions.csv"  # Hardcoded filename
    return send_file(BytesIO(csv_data.encode()), as_attachment=True, attachment_filename=csv_filename)

if __name__ == '__main__':
    # Start the Tkinter main loop in a separate thread
    import threading
    tkinter_thread = threading.Thread(target=run_tkinter)
    tkinter_thread.start()

    # Run Flask in a separate thread
    app.run(debug=True, threaded=True)
