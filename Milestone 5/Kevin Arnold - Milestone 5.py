## Kevin Arnold
## DSC410-T301 Predictive Analytics
## Milestone 5 -- Dash app

import dash
from dash import html, dcc, callback, Output, Input
import joblib

# load your pre-trained model from Milestone 4
model = joblib.load('./model/ml_reg.h5')

# initialize Dash app
app = dash.Dash(__name__)

# define the layout of the web application
app.layout = html.Div([
    html.H1("Crime Incident Predictor"),

    html.Label("Enter today's temperature (in Fahrenheit) to get the prediction of how many criminal incidents will take place in the Omaha Northeast Precinct:"),
    html.P(),
    dcc.Input(id='temperature', type='number', value=0),

    html.Br(),
    html.P(),
    html.Div(id='result')
])


# define the callback to update the output based on the temperature input
@app.callback(Output('result', 'children'),
              [Input('temperature', 'value')])
def update_result(temperature):
    # Make a prediction using the pre-trained model
    temperature = float(temperature)  # Convert input to float
    prediction = model.predict([[temperature]])
    pred_rounded = round(prediction[0])

    # display the prediction
    return f"The predicted number of incidents is: {pred_rounded}"


# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
