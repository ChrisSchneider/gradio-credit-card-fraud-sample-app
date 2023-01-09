import os
import requests
import gradio as gr
import pandas as pd

examples_df = pd.read_csv("examples.csv")

WML_API_KEY = os.getenv("WML_API_KEY")
if not WML_API_KEY:
  raise "Env variable WML_API_KEY not specified"

WML_URL = os.getenv("WML_URL")
if not WML_URL:
  raise "Env variable WML_URL not specified"

# Sends the input data from the input components to the
# WML scoring endpoint and then readies the data for the output components
def fraud_detector(activity_df, threshold):
  activity_df = activity_df.drop(["True Class"], axis=1)

  # Create a temporary access token with out API key (valid for 1 hour)
  res = requests.post('https://iam.cloud.ibm.com/identity/token', data={
    "apikey": WML_API_KEY,
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
  })
  res.raise_for_status()
  access_token = res.json()["access_token"]

  # Score the data
  res = requests.post(WML_URL, json={
    "input_data": [{
      "fields": list(activity_df.columns),
      "values": activity_df.values.tolist(),
    }]
  }, headers={
      "Authorization": "Bearer " + access_token
  })
  res.raise_for_status()
  pred = res.json()["predictions"][0]

  pred_df = pd.DataFrame([{
    "Time": activity_df["Time"][i],
    "Pred": x[1][1],
    "Fraud": x[1][1] > threshold,
  } for i, x in enumerate(pred["values"])])
  return (
      f"{pred_df.Fraud.sum()} / {len(pred_df)}",
      pred_df,
  )

# Components to specify inputs for the algorithm
inputs = [
  gr.Dataframe(
    row_count=(5,"dynamic"),
    col_count=(31,"fixed"),
    headers=list(examples_df.columns),
    label="Transactions"),
  gr.Slider(0.0, 1.0, 0.6),
]

# Components to display results of the algorithm
outputs = [
    gr.Label(label="Fraudulent Transactions"),
    gr.Dataframe(
      row_count=(2,"dynamic"),
      col_count=(3,"fixed"),
      label="Predictions"),
]

# Create and launch the UI
gr.Interface(
  fraud_detector,
  inputs,
  outputs,
  title="Credit Card Fraud Detection",
  examples = [[examples_df, 0.6]],
  allow_flagging="never",
).launch(
  server_name="0.0.0.0",
  server_port=int(os.getenv("PORT")) if "PORT" in os.environ else None,
)
