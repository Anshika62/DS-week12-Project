import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open(r"E:/DEVELOPERS_ARENA(INTERNSHIP)PROJECTS/Week-12/deployment/model.pkl", "rb"))
scaler = pickle.load(open(r"E:/DEVELOPERS_ARENA(INTERNSHIP)PROJECTS/Week-12/deployment/scaler.pkl", "rb"))
pca = pickle.load(open(r"E:/DEVELOPERS_ARENA(INTERNSHIP)PROJECTS/Week-12/deployment/pca.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        input_data = [
            int(request.form["Contract"]),
            int(request.form["PaymentMethod"]),
            int(request.form["PaperlessBilling"]),
            int(request.form["SeniorCitizen"]),
            float(request.form["Tenure"]),
            float(request.form["MonthlyCharges"]),
            float(request.form["TotalCharges"])
        ]

        input_array = np.array(input_data).reshape(1, -1)


        scaled = scaler.transform(input_array)
        pca_data = pca.transform(scaled)

        prediction = model.predict(pca_data)
        result = "There are more chances to Customer will Churn " if prediction[0] == 1 else "Here the customer is loyal for Comapany,and he will NOT Churn "

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
