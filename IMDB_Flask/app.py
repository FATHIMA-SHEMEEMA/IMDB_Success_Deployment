from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('model_out.pkl','rb'))

# Dictionary to map numerical values back to categories
# category_mapping = {
#     0: "Avg",
#     1: "Flop",
#     2: "Hit",

# }




app=Flask(__name__)


@app.route('/')
def main():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def home():



    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']
    data8=request.form['h']
    data9=request.form['i']
    data10=request.form['j']
    data11=request.form['k']
    data12=request.form['l']
    data13=request.form['m']
    data14=request.form['n']
    data15=request.form['o']
    arr=np.array([[data1,data2,data3,data4,
                   data5,data6,data7,data8,data9,
                   data10,data11,data12,data13,
                   data14,data15]])
    pred=model.predict(arr)
    return render_template("result.html",data=pred)


if __name__ == "__main__":
    app.run(debug=True)




# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# # Load the trained model
# model = pickle.load(open('model_out.pkl', 'rb'))

# # Dictionary to map numerical values back to categories
# category_mapping = {
#     0: "Avg",
#     1: "Flop",
#     2: "Hit",
# }

# app = Flask(__name__)

# @app.route('/')
# def main():
#     return render_template("home.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data
#     data1 = request.form['a']
#     data2 = request.form['b']
#     data3 = request.form['c']
#     data4 = request.form['d']
#     data5 = request.form['e']
#     data6 = request.form['f']
#     data7 = request.form['g']
#     data8 = request.form['h']
#     data9 = request.form['i']
#     data10 = request.form['j']
#     data11 = request.form['k']
#     data12 = request.form['l']
#     data13 = request.form['m']
#     data14 = request.form['n']
#     data15 = request.form['o']
    
#     # Combine the data into a NumPy array
#     arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]])
    
#     # Make the prediction
#     pred = model.predict(arr)[0]  # Get the first (and only) prediction
    
#     # Map the numerical prediction to the corresponding category
#     category = category_mapping.get(pred, "Unknown")
    
#     # Render the result page with the decoded prediction
#     return render_template("result.html", data=category)

# if __name__ == "__main__":
#     app.run(debug=True)
