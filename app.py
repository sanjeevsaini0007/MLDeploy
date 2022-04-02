
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            gre_score=float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            is_research = request.form['research']
            if(is_research=='yes'):
                research=1
            else:
                research=0
            filename = 'finalized_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            #my line
            df= pd.read_csv('Admission_Prediction_required.csv')
            x= df.drop(columns=['Chance of Admit'])
            st_sc= StandardScaler()
            st_sc.fit_transform(x)
            #my line ends
            prediction=loaded_model.predict(st_sc.transform([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]]))
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=round(100*prediction[0]))
            #return render_template('results.html', prediction= loaded_model.coef_)
        except Exception as e:
            #print('The Exception message is: ',e)
            return 'something is wrong' + 'The Exception message is: '+str(e)
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app