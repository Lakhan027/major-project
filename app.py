from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, jsonify, render_template
##from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('RF_Speech22.pkl','rb'))
model = pickle.load(open('DT_Speech22.pkl','rb'))
model = pickle.load(open('KNN_Speech22.pkl','rb'))
model = pickle.load(open('SVM_Speech22.pkl','rb'))
model = pickle.load(open('NB_Speech22.pkl','rb'))
#run_with_ngrok(app)

dataset= pd.read_csv('train.csv')

@app.route('/')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')

@app.route('/final')
def final():
    return render_template('final.html')

@app.route('/ml')
def ml():
    return render_template('ml.html')

@app.route('/links')
def links():
    return render_template('links.html')
  
@app.route('/predict',methods=['GET'])
def predict():
    
    twt = (request.args.get('twt'))
  
   # Cleaning the texts for all review using for loop

    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      print(review)
      corpus.append(review)
    
    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    print(cv)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    #Random Forest
    
    # Fitting Random Forest to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier_RF.fit(X_train, y_train)


    #-------------------------------------------------------------------------------------------

    #Decision Tree
    
    from sklearn.tree import DecisionTreeClassifier
    classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_DT.fit(X_train, y_train)

   
    #--------------------------------------------------------------------------------------

    #KNN
    
    from sklearn.neighbors import KNeighborsClassifier
    classifier_KNN =  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier_KNN.fit(X_train, y_train)

    
    #-------------------------------------------------------------------------------------

    #SVM
    from sklearn.svm import SVC
    classifier_SVM = SVC(kernel = 'linear', random_state = 0)
    classifier_SVM.fit(X_train, y_train)

    
    #--------------------------------------------------------------------------------------

    #Nave Bayes

    from sklearn.naive_bayes import GaussianNB
    classifier_NB = GaussianNB()
    classifier_NB.fit(X_train, y_train)

   
    #----------------------------------------------------------------------------------
   
    #----------------------------------------------------------------------------------

    Model = (request.args.get('Model'))

    if Model=="Random Forest Classifier":
        input_data = [twt] 
        input_data = cv.transform(input_data).toarray()
        input_pred = classifier_RF.predict(input_data)
        input_pred = input_pred.astype(int)

    elif Model=="Decision Tree Classifier":
        input_data = [twt] 
        input_data = cv.transform(input_data).toarray()
        input_pred = classifier_DT.predict(input_data)
        input_pred = input_pred.astype(int)

    elif Model=="KNN Classifier":
        input_data = [twt] 
        input_data = cv.transform(input_data).toarray()
        input_pred = classifier_KNN.predict(input_data)
        input_pred = input_pred.astype(int)

    elif Model=="SVM Classifier":
        input_data = [twt] 
        input_data = cv.transform(input_data).toarray()
        input_pred = classifier_SVM.predict(input_data)
        input_pred = input_pred.astype(int)

    else:
        input_data = [twt] 
        input_data = cv.transform(input_data).toarray()
        input_pred = classifier_NB.predict(input_data)
        input_pred = input_pred.astype(int)

    
    if input_pred[0] == 1:
      return render_template('final.html', prediction_text='This is a Positive Review', extra_text ="-> Prediction by " + Model)
    
    else:
      return render_template('final.html', prediction_text='This is a Negative Review', extra_text ="-> Prediction by " + Model)


if __name__ == "__main__":
    app.run(debug=True)

