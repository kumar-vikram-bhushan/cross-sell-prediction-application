from flask import Flask, render_template, url_for , request, redirect, jsonify
import pickle
import numpy as np

app = Flask ( __name__ )
model = pickle.load(open('model.pkl', 'rb'))
@app.route ( '/' )
def home():
    return render_template('index.html')

@app.route( '/predict',methods=['POST'])
def predict():

    temp_array =list()
    if request.method== 'POST':
         Gender = request.form['Gender']
         if Gender =='Female':
            temp_array= temp_array+[1]
         else:
            temp_array = temp_array + [0]

         Age =int(request.form['Age'])
         temp_array = temp_array + [Age]

         Driving_Licence = request.form['Driving_Licence']
         temp_array = temp_array + [Driving_Licence]

         # if Driving_Licence  =='yes':
         #    temp_array= temp_array+[1]
         # else:
         #    temp_array = temp_array + [0]


         
         Region_Code = int(request.form['Region_Code'])
         temp_array = temp_array + [Region_Code]

         Previously_Insured = request.form['Previously_Insured']
         temp_array = temp_array + [Previously_Insured]
         # if Previously_Insured  =='yes':
         #    temp_array= temp_array+[1]
         # else:
         #    temp_array = temp_array + [0]


         
         Vehicle_Age = request.form['Vehicle_Age']
         if Vehicle_Age  == '< 1 Year':
            temp_array= temp_array+[1]
         elif Vehicle_Age == '1-2 Year':
            temp_array = temp_array+[3] 
         else:
            temp_array = temp_array + [2]
         

         Vehicle_Damage = int(request.form['Vehicle_Damage'])
         temp_array = temp_array + [Vehicle_Damage]
         # if Vehicle_Damage  == yes:
         #    temp_array= temp_array+[1]
         # else:
         #    temp_array = temp_array + [0]
         
         Annual_Premium = int(request.form['Annual_Premium'])
         temp_array = temp_array + [Annual_Premium]

         Policy_Sales_Channel = int(request.form['Policy_Sales_Channel'])
         temp_array = temp_array + [Policy_Sales_Channel]

         temp_array = temp_array +[100 ] 
         # print(temp_array)

         final_features = np.array([temp_array])
         prediction = model.predict( final_features )
         output = prediction[0]

         if output == 1:
                  output = 'Coustmer is interested'
         else:
                  output = 'Coustmer is not interested'

         return render_template ( 'result.html', prediction_text='  {}'.format( output ) )
         
    else:
      return render_template('index.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




if __name__ == '__main__':
    app.run (debug = True)