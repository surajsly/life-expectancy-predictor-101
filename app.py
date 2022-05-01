import pandas as pd
from flask import Flask, jsonify, request, after_this_request , Response
import pickle
from sklearn.preprocessing import scale
import json

# load model
model = pickle.load(open('gradientBoostingModel.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():

    
    # get data
    data = request.get_json(force=True)
    print(data)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df =  pd.DataFrame.from_dict(data, orient='index')
    data_df = scale(data_df)
    # predictions
    result = model.predict(data_df.reshape(1, -1))

    # send back to browser
    output = {'results': result[0]}
    output = jsonify(json.dumps(output))
    output.headers.add('Access-Control-Allow-Origin', '*')
    print(output)

    # return data
    return output

if __name__ == '__main__':
    app.run()
