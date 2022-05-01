import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/Cyberbullyingdetection_sv.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open(f'model/cvec.pkl', 'rb') as i:
    loaded_vec = pickle.load(i)
    
# Initialise the Flask app
app = flask.Flask(__name__, template_folder='template')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        string = flask.request.form['string']
	string = string.enocode('TIS-620')
	string = text_process(string)
	token = loaded_vec.transform(pd.Series([string]))
	result_pred = model.predict(token)
	result_pred = str(result_pred)
	return render_template('main.html',prediction = result_pred)

if __name__ == '__main__':
    app.run()
