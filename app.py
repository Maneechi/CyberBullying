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
		result_pred = model.predict(loaded_vec.transform([string]))
		return render_template('main.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run()
