from flask import Flask, render_template, request
import copy
import pickle
import numpy as np
import question_set

app = Flask(__name__, static_url_path='', static_folder='')

questions = copy.deepcopy(question_set.original_questions)
type_questions = copy.deepcopy(question_set.type_questions)
default_values = copy.deepcopy(question_set.default_values)
selected_surfactant = 'AFFF'


def generate_question_id():
    q_inx = 0
    q_id = {}
    for question_group in questions.keys():
        sub_questions = questions[question_group]
        for quest in sub_questions.keys():
            q_type = type_questions[quest]
            if q_type == 'checkbox':
                answers = sub_questions[quest]
                for a in answers:
                    q_id[quest + "_" + a] = q_inx
                    q_inx += 1
            else:
                q_id[quest] = q_inx
                q_inx += 1
    return q_id

@app.route('/')
def flask_app():
    return render_template('main.html', questions=questions, questions_type=type_questions,
                           result_OV_Regressor=0,
                           result_OV_Classifier='None',
                           result_Turbidity_Regressor=0,
                           default_values=default_values,
                           selected_surfactant=selected_surfactant)


@app.route("/select_surfactant", methods=['POST'])
def select_surfactant():
    global selected_surfactant

    selected_surfactant = request.form['sel_value']

    return render_template('main.html', questions=questions, questions_type=type_questions,
                           result_OV_Regressor=0,
                           result_OV_Classifier='None',
                           result_Turbidity_Regressor=0,
                           default_values=default_values,
                           selected_surfactant=selected_surfactant)

@app.route('/flask_app', methods=['POST'])
def flask_app_answers():
    q_id = generate_question_id()

    X_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for question_group in questions.keys():
        sub_questions = questions[question_group]

        for quest in sub_questions.keys():
            if type_questions[quest] == 'combobox':
                answered = request.form[quest]
                if answered.isdigit():
                    X_test[q_id[quest]] = float(answered)
                else:
                    answer_inx = sub_questions[quest].index(answered)
                    X_test[q_id[quest]] = answer_inx
            elif type_questions[quest] == 'editbox':
                answered = request.form[quest]
                if not answered:
                    X_test[q_id[quest]] = 0
                else:
                    X_test[q_id[quest]] = float(answered)
            elif type_questions[quest] == 'checkbox':
                answers = sub_questions[quest]
                for a in answers:
                    quest_a = quest + "_" + a
                    if quest_a in request.form:
                        X_test[q_id[quest_a]] = 1
            else:
                if quest in request.form:
                    answered = request.form[quest]
                    answers = questions[quest]
                    inx = answers.index(answered)
                    X_test[q_id[quest]] = inx

    surfactant_names = ['AFFF', 'B&B', 'Blast', 'Calla', 'Powergreen', 'PRC', 'SDS', 'Surge', 'Triton-X-100', 'Type 1']
    sf_index = surfactant_names.index(selected_surfactant)
    X_test.insert(0, sf_index)

    print(X_test)

    # load it ML alg
    with open('ml_models/ML_Alg_OV_Regressor.pkl', 'rb') as fid:
        ML_Alg_OV_Regressor = pickle.load(fid)

    with open('ml_models/ML_Alg_OV_Classifier.pkl', 'rb') as fid:
        ML_Alg_OV_Classifier = pickle.load(fid)

    with open('ml_models/ML_Alg_Turbidity_Regressor.pkl', 'rb') as fid:
        ML_Alg_Turbidity_Regressor = pickle.load(fid)

    # On web
    # with open('/home/woohyounglee/mysite/ml_models/ML_Alg_OV_Regressor.pkl', 'rb') as fid:
    #     ML_Alg_OV_Regressor = pickle.load(fid)
    #
    # with open('/home/woohyounglee/mysite/ml_models/ML_Alg_OV_Classifier.pkl', 'rb') as fid:
    #     ML_Alg_OV_Classifier = pickle.load(fid)
    #
    # with open('/home/woohyounglee/mysite/ml_models/ML_Alg_Turbidity_Regressor.pkl', 'rb') as fid:
    #     ML_Alg_Turbidity_Regressor = pickle.load(fid)


    # Change to numpy array
    X_test = np.array(X_test)

    # Ceshape for a single sample with many features.
    X_test = X_test.reshape(1, -1)

    # Test prediction
    pred_OV_Regressor = ML_Alg_OV_Regressor.predict(X_test)*100

    prob = ML_Alg_OV_Classifier.predict_proba(X_test)[0][1]
    if 0 <= prob and prob < 0.33:
        pred_OV_Classifier = 'Low'
    elif 0.33 <= prob and prob < 0.66:
        pred_OV_Classifier = 'Mid'
    elif 0.66 <= prob:
        pred_OV_Classifier = 'High'

    pred_Turbidity_Regressor = ML_Alg_Turbidity_Regressor.predict(X_test)*100

    if pred_Turbidity_Regressor > 100:
        pred_Turbidity_Regressor = 0

    return render_template('main.html', scroll='break_point',
                           questions=questions, questions_type=type_questions,
                           result_OV_Regressor=pred_OV_Regressor,
                           result_OV_Classifier=pred_OV_Classifier,
                           result_Turbidity_Regressor=pred_Turbidity_Regressor,
                           default_values=default_values,
                           selected_surfactant=selected_surfactant)

if __name__ == '__main__':
    print(app.root_path)
    app.run(debug=True)