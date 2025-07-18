from flask import Flask, request, render_template
import pandas as pd
import pickle
import os  # Render'ın PORT ortam değişkeni için gerekli

app = Flask(__name__)

# Model yükleniyor
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Tahmin fonksiyonu
def predict_personality(time_alone, social_events, going_outside, friends_circle, post_freq, stage_fear, drained_after_social):
    sample = pd.DataFrame({
        "Time_spent_Alone": [time_alone],
        "Social_event_attendance": [social_events],
        "Going_outside": [going_outside],
        "Friends_circle_size": [friends_circle],
        "Post_frequency": [post_freq],
        "Stage_fear_No": [stage_fear == "No"],
        "Stage_fear_Yes": [stage_fear == "Yes"],
        "Drained_after_socializing_No": [drained_after_social == "No"],
        "Drained_after_socializing_Yes": [drained_after_social == "Yes"],
    })

    prediction = model.predict(sample)[0]
    label_map = {0: "İçe Dönük", 1: "Dışa Dönük"}
    return label_map[prediction]

# Ana route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        time_alone = float(request.form['time_alone'])
        social_events = float(request.form['social_events'])
        going_outside = float(request.form['going_outside'])
        friends_circle = float(request.form['friends_circle'])
        post_freq = float(request.form['post_freq'])
        stage_fear = request.form['stage_fear']
        drained_after_social = request.form['drained_after_social']

        result = predict_personality(time_alone, social_events, going_outside, friends_circle, post_freq, stage_fear, drained_after_social)

    return render_template('index.html', result=result)

# Render için port bind ayarı
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
