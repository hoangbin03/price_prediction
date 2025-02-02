from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load mô hình đã huấn luyện 
model = joblib.load('model/best_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Lấy dữ liệu từ frontend
    export_volume = data['exportVolume']
    import_volume = data['importVolume']
    seasons = data['seasons']
    location = data['location'] 

    # Tạo vector tính năng cho mô hình 
    features = [export_volume, import_volume] + [1 if season in seasons else 0 for season in ['north', 'central', 'south']]
    
    # Thêm location vào tính năng
    location_features = [1 if loc == location else 0 for loc in ['BDH', 'HNI', 'HDG']]
    features += location_features
    
    # Dự đoán giá
    prediction = model.predict([features])

    return jsonify({'predictedPrice': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
