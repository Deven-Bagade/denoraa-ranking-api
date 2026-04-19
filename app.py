import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================

print("🚀 Loading model artifacts...")

# For development, load from local file
# For production, you'll upload the model file to Render
MODEL_PATH = os.environ.get('MODEL_PATH', 'denoraa_ranker_model.pkl')

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    preprocessor = artifacts['preprocessor']
    feature_columns = artifacts['feature_columns']
    best_threshold = artifacts.get('best_threshold', 0.5)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None
    preprocessor = None
    feature_columns = []
    best_threshold = 0.5

# ============================================================
# FIREBASE INITIALIZATION (Optional - for reading provider data)
# ============================================================

# Use environment variables for Firebase credentials
firebase_credentials = os.environ.get('FIREBASE_CREDENTIALS')
if firebase_credentials:
    cred = credentials.Certificate(json.loads(firebase_credentials))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized")
else:
    print("⚠️ Firebase credentials not provided")
    db = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wilson_score(rating, n):
    """Calculate Wilson Score for reliability"""
    if n == 0:
        return 0
    z = 1.96
    p = rating / 5
    numerator = p + (z*z)/(2*n) - z * np.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    denominator = 1 + (z*z)/n
    return numerator / denominator

def extract_features(provider_data, request_data):
    """Extract features from provider and request data"""
    features = {}
    
    # Basic features
    features['provider_rating'] = provider_data.get('providerRating', 5.0)
    features['provider_total_jobs'] = provider_data.get('providerTotalJobs', 0)
    features['provider_exp_months'] = provider_data.get('providerExpMonths', 0)
    features['provider_price'] = provider_data.get('providerPrice', 0)
    features['provider_is_new'] = 1 if provider_data.get('providerTotalJobs', 0) < 10 else 0
    features['provider_acceptance_rate'] = provider_data.get('providerAcceptanceRate', 0)
    features['provider_response_rate'] = provider_data.get('providerResponseRate', 0)
    features['provider_reliability_score'] = provider_data.get('providerReliabilityScore', 0.5)
    features['provider_reports_count'] = provider_data.get('providerReportsCount', 0)
    features['provider_complaints_count'] = provider_data.get('providerComplaintsCount', 0)
    features['provider_cancellations_last_7d'] = provider_data.get('providerCancellationsLast7d', 0)
    features['provider_no_shows_last_7d'] = provider_data.get('providerNoShowsLast7d', 0)
    features['provider_ignores_last_7d'] = provider_data.get('providerIgnoresLast7d', 0)
    features['provider_recent_risk'] = provider_data.get('providerRecentRisk', 0)
    
    # Request features
    features['request_mode'] = request_data.get('requestMode', 'manual')
    features['is_urgent'] = 1 if request_data.get('isUrgent', False) else 0
    features['category_lvl_1'] = request_data.get('categoryLvl1', '')
    features['category_lvl_2'] = request_data.get('categoryLvl2', '')
    features['category_lvl_3'] = request_data.get('categoryLvl3', '')
    features['user_min_price'] = request_data.get('userMinPrice', 0)
    features['user_max_price'] = request_data.get('userMaxPrice', 10000)
    
    # User features
    user_data = request_data.get('user', {})
    features['user_reports_count'] = user_data.get('userReportsCount', 0)
    features['user_complaints_count'] = user_data.get('userComplaintsCount', 0)
    features['user_cancellations_last_7d'] = user_data.get('userCancellationsLast7d', 0)
    
    # Distance (provided by client or calculated)
    features['distance_km'] = request_data.get('distanceKm', 5)
    features['distance_score'] = request_data.get('distanceScore', 0.8)
    
    # Price match
    price_match = 1 if (features['provider_price'] >= features['user_min_price'] and 
                        features['provider_price'] <= features['user_max_price']) else 0
    features['price_match'] = price_match
    
    # Price diff
    if features['user_min_price'] > 0:
        features['price_diff_percent'] = ((features['provider_price'] - features['user_min_price']) / features['user_min_price']) * 100
    else:
        features['price_diff_percent'] = 0
    
    if features['user_max_price'] > features['user_min_price']:
        features['price_diff_normalised'] = (features['provider_price'] - features['user_min_price']) / (features['user_max_price'] - features['user_min_price'])
    else:
        features['price_diff_normalised'] = 0
    
    # Wilson Score
    features['wilson_score'] = wilson_score(features['provider_rating'], features['provider_total_jobs'])
    
    return features

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/rank', methods=['POST'])
def rank():
    """Rank a single provider"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        provider = data.get('provider', {})
        request_data = data.get('request', {})
        
        features = extract_features(provider, request_data)
        
        # Create feature vector in correct order
        feature_vector = []
        for col in feature_columns:
            value = features.get(col, 0)
            if isinstance(value, str):
                feature_vector.append(value)
            else:
                feature_vector.append(float(value))
        
        # Convert the list to a DataFrame with the correct column names
        df_features = pd.DataFrame([feature_vector], columns=feature_columns)
        X = preprocessor.transform(df_features)
        score = float(model.predict(X)[0])
        
        return jsonify({
            'score': score,
            'is_good_match': score >= best_threshold,
            'wilson_score': features.get('wilson_score', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rank_batch', methods=['POST'])
def rank_batch():
    """Rank multiple providers at once"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        providers = data.get('providers', [])
        request_data = data.get('request', {})
        
        results = []
        feature_vectors = []
        
        for provider in providers:
            features = extract_features(provider, request_data)
            
            feature_vector = []
            for col in feature_columns:
                value = features.get(col, 0)
                if isinstance(value, str):
                    feature_vector.append(value)
                else:
                    feature_vector.append(float(value))
            feature_vectors.append(feature_vector)
        
        # Batch transform and predict
        df_features = pd.DataFrame(feature_vectors, columns=feature_columns)
        X = preprocessor.transform(df_features)
        scores = model.predict(X).tolist()
        
        for i, provider in enumerate(providers):
            results.append({
                'provider_id': provider.get('provider_id', provider.get('uid')),
                'score': scores[i],
                'is_good_match': scores[i] >= best_threshold
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'results': results,
            'best_threshold': best_threshold
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/wilson', methods=['POST'])
def wilson():
    """Calculate Wilson Score for a provider"""
    try:
        data = request.json
        rating = data.get('rating', 5.0)
        total_jobs = data.get('total_jobs', 0)
        
        score = wilson_score(rating, total_jobs)
        
        return jsonify({
            'wilson_score': score,
            'rating': rating,
            'total_jobs': total_jobs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/providers/ranked', methods=['POST'])
def get_ranked_providers():
    """Get ranked list of providers for a request"""
    if db is None:
        return jsonify({'error': 'Firebase not configured'}), 500
    
    try:
        data = request.json
        category = data.get('categoryLvl1')
        user_location = data.get('userLocation')
        radius_km = data.get('radiusKm', 10)
        user_min_price = data.get('userMinPrice', 0)
        user_max_price = data.get('userMaxPrice', 10000)
        
        # Query providers from Firestore
        providers_ref = db.collection('providers')
        providers = []
        
        # Simple query (you can add more filters)
        query = providers_ref.where('isVerified', '==', True).limit(50)
        docs = query.stream()
        
        for doc in docs:
            provider_data = doc.to_dict()
            provider_data['provider_id'] = doc.id
            providers.append(provider_data)
        
        # Prepare request data for ranking
        request_data = {
            'categoryLvl1': category,
            'userMinPrice': user_min_price,
            'userMaxPrice': user_max_price,
            'userLocation': user_location
        }
        
        # Rank providers
        response = rank_batch_internal(providers, request_data)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def rank_batch_internal(providers, request_data):
    """Internal function to rank providers"""
    if model is None:
        return {'error': 'Model not loaded'}
    
    results = []
    feature_vectors = []
    
    for provider in providers:
        features = extract_features(provider, request_data)
        
        feature_vector = []
        for col in feature_columns:
            value = features.get(col, 0)
            if isinstance(value, str):
                feature_vector.append(value)
            else:
                feature_vector.append(float(value))
        feature_vectors.append(feature_vector)
    
    X = preprocessor.transform(feature_vectors)
    scores = model.predict(X).tolist()
    
    for i, provider in enumerate(providers):
        results.append({
            'provider_id': provider.get('provider_id'),
            'name': provider.get('username'),
            'rating': provider.get('providerRating', 5.0),
            'price': provider.get('providerPrice', 0),
            'score': scores[i],
            'is_good_match': scores[i] >= best_threshold
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'results': results,
        'best_threshold': best_threshold,
        'count': len(results)
    }

# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
