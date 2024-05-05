from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = load('breast_cancer.pkl')

# Define list of features
features_list = [
    'texture_mean', 'smoothness_mean', 'compactness_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'texture_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se', 'texture_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]


class Features(BaseModel):
    texture_mean: float
    smoothness_mean: float
    compactness_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    texture_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    texture_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float



# Define prediction route
@app.post("/predict/")
async def predict(features: Features):
    try:        
        # chick input 
        input_data = np.array([[getattr(features, feature) for feature in features_list]])        

        # Make prediction
        prediction = model.predict(input_data)
        
        # Determine diagnosis result
        diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'
        
        # Return prediction as JSON
        
        return {"diagnosis": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define 404 error handler
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(status_code=404, content={"Project name": "Breast Cancer Prediction"})

if __name__ == '__main__':
    uvicorn.run(app, port=5012)
