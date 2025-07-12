from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import joblib
import os
import io
from typing import Optional

# Initialize FastAPI
app = FastAPI(
    title="Dog Breed Classifier",
    description="API for dog breed classification using ResNet18 features and Logistic Regression",
    version="1.0"
)

# Constants
MODEL_PATH = '/app/model/best_model.pth'
CLASS_NAMES_PATH = '/app/model/class_names.pkl'



class DogBreedClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self.feature_extractor = self._init_feature_extractor()
        self.transform = self._get_transform()

        # Инициализация scaler как None, если он не используется
        self.scaler = None
        # Если scaler действительно нужен, раскомментируйте следующую строку:
        # self.scaler = self._load_scaler() if os.path.exists(SCALER_PATH) else None

    def _load_class_names(self):
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")
        return joblib.load(CLASS_NAMES_PATH)

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        # 1. Загружаем сохранённые веса
        saved_state = torch.load(MODEL_PATH, map_location=self.device)

        # 2. Создаём модель с правильной архитектурой
        class DogBreedModel(nn.Module):
            def __init__(self, input_dim=512, output_dim=75):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                return self.linear(x)

        # 3. Инициализируем модель
        model = DogBreedModel(
            input_dim=512,
            output_dim=len(self.class_names)
        ).to(self.device)

        # 4. Загружаем веса без изменений
        model.load_state_dict(saved_state)
        model.eval()

        return model

    def _init_feature_extractor(self):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        return nn.Sequential(*list(model.children())[:-1]).eval().to(self.device)

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Метод _load_scaler больше не требуется, если scaler не используется
    # Если scaler нужен, раскомментируйте:
    # def _load_scaler(self):
    #     if not os.path.exists(SCALER_PATH):
    #         return None
    #     return joblib.load(SCALER_PATH)

    def predict(self, image_bytes):
        try:
            # Извлекаем фичи
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                features = features.squeeze().cpu().numpy().flatten()

                # Если scaler используется:
                # if self.scaler:
                #     features = self.scaler.transform(features.reshape(1, -1))

                # Делаем предсказание
                pred = self.model(torch.FloatTensor(features).to(self.device))
                prob = torch.softmax(pred, dim=0)
                confidence, pred_class = torch.max(prob, 0)

                return {
                    'breed': self.class_names[pred_class.item()],
                    'confidence': confidence.item(),
                    'status': 'success'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# Initialize classifier
try:
    classifier = DogBreedClassifier()
except Exception as e:
    raise RuntimeError(f"Failed to initialize classifier: {str(e)}")


@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        result = classifier.predict(contents)

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)