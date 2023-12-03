python
import torch
from models.experimental import attempt_load

class ModelAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
    
    def load_model(self):
        model = attempt_load(self.model_path, map_location=torch.device('cpu'))
        return model
    
    def get_anchor_grid(self, model):
        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
        return m.anchor_grid

# Usage
model_path = './best.pt'
analyzer = ModelAnalyzer(model_path)
model = analyzer.load_model()
anchor_grid = analyzer.get_anchor_grid(model)
print(anchor_grid)
