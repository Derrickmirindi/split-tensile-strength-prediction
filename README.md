# split-tensile-strength-prediction

NODE-based Split Tensile Strength Prediction System for rubberized brick-powder concrete. Developed by Derrick Mirindi.

## Overview

This web application uses a trained Neural Oblivious Decision Ensemble (NODE) model to predict the split tensile strength (MPa) of rubberized brick-powder concrete from seven mix-design and curing parameters. It runs entirely in the browser using TensorFlow.js and serves as a fast decision-support tool for material design.

## Inputs (7 features)

1. Cement (kg)
2. Brick powder (kg)
3. Water (kg)
4. Fine aggregate (kg)
5. Coarse aggregate (kg)
6. Waste tire rubber (kg)
7. Age (day)

## Output

- Split tensile strength (MPa)

## Model

The NODE model is an ensemble of independent sub-networks (HeNormal-seeded) with LayerNormalization, swish activation, dropout, and a learned combination layer (Concatenate + Dense). Features are standardized with a StandardScaler and the target is z-score normalized; predictions are inverse-transformed back to MPa in the browser.

## Required files

The website (`index.html`) expects three additional files in the repository root:

- `model.json` - TensorFlow.js model topology
- `group1-shard1of1.bin` - model weights
- `scaler_params.json` - standardization parameters (`X_mean`, `X_std`, `Y_mean`, `Y_std`)

## Exporting the model from Colab

After training `node_model` in your notebook, run:

```python
!pip install tensorflowjs
import tensorflowjs as tfjs, json
tfjs.converters.save_keras_model(node_model, '/content/tfjs_model')

scaler_params = {
    "X_mean": scaler.mean_.tolist(),
    "X_std":  scaler.scale_.tolist(),
    "Y_mean": float(y_mean),
    "Y_std":  float(y_std)
}
with open('/content/tfjs_model/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)
```

Then download `model.json`, the `.bin` shard, and `scaler_params.json` and upload them to this repository root.

## Deployment

The site is served via GitHub Pages from the `main` branch. Once the model files are uploaded, open the published URL to use the predictor.

Developed by Derrick Mirindi.
