## Features
Two feature extraction methods are implemented:
1. **Pixel Features** - Each character converted to binary (space=0, symbol=1). Digits: 784 features, Faces: 4200 features
2. **Grid Features** - Image divided into 8x8 grid, each cell stores symbol density. 64 features per image

## How to Run
```bash
python3 main.py
```

## Results Summary
| Dataset | Algorithm | Feature | Accuracy |
|---------|-----------|---------|----------|
| Digits  | Naive Bayes | Pixel | ~77% |
| Digits  | Perceptron  | Pixel | ~80% |
| Faces   | Naive Bayes | Pixel | ~90% |
| Faces   | Perceptron  | Pixel | ~88% |
| Digits  | Naive Bayes | Grid  | ~66% |
| Digits  | Perceptron  | Grid  | ~65% |
| Faces   | Naive Bayes | Grid  | ~71% |
| Faces   | Perceptron  | Grid  | ~74% |

## Requirements
- Python 3
- NumPy

## Notes
- No ML libraries used — all algorithms implemented from scratch
- Test data is never accessed during training
- Each experiment runs 5 random trials at 10%-100% training data increments