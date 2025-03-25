# MLflow Integration and Hyperparameter Tuning

## Recent Commits Summary
Recent updates to the [aipnd-project repository](https://github.com/jtayl222/aipnd-project/tree/master) integrated MLflow to systematically track and optimize machine learning experiments. These commits specifically enabled:

- MLflow tracking setup for capturing hyperparameters, performance metrics, and artifacts.
- Systematic logging of training experiments.
- Storage of experiment details and results within the `mlruns` folder.

## Hyperparameter Grid
The recent experiments leveraged a systematic grid search approach using the following hyperparameters:

```python
lr_values = [0.0001, 0.001, 0.01]
hidden_units_values = [256, 512, 1024]
```

## Key Findings
Analysis of experiments recorded in MLflow revealed:

- **Optimal Hyperparameters:**
  - Learning Rate: `0.0001`
  - Hidden Units: `512`
  - Highest achieved accuracy: `~91.93%`

- **Performance Patterns:**
  - Lower learning rates consistently outperformed higher learning rates.
  - Moderate hidden layer sizes (e.g., 512) provided the best performance balance.

## Recommended Next Steps

- To further enhance the performance, we have two promising directions:

### 1. Further Hyperparameter Tuning (Refining Current Model):
Since we found that a smaller learning rate (0.0001) and 512 hidden units performed best, you could:

- Experiment with even smaller learning rates:

   - Try values like 5e-5, 1e-5.

- Fine-tune hidden layers and units:

   - Test intermediate hidden unit sizes such as 384, 768, or a combination of multiple hidden layers.

- Adjust regularization and dropout:

   - Experiment with dropout rates (0.1, 0.3, 0.5) to manage potential overfitting.

- Change optimizers:

   - Test optimizers like AdamW, SGD with momentum, or RMSprop.

### 2. Different Architecture (e.g., MobileNet):
Considering a different architecture such as MobileNet could lead to significant performance improvements, especially if your current model plateaus despite hyperparameter tuning:

- Advantages of MobileNet:

   - Lightweight with fewer parameters (faster training and inference).

   - Specifically optimized for efficiency and accuracy.

   - Better generalization potential due to transfer learning from ImageNet.

- Approach:

   - Use MobileNet with pre-trained weights as a feature extractor.

   - Experiment with fine-tuning different layers of MobileNet.

   - Tune hyperparameters specifically optimized for MobileNet architecture, starting with a lower learning rate (1e-4 or 5e-5).

### Recommended Next Step:
If the current model’s accuracy (~92%) isn't meeting the goal, shifting to MobileNet (or another transfer learning-based architecture) would likely provide more significant gains compared to incremental hyperparameter improvements.

However, if your current model shows continuous improvement with subtle hyperparameter adjustments, further fine-tuning would be beneficial before moving on.

#### Suggested immediate action:

- Run a small-scale comparison experiment:
Current model with finer hyperparameter tuning vs. MobileNet with basic transfer learning.

This dual approach will quickly indicate the most promising path forward.