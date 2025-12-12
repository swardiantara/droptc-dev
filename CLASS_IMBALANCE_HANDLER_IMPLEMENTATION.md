# Class Imbalance Handler Implementation Summary

## Overview
Successfully incorporated class imbalance handling with support for both **CrossEntropyLoss** and **Focal Loss**, along with three class weighting strategies.

## Changes Made

### 1. **New Command-Line Arguments**

Added to `get_args()`:
```
--loss_fn {ce, focal}  Loss function (default: ce)
--class_weight {uniform, balanced, inverse}  Class weighting strategy (default: uniform)
```

### 2. **New Helper Functions**

#### `FocalLoss` (torch.nn.Module)
- Custom implementation of Focal Loss for handling severe class imbalance
-- Parameters:
   - `weight`: Class weights tensor used as per-class alpha (determined from `--class_weight`)
   - `gamma`: Focusing parameter (fixed to 2.0)
   - `reduction`: 'mean', 'sum', or None

#### `compute_class_weights(labels, weight_type='uniform', num_classes=7)`
- Computes class weights based on class distribution
- Strategies:
  - **uniform**: No weighting (returns None)
  - **balanced**: Inverse of class frequency, normalized
  - **inverse**: Total samples / (num_classes × class_count)
- Returns: `torch.FloatTensor` of class weights or None

#### `get_loss_fn(loss_type='ce', class_weights=None)`
- Factory function to instantiate the correct loss function
- Returns: Loss function instance (CrossEntropyLoss or FocalLoss)

### 3. **Modified `train()` Function**

**Signature change:**
```python
# Before
def train(model, train_loader, test_loader, args, device):

# After
def train(model, train_loader, test_loader, train_df, args, device):
```

**Key changes:**
- Now accepts `train_df` parameter for computing class weights
- Automatically computes class weights based on training data
- Selects appropriate loss function using the factory pattern
- Moves FocalLoss to device if needed

### 4. **Updated Folder Structure**

**Old structure:**
```
experiments/scenario/feature_col/embedding/freeze/seed/
```

**New structure:**
```
experiments/scenario/feature_col/embedding/freeze/loss_cw_handler/seed/
```

**Example paths:**
- `experiments/droptc/sentence/bert-base-uncased/unfreeze/ce-uniform/42/`
- `experiments/droptc/sentence/bert-base-uncased/unfreeze/focal-balanced/42/`
- `experiments/droptc/sentence/all-mpnet-base-v2/freeze/ce-inverse/42/`

**Benefits:**
- Self-documenting (loss function and weighting strategy visible in path)
- Easy to organize and compare experiments
- Facilitates batch analysis and recap
- No need to inspect JSON files to understand configuration

### 5. **Updated `main()` Function**

**Changes:**
- Creates `loss_cw_handler` string: `f"{args.loss_fn}-{args.class_weight}"`
- Inserts it into the folder hierarchy
- Passes `train_df` to the `train()` function

## Usage Examples

### Default behavior (unchanged):
```bash
python train_classifier.py --scenario droptc
# Folder: experiments/droptc/sentence/bert-base-uncased/unfreeze/ce-uniform/42/
```

### Using Focal Loss with balanced class weights:
```bash
python train_classifier.py --scenario droptc --loss_fn focal --class_weight balanced
# Folder: experiments/droptc/sentence/bert-base-uncased/unfreeze/focal-balanced/42/
```

### Using CrossEntropyLoss with inverse class weights:
```bash
python train_classifier.py --scenario droptc --class_weight inverse
# Folder: experiments/droptc/sentence/bert-base-uncased/unfreeze/ce-inverse/42/
```

### Notes on Focal Loss behavior:
- Focal Loss uses class weights computed from the training labels as per-class alpha values.
- Gamma is fixed to 2.0. To change focus on hard examples, adjust `--class_weight` (balanced or inverse).

## Key Features

1. **Class Weight Computation**
   - Automatically computed from training data distribution
   - Three weighting strategies for different imbalance scenarios

2. **Loss Function Flexibility**
   - CrossEntropyLoss: Standard loss with optional class weighting
   - Focal Loss: Designed for harder examples, adjustable focus

3. **Configuration Persistence**
   - All parameters saved in `scenario_arguments.json`
   - Includes loss function and class weight strategy

4. **Reproducible Organization**
   - Folder structure encodes all imbalance handling settings
   - Easy to identify and analyze results across different configurations

## Backward Compatibility

**Fully backward compatible** - existing scripts will use default values:
- `--loss_fn ce` (CrossEntropyLoss)
- `--class_weight uniform` (no weighting)

## Testing Recommendations

1. Compare results with `ce-uniform` vs `ce-balanced` to measure impact
2. Experiment with different `--class_weight` strategies (balanced/inverse) when using focal; gamma is fixed to 2.0
3. Validate performance improvements on minority classes
4. Monitor training curves for convergence differences

## Implementation Details

### Class Weight Formulas

**Balanced:**
```
weights[c] = (1 / count[c]) / sum(1 / count[*]) * num_classes
```

**Inverse:**
```
weights[c] = total_samples / (num_classes * count[c])
```

### Focal Loss Formula

```
FL(pt) = -α * (1 - pt)^γ * log(pt)
```
where `pt` is the model's predicted probability for the target class.

Note: In this implementation α is provided per-class from the computed class weights; γ is fixed to 2.0.
