# 💳 Credit Risk Assessment using Cost-Sensitive Composite Metric Optimization

This project presents a **cost-sensitive, multi-metric evaluation framework** for credit risk prediction using traditional machine learning classifiers. The goal is to assess model performance not just with standard metrics like AUC or Precision, but via a **dynamically optimized composite score** that reflects domain-specific cost considerations (i.e., False Negatives vs False Positives).

## Datasets Used

- **German Credit Dataset**  
  Classifies individuals as *Good* or *Bad* credit risks based on 20 features (credit amount, duration, purpose, personal attributes, etc.).

- **Taiwan Default Credit Dataset**  
  Predicts credit card default based on 6 months of billing and payment history for 30,000 clients.

## Key Features

- **Models**: Gradient Boosting, Random Forest, Logistic Regression, SVC  
- **Cost-sensitive setup**: False Negatives (FN) penalized **5x** more than False Positives (FP)  
- **Metrics Evaluated**: AUC-ROC, MCC, Precision, Brier Score  
- **Composite Score Optimization**: Z-score + constrained correlation-based weights  
- **Train-Test Splits**: 80-20, 70-30, 60-40  
- **Visualizations**: metric comparisons, stability across splits

## Methodology

### Data Preprocessing
- One-hot encoding of categorical features  
- Feature scaling with `StandardScaler`

### Model Training & Caching
- `joblib` used to store and reload models

### Cost-Aware Threshold Tuning
- Best threshold selected from range [0.1 to 0.89] to minimize cost

### Composite Score Optimization
- Uses `scipy.optimize.minimize` with constraints to assign metric weights

## Evaluation Outputs

- Cost vs Threshold plots  
- Composite vs Traditional Metrics  
- Heatmaps and correlation analysis  
- Performance stability across splits

## File Structure
├── processed_german_credit_data.csv  
├── default of credit card clients.xls  
├── CRA_GD.ipynb  
├── CRA_TWD.ipynb  
├── README.md  

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib

