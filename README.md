# SETalyze

## Overview
SETalyze is a project aimed at predicting daily stock prices for Thailand's SET50 index using a combination of news, fundamental, and technical data. By employing advanced machine learning, SETalyze can make precise time series predictions, benefiting both traders and investors.

## Objective
- Accuracy: Ensure high accuracy in predicting daily stock prices for trading and investment decisions.
- Directional Insight: Ascertain the probable directional movement (rise or fall) of a stock price.

## Scope
- Focuses on predicting the stock's closing prices.
- Restricted to the SET50 index of the Thai stock market.

## Getting Started

### Prerequisites
- Python 3.x
- Pandas
- Darts
- Scikit-learn
- Os
- Numpy

You can install the required packages using pip:

```
pip install pandas darts scikit-learn os numpy
```

### Execution
To run SETalyze:

```
python prediction.py
```

Upon successful execution, predictions will be organized and stored in the result/ directory as Prediction.csv.

## Methodology
1. **Data Preprocessing**:
   The system meticulously cleans the data, ensuring all irrelevant or incomplete data points are excluded. Subsequently, it scales the dataset and identifies key components to be retained for analysis.

2. **Machine Learning**:
   SETalyze harnesses the power of the BlockRNNModel from the Darts library to deliver accurate time series predictions.

3. **Post-Processing & Output**:
   After the prediction phase, SETalyze structures the outputs, ensuring they're intuitive and ready for any subsequent financial analyses. The results are then saved in a well-formatted CSV file.

