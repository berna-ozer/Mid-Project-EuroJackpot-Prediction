<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Eurojackpot Prediction

<div style="text-align:center;">
    <img src="Assets/streamlit-header.png" alt="Eurojackpot" width="500"/>
</div>

## Overview

The Eurojackpot Prediction is a tool designed to help users evaluate the profitability of participating in the Eurojackpot lottery. By analyzing historical data and calculating potential profits, this tool assists in making informed decisions on whether to participate in the upcoming draw.

## Features

- **Last Draw Information**: Get details about the last Eurojackpot draw, including the jackpot prize, total deposit, and total potential profit.
- **Next Draw Prediction**: View the predicted potential profit for the next draw based on historical data and trends.
- **Previous Draw Insights**: Access information about previous draws to analyze trends and patterns in lottery participation.
- **Profit Estimation**: Estimate how many draws are needed to reach a target potential profit based on the current jackpot rollover situation.

## How It Works

1. **Data Analysis**: The tool uses historical Eurojackpot results and applies statistical models, including multilinear regression, to identify trends and predict future outcomes by leveraging with Machine Learning.
   
2. **Profit Calculation**: The potential profit is calculated by multiplying the winning probability by the total prize won in the last draw for each tier and summing the results. If this value exceeds the ticket cost (2 euros), it indicates a potentially profitable draw.

3. **Predictive Modeling**: The tool predicts the potential profit for future draws, particularly focusing on scenarios where the jackpot has rolled over multiple times, thereby increasing the chances of higher returns on lower-tier prizes.

## Usage

- **Eurojackpot Analyzer Interface**: The tool provides a user-friendly interface where you can check recent draw results, predict potential profits for the next draw, and explore historical data trends.

- **Decision-Making**: Use the predictions and analysis to decide whether to participate in the lottery based on the expected profitability of upcoming draws.

## Conclusion

The Eurojackpot Prediction is a powerful tool for those interested in maximizing their chances of making a profitable lottery play. By leveraging historical data and statistical models, it provides valuable insights and predictions that can guide lottery participation decisions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/berna-ozer/eurojackpot-analyzer.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run eurojackpot-visual.py
   ```



