import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import ttest_1samp

# Data Function Definitions

def fetchLastDraw():
    url = 'https://www.lottonumbers.com/eurojackpot-results-' + str(2024)
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    date_box = soup.find('table', attrs= {'class':'lotteryTable'}).find('tbody').find('a')
    date = date_box.get('href').split('/')[3]
    # print(date_list)
            
    # Retrieve 'Draw' information using URL
    url = 'https://www.lottonumbers.com/eurojackpot/results/' + date
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    balls = [li.text.strip().split()[0] for li in soup.find('ul', attrs={'class':"balls -lg -cn"}).find_all('li')]

    #Retrieve 'Prize Tier', 'Per Winner Prize', 'Total Winners' information using table in soup, and append into df
    prize_breakdown = soup.find('table', attrs= {'class':"table-breakdown"})
    draw_df =  pd.read_html(str(prize_breakdown))[0]
    #Append 'Balls' and 'Date' information into df
    draw_df['Balls'] = str(','.join(balls))
    draw_df['Date'] = date
    # print(draw_df.head())

    ###DATA CLEANING
    #remove rows where ['Prize Tier'] = 'Totals'
    draw_df = draw_df[draw_df['Prize Tier'] != 'Totals']

    #clean Prize Tier column
    draw_df['Prize Tier'] = draw_df['Prize Tier'].apply(lambda x: '+'.join([n if n.isdigit() else '0' for n in x.split()[1:] if n.isdigit()]) if 'and' in x else f'{x.split()[1]}+0')

    draw_df.loc[draw_df['Total Winners'] == 'Rollover  0', 'Total Winners'] = 0
    draw_df['Total Winners'] = pd.to_numeric(draw_df['Total Winners'], errors='coerce').astype('Int64')
    draw_df['Per Winner Prize'] = draw_df['Per Winner Prize'].str.replace('[^0-9.]', '', regex=True).astype(float)
    draw_df['Date'] = draw_df['Date'].astype('datetime64')
    return draw_df

def calculateLastDrawSummary(draw_df):
    winning_probabilities = {
        '5+2': 1/139838160,
        '5+1': 1/6991908,
        '5+0': 1/3107515,
        '4+2': 1/621503,
        '4+1': 1/31075,
        '4+0': 1/13811,
        '3+2': 1/14125,
        '3+1': 1/706,
        '3+0': 1/314,
        '2+2': 1/985,
        '2+1': 1/49,
        '1+2': 1/188
    }

    draw_df['Total Potential Profit'] = draw_df.apply(lambda row: row['Per Winner Prize'] * winning_probabilities[row['Prize Tier']], axis=1)

    total_potential_profit_draw = draw_df.groupby('Date')['Total Potential Profit'].sum().reset_index()
    total_potential_profit_draw = total_potential_profit_draw[['Date', 'Total Potential Profit']]

    # calculate total_reward and total_deposit and append into a df
    total_deposit = draw_df[draw_df['Prize Tier'] == '2+1']
    total_deposit['Total Deposit'] = total_deposit.apply(lambda row: (row['Per Winner Prize']) * (row['Total Winners']) * (100/20.3) * 2, axis=1)
    total_deposit = total_deposit[['Date', 'Total Deposit']]

    big_df = draw_df[draw_df['Prize Tier'] == '5+2'][['Date', 'Per Winner Prize']]
    big_df = big_df.rename(columns={'Per Winner Prize': 'Jackpot Prize'})
    big_df = big_df[['Date', 'Jackpot Prize']]


    last_draw_df = total_potential_profit_draw.merge(total_deposit, on='Date')
    last_draw_df = last_draw_df.merge(big_df, on='Date')

    return last_draw_df, draw_df

# Data Codes Runs 

draw_df = fetchLastDraw()
latest_remainder = draw_df.loc[(draw_df['Total Winners'] == 0) & (draw_df['Prize Tier'] == '5+2'), 'Per Winner Prize'].iloc[-1] if len(draw_df) > 0 else 10000000.0
summary_df, draw_df = calculateLastDrawSummary(draw_df)

# Streamlit Codes

def showLastDrawProfits(summary_df, draw_df):
     

def main():
    # Streamlit App
    st.title('EuroJackpot Lottery Analyzer')

# Run Block

main()    




