import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import time
import random

year_list = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]

def fetch_dates(year):
    # Retrieve all dates for each year using the URL and append them to a list
    url = 'https://www.lottonumbers.com/eurojackpot-results-' + str(year)
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    date_box = soup.find('table', attrs= {'class':'lotteryTable'}).find('tbody').find_all('a')
    date_list = []
    for date in date_box:
        date_list.append(date.get('href').split('/')[3])

    return date_list

# Data Function Definitions
def fetch_last_draw():
    url = 'https://www.lottonumbers.com/eurojackpot-results-' + str(2024)
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    date_box = soup.find('table', attrs= {'class':'lotteryTable'}).find('tbody').find('a')
    date = date_box.get('href').split('/')[3]
    return fetch_draw(date)

def fetch_draw(date):     
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
    draw_df['Date'] = pd.to_datetime(draw_df['Date'])
    
    jackpot_df = draw_df.loc[(draw_df['Total Winners'] == 0) & (draw_df['Prize Tier'] == '5+2'), 'Per Winner Prize']
    if not jackpot_df.empty:
        latest_remainder = jackpot_df.iloc[-1]
    else:
        latest_remainder = 10000000.0
        
    return draw_df, latest_remainder

def calculate_last_draw_summary(draw_df):
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

    return last_draw_df

@st.cache_data
def nextDrawDate():
    def get_next_weekday(date, target_weekday):
        return date + timedelta(days=target_weekday)

    selected_date = pd.to_datetime(st.session_state.summary_df['Date'][0])

    # Check the day of the week of the selected date
    if selected_date.weekday() == 1:  # If it's a Tuesday
        return get_next_weekday(selected_date, 3)  # Next Friday
    else:
        return get_next_weekday(selected_date, 4)  # Next Tuesday
    
@st.cache_data
def predict_next_draw(prediction_df, latest_remainder):
    # Selecting relevant columns
    data = prediction_df[['Remainder Jackpot', 'Total Deposit', 'Jackpot Prize', 'Total Potential Profit']].copy()

    # Train a model for Total Deposit using only 'Remainder Jackpot'
    model_deposit = LinearRegression()
    model_deposit.fit(data[['Remainder Jackpot']], data['Total Deposit'])

    # Train a model for Jackpot Prize using only 'Remainder Jackpot'
    model_jackpot = LinearRegression()
    model_jackpot.fit(data[['Remainder Jackpot']], data['Jackpot Prize'])

    # For the latest day's prediction, use the provided 'latest_remainder_jackpot' value
    latest_remainder_jackpot = latest_remainder

    # Predict the 'Total Deposit' using only 'Remainder Jackpot'
    estimated_total_deposit = model_deposit.predict([[latest_remainder_jackpot]])[0]

    # Predict the 'Jackpot Prize' using only 'Remainder Jackpot'
    estimated_jackpot_prize = model_jackpot.predict([[latest_remainder_jackpot]])[0]

    # Now, for the 'Total Potential Profit', we can use a similar approach
    # Assuming 'Total Potential Profit' is also predicted using 'Remainder Jackpot', 'Total Deposit', and 'Jackpot Prize'
    model_profit = LinearRegression()
    model_profit.fit(data[['Remainder Jackpot', 'Total Deposit', 'Jackpot Prize']], data['Total Potential Profit'])

    # Predict the 'Total Potential Profit'
    predicted_total_profit_latest = model_profit.predict([[latest_remainder_jackpot, estimated_total_deposit, estimated_jackpot_prize]])[0]

    return estimated_total_deposit, estimated_jackpot_prize, predicted_total_profit_latest


def estimate_draws_need_for_target(target_total_profit, latest_remainder, always_rollover):
    # Selecting relevant columns
    data = prediction_df[['Remainder Jackpot', 'Total Deposit', 'Jackpot Prize', 'Total Potential Profit']].copy()

    # Train a model for Total Deposit using only 'Remainder Jackpot'
    model_deposit = LinearRegression()
    model_deposit.fit(data[['Remainder Jackpot']], data['Total Deposit'])

    # Train a model for Jackpot Prize using only 'Remainder Jackpot'
    model_jackpot = LinearRegression()
    model_jackpot.fit(data[['Remainder Jackpot']], data['Jackpot Prize'])

    # For the latest day's prediction, use the provided 'latest_remainder_jackpot' value
    latest_remainder_jackpot = latest_remainder

    # Predict the 'Total Deposit' using only 'Remainder Jackpot'
    estimated_total_deposit = model_deposit.predict([[latest_remainder_jackpot]])[0]

    # Predict the 'Jackpot Prize' using only 'Remainder Jackpot'
    estimated_jackpot_prize = model_jackpot.predict([[latest_remainder_jackpot]])[0]

    # Now, for the 'Total Potential Profit', we can use a similar approach
    # Assuming 'Total Potential Profit' is also predicted using 'Remainder Jackpot', 'Total Deposit', and 'Jackpot Prize'
    model_profit = LinearRegression()
    model_profit.fit(data[['Remainder Jackpot', 'Total Deposit', 'Jackpot Prize']], data['Total Potential Profit'])

    # Predict the 'Total Potential Profit'
    predicted_total_profit_latest = model_profit.predict([[latest_remainder_jackpot, estimated_total_deposit, estimated_jackpot_prize]])[0]

    # Initialize variables for the iterative process
    draw_limit = 10
    predicted_profit = 0.0
    number_of_draws = 0

    # Function to estimate the next 'Remainder Jackpot'
    def next_remainder_jackpot(current_jackpot, deposit):
        if always_rollover:
            return current_jackpot
        
        ticket_count = (deposit / 2) # ticket prize 2 euro
        jackpot_prob = 1/139838160 # constant probability already calculated

        if random.random() < jackpot_prob * ticket_count:
            # Jackpot Fell, next week's min prize
            return 10000000
        else:
            # Jackpot roll over to next week
            return current_jackpot

    # Iterate until the target total profit is reached or exceeded

    # Loop until the target profit is reached
    while predicted_profit < target_total_profit and number_of_draws < draw_limit :
        # Predict 'Total Deposit' and 'Jackpot Prize' for the current 'Remainder Jackpot'
        estimated_total_deposit = model_deposit.predict([[latest_remainder_jackpot]])[0]
        estimated_jackpot_prize = model_jackpot.predict([[latest_remainder_jackpot]])[0]

        # Predict 'Total Potential Profit' for the current draw
        predicted_profit = model_profit.predict([[latest_remainder_jackpot, estimated_total_deposit, estimated_jackpot_prize]])[0]

        print(f"----------")
        print(f" Number of draw {number_of_draws + 1}")
        print(f" estimated_total_deposit {estimated_total_deposit}")
        print(f" estimated_jackpot_prize {estimated_jackpot_prize}")
        print(f" Predicted Profit {predicted_profit}")

        # Update draw count
        number_of_draws += 1

        # Update 'latest_remainder_jackpot' for the next draw
        latest_remainder_jackpot = next_remainder_jackpot(estimated_jackpot_prize, estimated_total_deposit)

    # Output the number of draws needed to reach the target total profit
    print(f"Number of draws needed to reach a total profit of {target_total_profit}: {number_of_draws}")
    success = target_total_profit <= predicted_profit
    return number_of_draws, success

@st.cache_data
def read_previous_data():
    return pd.read_excel('Data/Prediction.xlsx')

# Streamlit Codes

def show_last_draw_profit():
    if st.session_state.last_draw_fetched == False:
        with st.spinner("Fetching last draw..."):
            time.sleep(2.0)
            draw_df, st.session_state.latest_remainder = fetch_last_draw()
            st.session_state.summary_df = calculate_last_draw_summary(draw_df)
            st.session_state.last_draw_fetched = True
    
    summary_df = st.session_state.summary_df
    
    if summary_df.empty:
        st.write("No data available for the last draw.")
        return

    # Ensure Date is in the correct format
    summary_df['Date'] = pd.to_datetime(summary_df['Date']).dt.date

    # Displaying the summary in a more appealing format
    st.write("## Last Draw Summary")

    # Assuming there's only one row in the summary_df
    if len(summary_df) == 1:
        date = summary_df.iloc[0]['Date']
        total_potential_profit = f"€{summary_df.iloc[0]['Total Potential Profit']:,.2f}"
        total_deposit = f"€{summary_df.iloc[0]['Total Deposit']:,.0f}"
        jackpot_prize = f"€{summary_df.iloc[0]['Jackpot Prize']:,.0f}"

        st.write(f"### Date: {date}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Jackpot Prize", value=jackpot_prize)
        with col2:
            st.metric(label="Total Deposit", value=total_deposit)
        with col3:
            st.metric(label="Total Potential Profit", value=total_potential_profit)

def show_next_draw_section():
    st.write("## Next Draw")
    
    date_text = nextDrawDate().strftime('%Y-%m-%d')
    st.write(f"### Date: {date_text}")

    button_container = st.empty()

    if st.session_state.show_calculate_button:
        if button_container.button("Predict Potential Profit"):
            st.session_state.show_calculate_button = False
            st.session_state.show_next_draw = True
            button_container.empty()
            
    if st.session_state.show_next_draw:
        show_next_draw_predictions() 
        

def show_next_draw_predictions():
    if not st.session_state.next_draw_calculated:
        with st.spinner('Predicting...'):
            # Sleep for loading 
            time.sleep(3)
            st.session_state.total_deposit, st.session_state.jackpot, st.session_state.potential_profit = predict_next_draw(prediction_df, st.session_state.latest_remainder)
            st.session_state.next_draw_calculated = True
    
    total_deposit = f"€{st.session_state.total_deposit:,.0f}"
    jackpot_prize = f"€{st.session_state.jackpot:,.0f}"
    total_potential_profit = f"€{st.session_state.potential_profit:,.2f}"
    
                
    if st.session_state.potential_profit > 1.9: 
        st.success('You are on lucky day!', icon='🍀')
        st.balloons()
    else:
        st.warning("Maybe it's better to wait the next time", icon='🤔')
        st.snow()
        
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.metric(label="Jackpot Prize", value=jackpot_prize)
    with col2:
        st.metric(label="Total Deposit", value=total_deposit)
    with col3:
        st.metric(label="Total Potential Profit", value=total_potential_profit)

def show_predict_draw_count_section():
    st.write("## Target Profit Prediction")
    st.write("Estimates how many draws later your target value will possibly be reached.")
    
    form = st.form('target_profit_form')
    always_rollover = form.toggle('Jackpot Always Rollovers')
    target_profit = form.number_input('Target Potential Profit:')    
    submitted = form.form_submit_button("Submit")
        
    if submitted:
        with st.spinner("Estimation running..."):
            time.sleep(2.0)
            number_of_draws, success = estimate_draws_need_for_target(target_profit, st.session_state.latest_remainder, always_rollover)
        if success:
            st.success(f"### 😎  {number_of_draws} draws needed to reach the target profit!")
        else:
            st.error(f"### 😔  It's not likely to reach the target in limit of {number_of_draws} draws")            
    

def show_previous_draws():
    year = int(st.selectbox('Select Year', [str(year) for year in sorted(year_list, reverse=True)]))
    with st.spinner('Loading draw dates...'):
        time.sleep(1.0)
        date_list = fetch_dates(year)
    form = st.form('date_form')
    date = form.selectbox('Select Date', date_list)
    submit = form.form_submit_button('Fetch')
    if submit:
        show_draw_stats(date)

def show_draw_stats(date):
    with st.spinner("Fetching the draw..."):
        time.sleep(1.0)
        draw_df, _ = fetch_draw(date)
        summary_df = calculate_last_draw_summary(draw_df)
    
    if summary_df.empty:
        st.write("No data available for this draw.")
        return

    # Ensure Date is in the correct format
    summary_df['Date'] = pd.to_datetime(summary_df['Date']).dt.date

    # Displaying the summary in a more appealing format
    st.write("## Draw Summary")

    # Assuming there's only one row in the summary_df
    if len(summary_df) == 1:
        date = summary_df.iloc[0]['Date']
        total_potential_profit = f"€{summary_df.iloc[0]['Total Potential Profit']:,.2f}"
        total_deposit = f"€{summary_df.iloc[0]['Total Deposit']:,.0f}"
        jackpot_prize = f"€{summary_df.iloc[0]['Jackpot Prize']:,.0f}"

        st.write(f"### Date: {date}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Jackpot Prize", value=jackpot_prize)
        with col2:
            st.metric(label="Total Deposit", value=total_deposit)
        with col3:
            st.metric(label="Total Potential Profit", value=total_potential_profit)


# Data Codes Runs 

prediction_df = read_previous_data()

# Assing initial session state variables
if 'show_calculate_button' not in st.session_state:
    st.session_state.show_calculate_button = True
    
if 'show_next_draw' not in st.session_state:
    st.session_state.show_next_draw = False

if 'last_draw_fetched' not in st.session_state:
    st.session_state.last_draw_fetched = False

if 'next_draw_calculated' not in st.session_state:
    st.session_state.next_draw_calculated = False

if 'celebrated' not in st.session_state:
    st.session_state.celebrated = False
    
# Streamlit App UI 
st.title('EuroJackpot Lottery Analyzer')
st.image('Assets/streamlit-header.png', caption='', use_column_width=True)


tab1, tab2, tab3, tab4 = st.tabs(['Last Draw', 'Next Draw', 'Previous Draws', 'Target Profit'])

with tab1:
    with st.container(border=True):
        show_last_draw_profit()
        
with tab2:
    with st.container(border=True):
        show_next_draw_section()

with tab3:
    with st.container(border=True):
        show_previous_draws()

with tab4:
    with st.container(border=True):
        show_predict_draw_count_section() 