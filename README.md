## Discounted Cash Flow Analysis

This is an interactive web appliation develped with streaml that calculates the implied shares price suing the discounted cash flow (DCF) model. The script fetches the stock price and financital data from the yahoo finance. Based on these data, future free cash flow and present value of future cash flow are computed. For simplicity, future free cash flows are estimated using the linear interpolation of the last reported data. Lastly, the implied shares price is calculated.

The DCF method relies on projecting future cash flows and discounting them back to their present value. Since assumptions about growth rates, discount rates, terminal values, and other factors significantly influence estimation, sensitivity plots enable you to simulated how there parameters influence estimation.

check the application -> [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://discounted-cash-flow-model-e7vmmuqpvmh8rbzerp9hbe.streamlit.app//)


### Background

Discounted cash flow (DCF) is a valuation method that estimates the value of an investment using its expected future cash flows. Analysts can use the present price of the future cash flows to determine whether the future cash flows of an investment or project are greater than the value of the initial investment.

### Usage

This is an interactive web application developed with streamlit. -> [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://discounted-cash-flow-model-e7vmmuqpvmh8rbzerp9hbe.streamlit.app/.streamlit.app/)

1. Type a US security symbol

2. Tune your projection parameters to estimated future free cash flow.

3. Tune your parameters for Weighted Average Cost of Capitals

4. Set a terminal growth rate to calculate implied share price

5. Lastly, similate how each parameter influences the implied share price

### Required Libraries

* numpy
* pandas
* plotly
* streamlit
* yfinance
