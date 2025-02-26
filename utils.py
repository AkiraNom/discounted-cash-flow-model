# solve yfinance error on streamlit cloud
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

class DiscountedCashFlow():

    @staticmethod
    def linear_interpolate(initial_value :int | float, terminal_value: int |float, years: int) -> np.ndarray:
        return np.linspace(initial_value, terminal_value, years)

    @staticmethod
    def calculate_cost_of_equity(rf: float, beta: float, ERP: float) -> float:
        """
        Cost of equity (Re) = Risk-free rate (rf) + beta * Equity risk premium (ERP)
        """
        return (rf + beta * ERP) /100

    @staticmethod
    def calculate_yield_to_maturatity(C: float, F: float, P: float, n: float) -> float:
        """
        Calculate Yield To Maturity (YTM):

            YTM = (C + {F-P} / n) / ({F + P} / 2)

                C  (float) : annual coupon payment
                F  (float) : Face value
                P  (float) : Price
                n  (float) : years to maturity

            approximation of semi-annual YTM
        """
        YTM = (C + (F - P)/ n) / ((F + P) / 2)

        return YTM

    @staticmethod
    def calculate_weighted_average_cost_of_capital(marketCap: float, cost_of_equity: float, debt: float, cost_of_debt: float, tax_rate: float) -> float:
        """
        compute Weighted Average Cost of Capital
        wacc = cost of debt * (1 - corporate tax) * Long term debt / (market cap + long term debt) \\
                + cost of equity * market cap / (market cap + long term debt)

        parameters:
            cost_of_debt (float) : Cost of debt
            cost_of_equity (float) : Cost of equity
            debt  (float) : Market value of the firm's debt (Long term debt)
            marketCap  (float) : Market value of the firm's equity
            tax_rate (float) : Corporate tax rate

        Return:
            wacc (float) : weighted average cost of capital
        """

        wacc = cost_of_debt * (1 - tax_rate) * debt/ (debt + marketCap) + cost_of_equity * marketCap/ (debt + marketCap)
        return wacc

    @staticmethod
    def calculate_present_values(cash_flows: list | pd.Series, discount_rate: float) -> list:
        """
        calculate the current value of a future cashflow
        present value = cash flow/ (1 + discount_rate)**1 + cash flow/ (1 + discount_rate)**2  .....

        Return
            list : present values

        for calculation of a single present value,
            you can use calculate_present_value(value, discount_rate: float, t: float) function,
            which returns a float value

        """
        present_values_cf = [cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, start=1)]
        return present_values_cf

    @staticmethod
    def calculate_present_value(value: float, discount_rate: float, t: int) -> float:
        """
        Calculate the current value of a future cashflow at time, t

        value (flaot): value at time t
        discount_rate (float): discount_rate to discount back to the present value
        t (int) : time at a given value

        Return:
            present value (float) : present value of future value at time, t
        """
        return value/(1 + discount_rate)**t

    @staticmethod
    def calculate_terminal_value(free_cash_flow, terminal_growth_rate, discount_rate):
        """
        Calculate terminal value which is computed by the following formula:
            TV = FCF / (Tg - d)

            TV : terminal value
            Tg : terminal growth_rate
            d : disount rate
        """

        TV = free_cash_flow / (discount_rate - terminal_growth_rate)
        return TV

    @staticmethod
    def calculate_enterprise_value(pv_FCF, pv_TV) -> float:
        """
        entterprise value (EV) is sum of present value of company's future cashflow
        plus present value of terminal value measuring a company's total value including
        marketcap and all short/long term debt.

        Alternatively,
        EV = MC + total Debt - C
        where:
            MC : market capitalization
            total debt: the sum of short and long term debt
            C : cash and cash equivalents

        """
        return np.sum(pv_FCF).item() + pv_TV

    @staticmethod
    def calculate_equity_value(cash, debt, enterprise_value) -> float:
        """
        calculate market value of equity
        Equity value = Share Price * Number of Outstanding shares
        or
        Equity value = Enterprise Value - Total debt + cash and cash equivalents
        """
        return enterprise_value - debt + cash

    @staticmethod
    def estimate_implied_share_price(equity_value, outstandingShares) ->float:
        """
        Estimate the implied price per share which is the value of each sahre of a compnany's stock

        implied share price = equity value / outshanding shares
        """
        return equity_value/outstandingShares

    @classmethod
    def perform_discount_cash_flow_analysis(
        cls,
        n : int,
        free_cash_flow : float | int,
        terminal_growth_rate : float,
        wacc : float,
        pv_FCF : list | pd.Series,
        cash : float | int,
        debt : float | int,
        outstandingShares : float | int
        ) -> float:

        """
        Performs a Discounted Cash Flow (DCF) analysis to estimate the implied share price.

        This method calculates the terminal value, present value of terminal value, enterprise value,
        equity value, and finally the implied share price based on the given financial parameters.

        Parameters:
            n (int): Number of years for the projection period.
            free_cash_flow (float | int): The free cash flow value for the final projected year.
            terminal_growth_rate (float): The expected growth rate beyond the projection period.
            wacc (float): Weighted Average Cost of Capital, used as the discount rate.
            pv_FCF (list | pd.Series): Present values of free cash flows for the projection period.
            cash (float | int): Cash and cash equivalents.
            debt (float | int): Total debt of the company.
            outstandingShares (float | int): Number of outstanding shares.

        Returns:
            float: The estimated implied share price based on the DCF analysis.
        """

        TV  = cls.calculate_terminal_value(free_cash_flow, terminal_growth_rate, wacc)
        pv_TV = cls.calculate_present_value(TV, wacc, n)


        EV = cls.calculate_enterprise_value(pv_FCF, pv_TV)
        equity_value = cls.calculate_equity_value(cash, debt, EV)

        return cls.estimate_implied_share_price(equity_value, outstandingShares)


class DataProcessing:

    @staticmethod
    def fetch_ticker_data(ticker: str) -> yf.ticker.Ticker:
        return yf.Ticker(ticker)

    # @classmethod
    # def check_ticker_availability(cls, ticker: str):
    #     """
    #     Check if a ticker is available on yahoo finance.

    #     Parameters:
    #         ticker (str): ticker symbol
    #     Returns
    #         yf.ticker.Ticker if ticker is available. Otherwise it will return a warning message
    #     """

    #     ticker_obj = cls.fetch_ticker_data(ticker)
    #     try:
    #         ticker_obj.info
    #         return ticker_obj
    #     except YFException as e:
    #         st.warning(f"Ticker '{ticker_obj}' may not be available on Yahoo Finance. Error: {str(e)}")
    #         st.stop()
    #         return None

    @staticmethod
    def key_metrics(data):

        company_info = {
            'shortName' : data.info.get('shortName'),
            'BusinessSummary' : data.info.get('longBusinessSummary'),
            'CEO' : data.info.get('companyOfficers')[0]['name'],
            'address1' :  data.info.get('address1'),
            'city' : data.info.get('city'),
            'state' : data.info.get('stete'),
            'zip' : data.info.get('zip'),
            'country' : data.info.get('country'),
            'website' : data.info.get('website'),
            'industry' : data.info.get('industry'),
            'industryKey' : data.info.get('industryKey'),
            'industryDisp' : data.info.get('industryDisp'),
            'sector' : data.info.get('sector'),
            'sectorKey' : data.info.get('sectorKey'),
            'fullTimeEmployees' : data.info.get('fullTimeEmployees'),
            'currentPrice' : data.info.get('currentPrice'),
            'previousClose' : data.info.get('previousClose'),
            'beta' : data.info.get('beta'),
            'sharesOutstanding' : data.info.get('sharesOutstanding'),
            'marketCap' : data.info.get('marketCap'),
            'enterpriseValue' : data.info.get('enterpriseValue'),
            'PER' : data.info.get('trailingPE'),
            'EPS' : data.info.get('trailingEps'),
            'P/B' : data.info.get('priceToBook'),
            'P/S' : data.info.get('priceToSalesTrailing12Months'),
            'PEG' : data.info.get('trailingPegRatio'),
            'FCF' : data.info.get('freeCashflow')

        }

        return company_info

    @staticmethod
    def extract_financial_data(data: yf.ticker.Ticker) -> pd.DataFrame:

        """
        Get balance sheet, income statment, and cash flow
        Due to restriction of yahoo finance, years to be fetched are limited
        """

        balancesheet_data = data.balance_sheet.iloc[:,:-1]
        balancesheet_data = balancesheet_data[balancesheet_data.columns[::-1]]
        incomestmt_data = data.incomestmt.iloc[:,:-1]
        incomestmt_data = incomestmt_data[incomestmt_data.columns[::-1]]
        cashflow_data = data.cash_flow.iloc[:,:-1]
        cashflow_data = cashflow_data[cashflow_data.columns[::-1]]

        return incomestmt_data, balancesheet_data, cashflow_data

    @classmethod
    def compile_financial_statement_metrics(cls, data: yf.ticker.Ticker) -> pd.DataFrame:

        """
        Extract data values necessary for DCF analysis
        All feature names in reference dataframe are based on names on yahoo finance
        """

        df_ref = pd.concat([*cls.extract_financial_data(data)])

        df = pd.DataFrame(index=df_ref.columns)
        df.loc[:, 'total_revenue'] = df_ref.loc['Total Revenue']
        df.loc[:, 'ebit'] = df_ref.loc['EBIT']
        df.loc[:, 'total_debt'] = df_ref.loc['Total Debt']
        df.loc[:, 'capital_expenditures'] = df_ref.loc['Capital Expenditure'] * -1
        df.loc[:, 'depreciation_amortization'] = df_ref.loc['Depreciation And Amortization']
        df.loc[:, 'tax_provision'] = df_ref.loc['Tax Provision']

        # compute variables

        df.loc[:, 'rev_growth'] = df_ref.loc['Total Revenue'].infer_objects(copy=False).pct_change()

        # NWC = current assets - current liabilities * most popular
        df.loc[:, 'delta_nwc'] = (df_ref.loc['Current Assets'] - df_ref.loc['Current Liabilities']).diff()
        # # NWC = Accounts Receivalbe + inventroy - accoutns payable
        # df['delta_nwc'] = (df_ref.loc['Accounts Receivable',] + df_ref.loc['Inventory',] - df_ref.loc['Accounts Payable',]).diff()

        df.loc[:, 'ebit_sales'] = df_ref.loc['EBIT'] / df_ref.loc['Total Revenue']
        df.loc[:, 'dna_sales'] = df_ref.loc['Depreciation And Amortization'] / df_ref.loc['Total Revenue']
        df.loc[:, 'capex_sales'] = df_ref.loc['Capital Expenditure']/df_ref.loc['Total Revenue']
        df.loc[:, 'nwc_sales'] = df.loc[:,'delta_nwc'] / df_ref.loc['Total Revenue']
        df.loc[:, 'tax_ebit'] = df_ref.loc['Tax Provision'] / df_ref.loc['EBIT']
        df.loc[:, 'ebiat'] = df_ref.loc['EBIT'] - df_ref.loc['Tax Provision']

        return df

    @staticmethod
    def project_finantial_data(
        df: pd.DataFrame,
        years_projection : str,
        projection_method,
        revenue_growth_rate : float,
        ebit_rate: float,
        tax_rate: float,
        deorecuation_amortization_rate : float,
        capital_expenditures_rate : float,
        net_wroking_capital_rate : float,
        ) -> pd.DataFrame:

        """
        Project financial data
        years_projection : number of years to project financial data
        projection_method : define how to project financial data
        other parameters define the rate of change in parameters
        """

        base_year = df.index[-2].year
        df_ref = df.iloc[-2,:]
        years = range( base_year+ 1, base_year + 1 + years_projection)

        df_proj = pd.DataFrame(index = years, columns =df.columns)

        # linear interpolation
        df_proj["rev_growth"] = projection_method(df_ref["rev_growth"], revenue_growth_rate, years_projection)
        df_proj["ebit_sales"] = projection_method(df_ref["ebit_sales"], ebit_rate, years_projection)
        df_proj["dna_sales"] = projection_method(df_ref["dna_sales"], deorecuation_amortization_rate, years_projection)
        df_proj["capex_sales"] = projection_method(df_ref["capex_sales"], capital_expenditures_rate, years_projection)
        df_proj["tax_ebit"] = projection_method(df_ref["tax_ebit"], tax_rate, years_projection)
        df_proj["nwc_sales"] = projection_method(df_ref["nwc_sales"], net_wroking_capital_rate, years_projection)

        # cumulative values
        df_proj["total_revenue"] = df_ref["total_revenue"] *(1+df_proj["rev_growth"]).cumprod()
        df_proj["ebit"] = df_ref["ebit"] *(1+df_proj["ebit_sales"]).cumprod()
        df_proj["capital_expenditures"] = df_ref["capital_expenditures"] *(1+df_proj["capex_sales"]).cumprod()
        df_proj["depreciation_amortization"] = df_ref["depreciation_amortization"] *(1+df_proj["dna_sales"]).cumprod()
        df_proj["delta_nwc"] = df_ref["delta_nwc"] *(1+df_proj["nwc_sales"]).cumprod()
        df_proj["tax_provision"] = df_ref["tax_provision"] *(1+df_proj["tax_ebit"]).cumprod()
        df_proj["ebiat"] = df_proj["ebit"] - df_proj["tax_provision"]

        df_proj["freeCashFlow"] = df_proj["ebiat"] + df_proj["depreciation_amortization"] - df_proj["capital_expenditures"] - df_proj["delta_nwc"]

        return df_proj

    @st.cache_data
    @staticmethod
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    @staticmethod
    def clear_session_state():
        for key in st.session_state.keys():
            del st.session_state[key]

class PageLayout:

    def check_password():
        with open('./config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )

        if not st.session_state.get("authentication_status"):
            with st.container(border=True):
                st.code("""
                        Please use the following user information to test the application.
                            sample user information:
                                user 1:
                                    username: jsmith
                                    password: abc
                                user 2:
                                    username: rbriggs
                                    password: def
                            """)
        try:
            authenticator.login()

        except Exception as e:
            st.error(e)

        if st.session_state.get("authentication_status"):

            authenticator.logout(location="sidebar", key="logout-app-home")

            if not st.session_state.get("login_success"):
                success = st.success("You are logged in!", icon="✅")
                time.sleep(3)
                success.empty()
                st.session_state["login_success"] = True

        elif st.session_state["authentication_status"] is False:
            st.error("Username/password is incorrect")
            st.stop()

        elif st.session_state["authentication_status"] is None:
            st.warning("Please enter your username and password")
            st.stop()

    def key_metric_container(data: dict):

        with st.container():
            st.markdown(f"""### <center><strong>{data['shortName']}</strong></center>""", unsafe_allow_html=True)
            on = st.toggle("Company Description")
            if on:
                cols = st.columns([2,1])
                with cols[0]:
                    st.markdown(f"{data['BusinessSummary']}", unsafe_allow_html=True)
                with cols[1]:
                    st.metric('CEO', f"{data['CEO']}")
                    st.metric('Sector', f"{data['sector']}")
                    st.metric('Industory', f"{data['industry']}")
                    st.link_button('To Website', f"{data['website']}")
                st.divider()

            row1 = st.columns(4, gap = 'small')
            row2 = st.columns(4, gap = 'small')

            with row1[0]:
                st.metric('Current Price ($)', f"{data['currentPrice']:.2f}", f"{(data['currentPrice']-data['previousClose'])/data['previousClose']*100:.2f}%")
            with row1[1]:
                st.metric('Market Cap', f"{data['marketCap']/1e9:.2f}B")
            with row1[2]:
                st.metric('Beta (5Y Monthly)', f"{data['beta']:.2f}")
            with row1[3]:
                st.metric('Free Cash Flow', f"{data['FCF']/1e6:.2f}M")
            with row2[0]:
                st.metric('PE Ratio (TTM)', f"{data['PER']:.2f}")
            with row2[1]:
                st.metric('EPS (TTM)', f"{data['EPS']:.2f}")
            with row2[2]:
                st.metric('P/B', f"{data['P/B']:.2f}")
            with row2[3]:
                st.metric('P/S', f"{data['P/S']:.2f}")

    @staticmethod
    def warning_message(warning_text):
        st.write('')
        st.warning(warning_text)

    @staticmethod
    def projection_parameters():

        with st.form('Parameters for projection'):
            st.write('')
            st.markdown('##### Select input parameters for projection')
            row1 = st.columns(4, vertical_alignment='bottom')
            row2 = st.columns(4, vertical_alignment='bottom')
            with row1[0]:
                n_years = st.number_input('Number of years to project', min_value=0, max_value=30, value=10)
            with row1[1]:
                revenue_growth_rate = st.number_input('Revenue Growth (%)', value=0.07, step=0.01)
            with row1[2]:
                ebit_rate = st.number_input('EBIT to Sales (%)', value=0.23)
            with row1[3]:
                dna_rate = st.number_input('Depreciation and Amortization to Sales (%)', value=0.03)
            with row2[0]:
                capex_rate = st.number_input('Capital Expenditures to Sales (%)', value = 0.05)
            with row2[1]:
                nwc_rate = st.number_input('Delta Net Working Capital to Sales (%)', value = 0.05)
            with row2[2]:
                tax_rate = st.number_input('Tax of EBIT (%)', value = 0.21)
            # with row2[3]:
            #     terminal_growth_rate = st.number_input('Termanal Growth Rate(%)', value = 0.025)

            st.write('')
            submitted = st.form_submit_button('Apply', use_container_width=True)

            if submitted:
                st.session_state.projection_parameters = {
                    'n_years' : n_years,
                    'revenue_growth_rate' : revenue_growth_rate,
                    'ebit_rate' : ebit_rate,
                    'dna_rate' : dna_rate,
                    'capex_rate' : capex_rate,
                    'nwc_rate' : nwc_rate,
                    'tax_rate' : tax_rate,
                    # 'terminal_growth_rate'  : terminal_growth_rate
                    }

    @staticmethod
    def projection_plot(df, wacc_parameters):
        cols = st.columns([1,2,2])
        with cols[0]:
            st.metric('WACC (%)', value=(np.round((wacc_parameters['wacc']*100), 3)))
        with cols[1]:
            st.plotly_chart(px.bar(df, x=df.index, y='freeCashFlow').update_layout(title='Free Cash Flow',xaxis_title='Year', yaxis_title='Free Cash Flow ($)'))
        with cols[2]:
            st.plotly_chart(px.bar(df, x=df.index, y='pv_FCF').update_layout(title='Present Value of Free Cash Flow',xaxis_title='Year', yaxis_title='Present Value of Free Cash Flow ($)'))

    @staticmethod
    def wacc_parameters(data: dict):

        with st.form('Parameters for wacc'):
            st.write('')
            st.markdown('##### Select input parameters for WACC')
            row1 = st.columns(3, vertical_alignment='bottom')
            row2 = st.columns(3, vertical_alignment='bottom')
            with row1[0]:
                marketCap = st.number_input('Market Capitalization ($ in M)', min_value=0.0, value=(data['marketCap']/1e6))
            with row1[1]:
                beta = st.number_input('Beta', value=data['beta'], step=0.01)
            with row1[2]:
                sharesOutstanding = st.number_input('Shares of Outstanding (in M)', value=(data['sharesOutstanding']/1e6))
            with row2[0]:
                ticker_us_treasure_rate_10yr = '^TNX'
                us10Y_data = DataProcessing.fetch_ticker_data(ticker_us_treasure_rate_10yr)
                rf_rate = st.number_input("Risk Free Rate ($R_f$) (%)*", value=us10Y_data.history(period='1d')['Close'].values.item())
            with row2[1]:
                erp = st.number_input('Equity Risk Premium (ERP)', value = 4.0)
            with row2[2]:
                cost_of_debt = st.number_input('Cost of Debt (%)', value = 3.88)

            st.write('')
            st.markdown(f"*Risk Free Rate: <mark>:blue[{us10Y_data.info['shortName']}]</mark>", unsafe_allow_html=True)
            st.markdown(f"Cost of Equity: <mark>:blue[$$ beta*ERP + R_f$$]</mark>", unsafe_allow_html=True)
            submitted = st.form_submit_button('Apply', use_container_width=True)

            if submitted:
                st.session_state.wacc_parameters = {
                    'marketCap' : marketCap*1e6,
                    'beta' : beta,
                    'sharesOutstanding' : sharesOutstanding*1e6,
                    'rf_rate' : rf_rate,
                    'erp' : erp,
                    'cost_of_equity' : DiscountedCashFlow.calculate_cost_of_equity(rf_rate, beta, erp),
                    'cost_of_debt' : cost_of_debt/100,
                    }


    @staticmethod
    def dcf_analysis_section(df, wacc_parameters: dict, n_years: int):

        container = st.container(border=True)
        container.cols = st.columns([2,1,2])
        with container.cols[0]:
            terminal_growth_rate = st.number_input('Terminal Growth Rate (%)', value = 2.5)
        with container.cols[2]:
            val = DiscountedCashFlow.perform_discount_cash_flow_analysis(n_years,
                                                                        df['freeCashFlow'].values[-1].item(),
                                                                        terminal_growth_rate,
                                                                        wacc_parameters['wacc'],
                                                                        df['pv_FCF'],
                                                                        wacc_parameters['cash'],
                                                                        wacc_parameters['debt'],
                                                                        wacc_parameters['sharesOutstanding'])
            st.metric('Implied Share Price ($)', value=f"${val:.3f}")

        st.write('')

    @staticmethod
    def dcf_sensitivity_plot(df, wacc_parameters: dict, n_years: int):

        cols = st.columns([1,2,1,2,1,2])
        with cols[1]:
            wacc_min, wacc_max = st.slider("Select a range of WACC", 0.01, 0.2, (0.06, 0.18), help='Default range : 0.06 - 0.18')

        with cols[3]:
            tgr_min, tgrc_max = st.slider("Select a range of TGR", 0.01, 0.1, (0.01, 0.04), help='Default range : 0.01 - 0.04')
        with cols[5]:
            if st.button("Set Default", type="primary"):
                st.write('Default ranges are applied')
                wacc_max = 0.06
                wacc_min =0.18
                tgr_min = 0.01
                tgrc_max = 0.04


        waccs = np.linspace(wacc_min, wacc_max, 25)
        tgrs = np.linspace(tgr_min, tgrc_max, 25)

        # Create a meshgrid for cost of debt and growth rate
        waccs_mesh, tgrs_mesh = np.meshgrid(waccs, tgrs)

        # Calculate NPV for each combination of cost of debt and growth rate
        dcf_results = np.zeros_like(waccs_mesh)
        for i in range(len(tgrs)):
            for j in range(len(waccs)):
                dcf_results[i, j] = DiscountedCashFlow.perform_discount_cash_flow_analysis(n_years,
                                                                        df['freeCashFlow'].values[-1].item(),
                                                                        tgrs[i],
                                                                        waccs[j],
                                                                        df['pv_FCF'],
                                                                        wacc_parameters['cash'],
                                                                        wacc_parameters['debt'],
                                                                        wacc_parameters['sharesOutstanding'])

        fig = go.Figure(data=[go.Surface(z=dcf_results,
                                         x=waccs_mesh,
                                         y=tgrs_mesh,
                                         hovertemplate = 'WACC: %{x:.3f}'+\
                                                        '<br>TGR: %{y:.3f}'+\
                                                        '<br>Price: %{z:.3f}<extra></extra>',
                                         colorscale ='Viridis')])
        fig.update_layout(scene = dict(
                            xaxis=dict(title=dict(text='<b>Weighted Average Cost of Capital</b>')),
                            yaxis=dict(title=dict(text='<b>Terminal Growth Rate (%)</b>')),
                            zaxis=dict(title=dict(text='<b>Implied Share Price ($)</b>')),
                          ),
                         autosize=False,
                         height=800,
                         margin=dict(l=65, r=50, b=65, t=90))
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

        if 'dcf_results' not in st.session_state:
            st.session_state.dcf_results = {}

        if isinstance(dcf_results, np.ndarray) and dcf_results.size > 0:
            st.session_state.dcf_results['data'] = dcf_results
            st.session_state.dcf_results['waccs'] = waccs
            st.session_state.dcf_results['tgrs'] = tgrs

        st.plotly_chart(fig)

    @staticmethod
    def data_downloader(df, button_label, file_name, time):

        st.write('')
        csv = DataProcessing.convert_df(df)
        st.download_button(
            label=f"Download {button_label} as CSV",
            data=csv,
            file_name=f"{file_name}{time}.csv",
            mime="text/csv",
        )
        st.write('')
