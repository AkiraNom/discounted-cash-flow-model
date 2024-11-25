import streamlit as st
import pandas as pd
import datetime

from utils import DataProcessing, DiscountedCashFlow, PageLayout

st.set_page_config(page_title="Discounted Cash Flow Analysis & Visualization", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

col1,col2,col3=st.columns([1,20,1])
with col2:
    st.markdown("""## <center><strong>:bank: :grey[Discounted Cash Flow Analysis] :bank:</strong></center>""", unsafe_allow_html=True)

intro3,intro1,intro2=st.columns([1,5,1])
intro="""📢 The :blue[**Discounted cash flow (DCF)**] is a valuation method that estimates the value of an investment using its expected future cash flows.
            Analysts use DCF to determine the value of an investment today, based on projections of how much money that investment will generate in the future. \\
            \\
            Reference: [Investopedia-DCF](https://www.investopedia.com/terms/d/dcf.asp)
            """
with intro1:
    st.write("")
    st.info(intro)
    st.write("")

###############################
st.subheader('📁 Overview of Company')

tabs = st.tabs([
    '📑 Overview',
    '📑 Financial data'
])

with tabs[0]:
    cols = st.columns([1,3])

    ticker = ''
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ticker
    with cols[0]:

        with st.form('Search a company'):
            selected_ticker = st.text_input('Type a Ticker symbol', value='MSFT')
            submitted = st.form_submit_button('🔎 Search')

        if submitted:
            DataProcessing.clear_session_state()
            if 'ticker' not in st.session_state:
                st.session_state.ticker = selected_ticker

    ticker = st.session_state.ticker

    if ticker != '':
        data = DataProcessing.fetch_ticker_data(ticker)
        company_info = DataProcessing.key_metrics(data)
        incomestmt_data, balancesheet_data, cashflow_data = DataProcessing.extract_financial_data(data)

    with cols[1]:
        with st.expander('**Key Metrics**'):

            if ticker == '':
                PageLayout.warning_message('**Please search a company first**')

            else:
                PageLayout.key_metric_container(company_info)

if ticker != '':
    with tabs[1]:
        on = st.toggle("View Balance Sheet")
        if on:
            st.data_editor(balancesheet_data, use_container_width=True)

        on = st.toggle("View Income Statement")
        if on:
            st.data_editor(incomestmt_data, use_container_width=True)

        on = st.toggle("View Cash Flow")
        if on:
            st.data_editor(cashflow_data, use_container_width=True)
else:
    with tabs[1]:
        PageLayout.warning_message('**Please search a company first**')

    st.stop()

st.write('')
st.divider()

###########################################
st.markdown("""### <center><strong>:blue[Projection of Financial Data]</strong></center>""", unsafe_allow_html=True)

if 'projection_parameters' not in st.session_state:
    st.session_state.projection_parameters = {}

PageLayout.projection_parameters()
projection_parameters = st.session_state.projection_parameters

if projection_parameters == {}:
    PageLayout.warning_message('Please Select Input parameters and Apply')
    st.stop()

# data projection
df = DataProcessing.compile_financial_statement_metrics(data)
df_proj = DataProcessing.project_finantial_data(df,
                                                projection_parameters['n_years'],
                                                DiscountedCashFlow.linear_interpolate,
                                                projection_parameters['revenue_growth_rate'],
                                                projection_parameters['ebit_rate'],
                                                projection_parameters['tax_rate'],
                                                projection_parameters['dna_rate'],
                                                projection_parameters['capex_rate'],
                                                projection_parameters['nwc_rate'])

st.write('')
st.divider()

######################################
st.markdown("""### <center><strong>:blue[Weighted Average Cost of Capital (WACC)]</strong></center>""", unsafe_allow_html=True)

if 'wacc_parameters' not in st.session_state:
    st.session_state.wacc_parameters = None

PageLayout.wacc_parameters(company_info)

if st.session_state.wacc_parameters != None:

    st.session_state.wacc_parameters['cash'] = balancesheet_data.iloc[balancesheet_data.index.get_loc('Cash And Cash Equivalents'), 0]
    st.session_state.wacc_parameters['debt'] = balancesheet_data.iloc[balancesheet_data.index.get_loc('Total Liabilities Net Minority Interest'), 0]
    st.session_state.wacc_parameters['tax_rate'] = projection_parameters['tax_rate']

else:
    PageLayout.warning_message("Please click the apply button to set input parameters for WACC")
    st.stop()

wacc_parameters = st.session_state.wacc_parameters

st.session_state.wacc_parameters['wacc'] = DiscountedCashFlow.calculate_weighted_average_cost_of_capital(wacc_parameters['marketCap'],
                                                                                                         wacc_parameters['cost_of_equity'],
                                                                                                         wacc_parameters['debt'],
                                                                                                         wacc_parameters['cost_of_debt'],
                                                                                                         wacc_parameters['tax_rate']
                                                                                                         )


df_proj.loc[:, 'pv_FCF'] = DiscountedCashFlow.calculate_present_values(df_proj['freeCashFlow'], wacc_parameters['wacc'])

st.write('')
st.divider()

############################
st.markdown("""### <center><strong>:blue[Projection of Free Cash Flow & Present Value of Free Cash Flow]</strong></center>""", unsafe_allow_html=True)
PageLayout.projection_plot(df_proj, wacc_parameters)

st.markdown("""### <center><strong>:blue[DCF Analysis]</strong></center>""", unsafe_allow_html=True)

PageLayout.dcf_analysis_section(df_proj, wacc_parameters, projection_parameters['n_years'])

st.write('')
st.markdown(f"""### <center><strong>{company_info['shortName']} Sensitivity Analysis</strong></center>""", unsafe_allow_html=True)

PageLayout.dcf_sensitivity_plot(df_proj, wacc_parameters, projection_parameters['n_years'])

st.write('')
st.divider()

############################
st.markdown(f"""### <center><strong>Data Download</strong></center>""", unsafe_allow_html=True)

with st.expander('View Projected Finalcial Data'):

    st.write('')
    date = datetime.datetime.now().strftime('_%Y_%m_%d')
    PageLayout.data_downloader(df_proj.dropna(axis=1).T, 'projected data', 'projected_financial_data', date)
    st.data_editor(df_proj.dropna(axis=1).T)



with st.expander('View Parameters to estimate the implied share price'):

    simulated_parameters = {'time' : datetime.datetime.now()}
    simulated_parameters['ticker'] = st.session_state['ticker']

    parameters_list = [st.session_state.projection_parameters.items(), st.session_state.wacc_parameters.items()]
    for parameter in parameters_list:
        for key, val in parameter:
            simulated_parameters[key] = val

    df_simulated_parameters = pd.DataFrame(simulated_parameters, index=[0])
    PageLayout.data_downloader(df_simulated_parameters.T, 'simulated parameters', 'simulated_parameters', date)
    st.data_editor(df_simulated_parameters.T, use_container_width=True)



with st.expander('View Simulated DCF results'):

    st.write('')
    df_dcf_results = pd.DataFrame(st.session_state.dcf_results['data'], columns=st.session_state.dcf_results['waccs'], index=st.session_state.dcf_results['tgrs'])

    PageLayout.data_downloader(df_dcf_results, 'simulated DCF results', 'simulated_dcf_results', date)
    st.write('rows : TGRs, columns : WACCs')
    st.data_editor(df_dcf_results)
