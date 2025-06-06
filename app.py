import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import time

from simulation.model import WealthDistributionModel
from simulation.utils import calculate_gini, calculate_wealth_percentiles, calculate_wealth_shares
from simulation.visualization import plot_wealth_distribution, plot_gini_coefficient, plot_wealth_shares
from simulation.llm_analysis import get_simulation_presets, get_llm_analysis, initialize_gemini

# Set page configuration
st.set_page_config(
    page_title="Wealth Distribution Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title and description
st.title("Wealth Distribution Simulator")
st.markdown("""
This application simulates how wealth distribution evolves over 100 years when everyone 
starts with $1,000,000. The simulation uses agent-based modeling to capture economic, 
behavioral, and policy dynamics.
""")

# Initialize session state for storing selected preset
if 'selected_preset' not in st.session_state:
    st.session_state['selected_preset'] = 'default'
    
if 'custom_parameters' not in st.session_state:
    st.session_state['custom_parameters'] = False

# Get simulation presets
presets = get_simulation_presets()

# Sidebar for API key and presets
st.sidebar.header("Configuration")

# API key input for Gemini
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key (for AI analysis)",
    type="password",
    help="Required for AI-powered analysis of simulation results"
)

# Store API key in session state
if gemini_api_key:
    st.session_state['GEMINI_API_KEY'] = gemini_api_key

# Preset selector section
st.sidebar.subheader("Simulation Presets")

# Create two columns for better layout of preset buttons
preset_cols = st.sidebar.columns(2)

# Counter for alternating between columns
i = 0

# Current preset information display
current_preset_name = presets[st.session_state['selected_preset']]['name'] if not st.session_state.get('custom_parameters', False) else "Custom Parameters"

# Display all preset buttons
for preset_id, preset_data in presets.items():
    # Determine which column to place this preset button
    col_idx = i % 2
    
    # Create the button in the appropriate column
    if preset_cols[col_idx].button(
        preset_data['name'],
        key=f"preset_{preset_id}",
        help=preset_data['tooltip'],
        use_container_width=True,
        type="secondary" if st.session_state['selected_preset'] != preset_id or st.session_state.get('custom_parameters', False) else "primary"
    ):
        st.session_state['selected_preset'] = preset_id
        st.session_state['custom_parameters'] = False
        st.rerun()
    
    i += 1

# Show description of current preset
if not st.session_state.get('custom_parameters', False):
    st.sidebar.info(f"**{current_preset_name}**: {presets[st.session_state['selected_preset']]['description']}")

# Toggle for custom parameters
custom_params = st.sidebar.checkbox("Customize Parameters", value=st.session_state.get('custom_parameters', False))
if custom_params != st.session_state.get('custom_parameters', False):
    st.session_state['custom_parameters'] = custom_params
    st.rerun()

# Get parameters based on preset or use custom
if st.session_state.get('custom_parameters', False):
    # Sidebar parameters for custom configuration
    st.sidebar.header("Custom Parameters")
    
    # Demographics
    st.sidebar.subheader("Population")
    num_agents = st.sidebar.slider("Number of Agents", 100, 1000, 200, 100)
    
    # Economic Parameters
    st.sidebar.subheader("Economic Parameters")
    initial_interest_rate = st.sidebar.slider("Initial Interest Rate (%)", 1.0, 10.0, 3.0, 0.5)
    base_return_rate = st.sidebar.slider("Base Investment Return (%)", 3.0, 15.0, 7.0, 0.5)
    income_volatility = st.sidebar.slider("Income Volatility", 0.1, 0.5, 0.2, 0.05)
    initial_inflation_rate = st.sidebar.slider("Initial Inflation Rate (%)", 0.0, 10.0, 2.0, 0.5)
    
    # Behavioral Parameters
    st.sidebar.subheader("Behavioral Parameters")
    risk_aversion_mean = st.sidebar.slider("Average Risk Aversion", 0.0, 1.0, 0.5, 0.1)
    time_preference_mean = st.sidebar.slider("Average Time Preference (Patience)", 0.0, 1.0, 0.7, 0.1)
    consumption_preference_mean = st.sidebar.slider("Average Consumption Preference", 0.3, 0.8, 0.6, 0.05)
    
    # Policy Parameters
    st.sidebar.subheader("Policy Parameters")
    tax_rate = st.sidebar.slider("Income Tax Rate (%)", 0.0, 50.0, 25.0, 5.0) / 100
    wealth_tax_rate = st.sidebar.slider("Wealth Tax Rate (%)", 0.0, 5.0, 0.0, 0.5) / 100
    ubi_amount = st.sidebar.slider("Universal Basic Income (Annual $)", 0, 50000, 0, 1000)
    
    # Social Network
    st.sidebar.subheader("Social Network")
    social_influence_strength = st.sidebar.slider("Social Influence Strength", 0.0, 1.0, 0.3, 0.1)
    
    # Credit Market
    st.sidebar.subheader("Credit Market")
    max_loan_to_income_ratio = st.sidebar.slider("Max Loan-to-Income Ratio", 1.0, 10.0, 5.0, 0.5)
    credit_availability = st.sidebar.slider("Credit Availability", 0.0, 1.0, 0.7, 0.1)
else:
    # Use preset parameters
    preset_params = presets[st.session_state['selected_preset']]['parameters']
    
    # Extract parameters from preset
    num_agents = preset_params['num_agents']
    initial_interest_rate = preset_params['interest_rate'] * 100
    base_return_rate = preset_params['base_return_rate'] * 100
    income_volatility = preset_params['income_volatility']
    initial_inflation_rate = preset_params['inflation_rate'] * 100
    risk_aversion_mean = preset_params['risk_aversion_mean']
    time_preference_mean = preset_params['time_preference_mean']
    consumption_preference_mean = preset_params['consumption_preference_mean']
    tax_rate = preset_params['tax_rate']
    wealth_tax_rate = preset_params['wealth_tax_rate']
    ubi_amount = preset_params['ubi_amount']
    social_influence_strength = preset_params['social_influence_strength']
    max_loan_to_income_ratio = preset_params['max_loan_to_income_ratio']
    credit_availability = preset_params['credit_availability']

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulation Results", "Wealth Distribution", "Economic Indicators", "Agent Details", "AI Analysis"])

with tab1:
    st.header("Simulation Controls")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        simulate_button = st.button("Run Simulation", type="primary")
    with col2:
        years_to_simulate = st.number_input("Simulation Years", min_value=1, max_value=100, value=30)
    with col3:
        show_progress = st.checkbox("Show Progress", value=True)
    with col4:
        detailed_results = st.checkbox("Detailed Results", value=False)
    
    # Placeholder for simulation status
    simulation_status = st.empty()
    progress_bar = st.empty()
    
    # Run simulation when button is clicked
    if simulate_button:
        # Create model with selected parameters
        model = WealthDistributionModel(
            num_agents=num_agents,
            initial_wealth=1000000,
            interest_rate=initial_interest_rate / 100,
            base_return_rate=base_return_rate / 100,
            income_volatility=income_volatility,
            inflation_rate=initial_inflation_rate / 100,
            risk_aversion_mean=risk_aversion_mean,
            time_preference_mean=time_preference_mean,
            consumption_preference_mean=consumption_preference_mean,
            tax_rate=tax_rate,
            wealth_tax_rate=wealth_tax_rate,
            ubi_amount=ubi_amount,
            social_influence_strength=social_influence_strength,
            max_loan_to_income_ratio=max_loan_to_income_ratio,
            credit_availability=credit_availability,
            seed=None
        )
        
        # Set up data collection
        wealth_history = []
        gini_history = []
        gdp_history = []
        inflation_history = []
        interest_rate_history = []
        unemployment_history = []
        top1_share_history = []
        top10_share_history = []
        bottom50_share_history = []
        
        # Run the simulation
        if show_progress:
            progress = progress_bar.progress(0)
        
        for year in range(years_to_simulate):
            model.step()
            
            # Collect data
            agent_wealths = [agent.wealth for agent in model.agents]
            wealth_history.append(agent_wealths)
            gini_history.append(calculate_gini(agent_wealths))
            gdp_history.append(model.gdp)
            inflation_history.append(model.inflation_rate * 100)  # Convert to percentage
            interest_rate_history.append(model.interest_rate * 100)  # Convert to percentage
            unemployment_history.append(model.unemployment_rate * 100)  # Convert to percentage
            
            # Calculate wealth shares
            wealth_shares = calculate_wealth_shares(agent_wealths)
            top1_share_history.append(wealth_shares['top1'])
            top10_share_history.append(wealth_shares['top10'])
            bottom50_share_history.append(wealth_shares['bottom50'])
            
            if show_progress:
                progress.progress((year + 1) / years_to_simulate)
                simulation_status.text(f"Simulating Year {year + 1}/{years_to_simulate}")
                if year % 5 == 0:  # Update less frequently to improve performance
                    time.sleep(0.1)
        
        simulation_status.text("Simulation complete!")
        
        # Display results
        st.subheader("Simulation Summary")
        
        # Display metric cards for key results
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Final Gini Coefficient", f"{gini_history[-1]:.3f}")
        with metric_col2:
            st.metric("Final GDP", f"${gdp_history[-1]/1e9:.2f}B")
        with metric_col3:
            st.metric("Wealth Share - Top 1%", f"{top1_share_history[-1]:.1f}%")
        with metric_col4:
            st.metric("Wealth Share - Bottom 50%", f"{bottom50_share_history[-1]:.1f}%")
        
        # Create time series plots
        st.subheader("Economic Indicators Over Time")
        
        # Plot Gini coefficient over time
        fig_gini = plot_gini_coefficient(gini_history)
        st.plotly_chart(fig_gini, use_container_width=True)
        
        # Plot wealth shares over time
        fig_shares = plot_wealth_shares(top1_share_history, top10_share_history, bottom50_share_history)
        st.plotly_chart(fig_shares, use_container_width=True)
        
        # Store data in session state for other tabs
        st.session_state['simulation_run'] = True
        st.session_state['wealth_history'] = wealth_history
        st.session_state['gini_history'] = gini_history
        st.session_state['gdp_history'] = gdp_history
        st.session_state['inflation_history'] = inflation_history
        st.session_state['interest_rate_history'] = interest_rate_history
        st.session_state['unemployment_history'] = unemployment_history
        st.session_state['top1_share_history'] = top1_share_history
        st.session_state['top10_share_history'] = top10_share_history
        st.session_state['bottom50_share_history'] = bottom50_share_history
        st.session_state['model'] = model
        st.session_state['years_simulated'] = years_to_simulate
        
        # Clear analysis cache when running a new simulation
        if 'analysis_cache' in st.session_state:
            st.session_state['analysis_cache'] = {}
    else:
        if 'simulation_run' not in st.session_state:
            st.info("Click 'Run Simulation' to start the wealth distribution simulation.")

with tab2:
    st.header("Wealth Distribution Analysis")
    
    if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
        wealth_history = st.session_state['wealth_history']
        years_simulated = st.session_state['years_simulated']
        
        # Add a year selection slider
        selected_year = st.slider("Select Year", 0, years_simulated-1, years_simulated-1)
        
        # Distribution at selected year
        st.subheader(f"Wealth Distribution - Year {selected_year}")
        fig_dist = plot_wealth_distribution(wealth_history[selected_year])
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Display percentile data
        st.subheader("Wealth by Percentile")
        percentiles = calculate_wealth_percentiles(wealth_history[selected_year])
        
        # Create a table for percentiles
        percentile_df = pd.DataFrame({
            'Percentile': ['P10', 'P25', 'P50 (Median)', 'P75', 'P90', 'P99'],
            'Wealth': [
                f"${percentiles['p10']:,.0f}",
                f"${percentiles['p25']:,.0f}",
                f"${percentiles['p50']:,.0f}",
                f"${percentiles['p75']:,.0f}",
                f"${percentiles['p90']:,.0f}",
                f"${percentiles['p99']:,.0f}"
            ]
        })
        st.table(percentile_df)
    else:
        st.info("Run the simulation in the 'Simulation Results' tab to see wealth distribution analysis.")

with tab3:
    st.header("Economic Indicators")
    
    if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
        # Get data from session state
        years_simulated = st.session_state['years_simulated']
        gdp_history = st.session_state['gdp_history']
        inflation_history = st.session_state['inflation_history']
        interest_rate_history = st.session_state['interest_rate_history']
        unemployment_history = st.session_state['unemployment_history']
        
        # Create dataframe for economic indicators
        econ_df = pd.DataFrame({
            'Year': list(range(years_simulated)),
            'GDP': gdp_history,
            'Inflation Rate (%)': inflation_history,
            'Interest Rate (%)': interest_rate_history,
            'Unemployment Rate (%)': unemployment_history
        })
        
        # Allow selecting which indicators to display
        indicators = st.multiselect(
            "Select Indicators to Display",
            ['GDP', 'Inflation Rate (%)', 'Interest Rate (%)', 'Unemployment Rate (%)'],
            default=['GDP', 'Inflation Rate (%)']
        )
        
        if indicators:
            # Plot selected indicators
            fig = go.Figure()
            
            for indicator in indicators:
                if indicator == 'GDP':
                    # Format GDP as billions
                    fig.add_trace(go.Scatter(
                        x=econ_df['Year'],
                        y=econ_df['GDP'] / 1e9,
                        mode='lines',
                        name='GDP ($ Billions)'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=econ_df['Year'],
                        y=econ_df[indicator],
                        mode='lines',
                        name=indicator
                    ))
            
            fig.update_layout(
                title='Economic Indicators Over Time',
                xaxis_title='Year',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the data table
            st.subheader("Economic Data Table")
            formatted_df = econ_df.copy()
            formatted_df['GDP'] = formatted_df['GDP'].apply(lambda x: f"${x/1e9:.2f}B")
            st.dataframe(formatted_df)
    else:
        st.info("Run the simulation in the 'Simulation Results' tab to see economic indicators.")

with tab4:
    st.header("Agent Details")
    
    if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
        model = st.session_state['model']
        
        # Create a dataframe with agent details
        agent_data = []
        for agent in model.agents:
            agent_data.append({
                'ID': agent.unique_id,
                'Wealth': agent.wealth,
                'Income': agent.income,
                'Consumption': agent.consumption,
                'Savings': agent.savings,
                'Risk Aversion': agent.risk_aversion,
                'Time Preference': agent.time_preference,
                'Debt': agent.debt
            })
        
        agent_df = pd.DataFrame(agent_data)
        
        # Allow sorting by different columns
        sort_by = st.selectbox("Sort Agents By", ['Wealth', 'Income', 'Consumption', 'Savings', 'Debt', 'Risk Aversion', 'Time Preference'])
        ascending = st.checkbox("Ascending Order", value=False)
        
        # Sort and display the dataframe
        sorted_df = agent_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        
        # Format currency columns
        for col in ['Wealth', 'Income', 'Consumption', 'Savings', 'Debt']:
            sorted_df[col] = sorted_df[col].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(sorted_df, use_container_width=True)
        
        # Show demographic statistics
        st.subheader("Agent Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot risk aversion distribution
            risk_values = [agent.risk_aversion for agent in model.agents]
            fig_risk = px.histogram(
                x=risk_values,
                nbins=20,
                labels={'x': 'Risk Aversion'},
                title='Distribution of Risk Aversion'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with col2:
            # Plot time preference distribution
            time_pref_values = [agent.time_preference for agent in model.agents]
            fig_time = px.histogram(
                x=time_pref_values,
                nbins=20,
                labels={'x': 'Time Preference (Patience)'},
                title='Distribution of Time Preference'
            )
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Run the simulation in the 'Simulation Results' tab to see agent details.")

with tab5:
    st.header("AI-Powered Analysis")
    
    if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
        # Check if Gemini API key is provided
        if not initialize_gemini():
            st.warning("⚠️ Gemini API key is required for AI analysis")
            st.info("Please add your Gemini API key in the sidebar to enable AI-powered insights.")
        else:
            # Get simulation data from session state
            model = st.session_state['model']
            wealth_history = st.session_state['wealth_history']
            gini_history = st.session_state['gini_history']
            years_simulated = st.session_state['years_simulated']
            gdp_history = st.session_state['gdp_history']
            inflation_history = st.session_state['inflation_history']
            interest_rate_history = st.session_state['interest_rate_history']
            unemployment_history = st.session_state['unemployment_history']
            top1_share_history = st.session_state.get('top1_share_history', [])
            top10_share_history = st.session_state.get('top10_share_history', [])
            bottom50_share_history = st.session_state.get('bottom50_share_history', [])
            
            # Create a dictionary with the current simulation parameters
            current_params = {
                'num_agents': num_agents,
                'initial_wealth': 1000000,
                'interest_rate': initial_interest_rate / 100,
                'base_return_rate': base_return_rate / 100,
                'income_volatility': income_volatility,
                'inflation_rate': initial_inflation_rate / 100,
                'risk_aversion_mean': risk_aversion_mean,
                'time_preference_mean': time_preference_mean,
                'consumption_preference_mean': consumption_preference_mean,
                'tax_rate': tax_rate,
                'wealth_tax_rate': wealth_tax_rate,
                'ubi_amount': ubi_amount,
                'social_influence_strength': social_influence_strength,
                'max_loan_to_income_ratio': max_loan_to_income_ratio,
                'credit_availability': credit_availability
            }
            
            # Tabs for different types of analysis
            analysis_tabs = st.tabs([
                "Overall Summary", 
                "Inequality Analysis", 
                "Policy Implications", 
                "Behavioral Insights", 
                "Macroeconomic Trends", 
                "Future Scenarios"
            ])
            
            # Cache for storing generated analyses to avoid regenerating when switching tabs
            if 'analysis_cache' not in st.session_state:
                st.session_state['analysis_cache'] = {}
            
            # Helper function to get analysis with caching
            def get_cached_analysis(analysis_type):
                if analysis_type not in st.session_state['analysis_cache']:
                    with st.spinner(f"Generating {analysis_type} analysis..."):
                        analysis = get_llm_analysis(
                            analysis_type=analysis_type,
                            model=model,
                            wealth_history=wealth_history,
                            gini_history=gini_history,
                            years_simulated=years_simulated,
                            gdp_history=gdp_history,
                            inflation_history=inflation_history,
                            interest_rate_history=interest_rate_history,
                            unemployment_history=unemployment_history,
                            top1_share_history=top1_share_history,
                            top10_share_history=top10_share_history,
                            bottom50_share_history=bottom50_share_history,
                            parameters=current_params
                        )
                        st.session_state['analysis_cache'][analysis_type] = analysis
                return st.session_state['analysis_cache'][analysis_type]
            
            # Overall Summary
            with analysis_tabs[0]:
                st.subheader("Overall Simulation Summary")
                summary = get_cached_analysis("summary")
                st.markdown(summary)
                
                # Add option to regenerate analysis
                if st.button("Regenerate Summary", key="regen_summary"):
                    if "summary" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["summary"]
                    st.rerun()
            
            # Inequality Analysis
            with analysis_tabs[1]:
                st.subheader("Wealth Inequality Analysis")
                inequality_analysis = get_cached_analysis("inequality_analysis")
                st.markdown(inequality_analysis)
                
                if st.button("Regenerate Inequality Analysis", key="regen_inequality"):
                    if "inequality_analysis" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["inequality_analysis"]
                    st.rerun()
            
            # Policy Implications
            with analysis_tabs[2]:
                st.subheader("Policy Implications")
                policy_analysis = get_cached_analysis("policy_implications")
                st.markdown(policy_analysis)
                
                if st.button("Regenerate Policy Analysis", key="regen_policy"):
                    if "policy_implications" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["policy_implications"]
                    st.rerun()
            
            # Behavioral Insights
            with analysis_tabs[3]:
                st.subheader("Behavioral Insights")
                behavioral_analysis = get_cached_analysis("behavioral_insights")
                st.markdown(behavioral_analysis)
                
                if st.button("Regenerate Behavioral Analysis", key="regen_behavioral"):
                    if "behavioral_insights" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["behavioral_insights"]
                    st.rerun()
            
            # Macroeconomic Trends
            with analysis_tabs[4]:
                st.subheader("Macroeconomic Trends")
                macro_analysis = get_cached_analysis("macroeconomic_trends")
                st.markdown(macro_analysis)
                
                if st.button("Regenerate Macroeconomic Analysis", key="regen_macro"):
                    if "macroeconomic_trends" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["macroeconomic_trends"]
                    st.rerun()
            
            # Future Scenarios
            with analysis_tabs[5]:
                st.subheader("Suggested Future Scenarios")
                scenario_analysis = get_cached_analysis("future_scenarios")
                st.markdown(scenario_analysis)
                
                if st.button("Regenerate Scenario Suggestions", key="regen_scenarios"):
                    if "future_scenarios" in st.session_state['analysis_cache']:
                        del st.session_state['analysis_cache']["future_scenarios"]
                    st.rerun()
            
            # Add a button to clear the entire cache
            if st.button("Regenerate All Analyses", type="secondary"):
                st.session_state['analysis_cache'] = {}
                st.rerun()
                
    else:
        st.info("Run the simulation in the 'Simulation Results' tab to see AI-powered analysis.")
        st.markdown("""
        ### What to expect from AI analysis:
        
        The AI analysis provides the following insights about your simulation:
        
        - **Overall Summary**: Key patterns and dynamics from the simulation
        - **Inequality Analysis**: Factors affecting wealth distribution and inequality
        - **Policy Implications**: How policy choices impacted the economy
        - **Behavioral Insights**: Effects of behavioral traits on economic outcomes
        - **Macroeconomic Trends**: Analysis of GDP, inflation, and other indicators
        - **Future Scenarios**: Suggested simulations to try next based on your results
        
        > Note: A Gemini API key is required for these insights. You can add this in the sidebar.
        """)
