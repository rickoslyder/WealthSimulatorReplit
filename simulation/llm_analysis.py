import os
import google.generativeai as genai
import streamlit as st
import numpy as np
import pandas as pd

# Simulation expert persona from the provided description
SIMULATION_EXPERT_PERSONA = """
The ideal individual to analyze and interpret the outputs of this sophisticated agent-based wealth simulation would possess the following profile:

### Economic & Financial Expertise
- Macroeconomic Knowledge: Solid understanding of economic indicators (GDP growth, inflation, unemployment rates, etc.), macroeconomic theory, monetary policy mechanisms (Taylor rule), and fiscal policy.
- Financial Sector Acumen: Familiarity with credit markets, debt dynamics, financial regulation (e.g., loan-to-value ratios, macroprudential policies), banking operations, and systemic risk analysis.

### Behavioral Economics & Psychology
- Behavioral Economics Background: Deep understanding of prospect theory (loss aversion, reference-dependent preferences), hyperbolic discounting (time-inconsistency, present bias), positional (socially influenced) consumption, and herd behavior.
- Psychological Insight: Ability to interpret agent behavior, identifying and quantifying the impact of cognitive biases, social influences, and psychological traits on economic outcomes.

### Computational & Quantitative Skills
- Agent-Based Modeling (ABM) Proficiency: Expertise with ABM frameworks, particularly Mesa (Python).
- Data Analysis & Visualization: Strong skills in Python data manipulation and visualization.
- Calibration & Validation Experience: Familiarity with calibration methods and sensitivity analyses.

### Statistical & Mathematical Competence
- Distributional Analysis: Skill in analyzing wealth distributions, calculating metrics like the Gini coefficient, Pareto distributions, and identifying inequality dynamics.
- Statistical Fluency: Adept at interpreting statistical outcomes, emergent patterns, and correlations between model parameters and macroeconomic outcomes.

### Domain Integration & Policy Awareness
- Policy Evaluation Skills: Capability to translate model outcomes into actionable insights or policy recommendations.
- Interdisciplinary Approach: Ability to integrate insights from economics, finance, psychology, sociology (social network effects), and computational modeling into a cohesive narrative.

### Critical Thinking & Communication
- Analytical Mindset: Comfortable questioning assumptions, identifying critical model limitations, and assessing the robustness of emergent phenomena.
- Clear Communicator: Skilled at communicating technical results clearly to laypeople, researchers, and policymakers.
"""

# Initialize Gemini (ensuring API key is set)
def initialize_gemini():
    api_key = st.session_state.get('GEMINI_API_KEY')
    
    if not api_key:
        st.warning("⚠️ Gemini API key not found. AI-powered analysis is disabled.")
        st.info("To enable AI analysis features, please add your Gemini API key in the sidebar.")
        return False
    
    genai.configure(api_key=api_key)
    return True

def generate_system_prompt(analysis_type):
    """Generate an appropriate system prompt based on the analysis type."""
    
    base_prompt = f"""
    You are an expert in economics, finance, and wealth distribution analysis. 
    {SIMULATION_EXPERT_PERSONA}
    
    Your task is to analyze simulation results from an agent-based model of wealth distribution.
    
    Guidelines:
    1. Provide clear, jargon-free explanations accessible to educated non-experts.
    2. Support your analysis with specific data points from the simulation results.
    3. Highlight key relationships between parameters and outcomes.
    4. Avoid ideological bias - focus on empirical patterns and mechanisms.
    5. Be specific about limitations and uncertainties.
    6. Use bullet points and short paragraphs for readability.
    """
    
    analysis_prompts = {
        "summary": base_prompt + """
            Provide a concise summary (3-4 paragraphs) of the overall simulation results.
            Focus on the most striking patterns and their potential implications.
            Highlight unexpected or counterintuitive findings.
        """,
        "inequality_analysis": base_prompt + """
            Analyze the wealth inequality patterns in the simulation.
            Explain what factors contributed to the observed Gini coefficient and wealth distribution.
            Discuss how different percentiles fared relative to each other and why.
            Include references to the observed wealth shares (top 1%, top 10%, bottom 50%).
        """,
        "policy_implications": base_prompt + """
            Analyze the effectiveness of policy interventions in the simulation.
            Discuss how tax rates, UBI, and other policy levers affected outcomes.
            Provide balanced analysis of tradeoffs involved in different policy approaches.
            Suggest what policy combinations might be worth exploring further.
        """,
        "behavioral_insights": base_prompt + """
            Analyze how behavioral factors influenced the simulation outcomes.
            Discuss the impact of risk aversion, time preference, and consumption preferences.
            Explain how social influence dynamics affected the overall results.
            Highlight any emergent patterns from agent interactions.
        """,
        "macroeconomic_trends": base_prompt + """
            Analyze the macroeconomic trends in the simulation.
            Discuss the relationships between GDP growth, inflation, interest rates, and unemployment.
            Explain how these factors influenced and were influenced by wealth distribution.
            Highlight any boom-bust cycles or other dynamic patterns over time.
        """,
        "future_scenarios": base_prompt + """
            Based on the current simulation results, suggest 3-4 interesting alternative scenarios to explore.
            For each scenario, specify what parameters should be modified and why.
            Explain what hypotheses could be tested with these scenarios.
            Suggest how comparing these scenarios might provide new insights.
        """
    }
    
    return analysis_prompts.get(analysis_type, base_prompt)

def format_simulation_data(model, wealth_history, gini_history, years_simulated, 
                          gdp_history, inflation_history, interest_rate_history, 
                          unemployment_history, top1_share_history, top10_share_history, 
                          bottom50_share_history, parameters):
    """Format simulation data for LLM analysis."""
    
    # Format key metrics for readability
    final_gini = gini_history[-1]
    final_gdp = gdp_history[-1]/1e9  # Convert to billions
    final_top1 = top1_share_history[-1]
    final_top10 = top10_share_history[-1]
    final_bottom50 = bottom50_share_history[-1]
    
    # Calculate wealth percentiles at final step
    final_wealth = wealth_history[-1]
    percentiles = np.percentile(final_wealth, [10, 25, 50, 75, 90, 99])
    
    # Format data as text
    data_text = f"""
    SIMULATION PARAMETERS:
    - Number of agents: {parameters['num_agents']}
    - Initial wealth per agent: ${parameters['initial_wealth']:,}
    - Years simulated: {years_simulated}
    - Interest rate: {parameters['interest_rate']*100:.1f}%
    - Base investment return: {parameters['base_return_rate']*100:.1f}%
    - Income volatility: {parameters['income_volatility']:.2f}
    - Inflation rate: {parameters['inflation_rate']*100:.1f}%
    - Risk aversion (mean): {parameters['risk_aversion_mean']:.2f}
    - Time preference (mean): {parameters['time_preference_mean']:.2f}
    - Consumption preference (mean): {parameters['consumption_preference_mean']:.2f}
    - Income tax rate: {parameters['tax_rate']*100:.1f}%
    - Wealth tax rate: {parameters['wealth_tax_rate']*100:.2f}%
    - Universal basic income: ${parameters['ubi_amount']:,}/year
    - Social influence strength: {parameters['social_influence_strength']:.2f}
    - Max loan-to-income ratio: {parameters['max_loan_to_income_ratio']:.1f}
    - Credit availability: {parameters['credit_availability']:.2f}
    
    KEY FINAL RESULTS:
    - Gini coefficient: {final_gini:.3f}
    - GDP: ${final_gdp:.2f} billion
    - Wealth share (top 1%): {final_top1:.1f}%
    - Wealth share (top 10%): {final_top10:.1f}%
    - Wealth share (bottom 50%): {final_bottom50:.1f}%
    
    FINAL WEALTH PERCENTILES:
    - 10th percentile: ${percentiles[0]:,.0f}
    - 25th percentile: ${percentiles[1]:,.0f}
    - 50th percentile (median): ${percentiles[2]:,.0f}
    - 75th percentile: ${percentiles[3]:,.0f}
    - 90th percentile: ${percentiles[4]:,.0f}
    - 99th percentile: ${percentiles[5]:,.0f}
    
    TIME SERIES DATA (starting year, middle year, final year):
    - Gini coefficient: [{gini_history[0]:.3f}, {gini_history[len(gini_history)//2]:.3f}, {gini_history[-1]:.3f}]
    - GDP (billions): [${gdp_history[0]/1e9:.2f}, ${gdp_history[len(gdp_history)//2]/1e9:.2f}, ${gdp_history[-1]/1e9:.2f}]
    - Inflation rate: [{inflation_history[0]:.1f}%, {inflation_history[len(inflation_history)//2]:.1f}%, {inflation_history[-1]:.1f}%]
    - Interest rate: [{interest_rate_history[0]:.1f}%, {interest_rate_history[len(interest_rate_history)//2]:.1f}%, {interest_rate_history[-1]:.1f}%]
    - Unemployment rate: [{unemployment_history[0]:.1f}%, {unemployment_history[len(unemployment_history)//2]:.1f}%, {unemployment_history[-1]:.1f}%]
    - Top 1% wealth share: [{top1_share_history[0]:.1f}%, {top1_share_history[len(top1_share_history)//2]:.1f}%, {top1_share_history[-1]:.1f}%]
    - Top 10% wealth share: [{top10_share_history[0]:.1f}%, {top10_share_history[len(top10_share_history)//2]:.1f}%, {top10_share_history[-1]:.1f}%]
    - Bottom 50% wealth share: [{bottom50_share_history[0]:.1f}%, {bottom50_share_history[len(bottom50_share_history)//2]:.1f}%, {bottom50_share_history[-1]:.1f}%]
    """
    
    return data_text

def get_llm_analysis(analysis_type, model, wealth_history, gini_history, years_simulated, 
                    gdp_history, inflation_history, interest_rate_history, 
                    unemployment_history, top1_share_history, top10_share_history, 
                    bottom50_share_history, parameters):
    """Get LLM analysis of simulation results."""
    
    if not initialize_gemini():
        return "AI analysis unavailable. Please provide a Gemini API key to enable this feature."
    
    # Format data for the LLM
    data_text = format_simulation_data(
        model, wealth_history, gini_history, years_simulated, 
        gdp_history, inflation_history, interest_rate_history, 
        unemployment_history, top1_share_history, top10_share_history, 
        bottom50_share_history, parameters
    )
    
    # Generate system prompt
    system_prompt = generate_system_prompt(analysis_type)
    
    # Create generation config
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1024,
    }
    
    # Prepare message for Gemini
    prompt = f"{system_prompt}\n\nHere is the simulation data to analyze:\n{data_text}"
    
    # Safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# Define preset configurations for the simulation
def get_simulation_presets():
    """Return predefined simulation presets."""
    
    presets = {
        "default": {
            "name": "Default Balanced Economy",
            "description": "Standard economic conditions with moderate policies and behavior.",
            "tooltip": "A balanced economy with standard parameters - good starting point",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.03,
                "base_return_rate": 0.07,
                "income_volatility": 0.2,
                "inflation_rate": 0.02,
                "risk_aversion_mean": 0.5,
                "time_preference_mean": 0.7,
                "consumption_preference_mean": 0.6,
                "tax_rate": 0.25,
                "wealth_tax_rate": 0.0,
                "ubi_amount": 0,
                "social_influence_strength": 0.3,
                "max_loan_to_income_ratio": 5.0,
                "credit_availability": 0.7
            }
        },
        "nordic_model": {
            "name": "Nordic Social Democracy",
            "description": "High tax rates, strong safety nets, and regulated markets.",
            "tooltip": "Simulates Nordic-style economies with high taxes, strong welfare, and regulated markets",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.02,
                "base_return_rate": 0.06,
                "income_volatility": 0.15,
                "inflation_rate": 0.02,
                "risk_aversion_mean": 0.6,
                "time_preference_mean": 0.8,
                "consumption_preference_mean": 0.55,
                "tax_rate": 0.45,
                "wealth_tax_rate": 0.02,
                "ubi_amount": 20000,
                "social_influence_strength": 0.4,
                "max_loan_to_income_ratio": 4.0,
                "credit_availability": 0.6
            }
        },
        "laissez_faire": {
            "name": "Laissez-Faire Capitalism",
            "description": "Minimal taxation, deregulated markets, no safety nets.",
            "tooltip": "Free market economy with minimal government intervention",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.04,
                "base_return_rate": 0.09,
                "income_volatility": 0.3,
                "inflation_rate": 0.03,
                "risk_aversion_mean": 0.4,
                "time_preference_mean": 0.6,
                "consumption_preference_mean": 0.7,
                "tax_rate": 0.15,
                "wealth_tax_rate": 0.0,
                "ubi_amount": 0,
                "social_influence_strength": 0.2,
                "max_loan_to_income_ratio": 7.0,
                "credit_availability": 0.8
            }
        },
        "ubi_experiment": {
            "name": "Universal Basic Income",
            "description": "Standard economy with significant UBI implementation.",
            "tooltip": "Tests effects of a substantial universal basic income with moderate taxation",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.03,
                "base_return_rate": 0.07,
                "income_volatility": 0.2,
                "inflation_rate": 0.025,
                "risk_aversion_mean": 0.5,
                "time_preference_mean": 0.7,
                "consumption_preference_mean": 0.6,
                "tax_rate": 0.35,
                "wealth_tax_rate": 0.01,
                "ubi_amount": 30000,
                "social_influence_strength": 0.3,
                "max_loan_to_income_ratio": 5.0,
                "credit_availability": 0.7
            }
        },
        "wealth_tax": {
            "name": "Progressive Wealth Taxation",
            "description": "Economy with significant wealth taxation but moderate income taxes.",
            "tooltip": "Tests impact of substantial wealth taxation with moderate income taxes",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.03,
                "base_return_rate": 0.07,
                "income_volatility": 0.2,
                "inflation_rate": 0.02,
                "risk_aversion_mean": 0.5,
                "time_preference_mean": 0.7,
                "consumption_preference_mean": 0.6,
                "tax_rate": 0.25,
                "wealth_tax_rate": 0.03,
                "ubi_amount": 10000,
                "social_influence_strength": 0.3,
                "max_loan_to_income_ratio": 5.0,
                "credit_availability": 0.7
            }
        },
        "high_volatility": {
            "name": "High Economic Volatility",
            "description": "Turbulent economy with high volatility and inflation.",
            "tooltip": "Simulates unstable economic conditions with high volatility and inflation",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.06,
                "base_return_rate": 0.10,
                "income_volatility": 0.4,
                "inflation_rate": 0.05,
                "risk_aversion_mean": 0.5,
                "time_preference_mean": 0.6,
                "consumption_preference_mean": 0.65,
                "tax_rate": 0.20,
                "wealth_tax_rate": 0.0,
                "ubi_amount": 0,
                "social_influence_strength": 0.3,
                "max_loan_to_income_ratio": 6.0,
                "credit_availability": 0.6
            }
        },
        "social_influence": {
            "name": "Strong Social Networks",
            "description": "Economy with strong social influence on economic behavior.",
            "tooltip": "Tests how strong social networks and peer effects impact wealth distribution",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.03,
                "base_return_rate": 0.07,
                "income_volatility": 0.2,
                "inflation_rate": 0.02,
                "risk_aversion_mean": 0.5,
                "time_preference_mean": 0.7,
                "consumption_preference_mean": 0.6,
                "tax_rate": 0.25,
                "wealth_tax_rate": 0.0,
                "ubi_amount": 0,
                "social_influence_strength": 0.8,
                "max_loan_to_income_ratio": 5.0,
                "credit_availability": 0.7
            }
        },
        "easy_credit": {
            "name": "Easy Credit Economy",
            "description": "Economy with very easy access to credit and high loan limits.",
            "tooltip": "Simulates effects of loose credit policies and high borrowing limits",
            "parameters": {
                "num_agents": 200,
                "initial_wealth": 1000000,
                "interest_rate": 0.02,
                "base_return_rate": 0.08,
                "income_volatility": 0.25,
                "inflation_rate": 0.03,
                "risk_aversion_mean": 0.4,
                "time_preference_mean": 0.6,
                "consumption_preference_mean": 0.7,
                "tax_rate": 0.20,
                "wealth_tax_rate": 0.0,
                "ubi_amount": 0,
                "social_influence_strength": 0.3,
                "max_loan_to_income_ratio": 10.0,
                "credit_availability": 0.9
            }
        }
    }
    
    return presets