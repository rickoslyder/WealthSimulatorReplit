import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from simulation.agents import EconomicAgent
from simulation.utils import calculate_gini, calculate_wealth_percentiles, calculate_wealth_shares

class WealthDistributionModel(Model):
    """
    Agent-based model simulating wealth distribution dynamics over time.
    """
    
    def __init__(self, 
                 num_agents=200, 
                 initial_wealth=1000000,
                 interest_rate=0.03, 
                 base_return_rate=0.07,
                 income_volatility=0.2,
                 inflation_rate=0.02,
                 risk_aversion_mean=0.5,
                 time_preference_mean=0.7,
                 consumption_preference_mean=0.6,
                 tax_rate=0.25,
                 wealth_tax_rate=0.0,
                 ubi_amount=0,
                 social_influence_strength=0.3,
                 max_loan_to_income_ratio=5.0,
                 credit_availability=0.7):
        """
        Initialize a new WealthDistributionModel.
        
        Args:
            num_agents: Number of agents in the simulation
            initial_wealth: Starting wealth for each agent
            interest_rate: Base interest rate in the economy
            base_return_rate: Average return on investments
            income_volatility: Variance in income distribution
            inflation_rate: Annual inflation rate
            risk_aversion_mean: Average risk aversion in the population
            time_preference_mean: Average time preference (patience)
            consumption_preference_mean: Average consumption preference
            tax_rate: Income tax rate
            wealth_tax_rate: Wealth tax rate
            ubi_amount: Universal basic income amount
            social_influence_strength: Strength of peer effects
            max_loan_to_income_ratio: Maximum loan-to-income ratio
            credit_availability: Ease of obtaining credit (0-1)
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.initial_wealth = initial_wealth
        self.interest_rate = interest_rate
        self.base_return_rate = base_return_rate
        self.income_volatility = income_volatility
        self.inflation_rate = inflation_rate
        self.risk_aversion_mean = risk_aversion_mean
        self.time_preference_mean = time_preference_mean
        self.consumption_preference_mean = consumption_preference_mean
        self.tax_rate = tax_rate
        self.wealth_tax_rate = wealth_tax_rate
        self.ubi_amount = ubi_amount
        self.social_influence_strength = social_influence_strength
        self.max_loan_to_income_ratio = max_loan_to_income_ratio
        self.credit_availability = credit_availability
        
        # Macroeconomic variables
        self.gdp = num_agents * initial_wealth * 0.1  # Initial GDP is 10% of total wealth
        self.gdp_growth_rate = 0.025  # Initial GDP growth
        self.unemployment_rate = 0.05  # Initial unemployment
        self.income_mean = np.log(initial_wealth * 0.08)  # Log mean of income distribution
        
        # Initialize scheduler and data collection
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "Gini": lambda m: calculate_gini([a.wealth for a in m.schedule.agents]),
                "GDP": lambda m: m.gdp,
                "Inflation": lambda m: m.inflation_rate,
                "Interest_Rate": lambda m: m.interest_rate,
                "Unemployment": lambda m: m.unemployment_rate
            },
            agent_reporters={
                "Wealth": "wealth",
                "Income": "income",
                "Debt": "debt"
            }
        )
        
        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = EconomicAgent(i, self, initial_wealth)
            self.schedule.add(agent)
            self.agents.append(agent)
        
        # Create social network
        self._create_social_network()
        
        # Initial data collection
        self.datacollector.collect(self)
        
        # Tracking the number of years (steps) since start
        self.years = 0
    
    def _create_social_network(self):
        """Create a social network connecting agents."""
        # Create a small-world network
        # Small-world networks have local clustering and short average path lengths
        G = nx.watts_strogatz_graph(self.num_agents, k=10, p=0.2)
        
        # Assign neighbors to each agent
        for i, agent in enumerate(self.agents):
            # Get the neighbors from the network
            neighbor_indices = list(G.neighbors(i))
            agent.neighbors = [self.agents[idx] for idx in neighbor_indices]
    
    def step(self):
        """Execute one time step (year) in the simulation."""
        # Update macroeconomic conditions before agent actions
        self._update_macroeconomic_conditions()
        
        # Let agents make their decisions
        self.schedule.step()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Increment year counter
        self.years += 1
    
    def _update_macroeconomic_conditions(self):
        """Update macroeconomic variables based on current state."""
        # Calculate current GDP based on agent economic activity
        previous_gdp = self.gdp
        
        # GDP components: consumption, investment, government, net exports
        total_consumption = sum(agent.consumption for agent in self.agents)
        total_investment = sum(0.05 * agent.wealth for agent in self.agents)  # Simplified investment
        government_spending = sum(agent.income for agent in self.agents) * self.tax_rate
        
        # Simple GDP calculation
        self.gdp = total_consumption + total_investment + government_spending
        
        # Calculate GDP growth rate
        self.gdp_growth_rate = (self.gdp - previous_gdp) / previous_gdp if previous_gdp > 0 else 0
        
        # Update unemployment based on GDP growth
        # Higher growth reduces unemployment with some lag and stochasticity
        unemployment_change = -0.2 * self.gdp_growth_rate + np.random.normal(0, 0.01)
        self.unemployment_rate = max(0.02, min(0.15, self.unemployment_rate + unemployment_change))
        
        # Update inflation based on GDP growth and unemployment (Phillips curve)
        # High growth and low unemployment tend to increase inflation
        inflation_pressure = 0.3 * self.gdp_growth_rate - 0.2 * (self.unemployment_rate - 0.05)
        inflation_noise = np.random.normal(0, 0.005)
        self.inflation_rate = max(0, min(0.15, self.inflation_rate + inflation_pressure + inflation_noise))
        
        # Central bank adjusts interest rates using a Taylor rule
        # Responds to inflation and output gaps
        inflation_gap = self.inflation_rate - 0.02  # Target inflation of 2%
        output_gap = self.gdp_growth_rate - 0.025   # Target growth of 2.5%
        
        interest_rate_change = 0.5 * inflation_gap + 0.5 * output_gap
        self.interest_rate = max(0.001, min(0.15, self.interest_rate + interest_rate_change))
        
        # Update credit availability based on economic conditions
        # Credit availability increases in good times, tightens in bad times
        self.credit_availability = max(0.3, min(0.9, 
            self.credit_availability + 0.2 * self.gdp_growth_rate - 0.3 * max(0, inflation_gap)
        ))
        
        # Slightly adjust the income distribution mean based on economic performance
        self.income_mean += 0.01 * self.gdp_growth_rate
