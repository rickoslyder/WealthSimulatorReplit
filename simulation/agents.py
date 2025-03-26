import numpy as np
from mesa import Agent

class EconomicAgent(Agent):
    """
    Agent representing an individual in the economy with wealth, income, and behavioral traits.
    """
    
    def __init__(self, unique_id, model, initial_wealth=1000000):
        """
        Initialize a new economic agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: The model instance the agent belongs to
            initial_wealth: Starting wealth for the agent (default $1,000,000)
        """
        super().__init__(unique_id, model)
        
        # Economic attributes
        self.wealth = initial_wealth
        self.income = np.random.lognormal(mean=model.income_mean, sigma=model.income_volatility)
        self.consumption = 0
        self.savings = 0
        self.investments = 0
        self.debt = 0
        self.credit_score = np.random.uniform(600, 850)  # Initial credit score
        
        # Employment status (1 = employed, 0 = unemployed)
        self.employed = np.random.choice([0, 1], p=[model.unemployment_rate, 1 - model.unemployment_rate])
        
        # Behavioral traits - normally distributed around population means
        self.risk_aversion = np.clip(
            np.random.normal(model.risk_aversion_mean, 0.2), 
            0.1, 
            0.9
        )
        self.time_preference = np.clip(
            np.random.normal(model.time_preference_mean, 0.2),
            0.1,
            0.9
        )
        self.consumption_preference = np.clip(
            np.random.normal(model.consumption_preference_mean, 0.1),
            0.3,
            0.9
        )
        
        # Social network attributes - will be filled by the model
        self.neighbors = []
        
        # Track history for this agent (optional, for detailed analysis)
        self.wealth_history = [initial_wealth]
        self.income_history = [self.income]
        
    def step(self):
        """Execute agent actions for the current time step."""
        # Update employment status based on unemployment rate
        if not self.employed:
            # Try to find a job if unemployed
            self.employed = np.random.choice(
                [0, 1], 
                p=[self.model.unemployment_rate, 1 - self.model.unemployment_rate]
            )
        elif np.random.random() < self.model.unemployment_rate * 0.3:
            # Small chance of losing job if employed
            self.employed = 0
        
        # Update income based on employment status
        if self.employed:
            # Regular income plus income growth
            base_income_growth = max(self.model.gdp_growth_rate, 0)
            individual_variation = np.random.normal(0, 0.02)  # Individual performance variation
            income_growth = base_income_growth + individual_variation
            
            self.income = self.income * (1 + income_growth)
        else:
            # Unemployment benefits (a fraction of previous income)
            self.income = self.income * 0.3
        
        # Get UBI if available
        if self.model.ubi_amount > 0:
            self.income += self.model.ubi_amount
        
        # Pay taxes
        income_tax = self.income * self.model.tax_rate
        wealth_tax = self.wealth * self.model.wealth_tax_rate
        total_tax = income_tax + wealth_tax
        
        # Decide on consumption
        # Base consumption is a function of income, wealth, and consumption preference
        base_consumption = self.consumption_preference * self.income
        
        # Social influence on consumption (keeping up with the Joneses)
        if self.neighbors:
            neighbors_avg_consumption = np.mean([neighbor.consumption for neighbor in self.neighbors])
            social_pressure = self.model.social_influence_strength * max(0, neighbors_avg_consumption - base_consumption)
            base_consumption += social_pressure
        
        # Adjust for inflation
        inflation_adjustment = 1 + self.model.inflation_rate
        self.consumption = base_consumption * inflation_adjustment
        
        # Limit consumption by available resources
        available_resources = self.wealth + self.income - total_tax
        self.consumption = min(self.consumption, available_resources * 0.9)  # Ensure some minimum savings
        
        # Update wealth after consumption and taxes
        self.wealth = self.wealth + self.income - self.consumption - total_tax
        
        # Decide on savings and investments
        # Hyperbolic discounting and risk preferences influence saving vs immediate consumption
        savings_rate = self.time_preference * (1 - self.consumption_preference)
        self.savings = savings_rate * (self.income - total_tax - self.consumption)
        
        # Investment decision - riskier investments offer higher potential returns
        investment_allocation = 1 - self.risk_aversion  # Higher risk tolerance = higher allocation to risky assets
        safe_allocation = 1 - investment_allocation
        
        # Determine investment returns
        safe_return = self.model.interest_rate  # Risk-free rate
        risky_return = np.random.normal(
            self.model.base_return_rate,
            0.15 * (1 + investment_allocation)  # Higher allocation increases volatility
        )
        
        # Calculate total returns on current wealth
        investment_return = (safe_allocation * safe_return) + (investment_allocation * risky_return)
        
        # Apply returns to wealth, with a cap to prevent extreme outliers
        return_cap = 0.30  # Maximum 30% return in a single year
        capped_return = max(min(investment_return, return_cap), -0.5)  # Can't lose more than 50% in a year
        
        self.wealth = self.wealth * (1 + capped_return)
        
        # Decide on borrowing
        if (self.income > 0 and self.credit_score > 650 and 
            self.debt / max(1, self.income) < self.model.max_loan_to_income_ratio and
            np.random.random() < self.model.credit_availability):
            
            # Calculate how much the agent can borrow
            max_additional_debt = (self.model.max_loan_to_income_ratio * self.income) - self.debt
            
            # Agent decides to borrow based on their risk tolerance and opportunity
            borrowing_desire = (1 - self.risk_aversion) * (1 - self.debt / max(1, self.income))
            potential_borrowing = max_additional_debt * borrowing_desire
            
            # Random borrowing event (e.g., for a major purchase or investment)
            if np.random.random() < 0.1:  # 10% chance each year
                loan_amount = np.random.uniform(0, potential_borrowing)
                self.debt += loan_amount
                self.wealth += loan_amount  # Add loan to wealth
                
                # Update credit score (small negative impact from new debt)
                self.credit_score = max(600, self.credit_score - 5)
        
        # Repay debt
        if self.debt > 0:
            # Interest on debt
            interest_payment = self.debt * (self.model.interest_rate + 0.02)  # Interest rate premium
            
            # Determine repayment amount (at least the interest, more if wealthy)
            repayment_capacity = 0.1 * self.income + 0.02 * self.wealth
            repayment = min(self.debt + interest_payment, max(interest_payment, repayment_capacity))
            
            # Check if agent can afford the payment
            if repayment <= self.wealth:
                self.wealth -= repayment
                self.debt = max(0, self.debt + interest_payment - repayment)
                
                # Improve credit score with successful payment
                self.credit_score = min(850, self.credit_score + 2)
            else:
                # Can't afford payment - partial payment and credit score damage
                partial_payment = self.wealth * 0.9  # Use 90% of wealth for payment
                self.wealth -= partial_payment
                self.debt = self.debt + interest_payment - partial_payment
                
                # Damage credit score
                self.credit_score = max(300, self.credit_score - 50)
                
                # Potential default if seriously underwater
                if self.debt > self.income * 2 and np.random.random() < 0.3:
                    # Default - lose some wealth, clear debt, severely damage credit
                    self.wealth = self.wealth * 0.7  # Lose 30% of remaining wealth in bankruptcy
                    self.debt = 0
                    self.credit_score = max(300, self.credit_score - 150)
        
        # Ensure wealth can't go negative
        self.wealth = max(0, self.wealth)
        
        # Update history
        self.wealth_history.append(self.wealth)
        self.income_history.append(self.income)
