import numpy as np
from typing import Dict, Any

def calculate_tax_comparison(income: float, mortgage_interest: float, property_tax: float, 
                             charitable_donations: float, state_local_tax: float,
                             child_tax_credit: float = 2000) -> Dict[str, Any]:
    # ... (use the existing calculate_tax_comparison function from your code)
    # For brevity, I'm not repeating the entire function here
    pass

def should_buy_house(income: float, down_payment: float, home_price: float, 
                     interest_rate: float, property_tax_rate: float, 
                     maintenance_rate: float, rent: float, 
                     expected_appreciation: float, time_horizon: int) -> Dict[str, Any]:
    """
    Determine if buying a house is financially beneficial compared to renting.
    
    Parameters:
    income: Annual household income
    down_payment: Available down payment
    home_price: Price of the house
    interest_rate: Annual mortgage interest rate (as a decimal)
    property_tax_rate: Annual property tax rate (as a decimal)
    maintenance_rate: Annual maintenance cost as a percentage of home value (as a decimal)
    rent: Monthly rent if not buying
    expected_appreciation: Expected annual home value appreciation rate (as a decimal)
    time_horizon: Number of years planning to stay in the house
    
    Returns: A dictionary with the analysis results
    """
    # Calculate mortgage details
    loan_amount = home_price - down_payment
    monthly_mortgage_rate = interest_rate / 12
    num_payments = time_horizon * 12
    monthly_mortgage = loan_amount * (monthly_mortgage_rate * (1 + monthly_mortgage_rate)**num_payments) / ((1 + monthly_mortgage_rate)**num_payments - 1)
    
    # Calculate annual costs
    annual_mortgage = monthly_mortgage * 12
    annual_property_tax = home_price * property_tax_rate
    annual_maintenance = home_price * maintenance_rate
    annual_housing_cost = annual_mortgage + annual_property_tax + annual_maintenance
    
    # Calculate tax benefits
    mortgage_interest = loan_amount * interest_rate  # Simplified calculation
    tax_comparison = calculate_tax_comparison(income, mortgage_interest, annual_property_tax, 0, 0)
    tax_savings = tax_comparison['Difference'] if tax_comparison['Better Option'] == 'Itemized Deduction' else 0
    
    # Calculate total cost of buying vs renting
    total_buy_cost = (annual_housing_cost - tax_savings) * time_horizon
    total_rent_cost = rent * 12 * time_horizon
    
    # Calculate expected home value after time horizon
    final_home_value = home_price * (1 + expected_appreciation)**time_horizon
    
    # Calculate net cost of buying (considering selling the house)
    net_buy_cost = total_buy_cost + down_payment - (final_home_value - loan_amount)
    
    return {
        "Buy House": net_buy_cost < total_rent_cost,
        "Net Cost of Buying": net_buy_cost,
        "Total Rent Cost": total_rent_cost,
        "Difference": total_rent_cost - net_buy_cost,
        "Annual Tax Savings": tax_savings,
        "Expected Home Value After {} Years".format(time_horizon): final_home_value
    }

# Example usage
result = should_buy_house(
    income=250000,
    down_payment=100000,
    home_price=500000,
    interest_rate=0.06,
    property_tax_rate=0.01,
    maintenance_rate=0.01,
    rent=3000,
    expected_appreciation=0.03,
    time_horizon=10
)

print(result)
