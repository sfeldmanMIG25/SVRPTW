import numpy as np

# --- 1. PHYSICAL/GEOGRAPHIC CONSTANTS ---
COORDINATE_BOUNDS = (0, 100)
DEPOT_COORDINATE = (50, 50)
DISTANCE_UNIT = "miles"

# --- 2. TIME & COST PARAMETERS (All in MINUTES) ---
TIME_UNIT = "minutes"

# Operating Day Times (8:00 AM to 4:00 PM)
DEPOT_E_TIME = 480 
DEPOT_L_TIME = 960 

# Cost Rates
WAGE_COST_PER_MINUTE = 14.50 / 60.0
TRANSIT_COST_PER_MILE = 0.50

# Penalties
EARLY_WAITING_PENALTY_PER_MINUTE = WAGE_COST_PER_MINUTE
HARD_LATE_PENALTY = 1000.0

# --- 3. STOCHASTIC DISTRIBUTION PARAMETERS ---
# EXTREME VARIABILITY SETTINGS

# Travel Time: Log-Normal(log(mu), sigma^2)
# 0.6 is extremely volatile. A 20-min trip often becomes 45-60 mins.
TRAVEL_TIME_LN_SIGMA = 0.6

# Service Time: Normal(mu, sigma^2)
# Mean is ~10 min, but Sigma is 8 min. 
# Service times will frequently range from 0 to 30+ minutes.
SERVICE_TIME_BASE_MEAN = 10.0 
SERVICE_TIME_SIGMA = 8.0