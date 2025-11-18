import numpy as np

# --- 1. PHYSICAL/GEOGRAPHIC CONSTANTS ---
# Standard Euclidean 2D coordinates will be used.
COORDINATE_BOUNDS = (0, 100)
DEPOT_COORDINATE = (50, 50)
DISTANCE_UNIT = "miles"

# --- 2. TIME & COST PARAMETERS (All in MINUTES) ---
# Time is the primary unit for simulation
TIME_UNIT = "minutes"

# Cost Rates
# Wage Cost: $7.25/hr * 2 crew members = $14.50/hr -> $0.241666.../min
WAGE_COST_PER_MINUTE = 14.50 / 60.0
# Transit Cost: $5/gallon / 10 mpg = $0.50/mile
TRANSIT_COST_PER_MILE = 0.50

# Penalties
# Early arrival penalty is implicit: the cost of crew sitting idle
EARLY_WAITING_PENALTY_PER_MINUTE = WAGE_COST_PER_MINUTE
# Hard late penalty: incurred if arrival > L_i (latest time window) and service is denied.
HARD_LATE_PENALTY = 1000.0

# --- 3. STOCHASTIC DISTRIBUTION PARAMETERS ---
# Travel Time Distribution: Log-Normal(log(mu), sigma^2)
# mu_ij is the actual Euclidean distance
TRAVEL_TIME_LN_SIGMA = 0.2

# Service Time Distribution: Normal(mu, sigma^2)
# The mean service time will be calculated dynamically based on the time window,
# but we need a baseline sigma for the distribution.
SERVICE_TIME_BASE_MEAN = 10.0 # minutes
# Set sigma such that a 3-sigma event is rare relative to a typical 60 min window duration
SERVICE_TIME_SIGMA = 3.0