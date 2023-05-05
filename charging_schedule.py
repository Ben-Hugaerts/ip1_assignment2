from gurobipy import *
import pandas as pd
from datetime import datetime, timedelta

start_date = datetime.strptime('19/07/2022', '%d/%m/%Y')
end_date = datetime.strptime('26/08/2022', '%d/%m/%Y')

date_list = []
while start_date <= end_date:
    date_list.append(start_date.strftime('%d/%m/%Y'))
    start_date += timedelta(days=1)

day_before_start_date = datetime.strptime('18/07/2022', '%d/%m/%Y')
day_before_end_date = datetime.strptime('25/08/2022', '%d/%m/%Y')

charge_date_list = []
while day_before_start_date <= day_before_end_date:
    charge_date_list.append(day_before_start_date.strftime('%d/%m/%Y'))
    day_before_start_date += timedelta(days=1)

# Convert csv files to dataframes
locations_dataframe = pd.read_csv('locations.csv')
cars_dataframe = pd.read_csv('packages.csv')

lengths =[]

for date in date_list:
    # Extract relevant data
    filtered_packages_dataframe = cars_dataframe[cars_dataframe.iloc[:, 4].str.contains(date)]
    merged_packages_dataframe = pd.merge(locations_dataframe,filtered_packages_dataframe, left_on='LocationID', right_on='LocationID')
    aggregated_volumes_packages_dataframe = merged_packages_dataframe.groupby('LocationID').agg({'Volume_cm3': 'sum', 'Long': 'first', 'Lat': 'first'})

    location_ids = aggregated_volumes_packages_dataframe.index[:].tolist()
    lengths.append(len(location_ids))

cars = []

for i in range(len(lengths)):
    cars.append(round(2.5 + 0.0069*lengths[i]))
    if lengths[i] == 0:
        cars[i] = 0

# Create a model
m = Model("charging_schedule")

# Create a variable for the maximum production per day
max_prod = m.addVar(vtype=GRB.CONTINUOUS, name="max_prod")

# Create a list of variables for the production per day
prod = []
for i in range(len(cars)):
    prod.append(m.addVar(vtype=GRB.INTEGER, name=f"prod_{i+1}"))

# Create a list of variables for the inventory per day
inv = []
for i in range(len(cars)):
    inv.append(m.addVar(vtype=GRB.INTEGER, name=f"inv_{i+1}"))

# Create a variable for the total number of cars in inventory/depot
total_cars_in_inv = m.addVar(vtype=GRB.INTEGER, name="total_cars_in_inv")

# Create a variable for the total number of decharged cars in inventory/depot
total_cars_in_inv_d = m.addVar(vtype=GRB.INTEGER, name="total_cars_in_inv_d")

for i in range(len(cars)):
    m.addConstr(prod[i] + inv[i-1] <= max(cars))

# Add a constraint to calculate the total number of cars in inventory/depot
m.addConstr(total_cars_in_inv == quicksum(inv))

# Add a constraint to calculate the total number of cars in inventory/depot
m.addConstr(total_cars_in_inv_d == max(cars)*len(cars)-quicksum(inv)-quicksum(cars))

# Set the objective to minimize the maximum production per day
m.setObjective(max_prod, GRB.MINIMIZE)
# m.setObjective(max_prod + (total_cars_in_inv + total_cars_in_inv_d)/(2 * sum(cars)), GRB.MINIMIZE)
#   --> This also works and can lower the time a vehicle remains at the depot

# Add a constraint that the production per day must be less than or equal to the maximum production per day
for i in range(len(cars)):
    m.addConstr(prod[i] <= max_prod)

# Add a constraint that the inventory per day must be equal to the previous inventory plus the production minus the cars
for i in range(len(cars)):
    if i == 0:
        m.addConstr(inv[i] == prod[i] - cars[i])
    else:
        m.addConstr(inv[i] == inv[i-1] + prod[i] - cars[i])

# Add a constraint that the inventory per day must be non-negative
for i in range(len(cars)):
    m.addConstr(inv[i] >= 0)

# Optimize the model
m.optimize()

# Define the column widths
date_width = 15
charge_width = 20
inv_width = 35
demand_width = 20

# Print the headers
print(f"{'Date':<{date_width}}{'Charging':<{charge_width}}{'Vehicles at depot the next day':<{inv_width}}{'Vehicles needed the next day':<{demand_width}}")
print("-" * (date_width + charge_width + inv_width + demand_width + 3))

# Print the data
for i in range(len(cars)):
    charge_str = f"Charge {int(prod[i].x)} vehicles "
    inv_str = f"{int(inv[i].x)} charged, {int(max(cars) - inv[i].x - cars[i])} decharged "
    demand_str = f"{int(cars[i])} vehicles"
    print(
        f"{charge_date_list[i]:<{date_width}}{charge_str:<{charge_width}}{inv_str:<{inv_width}}{demand_str:<{demand_width}}")

print(f"\nAmount of chargers needed: {int(max_prod.x)}")
print("")


# This is just a test scenario to show that a vehicle doesn't necessary remain too long in the same state:

# Convert schedule results to lists
inv_values = [v.X for v in inv]
prod_values = [p.X for p in prod]

# List containing usage of each vehicle for every day
states = [['d'] * max(cars)] * (len(cars) + 1)

new_states = [list(states[0])] + [None] * (len(states) - 1)

for i in range(1, len(states)):
    new_states[i] = list(states[i])  # create a new list for each day's state
    demand = cars[i-1]
    demand_assigned = 0
    demand_assigned_c = 0
    if 'u' in new_states[i - 1]:  # use the new list for the previous day's state
        index_u = len(new_states[i - 1]) - new_states[i - 1][::-1].index('u') - 1
        if index_u == len(new_states[i]) - 1:
            for j, elem in enumerate(new_states[i-1]):
                if elem != 'u':
                    insert_index = j
                    break
        else:
            insert_index = index_u + 1
    else:
        insert_index = 0
    while demand_assigned < demand:
        new_states[i][insert_index] = 'u'
        demand_assigned += 1
        insert_index = (insert_index + 1) % len(new_states[i])

    if demand_assigned == demand:
        demand_c = inv_values[i-1]
        if demand_c > 0:
            if 'c' in new_states[i - 1]:  # use the new list for the previous day's state
                index_c = new_states[i - 1].index('c')
            else:
                index_c = len(new_states[i]) - new_states[i][::-1].index('u')
                if index_c > len(new_states[i]) - 1:
                    index_c = 0
            while demand_assigned_c < demand_c:
                if new_states[i][index_c] == 'u':
                    index_c = (index_c + 1) % len(new_states[i])
                else:
                    new_states[i][index_c] = 'c'
                    demand_assigned_c += 1
                    index_c = (index_c + 1) % len(new_states[i])

states = new_states
trans_states = list(zip(*states))

# initialize a dictionary to store the longest sequence for each state
longest_seqs = {'u': 0, 'd': 0, 'c': 0}

# iterate through each state for each vehicle
for vehicle in trans_states:
    for i in range(len(vehicle)):
        state = vehicle[i]
        if i == 0 or state != vehicle[i-1]:
            # reset current sequence length to 1 if state changes
            current_seq_length = 1
        else:
            # increment current sequence length if state matches previous state
            current_seq_length += 1
        # update longest sequence length for current state if current sequence is longer
        longest_seqs[state] = max(longest_seqs[state], current_seq_length)

# print the result for each state
print("The following results are for test purposes only:")
for state, longest_seq in longest_seqs.items():
    if state == 'c':
        print(f"The longest time a vehicle remains at the depot fully charged is {longest_seq} days")
    if state == 'd':
        print(f"The longest time a vehicle remains at the depot decharged is {longest_seq} days")
    if state == 'u':
        print(f"The longest time a vehicle is used consecutively is {longest_seq} days")


