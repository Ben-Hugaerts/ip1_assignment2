from gurobipy import *
import pandas as pd
from datetime import datetime, timedelta

# With time windows 12h (4x3h) or without time windows 8h
##########################################
time_windows = True
##########################################

# Make lists with the dates when packages need to be delivered and when vehicles have to be charged
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

# Determine the amount of locations that need to be visited on each day
lengths =[]

for date in date_list:
    # Extract relevant data
    filtered_packages_dataframe = cars_dataframe[cars_dataframe.iloc[:, 4].str.contains(date)]
    merged_packages_dataframe = pd.merge(locations_dataframe,filtered_packages_dataframe, left_on='LocationID', right_on='LocationID')
    aggregated_volumes_packages_dataframe = merged_packages_dataframe.groupby('LocationID').agg({'Volume_cm3': 'sum', 'Long': 'first', 'Lat': 'first'})

    location_ids = aggregated_volumes_packages_dataframe.index[:].tolist()
    lengths.append(len(location_ids))

# Determine the amount of vehicles necessary for a certain amount of locations
# Functions are fitted for this to results of the CVRP 8h and CVRP with time windows of 12h,
# both with service time of 2min
cars = []
if time_windows == False:
    for i in range(len(lengths)):
        cars.append(round(2.5 + 0.009*lengths[i]))
        if lengths[i] == 0:
            cars[i] = 0
else:
    for i in range(len(lengths)):
        cars.append(round(3.25 + 0.0137*lengths[i]))
        if lengths[i] == 0:
            cars[i] = 0

# Create a model which minimizes the amount of chargers needed,
# while maximizing the amount of charged vehicles at the depot
m = Model("charging_schedule")

# Create a variable for the maximum chargers needed per day
max_chargers = m.addVar(vtype=GRB.CONTINUOUS, name="max_chargers")

# Create a list of variables for the amount of vehicles charged each day
charg = []
for i in range(len(cars)):
    charg.append(m.addVar(vtype=GRB.INTEGER, name=f"charg_{i+1}"))

# Create a list of variables for the vehicles which remain charged at the depot each day
dep = []
for i in range(len(cars)):
    dep.append(m.addVar(vtype=GRB.INTEGER, name=f"dep_{i+1}"))

# Create a variable for the total number of vehicles that remain charged at the depot over the whole period
total_cars_in_dep = m.addVar(vtype=GRB.INTEGER, name="total_cars_in_dep")

# Add a constraint that prohibits that vehicles are charged which were already fully charged earlier
for i in range(len(cars)):
    m.addConstr(charg[i] + dep[i-1] <= max(cars))

# Add a constraint to calculate the total number of vehicles fully charged at the depot
m.addConstr(total_cars_in_dep == quicksum(dep))

# Set the objective to minimize the maximum amount of chargers used on a day
# A second term is added to maximize the amount of charged vehicles at the depot
m.setObjective(max_chargers - total_cars_in_dep/len(cars), GRB.MINIMIZE)

# Add a constraint that the chargers used on a day must be less than or equal to the maximum chargers used on a day
for i in range(len(cars)):
    m.addConstr(charg[i] <= max_chargers)

# Add a constraint that the vehicles that remain fully charged at the depot each day must be equal to the vehicles that
# remained fully charged at the depot the previous day + the extra vehicles that are charged for that day - the amount
# of vehicles used that day
for i in range(len(cars)):
    if i == 0:
        m.addConstr(dep[i] == charg[i] - cars[i])
    else:
        m.addConstr(dep[i] == dep[i-1] + charg[i] - cars[i])

# Add a constraint that the amount of vehicles that remain fully charged at the depot each day must be non-negative
for i in range(len(cars)):
    m.addConstr(dep[i] >= 0)

# Optimize the model
m.optimize()

# Print the result
# Define the column widths
date_width = 15
charge_width = 20
dep_width = 35
demand_width = 20

# Print the headers
print(f"{'Date':<{date_width}}{'Charging':<{charge_width}}{'Vehicles at depot the next day':<{dep_width}}{'Vehicles needed the next day':<{demand_width}}")
print("-" * (date_width + charge_width + dep_width + demand_width + 3))

# Print the data
for i in range(len(cars)):
    charge_str = f"Charge {int(charg[i].x)} vehicles "
    dep_str = f"{int(dep[i].x)} charged, {int(max(cars) - dep[i].x - cars[i])} decharged "
    demand_str = f"{int(cars[i])} vehicles"
    print(
        f"{charge_date_list[i]:<{date_width}}{charge_str:<{charge_width}}{dep_str:<{dep_width}}{demand_str:<{demand_width}}")

print(f"\nAmount of chargers needed: {int(max_chargers.x)}")
print("")


# This is just a test scenario to show that a vehicle doesn't necessary remain too long in the same state:

# Convert schedule results to lists
dep_values = [v.X for v in dep]
charg_values = [p.X for p in charg]

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
        demand_c = dep_values[i-1]
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


