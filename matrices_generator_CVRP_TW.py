import pandas as pd
from random import choice
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from functions import haversine

start_time = time.time()

# Chose the date
date = '01/08/2022'

# Coordinates depot
lat_depot = 50.892837
lon_depot = 4.794280

# Convert csv files to dataframes
locations_dataframe = pd.read_csv('locations.csv')
demand_dataframe = pd.read_csv('packages.csv')

# Extract relevant data
filtered_packages_dataframe = demand_dataframe[demand_dataframe.iloc[:, 4].str.contains(date)]
merged_packages_dataframe = pd.merge(locations_dataframe,filtered_packages_dataframe, left_on='LocationID', right_on='LocationID')
aggregated_volumes_packages_dataframe = merged_packages_dataframe.groupby('LocationID').agg({'Volume_cm3': 'sum', 'Long': 'first', 'Lat': 'first'})

# Make a list of the location IDs that need to be visited
location_ids = aggregated_volumes_packages_dataframe.index[:].tolist()
location_ids = ['depot'] + location_ids

# Make a list of the total package volumes to be delivered at the locations
volumes = aggregated_volumes_packages_dataframe.iloc[:, 0].tolist()
volumes = [0] + volumes

# Generate time windows
timewindows = []
# In hours
TotalTimeWindow = 8
TimeWindowPerLocation = 2
num_zones = 4

# Possible start times
possible_start_times = []
for i in range(0, TotalTimeWindow // TimeWindowPerLocation):
    possible_start_times.append(i * 60 * TimeWindowPerLocation)

matrixTimeWindow = []
for i in range(len(location_ids)):
   # Choose a random starting time from the list of possible starting times
   starting_time = choice(possible_start_times)
   timewindows.append((starting_time, starting_time + TimeWindowPerLocation*60))

# Rewrite the coordinates for Valhalla
valhalla_coordinates = [{'lat': float(row[2]), 'lon': float(row[1])} for i, row in aggregated_volumes_packages_dataframe.head(len(location_ids)).iterrows()]
valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}] + valhalla_coordinates

# Create a list of the location IDs and valhalla_coordinates belonging to each zone
zone1_location_ids = ['depot']
zone1_valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}]
zone1_volumes = [0]
zone1_time_windows = [(0, TotalTimeWindow*60)]

zone2_location_ids = ['depot']
zone2_valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}]
zone2_volumes = [0]
zone2_time_windows = [(0, TotalTimeWindow*60)]

zone3_location_ids = ['depot']
zone3_valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}]
zone3_volumes = [0]
zone3_time_windows = [(0, TotalTimeWindow*60)]

zone4_location_ids = ['depot']
zone4_valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}]
zone4_volumes = [0]
zone4_time_windows = [(0, TotalTimeWindow*60)]

ids_zones = [zone1_location_ids, zone2_location_ids, zone3_location_ids, zone4_location_ids]
valhalla_zones = [zone1_valhalla_coordinates, zone2_valhalla_coordinates, zone3_valhalla_coordinates, zone4_valhalla_coordinates]
volumes_zones = [zone1_volumes, zone2_volumes, zone3_volumes, zone4_volumes]

for i in range(len(location_ids)-1):
    # Determine which zone the location belongs to based on its coordinates
    if timewindows[i][1] <= (TotalTimeWindow // num_zones)*1*60:
        zone1_location_ids.append(location_ids[i])
        zone1_valhalla_coordinates.append(valhalla_coordinates[i])
        zone1_volumes.append(volumes[i])
        zone1_time_windows.append(timewindows[i])
    elif (TotalTimeWindow // num_zones)*1*60 < timewindows[i][1] <= (TotalTimeWindow // num_zones)*2*60:
        zone2_location_ids.append(location_ids[i])
        zone2_valhalla_coordinates.append(valhalla_coordinates[i])
        zone2_volumes.append(volumes[i])
        zone2_time_windows.append(timewindows[i])
    elif (TotalTimeWindow // num_zones)*2*60 < timewindows[i][1] <= (TotalTimeWindow // num_zones)*3*60:
        zone3_location_ids.append(location_ids[i])
        zone3_valhalla_coordinates.append(valhalla_coordinates[i])
        zone3_volumes.append(volumes[i])
        zone3_time_windows.append(timewindows[i])
    else:
        zone4_location_ids.append(location_ids[i])
        zone4_valhalla_coordinates.append(valhalla_coordinates[i])
        zone4_volumes.append(volumes[i])
        zone4_time_windows.append(timewindows[i])

print("---------------------------------------------------")
print(f" ** Sizes of the zones on {date} **")
print(f"Time block 1 consists of {len(zone1_location_ids)} locations")
print(f"Time block 2 consists of {len(zone2_location_ids)} locations")
print(f"Time block 3 consists of {len(zone3_location_ids)} locations")
print(f"Time block 4 consists of {len(zone4_location_ids)} locations")
print("---------------------------------------------------")
print("Starting building of matrices ...")

distance_matrix_zone1 = [[1e10] * len(zone1_location_ids) for _ in range(len(zone1_location_ids))]
distance_matrix_zone2 = [[1e10] * len(zone2_location_ids) for _ in range(len(zone2_location_ids))]
distance_matrix_zone3 = [[1e10] * len(zone3_location_ids) for _ in range(len(zone3_location_ids))]
distance_matrix_zone4 = [[1e10] * len(zone4_location_ids) for _ in range(len(zone4_location_ids))]

time_matrix_zone1 = [[1e10] * len(zone1_location_ids) for _ in range(len(zone1_location_ids))]
time_matrix_zone2 = [[1e10] * len(zone2_location_ids) for _ in range(len(zone2_location_ids))]
time_matrix_zone3 = [[1e10] * len(zone3_location_ids) for _ in range(len(zone3_location_ids))]
time_matrix_zone4 = [[1e10] * len(zone4_location_ids) for _ in range(len(zone4_location_ids))]

distance_matrices = [distance_matrix_zone1, distance_matrix_zone2, distance_matrix_zone3, distance_matrix_zone4]
time_matrices = [time_matrix_zone1, time_matrix_zone2, time_matrix_zone3, time_matrix_zone4]
time_windows = [zone1_time_windows, zone2_time_windows, zone3_time_windows, zone4_time_windows]

# Define a function that processes a single location
def process_location(i, loc_id, coord, valhalla_coordinates, location_ids, distance_matrix, time_matrix):
    distances = []
    for j, (other_id, other_coord) in enumerate(zip(location_ids, valhalla_coordinates)):
        if loc_id != other_id:
            # Calculate distance using haversine formula
            distance = haversine(coord['lat'], coord['lon'], other_coord['lat'], other_coord['lon'])
            distances.append((other_id, distance))
    # Sort distances and keep the k closest locations
    distances = [tup for tup in distances if 'depot' not in tup]
    distances.sort(key=lambda x: x[1])
    closest = distances[:k]
    target_location_ids = [t[0] for t in closest]  # extract location IDs from tuples
    target_coordinates = [{'lat': lat_depot, 'lon': lon_depot}]
    index_list = [0]
    for location_id in target_location_ids:
        index = location_ids.index(location_id)
        target_coordinates.append(valhalla_coordinates[index])
        index_list.append(index)

    # Construct request for Valhalla
    request = json.dumps({
        "sources": [{'lat': coord['lat'], 'lon': coord['lon']}],
        "targets": target_coordinates,
        "costing":"auto"
    })

    # Send request
    response = requests.get('http://localhost:8002/sources_to_targets?json=' + request)
    # Extract response
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Request failed with status code {response.status_code}: {response.content}")

    # Populate the matrices with the distances and times
    for l in range(len(index_list)):
        distance_matrix[i][index_list[l]] = data['sources_to_targets'][0][l]['distance']
        time_matrix[i][index_list[l]] = data['sources_to_targets'][0][l]['time']

    distance_matrix[i][i] = 0
    time_matrix[i][i] = 0


for p in range(len(ids_zones)):
    # Number of closest locations to find
    k = 100

    distance_matrices[p] = [[1e10] * len(ids_zones[p]) for _ in range(len(ids_zones[p]))]
    time_matrices[p] = [[1e10] * len(ids_zones[p]) for _ in range(len(ids_zones[p]))]

    threads = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Loop through each location and submit a task to the thread pool
        for i, (loc_id, coord) in enumerate(zip(ids_zones[p], valhalla_zones[p])):
            t = executor.submit(process_location, i, loc_id, coord, valhalla_zones[p], ids_zones[p], distance_matrices[p], time_matrices[p])
            threads.append(t)

    # Wait for all threads to complete
    completed_threads, _ = wait(threads)

    # Check if any threads raised an exception, and raise it if necessary
    for t in completed_threads:
        if t.exception() is not None:
            raise t.exception()

    # Depot request
    # Construct request for Valhalla
    start_time_depot = time.time()

    depot_request_template = json.dumps({
        "sources": [{'lat': lat_depot, 'lon': lon_depot}],
        "targets": [],
        "costing": "auto"
    })

    batch_size = 50  # Set the number of targets to include in each batch
    num_batches = (len(ids_zones[p]) + batch_size - 1) // batch_size  # Compute the number of batches

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(ids_zones[p]))

        # Fill in the targets field of the request template with the appropriate subset of targets
        batch_request = json.loads(depot_request_template)
        batch_request["targets"] = valhalla_coordinates[start_index:end_index]

        # Send request
        response = requests.get('http://localhost:8002/sources_to_targets?json=' + json.dumps(batch_request))

        # Extract response
        if response.status_code == 200:
            batch_data = response.json()
        else:
            print(
                f"Batch {batch_index + 1} of {num_batches} failed with status code {response.status_code}: {response.content}")
            continue

        # Process the results for this batch
        for i in range(start_index, end_index):
            j = i - start_index
            distance_matrices[p][0][i] = batch_data['sources_to_targets'][0][j]['distance']
            time_matrices[p][0][i] = batch_data['sources_to_targets'][0][j]['time']

    # Convert time matrix to whole minutes
    time_matrices[p] = [[round((1 / 60) * element) for element in row] for row in time_matrices[p]]

    print(p+1, "/", len(ids_zones), "of the matrices constructed")


# Save results
#print("Location IDs: ", location_ids)
with open('location_ids.json', 'w') as location_ids_json:
    json.dump(ids_zones, location_ids_json)

#print("Total volumes in cm3: ", volumes)
with open('total_volumes_in_cm3.json', 'w') as volumes_json:
    json.dump(volumes_zones, volumes_json)

#print("Distance matrix in km: ", distance_matrix)
with open('distance_matrix.json', 'w') as distance_matrix_json:
    json.dump(distance_matrices, distance_matrix_json)

#print("Time matrix in seconds: ", time_matrix)
with open('time_matrix.json', 'w') as time_matrix_json:
    json.dump(time_matrices, time_matrix_json)

with open('time_windows.json', 'w') as time_windows_json:
    json.dump(time_windows, time_windows_json)

end_time = time.time()

total_time = end_time - start_time
print("---------------------------------------------------")
print(f"** Complete!: Matrices of the {len(ids_zones)} time blocks constructed **")
print("")
print("Amount of locations processed:", len(location_ids))
print("Total time:", round(total_time), "seconds")
print("---------------------------------------------------")