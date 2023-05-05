import pandas as pd
import math
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from functions import join_adjacent_zones

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

# Rewrite the coordinates for Valhalla
valhalla_coordinates = [{'lat': float(row[2]), 'lon': float(row[1])} for i, row in aggregated_volumes_packages_dataframe.head(len(location_ids)).iterrows()]
valhalla_coordinates = [{'lat': lat_depot, 'lon': lon_depot}] + valhalla_coordinates

# Create a list of the location IDs and valhalla_coordinates belonging to each zone
zone1_location_ids = []
zone1_valhalla_coordinates = []
zone1_volumes = []

zone2_location_ids = []
zone2_valhalla_coordinates = []
zone2_volumes = []

zone3_location_ids = []
zone3_valhalla_coordinates = []
zone3_volumes = []

zone4_location_ids = []
zone4_valhalla_coordinates = []
zone4_volumes = []

zone5_location_ids = []
zone5_valhalla_coordinates = []
zone5_volumes = []

# Divide the coordinates into the zones
for i in range(1, len(location_ids)):
    # Get the coordinates of the location
    lat_location = valhalla_coordinates[i]['lat']
    lon_location = valhalla_coordinates[i]['lon']

    # Determine which zone the location belongs to based on its coordinates
    if lat_location >= lat_depot and lon_location >= lon_depot:
        zone1_location_ids.append(location_ids[i])
        zone1_valhalla_coordinates.append(valhalla_coordinates[i])
        zone1_volumes.append(volumes[i])
    elif lat_location >= lat_depot and lon_location < lon_depot:
        zone2_location_ids.append(location_ids[i])
        zone2_valhalla_coordinates.append(valhalla_coordinates[i])
        zone2_volumes.append(volumes[i])
    elif lat_location < lat_depot and lon_location < lon_depot:
        # Divide the zone of Leuven in North and South
        if lat_location > 50.871567:
            zone3_location_ids.append(location_ids[i])
            zone3_valhalla_coordinates.append(valhalla_coordinates[i])
            zone3_volumes.append(volumes[i])
        else:
            zone4_location_ids.append(location_ids[i])
            zone4_valhalla_coordinates.append(valhalla_coordinates[i])
            zone4_volumes.append(volumes[i])
    else:
        zone5_location_ids.append(location_ids[i])
        zone5_valhalla_coordinates.append(valhalla_coordinates[i])
        zone5_volumes.append(volumes[i])

# Group the zones in lists
ids_zones = [zone1_location_ids, zone2_location_ids, zone3_location_ids, zone4_location_ids, zone5_location_ids]
valhalla_zones = [zone1_valhalla_coordinates, zone2_valhalla_coordinates, zone3_valhalla_coordinates, zone4_valhalla_coordinates, zone5_valhalla_coordinates]
volumes_zones = [zone1_volumes, zone2_volumes, zone3_volumes, zone4_volumes, zone5_volumes]

# Adjacent zones with few locations are joined together
ids_zones, joined_zones = join_adjacent_zones(ids_zones, 300)
valhalla_zones, joined_zones = join_adjacent_zones(valhalla_zones, 300)
volumes_zones, joined_zones = join_adjacent_zones(volumes_zones, 300)

for i in range(len(ids_zones)):
    ids_zones[i] = ['depot'] + ids_zones[i]
    valhalla_zones[i] = [{'lat': lat_depot, 'lon': lon_depot}] + valhalla_zones[i]
    volumes_zones[i] = [0] + volumes_zones[i]

print("---------------------------------------------------")
print(f" ** Sizes of the zones on {date} **")
for i in range(len(joined_zones)):
    if len(joined_zones[i]) == 1:
        print(f"Zone {joined_zones[i][0]} consists of {len(ids_zones[i])} locations")
    else:
       print(f"Zones {', '.join(map(str,sorted(joined_zones[i])))} are joined together and combined consist of {len(ids_zones[i])} locations")
print("---------------------------------------------------")
print("Starting building of matrices ...")

distance_matrices = []
time_matrices = []

for zone in ids_zones:
    size = len(zone)
    distance_matrix = [[1e10] * size for _ in range(size)]
    time_matrix = [[1e10] * size for _ in range(size)]
    distance_matrices.append(distance_matrix)
    time_matrices.append(time_matrix)

def haversine(lat1, lon1, lat2, lon2):
    # This function calculates the distance as the crow flies between two coordinates

    R = 6371  # Earth radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R*c*1000  # Distance in meters


def process_location(i, loc_id, coord, valhalla_coordinates, location_ids, distance_matrix, time_matrix):
    # This function calculates the distance and time over real routes between one origin and k destinations

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
    # Loop over the zones

    # Number of closest locations to find
    k = 100

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

end_time = time.time()

total_time = end_time - start_time
print("---------------------------------------------------")
if len(joined_zones) > 1:
    print(f"** Complete!: Matrices of the {len(joined_zones)} zones constructed **")
else:
    print(f"** Complete!: Matrix of the zone constructed **")
print("")
print("Amount of locations processed:", len(location_ids))
print("Total time:", round(total_time), "seconds")
print("---------------------------------------------------")