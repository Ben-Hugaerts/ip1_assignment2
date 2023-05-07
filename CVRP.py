"""Capacited Vehicles Routing Problem (CVRP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json

"""Parameters"""
titleResultfile = 'CVRP_15_08_2022.txt'  # Title of the resultfile

# Vehicle:
capacity_LCV = 3500000 * 0.8   # Assume 80% of the vehicles volume (cm^3) can be used due to packing
max_range_LCV = 185             # Maximum range vehicle in km
charging_time = 5 * 60         # charging time at depot in min for 0 to 100%

# Costs
real_vehicle_cost = 40000       # Cost for 1 vehicle in EUR
cost_driven_km = 0.3            # Cost to drive 1 km in EUR

# parameters algorithm
max_time_per_vehicle = 8*60     # Max time that a vehicle can drive in min
num_vehicles = 12               # # Maximum number of vehicles solver can take in account per zone
searchLimit_sec = 10            # Maximum search limit in seconds per zone

# Delivery
service_time = 2                # Service time in min: time needed for the courier to deliver the package


"""Make a file to write the results"""

# resultfile will be use to write the result in a .txt file
resultfile = ""
resultfile += str("Parameters:\n")
resultfile += str("   Vehicle:\n")
resultfile += str("      capacity: {}m^3 \n      max range: {}km \n      charging time: {}h\n").format(round(capacity_LCV / (0.8*1000000), 2), max_range_LCV, charging_time / 60)
resultfile += str("   Costs:\n")
resultfile += str("      vehicle cost : {}EUR \n      cost per km: {}EUR\n").format(real_vehicle_cost, cost_driven_km)
resultfile += str("   Parameters algorithm:\n")
resultfile += str("      max time per vehicle: {}min \n      num_vehicle: {} \n      searchLimit_sec: {}\n").format(
    max_time_per_vehicle, num_vehicles, searchLimit_sec)
resultfile += str("   Delivery:\n")
resultfile += str("      Service time : {}min \n").format(service_time)

resultfile += str("\n \n")


"""Intermediate calculation data"""

adjusted_cost_driven_km = 1
adjusted_vehicle_cost = real_vehicle_cost * (adjusted_cost_driven_km / cost_driven_km)
with open('inputs_CVRP/15_08_2022/distance_matrix.json', 'r') as distance_matrix_file:
    all_distance_matrices = json.load(distance_matrix_file)
    num_zones = len(all_distance_matrices)  # Read the number of zones used



"""Variables to store results"""

total_vehicles_used = 0
total_distance_all_zones = 0


def create_data_model(num_vehicles, zone):
    """Stores the data for the problem."""
    data = {}
    with open('inputs_CVRP/15_08_2022/distance_matrix.json', 'r') as distance_matrix_file:
        distance_matrices = json.load(distance_matrix_file)
        data['distance_matrix'] = distance_matrices[zone]

    with open('inputs_CVRP/15_08_2022/total_volumes_in_cm3.json', 'r') as total_volumes_file:
        volumes = json.load(total_volumes_file)
        data['demands'] = volumes[zone]

    with open('inputs_CVRP/15_08_2022/time_matrix.json', 'r') as time_matrix_file:
        time_matrices = json.load(time_matrix_file)
        # Add service times to the routes
        for p in range(len(time_matrices)):
            for r in range(len(time_matrices[p])):
                for c in range(len(time_matrices[p])):
                    time_matrices[p][r][c] += service_time
        data['time_matrix'] = time_matrices[zone]

    with open('inputs_CVRP/15_08_2022/location_ids.json', 'r') as locations_file:
        location_ids_all = json.load(locations_file)
        data['locations_ids'] = location_ids_all[zone]

    data['vehicle_capacities'] = [capacity_LCV] * num_vehicles
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0

    return data


def print_solution(data, manager, routing, solution, zone):
    info_resultfile = "\n"
    """Prints solution on console."""
    print("---------------------------------------------------")
    info_resultfile+="\n---------------------------------------------------\n"
    print("** Zone", zone + 1, "**")
    info_resultfile+="** Zone" + str(zone + 1)+ "**\n"
    print(f'Objective: {solution.ObjectiveValue()}')
    info_resultfile+="Objective: {}\n".format({solution.ObjectiveValue()})
    num_cars_used = 0
    total_distance = 0
    total_load = 0
    total_time = 0
    informationReduceVehicle = []

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format((vehicle_id + 1))
        route_distance = 0
        route_load = 0
        route_time = 0  # add a variable to track time
        node_from = 0  # to calculate the travel time between the customers for each arc
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} & Load({1}) -> '.format(data['locations_ids'][node_index], round(route_load / 1000000, 3))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            # calculate time taken to travel from previous_index to index
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_to = node_index
            route_time += data['time_matrix'][node_from][node_to]  # update route_time
            node_from = node_to  # to know for next iteration what the node before was
            # print(route_distance,round(route_time/60))

        if route_distance > 0:  # when a route is used by a vehicle we subtract the cost of the vehicle to then calculate the real objective in which te cost of a vehicle is equal to the actual cost of 60000
            route_distance += -200000

        route_distance += routing.GetArcCostForVehicle(previous_index, routing.Start(vehicle_id), vehicle_id)
        route_time += data['time_matrix'][node_from][0]  # update route_time to going back to depot
        informationReduceVehicle.append([route_distance, route_time])

        if route_distance > 0:  # only print information for used vehicles
            plan_output += ' & Load({0})\n'.format(round(route_load / 1000000, 3))
            # plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),solution.Min(time_var), solution.Max(time_var))
            plan_output += 'Distance of the route: {}km\n'.format(route_distance)
            plan_output += 'Load of the route: {}m^3\n'.format(round(route_load / 1000000, 3))
            plan_output += 'Time driven for the route: {}min\n'.format(round(route_time))
            print(plan_output)
            info_resultfile += plan_output
            num_cars_used += 1
            total_distance += (route_distance)
            total_load += route_load
            total_time += route_time

    real_objective = num_cars_used * real_vehicle_cost + cost_driven_km * total_distance  # Real objective has coefficients of 60000 for the vehicles used and 0.3 euro per km driven
    info_resultfile+='\nReal objective: {}\n'.format(real_objective)
    print('Real objective:', real_objective)
    info_resultfile += 'Number of cars used: {}\n'.format(num_cars_used)
    print('Number of cars used: {}'.format(num_cars_used))
    info_resultfile +='Total distance of all routes: {}km\n'.format(total_distance)

    print('Total distance of all routes: {}km'.format(total_distance))
    info_resultfile +='Total load of all routes: {}m^3\n'.format(round(total_load / 1000000, 3))
    print('Total load of all routes: {}m^3'.format(round(total_load / 1000000, 3)))
    info_resultfile +='Total time of all routes: {}min\n'.format(total_time)
    print('Total time of all routes: {}min'.format(total_time))
    return informationReduceVehicle, num_cars_used, total_distance,info_resultfile


def main(zone):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(num_vehicles, zone)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    adjusted_vehicle_cost = 200000  # vehicle cost of 200000 because the cost of 1 driven km is 1 and by setting the cost to 200000 we have the same ratio as 60000 to 0.3
    for vehicle_idx in range(data['num_vehicles']):
        routing.SetFixedCostOfVehicle(adjusted_vehicle_cost, vehicle_idx)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_range_LCV,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(
        adjusted_cost_driven_km)  # the cost of 1 driven km equals to 1 because it can't be set to 0.3 because it needs to be an integer

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index_time = routing.RegisterTransitCallback(time_callback)

    time = 'Time'
    routing.AddDimension(
        transit_callback_index_time,
        0,  # allow waiting time
        max_time_per_vehicle,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.FromSeconds(1) #zelf weggedaan
    search_parameters.time_limit.seconds = searchLimit_sec  # zelf toegevoegd

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        informationReduceVehicle, num_cars_used, total_distance, info_resultfile = print_solution(data, manager, routing, solution, zone)

        info_resultfile+="\n---------------------------------------------------\n"
        print("---------------------------------------------------")
        print("")

        return num_cars_used, total_distance, informationReduceVehicle, info_resultfile


def reduceNumberOfVehiclesAllZones(informationReduceVehicleAllZones):
    # informationReduceVehicleAllZones: matrix with per zone: per row information about the vehicle:
    #       [total_distance,driving_time_vehicle]

    num_cars_reduced = 0
    combined_cars = []
    possible_cars_to_combined = []
    for i in range(num_zones):
        possible_cars_to_combined.append([*range(num_vehicles)])  # Keeps track of which vehicles can still be combined

    # If car is not used, remove from possible_cars_to_combined
    for zone_i in range(num_zones):
        for vehicle_j in range(num_vehicles):
            distance = informationReduceVehicleAllZones[zone_i][vehicle_j][0]
            if distance == 0:
                possible_cars_to_combined[zone_i].remove(vehicle_j)

    # Try to combine cars
    for zone_i in range(num_zones):
        for vehicle_A in range(len(informationReduceVehicleAllZones[zone_i])):

            # Check if vehicle_A is still in possible_cars_to_combined
            if vehicle_A in possible_cars_to_combined[zone_i]:
                # If vehilcle_A can be combined, try to combine
                [total_distance_vehicle_A, driving_time_vehicle_A] = informationReduceVehicleAllZones[zone_i][vehicle_A]

                TOT_distance = total_distance_vehicle_A
                TOT_time = driving_time_vehicle_A
                TOT_combined = [(zone_i, vehicle_A)]
                num_combined = 1

                # Try to combine vehicle_A with as many cars as possible
                zone_j = 0
                vehicle_B = 0
                while zone_j != num_zones:
                    # Check if vehicle_B is still in possible_cars_to_combined:
                    if vehicle_B in possible_cars_to_combined[zone_j]:
                        [total_distance_vehicle_B, driving_time_vehicle_B] = informationReduceVehicleAllZones[zone_j][
                            vehicle_B]

                        # check if time is OK to put vehicle_B after the other vehicle(s) & vehicles A and vehicle B are not the same vehicle
                        if ((TOT_time + driving_time_vehicle_B) < max_time_per_vehicle) and (
                                (zone_i, vehicle_A) != (zone_j, vehicle_B)):

                            # Calculate how long you can charge
                            time_charging_between_routes = min(
                                (max_time_per_vehicle - TOT_time - driving_time_vehicle_B), charging_time)
                            # Calculate how much extra distance you can do by charging
                            charging_distance = (time_charging_between_routes / charging_time) * max_range_LCV
                            # Calculate what the maximum distance is that you can drive
                            max_driving_distance = max_range_LCV + charging_distance

                            # Calculate if the 2 routes can be done with 1 vehicle:
                            if (TOT_distance + total_distance_vehicle_B) <= max_driving_distance:
                                num_cars_reduced += 1
                                num_combined += 1
                                TOT_combined.append((zone_j, vehicle_B))
                                TOT_time += driving_time_vehicle_B
                                TOT_distance += total_distance_vehicle_B
                                possible_cars_to_combined[zone_j].remove(vehicle_B)

                    # Go to next vehicle
                    if vehicle_B != (num_vehicles - 1):
                        vehicle_B += 1
                    else:
                        vehicle_B = 0
                        zone_j += 1

                possible_cars_to_combined[zone_i].remove(vehicle_A)
                if num_combined > 1:
                    combined_cars.append((num_combined, TOT_combined, TOT_distance, TOT_time))
    return num_cars_reduced, combined_cars


if __name__ == '__main__':
    informationReduceVehicleAllZones = []
    # loop over the zones
    for i in range(num_zones):
        num_cars_used, total_distance, informationReduceVehicle, inforesultFile = main(i)
        total_vehicles_used += num_cars_used
        total_distance_all_zones += total_distance
        informationReduceVehicleAllZones.append(informationReduceVehicle)
        resultfile+=inforesultFile
        if i == num_zones - 1:
            print('Total distance of all zones: {}km'.format(total_distance_all_zones))
            resultfile+='\nTotal distance of all zones: {}km\n'.format(total_distance_all_zones)
            print("Total amount of vehicles used before reducing:", total_vehicles_used)
            resultfile+='Total amount of vehicles used before reducing: {}\n'.format(total_vehicles_used)

    num_cars_reducedAllZones, combined_carsAllZones = reduceNumberOfVehiclesAllZones(informationReduceVehicleAllZones)
    print("Total amount of vehicles reduced: ", num_cars_reducedAllZones)
    resultfile+='Total amount of vehicles reduced: {}\n'.format(num_cars_reducedAllZones)
    print("Total amount of vehicles used after reducing: ", total_vehicles_used - num_cars_reducedAllZones)
    resultfile+='Total amount of vehicles used after reducing: {}\n'.format(total_vehicles_used - num_cars_reducedAllZones)
    print("Vehicles combined: ")
    resultfile+='Vehicles combined: \n'
    for k in range(len(combined_carsAllZones)):
        cumul_dist = 0
        (num_combined, TOT_combined, TOT_distance, TOT_time) = combined_carsAllZones[k]
        print('  Combination of {} vehicles: total distance {}km & total time {}min'.format(num_combined, TOT_distance,
                                                                                            round(TOT_time)))
        resultfile+='  Combination of {} vehicles: total distance {}km & total time {}min\n'.format(num_combined, TOT_distance,
                                                                                            round(TOT_time))
        for j in range(num_combined):
            (zone_A, vehicle_A) = TOT_combined[j]
            [total_distance_vehicle_A, driving_time_vehicle_A] = informationReduceVehicleAllZones[zone_A][vehicle_A]
            cumul_dist += total_distance_vehicle_A
            if cumul_dist >= max_range_LCV:
                print(f"     * Charging of vehicle required for at least {int((charging_time/max_range_LCV) * (TOT_distance - max_range_LCV))} min * ")
                resultfile += f"     * Charging of vehicle required for at least {int((charging_time/max_range_LCV) * (TOT_distance - max_range_LCV))} min * \n"
                cumul_dist = 0
            print(
                '     Vehicle (zone {},{}): distance: {}km & driving time: {}min'.format((zone_A + 1), (vehicle_A + 1),
                                                                                         total_distance_vehicle_A,
                                                                                         round(driving_time_vehicle_A)))
            resultfile+='     Vehicle (zone {},{}): distance: {}km & driving time: {}min\n'.format((zone_A + 1), (vehicle_A + 1),
                                                                                         total_distance_vehicle_A,
                                                                                         round(driving_time_vehicle_A))


location = 'Results/' + titleResultfile
with open(location, 'w') as file:
    file.write(resultfile)
file.close()