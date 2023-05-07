# TO DO:
#   - auto's laten opladen


"""Capacited Vehicles Routing Problem with Time Window (CVRPTW) """
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json

"""Parameters"""

titleResultfile = 'CVRP_TW 15_08_2022.txt'  # Title of the resultfile
# Vehicle:
capacity_LCV = 3500000 * 0.8  # Assume 80% of the vehicles volume (cm^3) can be used due to packing
max_range_LCV = 185  # Maximum range LCV in km
charging_time = 5 * 60  # charging time at depot in min for 0 to 100%
# Costs
real_vehicle_cost = 40000  # Cost for 1 vehicle in EUR
cost_driven_km = 0.3  # Cost to drive 1 km in EUR
# parameters algorithm
max_time_per_vehicle = 720  # Max time that a LCV can drive
num_vehicles = 50 # Maximum number of vehicles solver can take in account per zone
searchLimit_sec = 20  # Maximum search limit in seconds per zone
# Delivery
service_time = 2

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
resultfile += str("\n \n")

"""Intermediate calculation data"""

adjusted_cost_driven_km = 1
adjusted_vehicle_cost = real_vehicle_cost * (adjusted_cost_driven_km / cost_driven_km)
with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/distance_matrix.json', 'r') as distance_matrix_file:
    all_distance_matrices = json.load(distance_matrix_file)
    num_zones = len(all_distance_matrices)  # Read the number of zones used

"""Variables to store results"""

total_vehicles_used = 0
total_distance_all_zones = 0


def create_data_model(num_vehicles, zone):
    """Stores the data for the problem."""
    data = {}
    with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/distance_matrix.json', 'r') as distance_matrix_file:
        distance_matrices = json.load(distance_matrix_file)
        data['distance_matrix'] = distance_matrices[zone]

    with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/total_volumes_in_cm3.json', 'r') as total_volumes_file:
        volumes = json.load(total_volumes_file)
        data['demands'] = volumes[zone]

    with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/time_matrix.json', 'r') as time_matrix_file:
        time_matrices = json.load(time_matrix_file)
        # Add service times to the routes
        for p in range(len(time_matrices)):
            for r in range(len(time_matrices[p])):
                for c in range(len(time_matrices[p])):
                    time_matrices[p][r][c] += service_time
        data['time_matrix'] = time_matrices[zone]

    with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/time_windows.json', 'r') as time_windows_file:
        time_windows = json.load(time_windows_file)
        data['time_windows'] = time_windows[zone]

    with open('inputs_CVRP_TW/12h (4x3h)/15_08_2022/location_ids.json', 'r') as locations_file:
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
    total_distance = 0
    total_load = 0
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    num_cars_used = 0
    informationReduceVehicle = []

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = '\nRoute for vehicle {}:\n'.format(vehicle_id + 1)
        route_distance = 0
        route_load = 0
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})'.format(
            data['locations_ids'][manager.IndexToNode(index)], solution.Min(time_var),
            solution.Max(time_var))
        start_time_vehicle = solution.Max(time_var)  # nieuw    #Last possible start time
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += 'Load({0}) -> '.format(round(route_load / 1000000, 3))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if routing.GetArcCostForVehicle(previous_index, index, vehicle_id) > 0:
                route_distance += (routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id))
            else:
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            # route_distance += routing.GetArcCostForVehicle(
            #     previous_index, index, vehicle_id)
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})'.format(
                data['locations_ids'][manager.IndexToNode(index)], solution.Min(time_var),
                solution.Max(time_var))
            # plan_output += '& TimeWindow({0}) '.format(matrixTimeWindow[node_index])  #zelf: Toevoegen van TimeWindows aan output: printen
            # index = solution.Value(routing.NextVar(index)))

        if route_distance > 0:  # when a route is used by a vehicle we subtract the cost of the vehicle to then calculate the real objective in which te cost of a vehicle is equal to the actual cost of 60000
            route_distance += -200000  # vehicle cost of 200000 because the cost of 1 driven km is 1 and by setting the cost to 200000 we have the same ratio as 60000 to 0.3

            plan_output += ' & Load({0})\n'.format(round(route_load / 1000000, 3))
            # plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),solution.Min(time_var), solution.Max(time_var))
            plan_output += 'Distance of the route: {}km\n'.format(route_distance)
            plan_output += 'Load of the route: {}m^3\n'.format(round(route_load / 1000000, 3))
            time_var = time_dimension.CumulVar(index)
            plan_output += 'Start time of the route: {}min\n'.format(start_time_vehicle)
            plan_output += 'End Time of the route: {}min\n'.format(solution.Min(time_var))
            print(plan_output)
            info_resultfile += plan_output
            total_distance += route_distance
            total_load += route_load
            total_time += solution.Min(time_var)
            num_cars_used += 1

        end_time_vehicle = solution.Min(time_var)
        informationReduceVehicle.append([route_distance, start_time_vehicle, end_time_vehicle])

    real_objective = num_cars_used * 60000 + 0.3 * total_distance

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
    return informationReduceVehicle, num_cars_used, total_distance, info_resultfile


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

    vehicle_cost = 200000
    # vehicle_cost = 1
    for vehicle_idx in range(data['num_vehicles']):
        routing.SetFixedCostOfVehicle(vehicle_cost, vehicle_idx)

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
    distance_dimension.SetGlobalSpanCostCoefficient(1)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index_time = routing.RegisterTransitCallback(time_callback)

    "zelf: Time window constraints"
    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index_time,
        30,  # allow waiting time
        max_time_per_vehicle,  # maximum time per vehicle #!!!
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    "zelf: Solution"
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

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

        return num_cars_used, total_distance, informationReduceVehicle,info_resultfile
    else:
        print("No solution found!")

def reduceNumberOfVehiclesAllZones(informationReduceVehicleAllZones):
    # informationReduceVehicleAllZones: matrix with per zone: per row information about the vehicle:
    #       [total_distance,start_time_vehicle,end_time_vehicle]
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
                [total_distance_vehicle_A, start_time_vehicle_A, end_time_vehicle_A] = \
                informationReduceVehicleAllZones[zone_i][vehicle_A]

                TOT_distance = total_distance_vehicle_A
                TOT_timeWindow = [(start_time_vehicle_A, end_time_vehicle_A)]
                TOT_combined = [(zone_i, vehicle_A)]
                num_combined = 1

                # Try to combine vehicle_A with as many cars as possible
                zone_j = 0
                vehicle_B = 0
                while zone_j != num_zones:
                    # Check if vehicle_B is still in possible_cars_to_combined:
                    if vehicle_B in possible_cars_to_combined[zone_j]:
                        [total_distance_vehicle_B, start_time_vehicle_B, end_time_vehicle_B] = \
                        informationReduceVehicleAllZones[zone_j][vehicle_B]

                        # check if time windows of the vehicle B can put after last vehicle of window A
                        #      remark: Because our zones are split in increasing start time, we only have to look if vehicle can put after the last of the combined vehicles
                        (start_time_last_vehicle, end_time_last_vehicle) = TOT_timeWindow[len(TOT_combined) - 1]
                        if end_time_last_vehicle < start_time_vehicle_B and ((zone_i,vehicle_A) != (zone_j,vehicle_B)):

                            # Calculate how long you can charge
                            time_charging_between_routes = min((start_time_vehicle_B - end_time_last_vehicle),
                                                               charging_time)
                            # Calculate how much extra distance you can do by charging
                            charging_distance = (time_charging_between_routes / charging_time) * max_range_LCV
                            # Calculate what the maximum distance is that you can drive
                            max_driving_distance = max_range_LCV + charging_distance

                            # Calculate if the 2 routes can be done with 1 vehicle:
                            if (TOT_distance + total_distance_vehicle_B) <= max_driving_distance:
                                num_cars_reduced += 1
                                num_combined += 1
                                TOT_combined.append((zone_j, vehicle_B))
                                TOT_timeWindow.append((start_time_vehicle_B, end_time_vehicle_B))
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
                    combined_cars.append((num_combined, TOT_combined, TOT_distance, TOT_timeWindow))

    return num_cars_reduced, combined_cars


if __name__ == '__main__':
    # loop over the zones
    informationReduceVehicleAllZones = []
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
        (num_combined, TOT_combined, TOT_distance, TOT_timeWindow) = combined_carsAllZones[k]
        print('  Combination of {} vehicles: total distance {}km & time windows: {}'.format(num_combined, TOT_distance,TOT_timeWindow))
        resultfile+='  Combination of {} vehicles: total distance {}km & time windows: {}\n'.format(num_combined, TOT_distance,TOT_timeWindow)

        for j in range(num_combined):
            (zone_A, vehicle_A) = TOT_combined[j]
            [route_distance, start_time_vehicle, end_time_vehicle] = informationReduceVehicleAllZones[zone_A][vehicle_A]
            print('     Vehicle (zone {},{}): distance {}km, start & end time: ({},{})'.format((zone_A + 1),
                                                                                               (vehicle_A + 1),
                                                                                               route_distance,
                                                                                               start_time_vehicle,
                                                                                               end_time_vehicle))
            resultfile+='     Vehicle (zone {},{}): distance {}km, start & end time: ({},{})\n'.format((zone_A + 1),(vehicle_A + 1),route_distance,start_time_vehicle,end_time_vehicle)

location = 'Results/' + titleResultfile
with open(location, 'w') as file:
    file.write(resultfile)
file.close()

# Output explanation:
# Time(378,398):the numbers represent the time window for the corresponding visit or location in the route.
#       Bv Time(378,398): The corresponding location should be visited between 378 (minimum) and 398 (maximum) time units