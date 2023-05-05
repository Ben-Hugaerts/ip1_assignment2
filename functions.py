def join_adjacent_zones(zones, min_size):
    joined = True
    zone_numbers = [[i+1] for i in range(len(zones))]  # keep track of zone numbers
    while joined:
        joined = False
        for i in range(len(zones)):
            neighbors = []
            if i == 0:
                neighbors.extend([1, 4])
            elif i == 1:
                neighbors.extend([0, 2])
            elif i == 2:
                neighbors.extend([1, 4])
            elif i == 3:
                neighbors.extend([2, 4])
            elif i == 4:
                neighbors.extend([0, 2, 3])
            for j in neighbors:
                if j < len(zones) and len(zones[i]) + len(zones[j]) < min_size:
                    zones[i] += zones[j]
                    zones[j] = []
                    zone_numbers[i] += zone_numbers[j]
                    zone_numbers[j] = []
                    joined = True
        zones = [zone for zone in zones if zone != []]
        zone_numbers = [zone for zone in zone_numbers if zone != []]
    return zones, zone_numbers  # return both the zones and the zone numbers

import math
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R*c*1000  # Distance in meters