# INTEGRATED PRACTICUM 1: ASSIGNMENT 2

## Integrated practicum 1, assignment 2: Smart management of electrified last-mile delivery fleets + time windows

## Usage
The most important files are CVRP.py and CVRP_TW.py as these contain the solvers that solve the routing problem. The solvers need matrices and these are generated in matrices_generator_CVRP.py for CVRP.py and in matrices_generator_CVRP_TW for CVRP_TW.py. Next each file in the repository is explained independently, read this carefully before usage.

* CVRP.py

  This file contains the solver to solve the capacited vehicle routing problem for one day. The routing solver of Google ORTools is used. At the start of the file the paramaters are defined, such as the volume of the vehicles, the range of the vehicles, the charging time, ... These parameters can be tweaked to test different scenarios. In the "create_data_model" function the necessary data is retrieved. Some example data for different days and working day lengths are preloaded on the repository, simply change the names when you want to try different data. The file matrices_generator_CVRP.py can also be used to generate custom data, for more information see the section of matrices_generator_CVRP.py.

* CVRP_TW.py

  This file contains the solver to solve the capacited vehicle routing problem with time windows for one day. The routing solver of Google ORTools is used. At the start of the file the paramaters are defined, such as the volume of the vehicles, the range of the vehicles, the charging time, ... These parameters can be tweaked to test different scenarios. In the "create_data_model" function the necessary data is retrieved. Some example data for different days and working day lengths are preloaded on the repository, simply change the names when you want to try different data. The file "matrices_generator_CVRP.py" can also be used to generate custom data, for more information see the section of "matrices_generator_CVRP.py".

* matrices_generator_CVRP.py

  ! This file needs have a Valhalla docker image running in a container --> See instructions below !
  
  The file generates the inputs necessary to run "CVRP.py" for one day. These inputs are lists containing the location IDs of each zone, the volumes that should be delivered at each location for each zone abd the distance and time matrices of each zone. The locations for the day are divided into 5 zones based on their coordinates. Adjacent zones may be joined together if their combined size isn't too large. To construct the distance and time matrices with respect to real road layout, the Valhalla API is used. Note that not all the matrix elements are calculated, but only those closest, as the crow flies, to the origin coordinate. This file uses functions from the "functions.py" file. The only parameter you can change in this file is the date for which the inputs are generated.

* matrices_generator_CVRP_TW.py

he file generates the inputs necessary to run "CVRP_TW.py" for one day. These inputs are lists containing the location IDs of each time interval, the volumes that should be delivered at each location for each time interval, the time windows of each time interval and the distance and time matrices of each time interval. The time windows are generated randomly and one for every location. Locations are divided into time intervals based on their time window. Usually, but not necessarly the possible time windows and time intervals are the same. To construct the distance and time matrices with respect to real road layout, the Valhalla API is used. Note that not all the matrix elements are calculated, but only those closest, as the crow flies, to the origin coordinate (= each row corresponds to a different origin coordinate and row by row the matrices are constructed). This file uses functions from the "functions.py" file. The only parameter you can change in this file is the date for which the inputs are generated.

* functions.py

This file is referenced in "matrices_generator_CVRP.py" and "matrices_generator_CVRP_TW.py" and contains two functions. One is used to join together adjacent zones if possible and the other to calculate the Euclidian distance.

* locations.csv & packages.csv

The given data containing the locations and packages that need to be delivered respectively. ! Note that 2 small modifications were made in the locations.csv file. Coordinate 16632 and 23686 were shifted a very small amount (1m-5m) because otherwise the Valhalla API was unable to find a route to these locations. !

## Valhalla Docker image
To use "matrices_generator_CVRP.py" and "matrices_generator_CVRP_TW.py" you need to run the Valhalla Docker image in a container with a graoh if Belgium. First install Docker Desktop using the link below:

https://www.docker.com/products/docker-desktop/

Next run the following command in a terminal:

```terminal
$ docker run -dt --name valhalla_gis-ops -p 8002:8002 -v $PWD/custom_files:/custom_files -e tile_urls=https://download.geofabrik.de/europe/andorra-latest.osm.pbf ghcr.io/gis-ops/docker-valhalla/valhalla:latest

You can open Docker Desktop to check if the container is running and shut it down, restart or turn it back on when you want to. For more information check the documention from Valhalla:

https://github.com/gis-ops/docker-valhalla
