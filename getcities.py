# %% Thanks to AdrienVannson https://github.com/AdrienVannson
import numpy as np
import requests
from time import sleep
import os


class DivisibleError(Exception):
    pass


def get_cities(cities, cities_per_request=50, time_sleep=1, verbose=False, force_recompute=False):
    # len(cities) must be divisible by cities_per_request
    cities_count = len(cities)
    if os.path.exists('dists{}.npy') and not force_recompute:
        dists = np.load('dists{}.npy')
    else:
        print('Dist file does not exits: create it')
        # Build the distance matrix
        # dists[i][j] is the time to go from the i-th biggest city to the j-th biggest city
        dists = np.zeros((cities_count, cities_count))

        # Initialize the progress bar
        print('-' * 100)
        if not cities_count % cities_per_request == 0:
            msg = 'len(cities) must be divisible by cities_per_request, here len(cities) = {0}, but cities_per_request = {1}'.format(
                cities_count, cities_per_request)
            raise DivisibleError(msg)
        size = cities_count // cities_per_request
        number_shown = 0

        for i_request in range(size):
            for j_request in range(size):

                # Update the progress bar
                percentage = 100 * (i_request * size + j_request) / (size ** 2)
                while number_shown < percentage:
                    number_shown += 1

                coords = ';'.join(
                    [str(long) + ',' + str(lat) for _, long, lat in cities[i_request * cities_per_request:(i_request+1) * cities_per_request]] +
                    [str(long) + ',' + str(lat) for _, long, lat in cities[j_request *
                                                                           cities_per_request:(j_request+1) * cities_per_request]]
                )

                # Other site: https://routing.openstreetmap.de/routed-car/table/v1/driving/...

                url = 'http://router.project-osrm.org/table/v1/driving/' \
                    + coords \
                    + '?sources=' + ';'.join([str(i) for i in range(cities_per_request)]) \
                    + '&destinations=' + \
                    ';'.join([str(i) for i in range(
                        cities_per_request, 2*cities_per_request)])

                response = requests.get(url)
                local_dists = np.array(response.json()['durations'])
                if verbose:
                    print('-- Request ({}, {}) done ... (have to reach = ({},{}))--'.format(
                        i_request, j_request, size-1, size-1))
                for i in range(cities_per_request):
                    for j in range(cities_per_request):
                        dists[i_request * cities_per_request + i][j_request * cities_per_request + j] = \
                            local_dists[i][j]

                sleep(time_sleep)

        # Terminate the progress bar
        print((100 - number_shown) * '#')
        # Save everything
        np.save('dists{}'.format(cities_count), dists)
    return dists
