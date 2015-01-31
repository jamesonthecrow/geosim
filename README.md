# GeoSim
The GeoSim mobility model simulate the movement of people in cities taking individual and social preferences into account.

## Dependencies
GeoSim requires numpy, matplotlib, and networkx to run.

## Example
To run the mode with the sample network run the following code block in python:

```
import geosim

alpha_dists = []#, lambda: 0.0]
params = {
			'rho': lambda: 0.6,
			'gamma': 0.6,
			'alpha': lambda: numpy.random.exponential(0.2),
			'beta': 0.8,
			'tau': 17.0,
			'xmin': 1.0,
			'network_type': 'real',
			'fname': '/home/jameson/geo_social/data/model_network_clean.pickle',
			'nusers': 100,
			'nlocations': 250,
			'ncontacts': 20,
			'nsteps': 5001
	}
model = geosim.GeoSim(params)
model.run()
```
