#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from spline import *
from scipy.interpolate import CubicSpline

def evaluate_spline(x, x_i, a, b, c, d):
	return a + (b + (c + d * (x - x_i)) * (x - x_i)) * (x - x_i)

def correct_interval(x, intervals): # This returns the correct interval index and the x0 start value.
	# Now loop over each element and check the thing.
	for i in range(len(intervals)-1):
		if x >= intervals[i] and x <= intervals[i+1]:
			return i, intervals[i] # Return the index thing... 
	assert False

def calculate_spline(x, t_knots, splines):
	i, t0 = correct_interval(x, t_knots)
	#print("Correct interval: "+str(i))
	if len(splines[0]) == 5:
			a, b, c, d, _ = splines[0][i], splines[1][i], splines[2][i], splines[3][i], splines[4][i]
	else:
		a, b, c, d = splines[0][i], splines[1][i], splines[2][i], splines[3][i]# , splines[4][i]
	return evaluate_spline(x, t0, a, b, c, d)


def compute_t_knots(points):
	""" Computes t values based on cumulative distance along the curve. """
	t_knots = [0]
	for i in range(1, len(points)):
		dist = np.sqrt((points[i][0] - points[i - 1][0])**2 + (points[i][1] - points[i - 1][1])**2)
		t_knots.append(t_knots[-1] + dist)  # Accumulate distances
	return t_knots

def render_result(x_splines: list, y_splines: list, points: list, t_knots: list) -> None:
	x_knots = [p[0] for p in points] # Something like this????
	y_knots = [p[1] for p in points]
	n = len(points)
	t_values = np.linspace(min(t_knots), max(t_knots), 20000)
	# Compute interpolated x and y values
	x_things = [calculate_spline(t, t_knots, x_splines) for t in t_values]
	y_things = [calculate_spline(t, t_knots, y_splines) for t in t_values]
	plt.plot(x_things, y_things, label="Cubic Spline Curve")
	plt.scatter(x_knots, y_knots, color="red", label="Control Points")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.title('Spline Graph Using Manually Calculated Coefficients')
	plt.grid(True)
	plt.show()
	return

from math import sqrt

def compute_t_range(points):
	t_knots = [0]
	for i in range(1, len(points)):
		dist = sqrt((points[i][0] - points[i - 1][0])**2 + (points[i][1] - points[i - 1][1])**2)
		t_knots.append(t_knots[-1] + dist)  # Accumulate distances
	return t_knots

from plagiarized import * # Import the implementation which I copied from stackoverflow. It is stored in plagiarized.py

USE_MODE = 2

if __name__=="__main__":
	x_vals=[-0.83,0.14,-1.09,1.09,-0.54,2.03,3.0]
	y_vals=[-2.03,-2.06,0.71,1.49,2.06,2.43,3.0]

	assert len(x_vals) == len(y_vals)
	assert all([isinstance(x, float) for x in x_vals])
	assert all([isinstance(x, float) for x in y_vals])
	points = [[x_vals[i], y_vals[i]] for i in range(len(x_vals))]
	x_knots = [p[0] for p in points]
	y_knots = [p[1] for p in points]
	t_i = compute_t_knots(points)
	assert len(t_i) == len(points) == len(x_knots) == len(y_knots)
	points_x_t = [(t_i[i], x_knots[i]) for i in range(len(t_i))]
	points_y_t = [(t_i[i], y_knots[i]) for i in range(len(t_i))]
	spline_x_reference = CubicSpline(t_i, x_knots).c
	spline_y_reference = CubicSpline(t_i, y_knots).c
	spline_y = get_spline_natural_fifth_degree(t_i, y_knots)
	spline_x = get_spline_natural_fifth_degree(t_i, x_knots)
	spline_x_reference = list(spline_x_reference)
	spline_y_reference = list(spline_y_reference)
	spline_x_reference.reverse()
	spline_y_reference.reverse()
	spline_x_stuff = calc_spline_params(np.array(t_i), np.array(x_knots))
	spline_y_stuff = calc_spline_params(np.array(t_i), np.array(y_knots))
	if USE_MODE == 0: # Use the implementation which I took from stackoverflow
		render_result(spline_x_stuff, spline_y_stuff, points, t_i)
	elif USE_MODE == 1: # Use Scipy. This one actually displays the correct result.
		render_result(spline_x_reference, spline_y_reference, points, t_i)
	else: # Use my own implementation
		render_result(spline_x, spline_y, points, t_i)
	exit(0)
