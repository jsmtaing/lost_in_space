# cisc372: lost_in_space
A parallel implementation of a solution to the n-body problem; which, in this case, is simulated as a model of the solar system (sun + the eight planets) alongside a randomly high number of asteroids.

## N-Body Problem
The n-body problem is a classical computation model where we have a system with n objects in it (like our solar system), each object having a mass, a velocity in 3 dimensions, and a position in 3 dimensions. The idea is to track/compute over time the movement and speed of all of the objects in the system, taking into account the gravitational interaction between them. This is an O(n^2) algorithm, as for each object, we have to sum up all of the effects of the other (n-1) objects.
