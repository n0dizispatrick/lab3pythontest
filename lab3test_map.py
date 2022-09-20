import numpy as np
import math
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
#from PIL import Image

def get_hyperbola(node_1, node_2, time_1, time_2, v, delta_d, max_d):
    # two lines, x0/y0 and x1/y1 corresponding to the two intersections of the
    # circles. These will be concateneated at the end to form a single line.
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    # The radii have constant difference of t_delta_d. "time delta difference"
    t_delta_d = abs(time_1 - time_2) * v

    # Determine which node received the transmission first.
    if(time_1 < time_2):
        circle1 = (node_1[0], node_1[1], 0)
        circle2 = (node_2[0], node_2[1], t_delta_d)
    else:
        circle1 = (node_2[0], node_2[1], 0)
        circle2 = (node_1[0], node_1[1], t_delta_d)

    # Iterate over all potential radii.
    for _ in range(int(max_d)//int(delta_d)):
        intersect = circle_intersection(circle1, circle2)
        if(intersect is not None):
            x0.append(intersect[0][0])
            x1.append(intersect[1][0])
            y0.append(intersect[0][1])
            y1.append(intersect[1][1])

        circle1 = (circle1[0], circle1[1], circle1[2]+delta_d)
        circle2 = (circle2[0], circle2[1], circle2[2]+delta_d)

    # Reverse so the concatenated hyperbola is continous. Could reverse only
    # x1/y1 instead if you wanted.
    x0 = list(reversed(x0))
    y0 = list(reversed(y0))

    # Concatenate
    x = x0 + x1
    y = y0 + y1

    return [x, y]


def get_hyp(rec_times, nodes, v, delta_d, max_d):
    if(rec_times.shape[0] == 0):
        return [] # return no bola
    
    hyp = []

    # node that receives the transmission first.
    first_node = int(np.argmin(rec_times))
    # Iterate over all other nodes.
    for j in [x for x in range(nodes.shape[0]) if x!= first_node]:
      
             get_hyperbola(node_1=(nodes[first_node][0],
                                   nodes[first_node][1]),
                          node_2=(nodes[j][0], nodes[j][1]),
                          time_1=rec_times[first_node],
                          time_2=rec_times[j],
                          v=v, delta_d=delta_d, max_d=max_d)
    
      
    return hyp


def circle_intersection(circle1, circle2):
    
    x1,y1,r1 = circle1
    x2,y2,r2 = circle2
    # http://stackoverflow.com/a/3349134/798588
    # d is euclidean distance between circle centres
    dx,dy = x2-x1,y2-y1
    d = math.sqrt(dx*dx+dy*dy)
    if d > r1+r2:
        # print('No solutions, the circles are separate.')
        return None # No solutions, the circles are separate.
    elif d < abs(r1-r2):
        # No solutions because one circle is contained within the other
        # print('No solutions because one circle is contained within the other')
        return None
    elif d == 0 and r1 == r2:
        # Circles are coincident - infinite number of solutions.
        # print('Circles are coincident - infinite number of solutions.')
        return None

    a = (r1*r1-r2*r2+d*d)/(2*d)
    h = math.sqrt(r1*r1-a*a)
    xm = x1 + a*dx/d
    ym = y1 + a*dy/d
    xs1 = xm + h*dy/d
    xs2 = xm - h*dy/d
    ys1 = ym - h*dx/d
    ys2 = ym + h*dx/d

    return ((xs1,ys1),(xs2,ys2))

rover_square_side = 5e3
rx_square_side = 25e3

# Speed of Light
v = 3e8

num_nodes = 3
# Meter increments to radius of circles when generating hyperbola of
# circle intersection.
delta_d = int(100)

# Max distance a transmission will be from the node that first
# received the transmission. This puts an upper bound on the radii of the
# circle, thus limiting the size of the hyperbola to be near the nodes.
max_d = int(20e3)

# Noise for recieve time
rec_time_noise = 10e-9

plot_trilateration_circles = False
plot_lines_between_nodes = False

# Generate nodes with x and y coordinates.
nodes = np.array([(-10000,-10000),(5000,-7500),(7500,7500)])
print('nodes:\n', nodes)

# Location of rover
rover = (1250,-1000)
print('rover:', rover)

# Distances from each node to the rover

distances = np.array([ ( (x[0]-rover[0])**2 + (x[1]-rover[1])**2 )**0.5
                       for x in nodes])
print('distances:', distances)

# Time at which each node receives the transmission.
rec_times = distances/v 
# Add noise to receive times
rec_times += np.random.normal(loc=0, scale=rec_time_noise, size =num_nodes)
                              
print('rec_times:', rec_times)

# Get the bola.
bola = get_hyp(rec_times, nodes, v, delta_d, max_d)

im = plt.imread("map1.png")
fig, ax = plt.subplots()
ax.imshow(im,extent=[-12500,12500,-12500,12500])

max_width = max(rover_square_side, rx_square_side)/2
ax.set_ylim((max_width*-1, max_width))
ax.set_xlim((max_width*-1, max_width))
for i in range(nodes.shape[0]):
    x = nodes[i][0]
    y = nodes[i][1]
    ax.scatter(x, y)
    ax.annotate('Node '+str(i), (x, y))
ax.scatter(rover[0], rover[1])
ax.annotate('Rover', (rover[0], rover[1]))


c = np.argmin(rec_times)  # Tower that received first.
p_c = nodes[c]
t_c = rec_times[c]

x = np.linspace(nodes[i][0] - 50000, nodes[i][0] + 50000, 100)
y = np.linspace(nodes[i][1] - 50000, nodes[i][1] + 50000, 100)
x, y = np.meshgrid(x, y)

for i in range(num_nodes):
    if i == c:
        continue
        
    p_i = nodes[i]
    t_i = rec_times[i]
    
    plt.contour(
        x, y,
        (
           np.sqrt((x-p_c[0])**2 + (y-p_c[1])**2) 
         - np.sqrt((x-p_i[0])**2 + (y-p_i[1])**2) 
         + v*(t_i - t_c)
        ),
        [0])

# Solve the location of the transmitter.
c = np.argmin(rec_times)
p_c = np.expand_dims(nodes[c], axis=0)
t_c = rec_times[c]

# Remove the c node to allow for vectorization.
all_p_i = np.delete(nodes, c, axis=0)
all_t_i = np.delete(rec_times, c, axis=0)

def eval_solution(x):
    """ x is 2 element array of x, y of the transmitter"""
    return (
          np.linalg.norm(x - p_c, axis=1)
        - np.linalg.norm(x - all_p_i, axis=1) 
        + v*(all_t_i - t_c) 
    )

# Initial guess.
x_init = [0, 0]

# Find a value of x such that eval_solution is minimized.
res = least_squares(eval_solution, x_init)

print(f"Actual rover location:    ({rover[0]}, {rover[1]}) ")
print(f"Calculated rover locaion: ({res.x[0]:.1f}, {res.x[1]:.1f})")
print(f"Error: {np.linalg.norm(rover-res.x):.1f}")

