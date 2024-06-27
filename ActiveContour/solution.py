import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

WINDOW_SIZE = 5

def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))

def load_data(fpath, radius):
    """
    :param fpath: path to image
    :param radius: initial radius for a contour
    :return: image, V in the form (x, y)
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 25  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V

# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

class Vertex:
    def __init__(self, id, coords, parent=None, cost=0):
        self.id = id
        self.coords = coords
        self.root = parent
        self.cost = cost

    def add_cost(self, cost):
        self.cost += cost
    
    def set_root(self, root):
        self.root = root

class ActiveContour:
    def __init__(self, img, initial_contour, threshold=0):
        self.__iteration_step = 0
        self.__img = img
        self.__contour = initial_contour
        self.__threshold = threshold
        self.__is_converged = False
        self.__set_image_magnitude()
        self.__set_average_distance()
    
    def __set_image_magnitude(self):
        sobel_x = cv2.Sobel(self.__img,cv2.CV_64F,1,0,ksize=3)
        sobel_y = cv2.Sobel(self.__img,cv2.CV_64F,0,1,ksize=3)
        magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y)).astype("float32")
        normalized_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())*255
        normalized_magnitude[normalized_magnitude<self.__threshold]=0
        self.magnitude = normalized_magnitude
    
    def __set_convergence(self, new_contour):
        self.__is_converged = np.array_equal(self.__contour, new_contour)

    def __set_average_distance(self):
        dist_sum = 0
        for i in range(len(self.__contour)-1):
            x1,y1 = self.__contour[i]
            x2,y2 = self.__contour[i+1]
            dist_sum += math.sqrt((x2-x1)**2 + (y2-y1)**2)

        self.__distance = dist_sum / len(self.__contour)
    
    def __get_external_energy(self, point):
        """ returns external energy for a given point.
        :param (x,y).
        :return: external energy cost.
        """
        x,y = point
        return -self.magnitude[y, x]**2
    
    def __get_internal_energy(self, prev_point, this_point, next_point=None):
        """ returns internal energy for a given point.
        :param (x,y).
        :return: internal energy cost.
        """
        alpha = .1
        beta = 1
        # elasticity
        energy = alpha*(self.__distance - math.sqrt((this_point[0]-prev_point[0])**2 + (this_point[1]-prev_point[1])**2))**2
        if (next_point != None):
            # curvature
            energy += beta*((next_point[0] - 2*this_point[0] + prev_point[0])**2 + (next_point[1] - 2*this_point[1] + prev_point[1])**2)
        
        return energy
    
    def update_contour(self):
        if self.__is_converged:
            print(f"Snake has already converged at step #{self.__iteration_step}!")
            return
        
        self.__iteration_step += 1
        vertex_states = {i: [] for i in range(len(self.__contour))}

        # create unconnected graph with unary costs
        for vertex_id, vertex in enumerate(self.__contour):
            for w_y in range(WINDOW_SIZE):
                for w_x in range(WINDOW_SIZE):
                    # absolute position of vertex's neighbouring position in the image
                    abs_vertex_x = vertex[0] + w_x - WINDOW_SIZE//2
                    abs_vertex_y = vertex[1] + w_y - WINDOW_SIZE//2
                    # if neighbouring pixel is within image, push to array of possible states
                    if ((abs_vertex_x >= 0) and (abs_vertex_y >= 0) and (abs_vertex_x < self.__img.shape[1])
                        and (abs_vertex_y < self.__img.shape[0])):
                        v = Vertex(id=vertex_id, parent=None, coords=(abs_vertex_x,abs_vertex_y))
                        v.add_cost(self.__get_external_energy(v.coords))
                        vertex_states[vertex_id].append(v)
        
        # add weighted edges to the graph
        prev_states = vertex_states.pop(0)
        this_states = vertex_states.pop(1)
        for vertex_id in range(2,len(self.__contour)):
            next_states = vertex_states.pop(vertex_id)
            for this_state in this_states:
                best_match = None
                best_match_value = math.inf
                # for every V_i iterate over V_i-1 and V_i+1 and find lowest energy cost to update V_i to
                for prev_state in prev_states:
                    for next_state in next_states:
                        cost = prev_state.cost + self.__get_internal_energy(prev_state.coords, this_state.coords, next_state.coords)
                        if (cost < best_match_value):
                            best_match_value = cost
                            best_match = prev_state
                this_state.set_root(best_match)
                this_state.add_cost(best_match_value)
            prev_states = this_states
            this_states = next_states

        # reached i = len(contour)-1. Calculate this step separately without curvature energy cost 
        for this_state in this_states:
            best_match = None
            best_match_value = math.inf
            for prev_state in prev_states:
                cost = prev_state.cost + self.__get_internal_energy(prev_state.coords, this_state.coords)
                if (cost < best_match_value):
                    best_match_value = cost
                    best_match = prev_state
            this_state.set_root(best_match)
            this_state.add_cost(best_match_value)
        
        # get last vertex state with lowest cost
        best_cost = math.inf
        best_match = None
        for state in next_states:
            if (state.cost < best_cost):
                best_match = state
                best_cost = state.cost

        # backtrack from last vertex to the beginning, storing result path into new_contour
        new_contour = np.array([[best_match.coords[0], best_match.coords[1]]])
        root = best_match.root
        while (root != None):
            new_contour = np.insert(new_contour, 0, [root.coords[0], root.coords[1]], axis=0)
            root = root.root
        
        # update current contour, check for convergence
        self.__set_convergence(new_contour)
        self.__contour = new_contour
        self.__set_average_distance()
    
    def get_contour(self):
        """ returns current contour positions.
        :return: list of coordinates as (x,y).
        """
        return self.__contour
    
    def is_converged(self):
        """ returns if the contour has converged and won't update.
        :return: Boolean.
        """
        return self.__is_converged

# ------------------------

def run(fpath, radius, threshhold=0):
    """ run experiment
    :param fpath: path to image.
    :param radius: initial radius of contour.
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 25

    # ------------------------
    # your implementation here
    snake = ActiveContour(Im, V, threshhold)
    # ------------------------
    ax.imshow(Im, cmap='gray')
    ax.set_title('Initial frame')
    plot_snake(ax, V)
    plt.pause(0.2)
    for t in range(n_steps):
        # ------------------------
        # your implementation here
        if snake.is_converged():
            break

        snake.update_contour()
        V = snake.get_contour()
        # ------------------------

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('Frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    ax.set_title('Final frame ' + str(t))
    plt.pause(2)

if __name__ == '__main__':
    run('./images/ball.png', radius=120)
    # threshold magnitude values to reduce image noise
    run('./images/coffee.png', radius=100, threshhold=80)
