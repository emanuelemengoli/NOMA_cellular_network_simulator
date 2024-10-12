import numpy as np
import math
from numpy.linalg import norm 

l2_norm = lambda x,y:  norm(x-y) if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) else norm(np.array(x)-np.array(y))

l2_norm_sqrd = lambda x, y: l2_norm(x,y)**2

def bounce_back(x, y,theta, max_x, max_y, border_margin):
    """
    Adjusts the position and angle (theta) of an entity when it hits the edges of a defined area.

    Parameters:
    - x (float): The current x-coordinate of the entity.
    - y (float): The current y-coordinate of the entity.
    - theta (float): The current angle (in radians) of movement.
    - max_x (float): The maximum x-coordinate boundary.
    - max_y (float): The maximum y-coordinate boundary.
    - border_margin (float): The margin to trigger a bounce back.

    Returns:
    - Tuple[float, float, float]: The new x-coordinate, y-coordinate, and angle after the bounce back.
    """

    # Bounce off the edges
    if x <= border_margin: #0
        x = -x
        theta = np.pi - theta

    elif x >= max_x:
        x = 2 * max_x - x
        theta = np.pi - theta

    if y <= border_margin:
        y = -y
        theta = -theta
        
    elif y >= max_y:
        y = 2 * max_y - y
        theta = -theta

    return x, y,theta

def update_motion(x, y,velox, theta, max_x, max_y, border_margin):
    """
    Updates the position and angle of an entity based on its velocity and angle, 
    then adjusts for any edge collisions.

    Parameters:
    - x (float): The current x-coordinate of the entity.
    - y (float): The current y-coordinate of the entity.
    - velox (float): The velocity of the entity.
    - theta (float): The angle (in radians) of movement.
    - max_x (float): The maximum x-coordinate boundary.
    - max_y (float): The maximum y-coordinate boundary.
    - border_margin (float): The margin to trigger a bounce back.

    Returns:
    - Tuple[float, float, float]: The updated x-coordinate, y-coordinate, and angle (theta).
    """

    # Calculate the next position
    x += velox * np.cos(theta)
    y += velox * np.sin(theta)

    x, y, theta = bounce_back(x, y, theta, max_x, max_y, border_margin)

    return x, y, theta

def hybrid_gmm(position,destination_sets, dimensions,border_margin,rndm_gen, max_velox, min_velox): 
    """
    A variation of the Gauss-Markov Mobility Model that yields smooth trajectories.

    Parameters:
    - position (Tuple[float, float]): The initial (x, y) position of the entity.
    - destination_sets (List[Tuple[float, float]]): List of potential destinations.
    - dimensions (Tuple[int, int]): The (width, height) of the simulation area.
    - border_margin (float): The margin to trigger a bounce back.
    - rndm_gen (np.random.Generator): Random number generator instance.
    - max_velox (float): Maximum velocity.
    - min_velox (float): Minimum velocity.

    Yields:
    - Tuple[Tuple[float, float], float, float]: The next position (x, y), updated velocity, and angle (theta).
    """

    x, y = position

    chosen_index = rndm_gen.choice(range(len(destination_sets)))

    destination = destination_sets[chosen_index]
    dest_x, dest_y = destination

    max_x, max_y = dimensions

    velox = rndm_gen.uniform(min_velox, max_velox) #defined in km/s

    cluster_average_velox = np.mean([rndm_gen.uniform(min_velox, max_velox) for _ in range(1000)]) 
    
    theta = np.arctan2(dest_y - y, dest_x - x)

    alpha = rndm_gen.uniform(0.1, 1)

    cluster_variance = rndm_gen.uniform(0.5, 1.5)

    alpha2 = 1-alpha

    alpha3 = np.sqrt(1.0 - alpha**2) * cluster_variance

    while True:
        
        # Calculate the next position
        x, y, theta = update_motion(x, y,velox, theta, max_x, max_y, border_margin)

        # Check if the entity has reached the destination
        if l2_norm((x,y), destination) < 300:#m
            new_destination_set = [d for d in destination_sets if not np.array_equal(d, destination)]
            chosen_index = rndm_gen.choice(range(len(new_destination_set)))
            destination = new_destination_set[chosen_index]
            dest_x, dest_y = destination

        # Calculate the direction towards the destination
        dir_to_dest = np.arctan2(dest_y - y, dest_x - x)

        velox = (alpha * velox +
                    alpha2 * cluster_average_velox
                    + alpha3 * rndm_gen.normal(0.0, 1.0)
                    )

        theta = (alpha * theta +
                  (1-alpha) *  dir_to_dest
                   + alpha3 * rndm_gen.normal(0.0, 1.0)
                  )

        # Yield the next state
        yield (x, y)

def random_walk(position,dimensions,border_margin, rndm_gen, max_velox, min_velox):
    """
    Generates a random walk trajectory for an entity within specified boundaries.

    Parameters:
    - position (Tuple[float, float]): The initial (x, y) position of the entity.
    - dimensions (Tuple[int, int]): The (width, height) of the simulation area.
    - border_margin (float): The margin to trigger a bounce back.
    - rndm_gen (np.random.Generator): Random number generator instance.
    - max_velox (float): Maximum velocity.
    - min_velox (float): Minimum velocity.

    Yields:
    - Tuple[float, float]: The next position (x, y) of the entity.
    """
    #defines the next position in space
    x, y = position
    max_x, max_y = dimensions
    while True:
        velox = rndm_gen.uniform(min_velox, max_velox)  # Velocity for each step
        theta = rndm_gen.uniform(0, 2*np.pi)
        x, y, theta = update_motion(x, y,velox, theta, max_x, max_y,border_margin)
        yield x,y

def biased_random_walk(position, dimensions,border_margin,rndm_gen, max_velox, min_velox):
    """
    Generates a biased random walk trajectory for an entity, favoring a preferred direction.

    Parameters:
    - position (Tuple[float, float]): The initial (x, y) position of the entity.
    - dimensions (Tuple[int, int]): The (width, height) of the simulation area.
    - border_margin (float): The margin to trigger a bounce back.
    - rndm_gen (np.random.Generator): Random number generator instance.
    - max_velox (float): Maximum velocity.
    - min_velox (float): Minimum velocity.

    Yields:
    - np.ndarray: The next position as a NumPy array [x, y] of the entity.
    """
    # Defines the next position in space with a bias in a preferred direction
    x, y = position
    max_x, max_y = dimensions
    bias_direction = rndm_gen.uniform(0, 2 * np.pi)
    bias_strength= rndm_gen.uniform(0.3, 0.7)
    
    iteration = 0
    while True:
        if iteration % 25 == 0:
            bias_direction = rndm_gen.uniform(0, 2 * np.pi)
            bias_strength = rndm_gen.uniform(0.3, 0.7)

        velox = rndm_gen.uniform(min_velox, max_velox)  # Velocity for each step
        random_theta = rndm_gen.uniform(0, 2 * np.pi)
        
        # Introduce bias towards the preferred direction
        biased_theta = (1 - bias_strength) * random_theta + bias_strength * bias_direction

        x, y, biased_theta = update_motion(x, y,velox, biased_theta, max_x, max_y,border_margin)

        iteration += 1
        yield np.array([x, y])

def levy_step(rndm_gen, alpha:float = 1.5, max_step_length: float = None):
    """
    Generates a step length based on a Lévy distribution.

    Parameters:
    - rndm_gen (np.random.Generator): Random number generator instance.
    - alpha (float): The stability parameter; 1 < alpha < 2 indicates a Lévy stable distribution.
    - max_step_length (float, optional): Maximum allowable step length.

    Returns:
    - float: The generated step length, potentially truncated by max_step_length.
    """
    # Mategna algorithm for Levy stable process
    sigma_u = (
        math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
        (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
    ) ** (1 / alpha)
    
    # Draw u and v from normal distributions
    u = rndm_gen.normal(0, sigma_u)
    v = rndm_gen.normal(0, 1)
    
    # Calculate step length s
    s = u / abs(v) ** (1 / alpha)

    # Truncate step length
    if max_step_length is not None:
        if abs(s) > max_step_length:
            s = np.sign(s) * max_step_length
    
    return s

def levy_walk(position, dimensions,rndm_gen, border_margin,alpha:float =1, max_step_length: float = None): 
    #alpha = 1 Cauchy distribution, alpha = 2 Gaussian distribution == Brownian motion 
    """
    Generates a Lévy flight trajectory for an entity, where movement steps are defined by a Lévy distribution.

    Parameters:
    - position (Tuple[float, float]): The initial (x, y) position of the entity.
    - dimensions (Tuple[int, int]): The (width, height) of the simulation area.
    - rndm_gen (np.random.Generator): Random number generator instance.
    - border_margin (float): The margin to trigger a bounce back.
    - alpha (float): The stability parameter; 1 < alpha < 2 indicates a Lévy stable distribution.
    - max_step_length (float, optional): Maximum allowable step length.

    Yields:
    - np.ndarray: The next position as a NumPy array [x, y] of the entity.
    """
    # Define the next position in space using Lévy flight
    x, y = position
    max_x, max_y = dimensions
    while True:
        step_length = levy_step(alpha = alpha,max_step_length = max_step_length, rndm_gen = rndm_gen)
        theta = rndm_gen.uniform(0, 2 * np.pi)
        x, y, theta = update_motion(x, y,step_length, theta, max_x, max_y,border_margin)
        yield np.array([x, y])

def brownian_motion(position, dimensions,border_margin, rndm_gen):
    """
    Generates a Brownian motion trajectory for an entity, modeled as a Lévy flight with alpha = 2.

    Parameters:
    - position (Tuple[float, float]): The initial (x, y) position of the entity.
    - dimensions (Tuple[int, int]): The (width, height) of the simulation area.
    - border_margin (float): The margin to trigger a bounce back.
    - rndm_gen (np.random.Generator): Random number generator instance.

    Returns:
    - Generator: A generator yielding the next position as a NumPy array [x, y] of the entity.
    """
    # Define the next position in space using Brownian motion
    return levy_walk(position=position, dimensions=dimensions,border_margin=border_margin, alpha=2, rndm_gen=rndm_gen) # Brownian motion
    
