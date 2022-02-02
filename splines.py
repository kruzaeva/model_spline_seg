import numpy as np
import math
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from tifffile import imread
import cv2
import GeodisTK

def bilinear_interpolate(im, y, x):
    """bilinear interpolation function from cv2

    Args:
        im ([type]): [description]
        y ([type]): [description]
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return cv2.remap(im, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)




def coefficients_matrix_generator(control_points=6, num_points=50):
    """B-matrix generator, enables efficient discrete coordinates generation

    Args:
        control_points (int, optional): number of control points. Defaults to 6.
        num_points (int, optional): number of points per spline segment. Defaults to 50.

    Returns:
        [type]: [description]
    """    
    t = np.linspace(0, 1, num_points)
    t = np.tile(t, control_points)
    coefficients_matrix = np.zeros((control_points, control_points * num_points), dtype=np.float32)

    for k in range(control_points):
        for i in range(k * num_points, (k + 1) * num_points):
            index_0 = k % control_points
            index_1 = (k + 1) % control_points
            index_2 = (k + 2) % control_points
            index_3 = (k + 3) % control_points

            coefficients_matrix[index_0, i] = (1 - t[i]) ** 3 / 6
            coefficients_matrix[index_1, i] = (3 * t[i] ** 3 - 6 * t[i] ** 2 + 4) / 6
            coefficients_matrix[index_2, i] = (-3 * t[i] ** 3 + 3 * t[i] ** 2 + 3 * t[i] + 1) / 6
            coefficients_matrix[index_3, i] = (t[i]) ** 3 / 6

    return coefficients_matrix



def calculate_loss_green(parameters, alpha, integrx, integry, gradient, integrgmapx, 
                                        integrgmapy, coefficients_matrix,
                   area_coefficient=-1,gmap_coefficient=-1):
    """[summary]

    Args:
        parameters ([type]): parameters of your geometrical model
        alpha ([type]): rotation angle
        integrx ([type]): cumulative sum of the image along x direction
        integry ([type]): cumulative sum of the image along y direction
        gradient ([type]): image gradient
        integrgmapx ([type]): cumulative sum of the geodesic distance map along x direction
        integrgmapy ([type]): cumulative sum of the geodesic distance map along x direction
        coefficients_matrix ([type]): B-matrix
        area_coefficient (int, optional): [description]. Defaults to -1.
        gmap_coefficient (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """    
    coords = evaluate_spline(parameters.astype(np.float32), alpha, coefficients_matrix)

    x=coords[0]

    y=coords[1]

    xdif=-x+np.roll(x,1)
    ydif=-y+np.roll(y,1)
    len_ds=(xdif**2+ydif**2)**0.5
    
    binary_area=np.sum((x*ydif-y*xdif))/2
    weighted_area=np.sum((bilinear_interpolate(integrx,x,y).T*ydif-bilinear_interpolate(integry,x,y).T*xdif))/2
    average_area=weighted_area/binary_area
    weighted_area_gmap=np.sum((bilinear_interpolate(integrgmapx,x,y).T*ydif-bilinear_interpolate(integrgmapy,x,y).T*xdif))/2
    average_area_gmap=weighted_area_gmap/binary_area

    fixed = 1/np.sum(len_ds)*np.sum(len_ds / (bilinear_interpolate(gradient, coords[0], coords[1]) + 0.0001) ** 1.5)

    return fixed + area_coefficient * average_area+ gmap_coefficient * average_area_gmap


def just_area_green(parameters, alpha, integrx, integry, coefficients_matrix):
    """[summary]

    Args:
        parameters ([type]): parameters of your geometrical model
        alpha ([type]): rotation angle
        integrx ([type]): cumulative sum of the image(or geodesic distance map) along x direction
        integry ([type]): cumulative sum of the image(or geodesic distance map) along y direction
        coefficients_matrix ([type]): B-matrix

    Returns:
        [type]: average inside a given map (image or geodesic distance)
    """    
    coords = evaluate_spline(parameters.astype(np.float32), alpha, coefficients_matrix)
    x=coords[0]

    y=coords[1]

    xdif=x-np.roll(x,1)
    ydif=y-np.roll(y,1)
    binary_area=np.sum((x*ydif-y*xdif))/2
    weighted_area=np.sum((bilinear_interpolate(integrx,x,y).T*ydif-bilinear_interpolate(integry,x,y).T*xdif))/2
    average_area=weighted_area/binary_area

    return average_area
    
    
def evaluate_spline(parameters, alpha, coefficients_matrix):
    """[summary]

    Args:
        parameters ([type]): [description]
        alpha ([type]): [description]
        coefficients_matrix ([type]): [description]

    Returns:
        [type]: [description]
    """    
    center = [parameters[4], parameters[5]]

    points = np.array([
        [center[0] + parameters[2] * 0.5, center[1]],
        [center[0] - parameters[3] + parameters[2] * 0.5, center[1] + parameters[1]],
        [center[0] - parameters[3] - parameters[2] + parameters[2] * 0.5, center[1] + parameters[1]],
        [center[0] - parameters[2] + parameters[2] * 0.5, center[1]],
        [center[0] - parameters[6] - parameters[2] + parameters[2] * 0.5, center[1] - parameters[0]],
        [center[0] - parameters[6] + parameters[2] * 0.5, center[1] - parameters[0]],
    ], np.float32)

    points = np.concatenate((
        np.cos(alpha) * (points[:, 0] - center[0]) - np.sin(alpha) * (points[:, 1] - center[1]) + center[0],
        np.sin(alpha) * (points[:, 0] - center[0]) + np.cos(alpha) * (points[:, 1] - center[1]) + center[1],
    )).reshape((2, -1))

    # generate discrete spline coordinates
    coords = np.dot(points, coefficients_matrix)
    return coords



def initialize_parameters_from_shape(shape, idx=0):
    """[summary]

    Args:
        shape ([type]): [description]
        idx (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """    
    return np.array([
        shape[idx] * 0.5,
        shape[idx] * 0.5,
        17,
        0,
        shape[0] * 0.5,
        shape[1] * 0.5,
        0
    ], dtype=np.float32)


def get_constraints(bounds):
    """[summary]

    Args:
        bounds ([type]): [description]

    Returns:
        [type]: [description]
    """    
    constraints = []
    for factor in range(len(bounds) - 1):
        lower, upper = bounds[factor]
        constraints += [
            {'type': 'ineq', 'fun': lambda par, lb=lower, i=factor: par[i] - lb},
            {'type': 'ineq', 'fun': lambda par, ub=upper, i=factor: ub - par[i]}
        ]

    factor = len(bounds) - 1
    lower, upper = bounds[factor]

    constraints += [
        {'type': 'ineq', 'fun': lambda par, lb=lower, i=factor: par[0] + par[1] - lb},
        {'type': 'ineq', 'fun': lambda par, ub=upper, i=factor: ub - (par[0] + par[1])}
    ]

    return constraints

def objective(args,n):
    n_frame=n
    image_file_name="data/name1.tif"
    image_data = imread(image_file_name).astype('float32')
    image = image_data[n_frame,  :, :]
    image = (image - image.min()) / (image.max() - image.min())
    bbox_path='data/gtframes/{num}.txt'
    file = np.loadtxt(bbox_path.format(num=n_frame))
    data = file[:, 1:5]

    data[:, 0] *= image.shape[1]
    data[:, 1] *= image.shape[0]
    data[:, 2] *= image.shape[1]
    data[:, 3] *= image.shape[0]

    border = 5

    min_maxes = np.c_[
                data[:, 0] - 0.5 * data[:, 2] - border,
                data[:, 0] + 0.5 * data[:, 2] + border,
                data[:, 1] - 0.5 * data[:, 3] - border,
                data[:, 1] + 0.5 * data[:, 3] + border].astype(np.int32)

    bboxes = []

    for x_min, x_max, y_min, y_max in min_maxes:
        bboxes.append((x_min, x_max, y_min, y_max))
    bboxes = np.array(bboxes)

    plt.figure()
    plt.imshow(image,cmap="gray")
    
    for bbox in bboxes:
        x_min, x_max, y_min, y_max = bbox

        number_control_points = 6  # number of Control Points
        number_points = 10  #  10  # number of discrete points per spline segment
        coefficients_matrix = coefficients_matrix_generator(control_points=number_control_points, num_points=number_points)

        intensity = image[int(y_min):int(y_max), int(x_min):int(x_max)].astype(np.float32)

        # some preprocessing
        intensity = gaussian_filter(intensity, sigma=2)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

        center=np.zeros(intensity.shape).astype(np.uint8)
        center[int(0.5*center.shape[0]),int(0.5*center.shape[1])]=1
        gmap=GeodisTK.geodesic2d_raster_scan(intensity, center, 1, 2)
        gmap = (gmap - gmap.min()) / (gmap.max() - gmap.min())
        integrgmapx=np.cumsum(gmap, axis=0)


        integrgmapy=np.cumsum(gmap, axis=1)

        gradient = (np.gradient(intensity)[0] ** 2 + np.gradient(intensity)[1] ** 2) ** 0.5

        intensity = gaussian_filter(intensity, sigma=2)



        gradient = gaussian_filter(gradient, sigma=3)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        gradient = np.clip(gradient, 0, 0.45)
        gradient = gaussian_filter(gradient, sigma=3)


        integrx=np.cumsum(intensity, axis=0)


        integry=np.cumsum(intensity, axis=1)

        ishape = intensity.shape
        #generate parameters from shape
        def initial_guess(parameters_, alpha_):
            return just_area_green(parameters_, alpha_, integrgmapx,
                                  integrgmapy,
                                  coefficients_matrix)

        if ishape[1] > ishape[0] and abs(ishape[1] / ishape[0] - 1) > 0.2:
            parameters = initialize_parameters_from_shape(ishape, idx=1)
            alpha = 0
        elif ishape[1] < ishape[0] and abs(ishape[1] / ishape[0] - 1) > 0.2:
            parameters = initialize_parameters_from_shape(ishape, idx=0)
            alpha = math.pi / 2
        elif abs(ishape[1] / ishape[0] - 1) < 0.2:
            parameters = initialize_parameters_from_shape(ishape, idx=1)

            if initial_guess(parameters, -math.pi / 4) > initial_guess(parameters, math.pi / 4):
                alpha = math.pi / 4
            else:
                alpha = -math.pi / 4

        bounds = [
            [9, 50],
            [9, 50],
            [17, 20],
            [-10, 10],
            [ishape[0] * 0.5 - 15, ishape[0] * 0.5 + 15],
            [ishape[1] * 0.5 - 15, ishape[0] * 0.5 + 15],
            [-10, 10],
            [max(ishape[0], ishape[1]) -10, 1.7*max(ishape[0], ishape[1])-5]
        ]

        constraints = get_constraints(bounds)
        maximum_iterations = 3

        optimizer = 'Cobyla'
        optimizer_options = dict(maxiter=1000000)

        coords = evaluate_spline(parameters, alpha, coefficients_matrix)
       
        for _ in range(maximum_iterations):
            def find_contour_super_c(alpha_):
                return calculate_loss_green(parameters, alpha_.astype(np.float32)[0], integrx, 
                                      integry,
                                      gradient, integrgmapx, 
                                    integrgmapy, coefficients_matrix,
                                      area_coefficient=args[0],
                                           gmap_coefficient=args[1])

            alpha_result = minimize(fun=find_contour_super_c, x0=alpha, method=optimizer, options=optimizer_options)

            alpha = np.float32(alpha_result.x.item())

            def find_contour_super_p(parameters_):
                return calculate_loss_green(parameters_, alpha, integrx, 
                                            integry, 
                                            gradient,integrgmapx, 
                                            integrgmapy, coefficients_matrix,
                                      area_coefficient=args[0],
                                    gmap_coefficient=args[1])

            parameters_result = minimize(fun=find_contour_super_p, x0=parameters, constraints=constraints,
                                         method=optimizer, options=optimizer_options)
            parameters = parameters_result.x.astype(np.float32)

        coords = evaluate_spline(parameters, alpha, coefficients_matrix)
        coords=np.c_[coords[1] + x_min, coords[0] + y_min]
        plt.plot(coords[:,0],coords[:,1],"y",linewidth=0.5)
    plt.show()
    return 

for i in range(23,24):
    objective([500, 500],i)
