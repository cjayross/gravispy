import numpy as np;
from itertools import product

def raytrace(height, width, fov):
    #Height in pixel
    #Width in pixel
    #fov in degree

    rad = np.radians(fov);
    d = 1; #distance from viewer's origin to the pixel grid. This is hardcoded now, but could be passed as a parameter.

    #x and y are distance from the center to each edge of the rectangle
    x = np.abs(np.tan(rad/2)*d);
    y = np.abs(np.tan(rad/2)* np.sqrt(x**2 + d**2));

    #these measurements are the width and height of each pixel on the projection
    pX= (2*x)/width;
    pY = (2*y)/height;

    #creates a 3-d array to hold theta and phi for each pixel
    angMat = np.zeros((height,width,2));

    for i,j in product(range(width), range(height)):
        #xi and yj are coordinates of the centers of pixels. The origin is in the upper left corner of the pixel grid.
        xi = i*pX +pX/2;
        yj = j*pY +pY/2;
        #distance of vector from center of pixel to viewer
        rij = np.sqrt( (xi -x)**2 + (yj-y)**2 + d**2);
        #swapping coordinate bases
        xp = xi -x;
        yp = y - yj;
        #calculating the angles for pixel i,j
        phi = np.arctan(xp/d);
        theta= np.arccos(yp/rij);
        angMat[j][i] = [theta,phi];
    #End loop
    return angMat;

#End Function

#Testing

#print(raytrace(5,5,90));
