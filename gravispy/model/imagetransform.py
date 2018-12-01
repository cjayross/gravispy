import numpy as np
import itertools as it
from PIL import Image
from ..geom import pix2sph, sph2pix, wrap, unwrap

def generate_lens_map(lens, res, args=(), prec=3):
    coords = list(it.product(*map(np.arange,res)))
    x, y = np.asarray(coords).astype(int).T
    theta, phi = pix2sph(x, y, res)

    arccos2 = lambda a: np.sign(unwrap(a))*np.arccos(a)
    arcsin2 = lambda a, k: np.pi*k + (-1)**k*np.arcsin(a)

    # consider alphas to be equal if they are the same up
    # to 3 decimals (by default). this reduces the amount of
    # calls to lens from possibly millions to only about 3000
    alpha = np.round(arccos2(np.cos(theta)*np.cos(phi)), prec)
    # compress alpha
    alphaz = np.unique(alpha)
    betaz = np.fromiter(lens(alphaz, *args), np.float64)
    # expand betaz
    beta_map = dict(zip(alphaz, betaz))
    beta = np.fromiter(map(beta_map.__getitem__, alpha), np.float64)

    # we will intentionally fail invalid calculations,
    # they will be filtered out afterward.
    # as such, we don't need to be warned that they occurred.
    errstate = np.seterr(all='ignore')

    sigma = np.sin(beta)/np.sin(alpha)
    mu, nu = map(lambda a: sigma*np.sin(a), [theta, phi])
    # this choice of k's needs to be scrutinized
    k1,k2 = map(lambda a: np.abs(unwrap(a))>np.pi/2, [theta, phi])
    psi, gamma = map(arcsin2, [mu, nu], [k1,k2])

    # cut out invalid results
    idxs = np.logical_not(np.isnan(psi) | np.isnan(gamma))
    keys = zip(x[idxs], y[idxs])
    values = zip(*sph2pix(psi[idxs], gamma[idxs], res))
    np.seterr(**errstate)

    return dict(zip(keys, values))

def apply_lensing(img, lens_map, res=None, color_mod=1.):
    if not res:
        res = img.size
    pix = img.load()
    new = Image.new(img.mode, res)
    new_pix = new.load()
    for pix_coord in it.product(*map(range, res)):
        # we currently aren't implementing hidden images
        try:
            map_coord = tuple(map(int,lens_map[pix_coord]))
            new_pix[pix_coord] = pix[map_coord]
        except KeyError:
            # the pixel defaults to black
            continue
    # temporary result
    new.save('output.png')

#Inputs are
#   -Filename : Filepath of base image file
#   -Phi : Array of size n, where Phi[i] = [phi0,phi1]
#   -Theta : Array of size n, where Theta[i] = [theta0,theta1]
#Points at (phi0,theta0) are transformed to (phi1,theta1)

#Function can display resultant image and/or save resultant image to file.
def imageTransform(fileName, phi, theta):
    swapCheck = set();
    #The addition is just using a linear multipler currently. Adjust value to fit desire. Currently 0.
    multiplier = 0.0;
    img = Image.open(fileName);
    ref = Image.open(fileName);
    col = img.size[0];
    row = img.size[1];
    pixels = img.load();
    pRef = ref.load();
    for i in range(len(phi)):
         x0 = phiTransform(phi[i][0],col-1);
         x1 = phiTransform(phi[i][1],col-1);
         y0 = thetaTransform(theta[i][0],row-1);
         y1 = thetaTransform(theta[i][1],row-1);

         if (x1,y1) in swapCheck and not((x0,y0) in swapCheck):

             #First Swap
             old = np.array( pixels[x1,y1]);
             new = np.array( pRef[x0,y0]);
             old = (old + multiplier * new).astype(int);
             if old[0] > 255:
                 old[0] = 255;
             if old[1] > 255:
                 old[1] =255;
             if old[2] > 255:
                 old[2] = 255;
             old = tuple(old);
             pixels[x1,y1] = old;

             #Second Swap
             swapCheck.add((x0,y0));
             pixels[x0,y0] = pRef[x1,y1];

         elif not( (x1,y1) in swapCheck) and  (x0,y0) in swapCheck:

             #First Swap
             swapCheck.add((x1,y1));
             pixels[x1,y1] = pRef[x0,y0];

             #Second Swap
             old = np.array( pixels[x0,y0]);
             new = np.array( pRef[x1,y1]);
             old = (old + multiplier * new).astype(int);
             if old[0] > 255:
                 old[0] = 255;
             if old[1] > 255:
                 old[1] =255;
             if old[2] > 255:
                 old[2] = 255;
             old = tuple(old);
             pixels[x0,y0] = old;
         elif not((x1,y1) in swapCheck) and not((x0,y0) in swapCheck):
             swapCheck.add((x0,y0));
             swapCheck.add((x1,y1));
             pixels[x0,y0] = pRef[x1,y1];
             pixels[x1,y1] = pRef[x0,y0];

    #Display Image:
    #img.show();

        #Save Image:
    img.save("output.jpg");


#X
def phiTransform(phi,width):
    x= np.rint((phi/(2*(np.pi)))*width);
    return x;
#Y
def thetaTransform(theta, height):
    y =np.rint(((np.cos(theta)+1)/2)*height);
    return y
