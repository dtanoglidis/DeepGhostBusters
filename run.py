#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import os
import numpy as np
import pylab as plt
import pandas as pd
import scipy.ndimage as nd
#import fitsio

import collections
import subprocess

import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.image
from matplotlib.patches import Rectangle

def fov_geometry(release='sva1',size=[530,454]):
    """
    Return positions of each CCD in PNG image for
    a given data release.

    Parameters:
        release : Data release name (currently ['sva1','y1a1']
        size    : Image dimensions in pixels [width,height]
    Returns:
        list    : A list of [id, xmin, ymin, xmax, ymax] for each CCD
    """

    SIZE=size
    WIDTH=SIZE[0]
    HEIGHT=SIZE[1]
    # CCDs belonging to each row
    ROWS = [ [3,2,1],                #range(3,0,-1),
             [7,6,5,4],              #range(7,3,-1),
             [12,11,10,9,8],         #range(12,7,-1),
             [18,17,16,15,14,13],    #range(18,12,-1),
             [24,23,22,21,20,19],    #range(24,18,-1),
             [31,30,29,28,27,26,25], #range(31,24,-1),
             [38,37,36,35,34,33,32], #range(38,31,-1),
             [44,43,42,41,40,39],    #range(44,38,-1),
             [50,49,48,47,46,45],    #range(50,44,-1),
             [55,54,53,52,51],       #range(55,50,-1),
             [59,58,57,56],          #range(59,55,-1),
             [62,61,60],             #range(62,59,-1)
             ]

    if release.lower() == 'sva1':
        # These are the old SV pngs, not the ones made for Y2A1
        # Boder padding in x,y; assumed symmetric
        PAD = [0,0] 
        ROWS = [r[::-1] for r in ROWS[::-1]]
    else:
        PAD = [0.02*WIDTH,0.02*HEIGHT]
        ROWS = ROWS

    NROWS = len(ROWS) # Number of rows
    NCCDS = [len(row) for row in ROWS]
    CCD_SIZE = [float(WIDTH-2*PAD[0])/max(NCCDS),
                float(HEIGHT-2*PAD[1])/NROWS] # CCD dimension (assumed to span image)

    ret = []
    for i,ccds in enumerate(ROWS):
        for j,ccd in enumerate(ccds):
            xpad = (SIZE[0] - len(ccds)*CCD_SIZE[0])/2.
            ypad = PAD[1]
            xmin = xpad + j*CCD_SIZE[0]
            xmax = xmin + CCD_SIZE[0]
            ymin = ypad + i*CCD_SIZE[1]
            ymax = ymin + CCD_SIZE[1]
            # These are output as ints now
            ret += [[int(ccd), int(xmin), int(ymin), int(xmax), int(ymax)]]
    return sorted(ret)

def draw_png(url):
    png = os.path.basename(url)
    if os.path.exists(png): os.remove(png)
    subprocess.check_call('wget %s'%url,shell=True)
    image = matplotlib.image.imread(png)
    ax = plt.gca()
    ax.axis('off')
    ax.imshow(image,cmap='gray',interpolation='none')
    if os.path.exists(png): os.remove(png)
    ax.annotate('png',(0.05,0.9),xycoords='axes fraction',ha='left',fontsize=10,
                bbox={'boxstyle':'round','fc':'white'})
    return image

def draw_fov(png,ccds=[],release='y1a1'):
    ax = plt.gca()
    ax.axis('off')
    ax.imshow(fov,cmap='gray',interpolation='none')
    ret = fov_geometry(release='y1a1',size=png.shape[::-1])
    patches = []

    for i,x1,y1,x2,y2 in ret:
        WIDTH = np.abs(x2-x1)
        HEIGHT = np.abs(y2-y1)
        rect = Rectangle((x1,y1),WIDTH,HEIGHT,fc='none',ec='w',lw=2)
        ax.add_artist(rect)
        patches.append(rect)
        center = (x2+x1)/2.,(y2+y1)/2.
        ax.annotate(i,xy=center,color='w',ha='center',va='center',fontsize=10)

    for ccd in badccds:
        i,x1,y1,x2,y2 = ret[ccd-1]
        fov[y1:y2,x1:x2] = 1
        center = ((x1+x2)/2.,(y1+y2)/2.)
        ax.annotate(i,xy=center,color='k',ha='center',va='center',fontsize=10)
    ax.imshow(fov,cmap='gray',interpolation='none')
    ax.annotate('blacklist',(0.05,0.9),xycoords='axes fraction',ha='left',fontsize=10,
                bbox={'boxstyle':'round','fc':'white'})
    return fov,patches

def draw_mini(url):
    ax = plt.gca()
    mini = os.path.basename(url)
    if os.path.exists(mini): os.remove(mini)
    subprocess.check_call('wget %s'%url,shell=True)
    fp = fitsio.read(mini)
    fp_img = -1*np.ones((202,224))
    fp_img[5:-5,:] = fp[::-1].T
    vmin,median,vmax = np.percentile(fp[fp!=-1],q=[5,50,95])
    ax.axis('off')
    im = ax.imshow(fp_img,cmap='gray',interpolation='none',vmin=vmin,vmax=vmax)
    ax.annotate('mini',(0.05,0.9),xycoords='axes fraction',ha='left',fontsize=10,
                bbox={'boxstyle':'round','fc':'white'})
    if os.path.exists(mini): os.remove(mini)
    return fp_img, im, [vmin,vmax]

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('filename',default='ghost-scatter-y6.txt')
    parser.add_argument('-f','--force',action='store_true')
    args = parser.parse_args()

    bl = np.genfromtxt(args.filename,names=True,dtype=int)
    bl.dtype.names = map(str.upper,bl.dtype.names)
    bl = bl[np.argsort(bl['EXPNUM'])]

    urls = pd.read_csv('urls-y6a1.csv').to_records(index=False)
    urls = urls[np.argsort(urls['EXPNUM'])]
    urls = urls[np.in1d(urls['EXPNUM'],np.unique(bl['EXPNUM']))]

    outdir = 'pngs'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i,(expnum,url) in enumerate(urls):
        print ("(%i/%i)"%(i+1,len(urls)))
        outfile = os.path.join(outdir,os.path.basename(url).replace('_TN',''))

        if os.path.exists(outfile) and not args.force:
            print("Found %s; skipping..."%outfile)
            continue

        fig,ax = plt.subplots(1,2,figsize=(10,6))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)

        plt.sca(ax[0])
        image = draw_png(url)
     
        fov = np.zeros_like(image)
        ret = fov_geometry(size=fov.shape[::-1])
        badccds = bl['CCDNUM'][bl['EXPNUM'] == expnum].astype(int)
        print ("Blacklist CCDs:",badccds)
        plt.sca(ax[-1])
        draw_fov(fov,ccds=badccds)

        title = os.path.splitext(outfile)[0]
        plt.savefig(outfile,bbox_inches='tight')
        plt.close()

#plt.ion()
