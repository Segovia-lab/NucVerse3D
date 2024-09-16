# colorcodeRemoval.py
#
# Script for getting binary to color coded volume
#
# Implemented by David Ho, Soonam Lee, Chichen Fu 
# Date: 08-29-2017 7:00PM
#
# Copyright (c) 2017 The Board of Trustees of Purdue University
# All Rights Reserved
#--------------------------------------------------------------------------

import skimage.io as io
import numpy as np
from skimage import util
from skimage import measure
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation, ball
from scipy.interpolate import RegularGridInterpolator
# from itertools import count,ifilterfalse
import scipy.misc
import time
import os
from skimage.color import rgb2gray

def CC2(volume,thSmall = 0,B_morp = 0):
	X,Y,Z = volume.shape
	windowSize = 40
	thColormap = 50
	
	
	# if not os.path.exists(dirRGB):
	# 	os.makedirs(dirRGB)

	# Measure time
	start1 = time.time()

	## Read images
	bw = util.img_as_ubyte(np.zeros([X,Y,Z]))
	# bw_centroid = util.img_as_ubyte(np.zeros([X,Y,Z]))

	# for ii in range(0,Z):

	# 	filename = os.path.join(dirIn, "z%04d.png" % (ii+1))
	# 	# filename = dirIn + str(ii+1) + ".png"
	# 	bw[:,:,ii] = rgb2gray(io.imread(filename))
	bw = volume
#	print("read volume done")

	bw3 = util.img_as_ubyte(np.zeros([X,Y,Z,3])) # RGB volume
	# bw3_centroid = util.img_as_ubyte(np.zeros([X,Y,Z,3])) # RGB volume
	bw_r = np.zeros([X,Y,Z])
	bw_g = np.zeros([X,Y,Z])
	bw_b = np.zeros([X,Y,Z])
	bw = erosion(bw,ball(B_morp))

	## Connected components
	cc = measure.label(bw,connectivity = 1)
	props = measure.regionprops(cc)
#	print(np.amax(cc))
#	print("connected components done")

	print("pass cc")

	## Colormap - Read text files saved from MATLAB
	cmap = []
	ins = open("utils/cmap.txt", "r")
	for line in ins:
		line = line.strip().split("\t")
		line2 = [float(n) for n in line]
		line3 = [int(line2[0]), int(line2[1]), int(line2[2])]
		cmap.append(line3)

	ins.close()

	cmap2 = []

	# Dark color removal from colormap
	for i in range(0,len(cmap)):
		if cmap[i][0] > thColormap or cmap[i][1] > thColormap or cmap[i][2] > thColormap:
			cmap2.append(cmap[i])

	print("colormap done")


	## Assign color index and small area removal
	# thSmall = 50

	N = 0 # the number of nuclei

	for ii in range(0,np.amax(cc)):
	#	print(ii, props[ii].area)
		coord = props[ii].coords

		if props[ii].area > thSmall: # added by David on 9/5
			N = N+1
			# coord = np.argwhere(cc == ii)
			coord_max = np.amax(coord,axis = 0)
			coord_min = np.amin(coord,axis = 0)
			# Define a 3D window containing a nucleus
			bw_window = bw[np.amax([0,coord_min[0] - windowSize]):np.amin([X-1,coord_max[0] + windowSize]), np.amax([0,coord_min[1] - windowSize]):np.amin([Y-1,coord_max[1] + windowSize]) ,np.amax([0,coord_min[2] - windowSize]):np.amin([Z-1,coord_max[2] + windowSize])]
			# Choose a color which is not used in the cropped window
			# tt = next(ifilterfalse(set(bw_window.flatten()).__contains__, count(1)))
			kk = 1
			while kk in bw_window:
				kk = kk+1
			# bw[cc == ii] = kk

		else:	# small area removal process
			# bw[cc == ii] = 0
			kk = 0

		for i in range(props[ii].area):
			bw[coord[i,0], coord[i,1], coord[i,2]] = kk;	# Changed by Soonam on 9/14

	print("number of nuclei: " + str(N))
	print("small area removal done")


	## Assign colors
	for ii in range(0,X):
		for jj in range(0,Y):
			for kk in range(0,Z):
				if bw[ii,jj,kk] != 0:
					bw_r[ii,jj,kk] = cmap2[bw[ii,jj,kk]-1][0]
					bw_g[ii,jj,kk] = cmap2[bw[ii,jj,kk]-1][1]
					bw_b[ii,jj,kk] = cmap2[bw[ii,jj,kk]-1][2]

	# dilation
	bw_r = dilation(bw_r, ball(B_morp))
	bw_g = dilation(bw_g, ball(B_morp))
	bw_b = dilation(bw_b, ball(B_morp))

	bw3[:,:,:,0] = bw_r
	bw3[:,:,:,1] = bw_g
	bw3[:,:,:,2] = bw_b

	print("wrote color coded images")

	## Write images
	# for ii in range(0,Z):
	# 	filename = os.path.join(dirRGB, "z%04d" %(ii+1) + ".tif")
	# 	tifffile.imsave(filename,bw3[:,:,ii,:],compress=0)
	
#	print("wrote color coded images")


	## Measure time
	end1 = time.time()
	print("time: " + str(end1-start1) + " s")
	return bw3

# CC2(128,128,128)