#some useful function ......
import os
import numpy as num
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  


def set_axis(ax,labelsize=None,linewidth=None,direction=None,major_length=None,major_width=None,minor_length=None,minor_width=None):
	if not  labelsize==None:
		ax.tick_params(labelsize=labelsize)
	if not  direction==None:
		ax.tick_params(which='both',direction=direction)
	if not major_length==None: 
		ax.tick_params(which='major',length=major_length)
	if not major_width==None:
		ax.tick_params(which='major',width=major_width) 
	if not minor_length==None:
		ax.tick_params(which='minor',length=minor_length) 
	if not minor_width==None: 
		ax.tick_params(which='minor',width=minor_width) 
	
	if not linewidth==None:
		ax.spines['bottom'].set_linewidth(linewidth)
		ax.spines['left'].set_linewidth(linewidth)
		ax.spines['right'].set_linewidth(linewidth)
		ax.spines['top'].set_linewidth(linewidth)
def set_ticks(axis,major=None,minor=None,both=True):
	if not major==None:
		majorlocator=MultipleLocator(major)
		axis.set_major_locator(majorlocator)
		# if (length is not None):
		# 	axis.tick_params(which='major',length=length)
		# if (width is not None):
		# 	axis.tick_params(which='major',width=width)
		# if (labelsize is not None):
		# 	axis.tick_params(which='major',labelsize=labelsize)
	if not minor==None:
		minorlocator=MultipleLocator(minor)
		axis.set_minor_locator(minorlocator)
		# if (length is not None):
		# 	axis.tick_params(which='minor',length=length)
		# if (width is not None):
		# 	axis.tick_params(which='minor',width=width)
		# if (labelsize is not None):
		# 	axis.tick_params(which='minor',labelsize=labelsize)
	if both: 
		axis.set_ticks_position('both')
# purtrubing the flux within the error according to the normal ditribution or others
def MonteC(flux_dbsp,err_dbsp,model='normal'):
	flux=num.ones_like(flux_dbsp)
	if model=='normal':
		for i in range(0,len(flux_dbsp)):
			flux[i]=num.random.normal(flux_dbsp[i],err_dbsp[i])
	return flux
	
