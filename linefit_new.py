from astropy.io import fits
from create_fits import create_fits
import mpfitexpr as mpfit
import numpy as num
#import matplotlib 
#matplotlib.use('agg')
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from linefit_library import WAVE_LINE,SCALE_TIED,composite_spectra_width
import scipy,os,copy
import scipy.stats
from library import MonteC,set_axis,set_ticks
import multiprocessing as multip
from pickle import dumps
from scipy.stats import f as ft
import re
import traceback


'''
Input parameter: 
wave, flux, err 
path_savefig: the path to save fitting figure 
path_savefile: the path to save the initial and best fit parameters 
path_reconstruct: the path to the file which was created by the this procedure, and it will extract the file contents to use 
narrow_up: the upperlimits of the narrow lines 
broad_low: the lowerlimits of the broad lines 

'''
class linefit:
	# wave=None or wave here?,call the function must instantiate the obj?
	def __init__(self,wave=None,flux=None,err=None,path_savefig=None,path_savefile=None,path_reconstruct=None,show=False,quiet=True,narrow_up=800,broad_low=1000,modify_initial_model='random',fit1=True,modifyP=True,MC=False,MC_number=1000,MC_saveresults=False,MC_savefig=False,MC_pool_number=6,desired_chi2=1,covert2simple=True,telluric=False,redshift=None,extra_mask=None):
# the narrow_up and broad_low was FWHM limits
#If there is no input for some necessary parameters, like 'wave',please input 'None'
		self.quiet=quiet
		self.wave0=wave
		if str(self.wave0)==str(None): self.loglam0=None
		else: self.loglam0=num.log(wave)
		self.flux0=flux
		self.err0=err
		self.telluric=telluric 
		self.extra_mask=extra_mask
		#NOTE we can write a function named get_mask, which used to mask some bad pixels like telluric region 
		#NOTE HOW about when the wave0 is None ? TODO when we do the reconstruction? Maybe we need to save the mask information, maybe we can always 
		#NOTE save the mask information and wave0 
		if self.wave0 is not None:
			ind_mask=self.wave0>0 
			if self.telluric:
				if redshift==None: raise Exception('Redshift is necessary for mask telluric region') 
				else: 
					self.ind_telluric=( (self.wave0*(1+redshift)>6860) & (self.wave0*(1+redshift)<6960) ) | \
						( (self.wave0*(1+redshift)>7160) & (self.wave0*(1+redshift)<7340) ) | \
						( (self.wave0*(1+redshift)>7580) & (self.wave0*(1+redshift)<7750) ) | \
						( (self.wave0*(1+redshift)>8150) & (self.wave0*(1+redshift)<8250) ) 
						# ( (self.wave0*(1+redshift)>6276) & (self.wave0*(1+redshift)<6310) )
					ind_mask=ind_mask & (self.ind_telluric==False)
			if not self.extra_mask==None:
				if not extra_mask[0].__class__ ==list: extra_mask=[extra_mask] 
				self.ind_extramask=self.wave0<0 
				for maskrange in extra_mask:
					print('ExtraMask: %s'%maskrange)
					self.ind_extramask=self.ind_extramask |  ((self.wave0>maskrange[0])&(self.wave0<maskrange[1]))
			#	print(len(self.ind_extramask[self.ind_extramask==True]))
				ind_mask=ind_mask & (self.ind_extramask==False)
			self.wave,self.flux,self.loglam,self.err=self.wave0[ind_mask],self.flux0[ind_mask],self.loglam0[ind_mask],self.err0[ind_mask]
			self.mask_array=ind_mask 
		else:
			self.wave,self.flux,self.loglam,self.err=None,None,None,None
			self.mask_array=None
	#	ind= num.isnan(self.err) | num.isnan(self.flux) | num.isinf(self.err)
	#	print(ind)
	#	self.wave=self.wave[ind==False];self.flux=self.flux[ind==False];self.wave=self.wave[ind==False];self.loglam=self.loglam[ind==False]
		self.path_savefig=path_savefig
		self.path_savefile=path_savefile
		self.narrow_up=narrow_up
		self.broad_low=broad_low
		self.components,self.ValueLimits,self.SigmaLimits={},{},{}
		self.MC=MC 
		self.MC_number=MC_number
		self.MC_saveresults=MC_saveresults # if True, the temporary fitting results of MC procedure will be saved
		self.MC_pool_number=6 #the number of pool used to implements the multi-prcoessing fitting when do the MC procedure
		self.MC_savefig=MC_savefig
		self.modify_initialp_model=modify_initial_model
		self.show=show
		self.covert2simple=covert2simple #if True, it will used the old format to save the results
		self.reconstruct=False 
		self.plot_upperlimit=False
		self.desired_chi2=desired_chi2
		self.fit1=fit1 # if fit1==False, then we do not need to fit the source excpet for doing the MC procedure, it was proper for those re-contruct from an already fitting results
		self.path_reconstruct=path_reconstruct
		self.modifyP=modifyP #if False, we take the parameter values of fitting results as the initial value and don't implement the modify_initialp function when do the fitting
		if self.modify_initialp_model=='random':
			self.modify_initialp_total=20.0
		else:
			self.modify_initialp_total=10.0
			self.modify_initialp_ith=0
		#plt.switch_backend('Agg')

	def add_gauss(self,line_number,line_name,n_or_b,sigma=None,scale=None,value=None,sigma_tied=-1,scale_tied=-1,value_tied=-1,sigma_fixed=False,scale_fixed=False,value_fixed=False,value_limits=None,sigma_limits=None,scale_limits=None):
		#take default value for "sigma,scale and value"  if special values was not assigned
		if not value:
			value=num.log(WAVE_LINE[line_name])
		if not sigma:
			if sigma_limits:
				sigma=sigma_limits[0]/3e5/2.355*1.1 if n_or_b=='b' else (sigma_limits[0]+sigma_limits[1])/2./3e5/2.355
				self.SigmaLimits[float(line_number)]=sigma_limits
			else:
				if n_or_b=='b':
					sigma=self.broad_low/3e5/2.355*1.1					
				elif n_or_b=='n':
					sigma=self.narrow_up/3e5/2.355*0.5
				else:
					sigma=(self.narrow_up+self.broad_low)/2.
		if not scale:
			scale=1

		try:value_tied=float(value_tied)
		except:pass
		try:sigma_tied=float(sigma_tied)
		except:pass
		try:scale_tied=float(scale_tied)
		except:pass
#       component=[col0,      col1     2      3     4     5      6          7          8          9            10          11     ]
		component=[line_name,'gauss',n_or_b,value,sigma,scale,value_tied,sigma_tied,scale_tied,value_fixed,sigma_fixed,scale_fixed]
		if float(line_number) in self.components.keys():
			raise Exception('components number %s alread exists'%line_number)
		self.components[float(line_number)]=component # float the input line_number, because it will be floated when we save the results
		if  value_limits:
			self.ValueLimits[float(line_number)]=value_limits
		if sigma_limits:
			self.SigmaLimits[float(line_number)]=sigma_limits
	@staticmethod
	def flux_gauss(wave,value,sigma,scale):
		flux=scale*scipy.stats.norm.pdf(num.log(wave),value,sigma)
		flux_integrate=sum([flux[x]*(wave[x+1]-wave[x]) for x in range(0,len(flux)-1)])
		return flux_integrate
	def eval_comp(self,index,loglam=None):
		if loglam is None: loglam=self.loglam0
		component=self.components[index] # When you try to get something from the dict use the index, maybe the int format of index is alright, but when you print the key of a dictionary , it seems that it will be different depends on of which float or int you used
		if component[1]=='gauss':
			value,sigma,scale=component[3],component[4],component[5]
			print(value,sigma,scale)
			flux=scale*scipy.stats.norm.pdf(loglam,value,sigma)
		elif component[1]=='poly':
			order=component[0]
			flux=component[2]*num.ones_like(loglam)
			print('order',order)
			for i in range(1,int(order)+1):
				flux=flux+component[2+i]*loglam**i
		return flux
	def get_yfit(self,loglam=None):
		#TODO we can merge this with the function eval_comp
		if loglam is None: loglam=self.loglam0 
		yfit=0

		params=self.pmin['params']

		for key in self.components.keys():
			component=self.components[key]
			if component[1]=='gauss':
			#	value,sigma,scale=component[3],component[4],component[5] #NOTE only when we after the doing the self.save_results, this sentence will be right 
			#   but what if we use the fit1=False, and get the results from a savefile ?
				scale_index,value_index,sigma_index=component[12],component[13],component[14]
				scale,value,sigma=params[scale_index],params[value_index],params[sigma_index]

				print(value,sigma,scale)
				flux=scale*scipy.stats.norm.pdf(loglam,value,sigma)
			elif component[1]=='poly':
				order=int(component[0])
				flux=0
				for j in range(0,order+1):
					aj_index=component[j+(order+1)*3+2]
					aj=params[aj_index]
					flux=flux+aj*loglam**j
			yfit=yfit+flux 
		return yfit 
	def flux_component(self,wave,key=None,component_name=None):
		logw=num.log(wave)
		if key is not None:
			if key.__class__==list:
				keys=key
			else: keys=[key]
		elif component_name is not None:
			# only surpport the gauss components
			keys=[]
			for keyname in self.compoents.keys():
				lname=self.components[keyname][0]+'_'+self.components[keyname][2]
				if lname==component_name:
					keys.append(keyname)
		else:
			raise Exception('Must give the key or component_name')

		flux=0
		for key in keys:
			component=self.components[key]
			if component[1]=='gauss':
				flux_component=component[5]*scipy.stats.norm.pdf(logw,component[3],component[4])
			elif component[1]=='poly':
				order=component[0]
				flux_component=0
				for i in range(order+1):
					flux_component=flux_component+component[2+i]*logw**i 
			flux=flux+flux_component
		return flux
	def mocked_line_spec(self,model='gauss',wave=num.arange(3500,7500,0.01),err_scale={},dw=1,show=True):
		components=self.components
		flux,ferr=0,0
		line_flux,line_ferr={},{}
		for key in components.keys():
			component=components[key]
			if component[1]=='gauss':
				flux_component=component[5]*scipy.stats.norm.pdf(num.log(wave),component[3],component[4])
				flux_err_scale=err_scale.get(key,1/10)
				flux_err_scales=num.random.uniform(flux_err_scale*0.9,flux_err_scale*1.1,len(flux_component))
				flux_err=flux_err_scales*flux_component
				flux_component=MonteC(flux_component,flux_err,model='normal')
				flux=flux+flux_component
				ferr=(ferr**2+flux_err**2)**0.5
				lineflux=sum(flux_component*0.01)
			#	print(sum(flux_component)/sum(flux_err))
				lineferr=sum(flux_err**2)**0.5*0.01
				line_flux[str(key)]=lineflux
				line_ferr[str(key)]=lineferr
			elif component[1]=='lorentz':
				print('Add this part if neccessary')
			else:
				raise Exception('No such component type')
		index=num.arange(0,len(wave),int(dw/0.01))
		wave,flux,ferr=wave[index],flux[index],ferr[index]
		if show:
			plt.plot(wave,flux,wave,ferr)
			plt.show()
		
		return wave,flux,ferr,line_flux,line_ferr

	def add_lorentz(self,line_number):
		#add this part if neccessary
		return 
	def add_linear(self,component_number,component_name,k=None,b=None,k_tied=-1,b_tied=-1,k_fixed=False,b_fixed=False):
		#add a liner function component, usually for a local continuum 
		
		if not k:
			k=1
		if not b:
			b=1
	#   component=[col0,            1      2 3   4      5      6        7   ]	
		component=[component_name,'linear',k,b,float(k_tied),float(b_tied),k_fixed,b_fixed]
		if float(component_number) in self.components.keys():
			raise Exception('components number %s alread exists'%component_number)
		self.components[float(component_number)]=component
	def add_poly(self,component_number,order=0,**params):
		#the coefficient of the poly was expressed as: a0+a1*x+a2*x^2+....
		#NOTE, 是否需要考虑， 不同成分（poly, gaussian),初始参数下，量级相差太多（可能差的不多？）会导致拟合出现问题？
		if order<0:
			raise Exception('The order can not be negative')
		component=[int(order),'poly']
		pcs=['a%d'%i for i in range(0,order+1)] # pcs: the Poly Coeffcients
		for pc in pcs:
			if pc not in params.keys(): component.append(1)
			else: component.append(params[pc])
		for pc in pcs:
			if (pc+'_fixed') not in params.keys(): component.append(False)
			elif params[pc+'_fixed']==True: component.append(True)
			else: raise Exception('Wrong value of parameter %s_fixed: %s'%(pc,params[pc+'_fixed']))
		for pc in pcs:
			if (pc+'_tied') not in params.keys(): component.append(-1)
			else: component.append(float(params[pc+'_tied']))
			
		if float(component_number) in self.components.keys():
			raise Exception('components number %s alread exists'%component_number)
		self.components[float(component_number)]=component
	@staticmethod
	def re_construct(path_reconstruct,reconstruct_fitting=False):
		hdu=fits.open(path_reconstruct)
		#get the self.components
		components,paramsN={},0
		for key in hdu[2].data.dtype.names:
			dataarray=hdu[2].data[key]
			if dataarray[1]=='gauss':
				dataarray=dataarray[0:12]
				paramsN=paramsN+3
			elif dataarray[1]=='linear':
				dataarray=dataarray[0:8]
				paramsN=paramsN+2
			elif dataarray[1]=='poly':
				order=int(float(dataarray[0])) # Notation: int('3.0') will raise an Exception
				dataarray=dataarray[0:(order+1)*3+2]
				paramsN=paramsN+order+1
			elif dataarray[1]=='lorentz':
				print('Add this part if lorentz profile was necessary!!')
			else:
				raise Exception('No such component type: %s'%dataarray[1])

			component=[]
			for data in dataarray:
				try:
					component.append(float(data))
				except:
					if data=='False': component.append(False)
					elif data=='True': component.append(True)
					else: component.append(data)
		
			components[float(key)]=component

#		print(dataarray)
#		print(components)
		wave=hdu[1].data['wave'].copy()
		flux=hdu[1].data['flux'].copy()
		err=hdu[1].data['err'].copy()
		try:
			mask=hdu[1].data['mask'].copy() 
		except:
			mask=wave>0

		if reconstruct_fitting:
			yfitmin=hdu[1].data['flux_fit'].copy()
			linefs,lineferrs,line_sigma,line_scale,line_value={},{},{},{},{}
			for key in hdu[1].header.keys():
			#TODO files produced by old version didn't have the key of line_scale and line_value, maybe we can reconstruct them 
			#TODO from the data stored in the hdu[2].data  
				if 'flux' in key:
					linefs[key.replace('_flux','')]=hdu[1].header[key]
				if 'ferr' in key:
					lineferrs[key.replace('_ferr','')]=hdu[1].header[key]
				if '_sigma' in key:
					line_sigma[key.replace('_sigma','')]=hdu[1].header[key]
				if '_scale' in key: 
					line_scale[key.replace('_scale','')]=hdu[1].header[key]
				if '_value' in key:
					line_value[key.replace('_value','')]=hdu[1].header[key]

			chi2_min=hdu[0].header['chi2_min']
			dof=len(flux)-hdu[0].header['parameter_number'] # Notation: The parameter_number here is the number of free parameters
			fnorm=dof*chi2_min

			p={'dof':dof,'fnorm':fnorm,'params':[0]*paramsN,'perror':[0]*paramsN} #The elements in the list [0]*paramsN won't keep the same with each other, 
			#because zero '0' here is non-variable. But if you define a list like [{'value':0}]*3, due to the variable elements, the three elements would keep the same as each other
			for key in hdu[2].data.dtype.names:
				data=hdu[2].data[key]
				if data[1]=='gauss':
					value_index,sigma_index,scale_index=int(data[13]),int(data[14]),int(data[12])
					p['params'][value_index],p['params'][sigma_index],p['params'][scale_index]=float(data[3]),float(data[4]),float(data[5])
					p['perror'][value_index],p['perror'][sigma_index],p['perror'][scale_index]=float(data[15]),float(data[16]),float(data[17])
				elif data[1]=='lorentz':
					print('Add this part if neccessary')
				elif data[1]=='linear':
					k_index,b_index=int(data[8]),int(data[9])
					p['params'][k_index],p['params'][b_index]=float(data[2]),float(data[3])
					p['perror'][k_index],p['perror'][b_index]=float(data[10]),float(data[11])
				elif data[1]=='poly':
					order=int(float(data[0]))
					for j in range(0,order+1):
						aj_index=int(data[2+j+(order+1)*3])
						p['params'][aj_index],p['perror'][aj_index]=float(data[2+j]),float(data[2+j+(order+1)*4])
				else:
					raise Exception ('No such components type: %s'%data[1])
			#get the sigma_limits and value_limits from the hdu[2]
			sigma_limits,value_limits={},{}
			for key in hdu[2].header.keys():
				if 'sigmalimits' in key:
					sigma_limits[float(key.split('_')[0])]=[float(hdu[2].header[key].split()[0]),float(hdu[2].header[key].split()[1])]
				if 'valuelimits' in key:
					value_limits[float(key.split('_')[0])]=[float(hdu[2].header[key].split()[0]),float(hdu[2].header[key].split()[1])]
			fitting_results=[p,yfitmin,chi2_min,linefs,lineferrs,line_sigma,sigma_limits,value_limits,line_scale,line_value]
		else: fitting_results=None
					
		hdu.close()
		return components,wave,flux,err,fitting_results,mask

	def params_reconstruct(self):
		if not self.path_reconstruct:
			raise Exception('Please give the path to the file using to re-construct')
		if self.fit1:
			self.components,self.wave0,self.flux0,self.err0,fit_results,mask=self.re_construct(self.path_reconstruct)

			self.mask_array=mask; self.loglam0=num.log(self.wave0)
			self.wave=self.wave0[mask];self.flux=self.flux0[mask];self.err=self.err0[mask];self.loglam=self.loglam0[mask]

		else:
			self.components,self.wave0,self.flux0,self.err0,fit_results,mask=self.re_construct(self.path_reconstruct,reconstruct_fitting=True)

			self.mask_array=mask; self.loglam0=num.log(self.wave0)
			self.wave=self.wave0[mask];self.flux=self.flux0[mask];self.err=self.err0[mask];self.loglam=self.loglam0[mask]

			self.pmin,self.yfitmin,self.chi2min,self.linefs,self.lineferrs,self.line_sigma,self.SigmaLimits,self.ValueLimits,self.line_scale,self.line_value=fit_results
		
			self.yfitmin=self.yfitmin[mask]
			self.components_initialp()
	

	@staticmethod
	def is_number(a):
		try:
			float(a)
			return True
		except:
			return False
	def comParams(self,name,n_or_b=None):
		#we must using this function ,afte we have ran the function lf.running and lf.saveresults, or the results was reconstructed from a fitted lines 
		if n_or_b==None:
			choice=name 
		else:
			choice=name+'_'+n_or_b
		params=self.pmin['params']
		perror=self.pmin['perror']

		sigmas,scales,values={'value':[],'err':[]},{'value':[],'err':[]},{'value':[],'err':[]}
		for key in self.components.keys():
			component=self.components[key]
			if not component[1]=='gauss':
				continue
			if (component[0]+'_'+component[2])==choice:
				value_index,sigma_index,scale_index=component[13],component[14],component[12]
			#	values['value'].append(component[3]);sigmas['value'].append(component[4]);scales['value'].append(component[5])
				values['value'].append(params[value_index]);sigmas['value'].append(params[sigma_index]);scales['value'].append(params[scale_index])
				values['err'].append(perror[value_index]);sigmas['err'].append(perror[sigma_index]);scales['err'].append(perror[scale_index])
		return values,sigmas,scales

	def FWHM(self,name,n_or_b,MC=False,MC_number=300,show=False):
		values,sigmas,scales=self.comParams(name,n_or_b)
#		print(values)
		if len(sigmas['value'])==1:
		#	print(sigmas['value'][0])
			return sigmas['value'][0]*2.355*3e5,sigmas['err'][0]*2.355*3e5
		elif len(sigmas['value'])==0:
			return None,None
		flux,wave=0,num.arange(1000,10000,0.1)
		flux_broad,wmax=0,0
		if not MC:
			for i in range(0,len(values['value'])):
				linef=self.flux_gauss(wave,values['value'][i],sigmas['value'][i],scales['value'][i])
				wmax=wmax+linef*num.e**values['value'][i]
				flux_broad=flux_broad+linef
				flux=flux+scales['value'][i]*scipy.stats.norm.pdf(num.log(wave),values['value'][i],sigmas['value'][i])
			wmax=wmax/flux_broad
			if show:
				plt.plot(wave[flux>0],flux[flux>0])
				plt.show()
			fmax=num.max(flux)
		#	wmax=wave[flux==fmax][0]
			whalf=wave[flux>fmax/2]
			if whalf[len(whalf)-1]>=9999.9:
				raise Exception('The wavelength upper limits 10000 was not enough for the line')
			print(wmax)
			fwhm=(whalf[len(whalf)-1]-whalf[0])/wmax*3e5
			return fwhm, 0
		else:
			# Maybe we can change this procedure using multiprocessing 
			#TODO we need to correct this part 
			raise Exception('Warning: The error given by the MC procedure was not exact,since different components was not indenpendent')
			fwhms,fwhm0s=[],[]
			for j in range(0,MC_number):
				flux,wmax,flux_broad=0,0,0
				for i in range(0,len(values['value'])):
					value=num.random.normal(values['value'][i],values['err'][i])
					scale=num.random.normal(scales['value'][i],scales['err'][i])
					sigma=num.random.normal(sigmas['value'][i],sigmas['err'][i])
					flux=flux+scale*scipy.stats.norm.pdf(num.log(wave),value,sigma)
					linef=self.flux_gauss(wave,value,sigma,scale)
					flux_broad=flux_broad+linef
					wmax=wmax+linef*num.e**value
				wmax=wmax/flux_broad
				print(wmax)
				fmax=num.max(flux)
			#	wmax=wave[flux==fmax][0]
				whalf=wave[flux>fmax/2]
				fwhm=whalf[len(whalf)-1]-whalf[0]
			
				if whalf[len(whalf)-1]>=9999.9:
					raise Exception('The wavelength upper limits 10000 was not enough for the line')
				fwhm0s.append(fwhm)
				fwhms.append(fwhm/wmax*3e5)
			fwhm=num.mean(fwhms)
			fwhm0_err=num.std(fwhm0s,ddof=1)
			if fwhm0_err<=0.5:
				print('Warning: the wavelength resolution was not enough to get a robust error!!!')
			fwhm_err=num.std(fwhms,ddof=1)
			
	
			return fwhm, fwhm_err		


		
	def tied_decomp(self,Tie,get_index=False):
		#this function was wrotten for the gaussian components
		Tie=Tie.lower()
		print(Tie)
		Tie=Tie.replace('+-','-').replace('-+','-').replace('--','+').replace('++','+')
		ties=Tie.replace('e+','e').replace('e-','e').replace('(','').replace(')','').replace('/','+').replace('-','+').replace('*','+').split('+')
		Indexs=[]
		for ti in ties:
	#		print(ti)
			key=float(re.sub('\D','',ti))
			t=re.sub('\d','',ti)
	#		print(ti)
			if t in ['sc','sca','scal','scale']:
				index=self.components[key][12]
				Indexs.append(key)
			elif t in ['si','sig','sigm','sigma']:
				index=self.components[key][14]
				Indexs.append(key)
			elif t in ['va','val','valu','value']:
				index=self.components[key][13]
	#			print(ti)
				Indexs.append(key)
			elif linefit.is_number(ti):
				continue
			elif t in ['.','','.e']:
				continue
			else:
				raise Exception('when do the ties decomposition,Can not match the item:%s'%t)
	#		print(ti,index)
			Tie0=Tie.replace(ti,'p[%d]'%index,1)
			if Tie0==Tie:
				raise Exception('substituion was not implemented for the item :%s'%ti)
			Tie=Tie0
		print(Tie)
		if get_index==True:
			return Indexs 
		return Tie

	def components_initialp(self):
		Par=[]
		func=''
		index=0
		self.line_keys=[] #record the keys of the line components

		components_keys=[]
		for key in self.components.keys():
			components_keys.append(key)
		components_keys.sort() #I think we should get the same keys in a same order when we run the self.components. 
		#But these sentence was runned to ensure that the components_key was in a same order when we run the procedure at different time
		for key in components_keys:
			component=self.components[key]
			if component[1]=='gauss':
				self.line_keys.append(key)
				self.components[key]=component[0:12]
				Par=Par+[{'value':0},{'value':0},{'value':0}]
				Par[index]['value']=component[3]
				Par[index]['parname']=component[0]+'_wave'
				Par[index]['limited']=[1,1]
				try:
					Par[index]['limits']=[self.ValueLimits[key][0]/num.e**component[3]+component[3],self.ValueLimits[key][1]/num.e**component[3]+component[3]]
				except:
					Par[index]['limits']=[-10./num.e**component[3]+component[3],10./num.e**component[3]+component[3]]
				Par[index]['fixed']=0 if component[9]==False else 1
				if (component[3]<Par[index]['limits'][0] )| (component[3]>Par[index]['limits'][1]):
					print(index,'value_limits',num.e**Par[index]['value'],[num.e**Par[index]['limits'][0],num.e**Par[index]['limits'][1]])
				
				Par[index+1]['value']=component[4]
				Par[index+1]['parname']=component[0]+'_sigma'
				Par[index+1]['limited']=[1,1]
				try:
					Par[index+1]['limits']=[self.SigmaLimits[key][0]/3e5/2.355,self.SigmaLimits[key][1]/3e5/2.355]
				except:
					Par[index+1]['limits']=[self.broad_low/3e5/2.355,7000./3e5/2.355] if component[2]=='b' else [60./3e5/2.355,self.narrow_up/3e5/2.355]
				Par[index+1]['fixed']=0 if component[10]==False else 1
				if (component[4]<Par[index+1]['limits'][0] )| (component[4]>Par[index+1]['limits'][1]):
					print(index+1,'sigma_limits',component[4],Par[index+1]['limits'])
				Par[index+2]['value']=component[5] if component[5]>1e-128 else 1e-128 # The fitting results can be lower than 1e-128 due to the limits of numerical computation, and so when re-construct it will raise an error
				Par[index+2]['parname']=component[0]+'_scale'
				Par[index+2]['limited']=[1,0]
				Par[index+2]['limits']=[1e-128,0] if component[2]=='n' else [0,0] # set the low limits of narrow lines >0, and therefore the tied between two groups of narrow lines won't raise runtime warning due to 0/0
				Par[index+2]['fixed']=0 if component[11]==False else 1
				if ((component[5]<Par[index+2]['limits'][0] )&(Par[index+2]['limited'][0]==1)) | ((component[5]>Par[index+2]['limits'][1])&(Par[index+2]['limited']==1)):
					print(index+2,'scale_limits',component[5],Par[index+2]['limits'])
				func=func if index==0 else func+'+'
				func=func+'p[%d]*scipy.stats.norm.pdf(x,p[%d],p[%d])'%(index+2,index,index+1)

				self.components[key]=self.components[key]+[index+2,index,index+1]
				index=index+3
				continue
			elif component[1]=='lorentz':
				self.line_keys.append(key)
				print('Add this part if neccessary')
			elif component[1]=='linear':
				self.components[key]=component[0:8]
				Par=Par+[{'value':0},{'value':0}]
				
				Par[index]['value']=component[2]
				Par[index]['parname']=component[0]+'_k'
				Par[index]['limited']=[0,0]
				Par[index]['fixed']=0 if component[6]==False else 1
				
				Par[index+1]['value']=component[3]
				Par[index+1]['parname']=component[0]+'_b'
				Par[index+1]['limited']=[0,0]
				Par[index+1]['fixed']=0 if component[7]==False else 1

				func=func if index==0 else func+'+'
				func=func+'p[%d]*x+p[%d]'%(index,index+1)
				self.components[key]=self.components[key]+[index,index+1]
				index=index+2
				continue
			elif component[1]=='poly':
				order=int(component[0])
				self.components[key]=component[0:2+(order+1)*3]
				for i in range(0,order+1):
					Par=Par+[{'value':0}]

					Par[index]['value']=component[2+i]
					Par[index]['parname']='a%d'%i
					Par[index]['limited']=[0,0]
					Par[index]['fixed']=0 if component[2+order+1+i]==False else 1

					func=func if index==0 else func+'+'
					func=func+('p[%d]'%index if i==0 else 'p[%d]*x**%d'%(index,i)) # Notation: the bracket was must
					self.components[key]=self.components[key]+[index]
					index=index+1

			else:
				raise NameError('There is no matching component type: %s'%component[1])
		self.Par=Par
		self.func=func

		for key in self.components.keys():
			component=self.components[key]
			if component[1]=='gauss':
				if not component[6]==-1:
					if component[6].__class__.__name__=='float':
						tied_component=self.components[component[6]]
						if not tied_component[1]=='gauss':
							raise NameError('Components tied with each other must have the same type')
						self.Par[component[13]]['tied']='p[%d]+'%tied_component[13]+str(component[3]-tied_component[3])
					elif component[6].__class__.__name__=='str':
						self.Par[component[13]]['tied']=self.tied_decomp(component[6])
					else:
						raise Exception('Wrong type of the item that indiacated the tied relation :%s'%component[6])
		#			print(self.Par[component[13]]['parname'])
		#			print(self.Par[component[13]]['tied'])
		#			print(tied_component[0])
				if not component[7]==-1:
					if component[7].__class__.__name__=='float':
						tied_component=self.components[component[7]]
						if not tied_component[1]=='gauss':
							raise NameError('Components tied with each other must have the same type')
						self.Par[component[14]]['tied']='p[%d]'%tied_component[14]
					elif component[7].__class__.__name__=='str':
						self.Par[component[14]]['tied']=self.tied_decomp(component[7])
					else:
						raise Exception('Wrong type of the item that indiacated the tied relation :%s'%component[7])
		#			print(self.Par[component[14]]['parname'])
		#			print(self.Par[component[14]]['tied'])
		#			print(tied_component[0])
				if not component[8]==-1:
					if component[8].__class__.__name__=='float':
						tied_component=self.components[component[8]]
						if not tied_component[1]=='gauss':
							raise NameError('Components tied with each other must have the same type')
						self.Par[component[12]]['tied']=SCALE_TIED['%s,%s'%(component[0],tied_component[0])]%tied_component[12]
					elif component[8].__class__.__name__=='str':
						self.Par[component[12]]['tied']=self.tied_decomp(component[8])
					else:
						raise Exception('Wrong type of the item that indiacated the tied relation :%s'%component[8])
		#			print(self.Par[component[12]]['parname'])
		#			print(self.Par[component[12]]['tied'])
		#			print(tied_component[0])
			elif component[1]=='lorentz':
				print('Add this part if necessary')
			
		self.MC_need=False
		lines_components=[]
		for key in self.components.keys():
			component=self.components[key]
			if component[1]=='gauss':
				line_component=component[0]+component[2]
				if line_component not in lines_components:
					lines_components.append(line_component)
				else:
					self.MC_need=True
					break
		

	def modify_initialp(self):
		if self.modify_initialp_model=='step':
			for par in self.Par:
				if par['fixed']==1:
					continue
				if par['limited']==[1,0]:
					par['value']=par['value']*1.5
				elif par['limited']==[1,1]:
					par['value']=par['limits'][0]+(par['limits'][1]-par['limits'][0])/self.modify_initialp_total*self.modify_initialp_ith
				else:
					continue
			return ':)'
		elif self.modify_initialp_model=='random':
			for par in self.Par:
				if par['fixed']==1:
					continue
				if par['limited']==[1,0]:
					#how to determine a proper error? Or using other probability distibution function?
					par['value']=abs(num.random.normal(par['limits'][0],1)-par['limits'][0])+par['limits'][0]
				elif par['limited']==[1,1]:
					par['value']=num.random.uniform(par['limits'][0],par['limits'][1])
				else:
					continue 
			return ':)'
		else: 
			raise NameError('No such model for modifying the initial parameter values: %s'%self.modify_initialp_model)


	def fitting(self):
		self.pmin,self.chi2min,self.yfitmin=[],1e55,[]
		self.modify_initialp_ith=0
		if self.modifyP:
			while 1:
				if self.modify_initialp_ith !=0:
					self.modify_initialp()
				#why if there are somthing wrong with following function, it will raise the error in the second itration? For example, the lengths of err and flux was different. Because
				#the MPFIT won't stop the procedure and raise a error for some special errors, and instead it will return the error message in the p.errmsg
				p,yfit=mpfit.mpfitexpr(self.func,self.loglam,self.flux,self.err,None,quiet=self.quiet,check=False,full_output=True,imports=None,parinfo=self.Par)
				#print('fitting',yfit.__class__)
				if not p.errmsg=='':
					raise Exception('Mpfit return an error message: %s'%p.errmsg) # I have added this part in the mpfitexpr.py file and actually here it can be removed
				if p.status<=0:
					raise Exception('The returned status value is %s, less than zero'%(p.status))
				if p.status==5: 
					print('Warining:  The maximum number of iterations has been reached')
					print('Warining:  The maximum number of iterations has been reached')
	#			print('hello2',p.fnorm/p.dof)
				if not p.dof==0: #when the case 'pdof=0' happen?!!!. The initial p.dof value was set as zero, and it means some error happened causing that the MPFIT return the p directly without running
					print(p.fnorm/p.dof)
					if p.fnorm/p.dof <self.chi2min:
						self.chi2min=p.fnorm/p.dof
						self.pmin=p
						self.yfitmin=yfit
	#				if (self.chi2min<1.2) & (self.modify_initialp_ith>5):
					if self.chi2min<self.desired_chi2: #the criteria maybe can be produced by the SN of the input line spectra,or the criteria can be produced by the ftest ?
						self.pmin={'params':self.pmin.params,'perror':self.pmin.perror,'fnorm':self.pmin.fnorm,'dof':self.pmin.dof} #There are some fortran object in the self.pmin which can not be pickled using function pickle.dumps, so I just choose some infromation useful
						break
				
				self.modify_initialp_ith=self.modify_initialp_ith+1
				if self.modify_initialp_ith > self.modify_initialp_total:
					print('Warning the result is bad also ,though the intial values have been changed %s times!!!'%self.modify_initialp_total)
					self.pmin={'params':self.pmin.params,'perror':self.pmin.perror,'fnorm':self.pmin.fnorm,'dof':self.pmin.dof}
					break	
		else:
			p,yfit=mpfit.mpfitexpr(self.func,self.loglam,self.flux,self.err,None,quiet=self.quiet,check=False,full_output=True,imports=None,parinfo=self.Par)
		#	print('debug',p.params,p.perror)
			if not p.errmsg=='':
				raise Exception('Mpfit return an error message: %s'%p.errmsg)
			self.chi2min=p.fnorm/p.dof 
			self.pmin={'params':p.params,'perror':p.perror,'fnorm':p.fnorm,'dof':p.dof}
			self.yfitmin=yfit 

		if (self.yfitmin.__class__==num.float64)|(self.yfitmin.__class__==float): 
			self.yfitmin=num.ones_like(self.wave)*self.yfitmin # Sometimes when the function is a constant, then the best results would be a constant and we need to reshape it to an array.
		self.linefs,self.lineferrs,self.line_sigma,self.line_value,self.line_scale,line_number={},{},{},{},{},{}
		self.err_status='estimated from error of scale'
		for key in self.line_keys:
			component=self.components[key]
			if component[1]=='gauss':
				lname=self.components[key][0]+'_'+self.components[key][2] 
				line_number[lname]=line_number.get(lname,0)+1
				value,sigma,scale=self.pmin['params'][component[13]],self.pmin['params'][component[14]],self.pmin['params'][component[12]]
				linef=self.flux_gauss(num.arange(self.wave[0],self.wave[-1],0.01),value,sigma,scale)
				if scale==0:
					lineferr=0
				else:
			#		print(self.pmin['perror'])
		#			print(lname,line_number[lname],scale,self.pmin['perror'][component[12]])
					lineferr=self.pmin['perror'][component[12]]/scale*linef
				self.line_sigma[lname+'_%s'%line_number[lname]]=sigma
				self.line_value[lname+'_%s'%line_number[lname]]=value
				self.line_scale[lname+'_%s'%line_number[lname]]=scale
			elif component[1]=='lorentz':
				print('Add this part if necessary!!!')
			else:
				raise Exception('The component type %s do not have line flux'%component[1])

			self.linefs[lname+'_%s'%line_number[lname]]=linef
			self.lineferrs[lname+'_%s'%line_number[lname]]=lineferr
			self.linefs[lname+'_total']=self.linefs.get(lname+'_total',0)+linef
			self.lineferrs[lname+'_total']=0 
#		return self.linefs,self.line_sigma
	def fitting_MC1(self,i,flux):
		#i represents the ith MC fitting
		try:
			self.path_savefile=os.path.join(self.MC_path,'%s.fits'%i)
			self.flux=MonteC(flux,self.err,model='normal')
			self.fitting()
			self.save_results()
			if self.MC_savefig:
				self.path_savefig=os.path.join(self.MC_path,'%s.png'%i)
				self.show=False
				self.plot_results()
		except Exception as e:
			traceback.print_exc()
			print(e)
		return 	self.linefs,self.line_sigma,self.line_scale,self.line_value
	def fitting_MC2(self,flux):
		#i represnts the ith MC fitting
		self.flux=MonteC(flux,self.err,model='normal')
		self.fitting()
		return  self.linefs,self.line_sigma,self.line_scale,self.line_value

	def running(self):
		#notation, if self.fit1=True and self.MC=True, the value of self.modifyP for the two process was same as each other
		# if we want it to be different, we must separate the two process, namely when you do the fit1, set the self.MC=False 
		#and set a value of self.modifyP you wanted
		if self.fit1:
			self.components_initialp()
		#	a=self.components[1][4]
			self.fitting()
			self.plot_results()
			self.save_results()
			self.components_initialp() #when do the step self.save_results, the intial values in the components have been changed into the fitted results, and self.components_initialp can re-construct the self.Par
		#	print(a,self.components[1][4])
		if self.MC:
		#	print(self.line_scale)
			pool=multip.Pool(processes=self.MC_pool_number)
			if not self.MC_need:
				print('MC procedure seems not necessary to calculate the error of parameters')
			flux=self.flux.copy() # in fact,we do not need to add the suffix 'copy', because the multiprocessing won't change the value in the main process
			if self.MC_saveresults:
				#NOTE when we do MC, sometimes it will raise an exception at self.save_resuts() like' There is no key 'Ha_n_1' in self.line_scale'. The 
				# error was caused by that there is no line_scale in the header of input file. 

				self.MC_path=os.path.join(self.path_savefile,'..','MC_temp')
				path_savefile,path_savefig=self.path_savefile,self.path_savefig
				if os.path.exists(self.MC_path):
					raise Exception('MC_temp direcotry already exists!!!')
				else:
					os.mkdir(self.MC_path)

				results_list=pool.starmap_async(func=self.fitting_MC1,iterable=[[x,flux.copy()] for x in range(0,self.MC_number)]).get()
				pool.close()
				pool.join()
				self.path_savefig,self.path_savefile=path_savefig,path_savefile		#In fact, this sentence can be ignored, becuase the self.path_save won't be changed though it will be changed in the fitting_MC  
			else:
				results_list=pool.map_async(func=self.fitting_MC2,iterable=[flux.copy() for x in range(0,self.MC_number)]).get()
				pool.close()
				pool.join()
			
			self.flux=flux
			self.err_status='MC_estimated'

			print('hello')
			line_number,linefs,lineferrs,line_sigma,line_scale,line_value={},{},{},{},{},{}
			for key in self.linefs.keys():
				linefs[key]=[]
				if 'total' in key:
					continue
				line_sigma[key]=[]
				line_scale[key]=[]
				line_value[key]=[]
				line_number[key.split('_')[0]+'_'+key.split('_')[1]]=line_number.get(key.split('_')[0]+'_'+key.split('_')[1],0)+1

			for lname in line_number.keys():
				for result in results_list:
					sigma=num.array([result[1][lname+'_%s'%i] for i in range(1,line_number[lname]+1)])
					line_flux=num.array([result[0][lname+'_%s'%i] for i in range(1,line_number[lname]+1)])
					scale=num.array([result[2][lname+'_%s'%i] for i in range(1,line_number[lname]+1)])
					value=num.array([result[3][lname+'_%s'%i] for i in range(1,line_number[lname]+1)])
					ind=num.argsort(sigma)
					line_flux,sigma,scale,value=line_flux[ind],sigma[ind],scale[ind],value[ind]
					for i in range(1,line_number[lname]+1):
						linefs[lname+'_%s'%i].append(line_flux[i-1])
						line_sigma[lname+'_%s'%i].append(sigma[i-1])
						line_scale[lname+'_%s'%i].append(scale[i-1])
						line_value[lname+'_%s'%i].append(value[i-1])
					linefs[lname+'_total'].append(result[0][lname+'_total'])
			print('hello again')
#			return results_list,linefs
			for key in linefs.keys():
#				plt.hist(linefs[key])
#				plt.title(key)
#				plt.show()
				lineferrs[key]=num.std(linefs[key],ddof=1)
				linefs[key]=num.mean(linefs[key])
				if 'total' not in key:
					line_sigma[key]=num.mean(line_sigma[key])
					line_scale[key]=num.mean(line_scale[key])
					line_value[key]=num.mean(line_value[key])
#			print('hello again again')

#			self.line_sigma,self.linefs,self.lineferrs,self.line_scale,self.line_value=line_sigma,linefs,lineferrs,line_scale,line_value
			for key in linefs.keys():
				if 'total' in key:
					self.linefs[key]=linefs[key]
					self.lineferrs[key]=lineferrs[key] 

			self.save_results()

	@staticmethod
	def separate_wave(wave,gap=300):
		I,inds=[],[]
		for i in range(0,len(wave)-1):
			if (wave[i+1]-wave[i])>gap:
				I.append(i)
				if len(I)==1:
					ind=wave<wave[i]
				else:
					ind=(wave>wave[I[len(I)-2]]) &(wave<wave[i])
				inds.append(ind)
		if len(I)==0:
			return 1,[wave>0]
		ind=wave>wave[I[len(I)-1]]
		inds.append(ind)
		return len(I)+1,inds

	
	def plot_results(self):
		config = {"font.family":'Times New Roman',"font.size": 20,"mathtext.fontset":'stix',"font.serif": ['SimSun']}
		rcParams.update(config)
		N,inds =self.separate_wave(self.wave )
		N,inds0=self.separate_wave(self.wave0)
		fig,axs=plt.subplots(ncols=N,nrows=1,figsize=(N*7.6,7.5))
		if N==1:
			axs=[axs]
		for ax in axs:
			set_axis(ax,labelsize=15,linewidth=1.8)

		fig.text(0.4,0.02,r'wavelength ($\rm \AA$)')
		fig.text(0.02,0.3,r'flux $\rm 10^{-17}erg\ s^{-1}\ cm^{-2}\ {\AA}^{-1}$',rotation=90)
		
		for ind0,ind,i in zip(inds0,inds,range(0,len(inds))):
			pl1,=axs[i].plot(self.wave0[ind0],self.flux0[ind0],'r')
	#		print(self.yfitmin)
			pl2,=axs[i].plot(self.wave0[ind0],self.get_yfit()[ind0],'b')
			pls,labels=[pl1,pl2],['data','best fit']
			ylim=plt.ylim()

			sign=0
			if self.telluric:
				pl_tel=axs[i].fill_between(self.wave0[ind0],ylim[0],ylim[1],where=self.ind_telluric[ind0],color='gray',alpha=0.7)
				pls.append(pl_tel);labels.append('telluric')
				sign=1
			if not (self.extra_mask==None):
				pl_extra=axs[i].fill_between(self.wave0[ind0],ylim[0],ylim[1],where=self.ind_extramask[ind0],color='pink',alpha=0.5)
				pls.append(pl_extra);labels.append('extra_mask')
				sign=1

			if (len(self.wave0[self.mask_array])<len(self.wave0)) &(sign==0):
				pl_mask =axs[i].fill_between(self.wave0[ind0],ylim[0],ylim[1],where=(self.mask_array==False)[ind0],color='pink',alpha=0.5)
				pls.append(pl_mask);labels.append('mask')

			if self.plot_upperlimit:
				print('hello upperlimits')
				pl_upper,=axs[i].plot(self.wave[ind],self.flux_upper[ind],label='upper limit')
			params=self.pmin['params']
			for key in self.components.keys():
				#TODO we can use the function eval_comp to do the procedure below
				component=self.components[key]
				if component[1]=='gauss':
					scale_index,value_index,sigma_index=component[12],component[13],component[14]
					scale,value,sigma=params[scale_index],params[value_index],params[sigma_index]
					if not ((value>self.loglam[ind][0]) & (value<num.max(self.loglam[ind]))):
						continue
					y=scale*scipy.stats.norm.pdf(self.loglam0,value,sigma)
					if component[2]=='n':
						pl3,=axs[i].plot(self.wave0[ind0],y[ind0],'g')
					else:
						pl4,=axs[i].plot(self.wave0[ind0],y[ind0],'m')
					continue
				if component[1]=='linear':
					k_index,b_index=component[8],component[9]
					k,b=params[k_index],params[b_index]
					y=k*self.loglam0+b
					pl5,=axs[i].plot(self.wave0[ind0],y[ind0],'k')
					continue
				if component[1]=='poly':
					order=int(component[0])
					y=0
					for j in range(0,order+1):
						aj_index=component[j+(order+1)*3+2]
						aj=params[aj_index]
						y=y+aj*self.loglam0**j
					pl6,=axs[i].plot(self.wave0[ind0],y[ind0],'k--')
		try:
			pls.append(pl3)
			labels.append('gauss components')
		except: print('No narrow gauss')
		try:
			pls.append(pl4)
			labels.append('gauss compoents')
		except: print('No broad gauss')
		try:
			pls.append(pl5)
			labels.append('local_conti')
		except: print('No linear component')
		try:
			pls.append(pl6)
			labels.append('poly')
		except: print('No poly component')
		try:
			pls.append(pl_upper)
			labels.append('upper_limits')
		except: print('No upperlimits component')

		plt.legend(handles=pls,labels=labels,fontsize=14)
	
		if self.path_savefig:
			if os.path.exists(self.path_savefig):
				os.remove(self.path_savefig)
			plt.savefig(self.path_savefig)

		if self.show:
			plt.show()

		plt.close()
		
	def save_results(self):
		#NOTE to improve this, we should the initial wave,flux,ferr, and the mask array 
		#NOTE and also the yfit save should have the same length as wave 
		# datas={0:None,1:{'wave':{'data':self.wave,'fmt':'D','unit':'Angstrom'},\
		# 				 'flux':{'data':self.flux,'fmt':'D','unit':'Angstrom'},\
		# 				 'flux_fit':{'data':self.yfitmin,'fmt':'D','unit':'Angstrom'},\
		# 				 'err':{'data':self.err,'fmt':'D','unit':'Angstrom'}}}
		# if (self.telluric) | (self.extra_mask is not None): 
		# 	datas[1]['wave0']={'data':self.wave0,'fmt':'D','unit':'Angstrom'}
		# 	datas[1]['flux0']={'data':self.flux0,'fmt':'D','unit':'Angstrom'}
		# 	datas[1]['err0']={'data':self.err0,'fmt':'D','unit':'Angstrom'}

		datas={0:None,1:{'wave':{'data':self.wave0,'fmt':'D','unit':'Angstrom'},\
						 'flux':{'data':self.flux0,'fmt':'D','unit':'Angstrom'},\
						 'flux_fit':{'data':self.get_yfit(),'fmt':'D','unit':'Angstrom'},\
						 'err':{'data':self.err0,'fmt':'D','unit':'Angstrom'},\
						 'mask':{'data':self.mask_array,'fmt':'L','unit':None}}}
		pfit,pfit_err=self.pmin['params'],self.pmin['perror']
		data2={}
		for key in self.components.keys():
			component=self.components[key]
			if component[1]=='gauss':
				component[3],component[4],component[5]=pfit[component[13]],pfit[component[14]],pfit[component[12]] #it would change the component value in the self.components
				component=component+[pfit_err[component[13]],pfit_err[component[14]],pfit_err[component[12]]] #it would not change the component value in the self.components

			elif component[1]=='linear':
				component[2],component[3]=pfit[component[8]],pfit[component[9]]
				component=component+[pfit_err[component[8]],pfit_err[component[9]]]
			elif component[1]=='poly':
				order=int(component[0])
				for j in range(0,order+1):
					component[2+j]=pfit[component[2+j+(order+1)*3]]
				for j in range(0,order+1):
					component=component+[pfit_err[component[2+j+(order+1)*3]]] # This step can not be processed in the above loop, because this step will 
					#change the address of the variable 'component', and therefore after the first circle , it will not changes the values in self.components in such way
			elif component[1]=='lorentz':
				print('Add this part if necessary')
			else: raise NameError('No such class for component: %s'%component[1])
			data2['%s'%key]={'data':component,'fmt':'26A','unit':None}
		datas[2]=data2

		self.deviation=sum((self.flux-self.yfitmin)**2)/len(self.flux)
		param0={'narrow_up':self.narrow_up,'broad_low':self.broad_low,'chi2_min':self.chi2min,\
				'deviation':self.deviation,'parameter_number':len(self.flux)-self.pmin['dof']}
		param2={'function':self.func}
		for key in self.ValueLimits.keys():
			param2[str(key)+'_valuelimits']='%s %s'%(self.ValueLimits[key][0],self.ValueLimits[key][1])
		for key in self.SigmaLimits.keys():
			param2[str(key)+'_sigmalimits']='%s %s'%(self.SigmaLimits[key][0],self.SigmaLimits[key][1])
		param1={'err_status':self.err_status}
		for key in self.linefs.keys():
			param1[key+'_flux']=self.linefs[key]
			param1[key+'_ferr']=self.lineferrs[key]			
			if 'total' not in key:
				param1[key+'_sigma']=self.line_sigma[key]
				param1[key+'_scale']=self.line_scale[key]
				param1[key+'_value']=self.line_value[key]

		if not self.covert2simple:
			params={0:param0,1:param1,2:param2}
		else:
			# Just designed for two groups of gauss components at most!!!
			for key in self.components.keys():
				component=self.components[key]
				if component[1]=='gauss':
					linename=component[0]
					if component[2]=='b':
						linename=linename+'0'
					if '%s_value'%linename in param0.keys():
						linename=linename+'2'
					param0['%s_value'%linename]='%s %s'%(pfit[component[13]],pfit_err[component[13]])
					param0['%s_sigma'%linename]='%s %s'%(pfit[component[14]],pfit_err[component[14]])
					param0['%s_scale'%linename]='%s %s'%(pfit[component[12]],pfit_err[component[12]])
				elif component[1]=='linear':
					param0['local_conti_kvalue']='%s %s'%(pfit[component[8]],pfit_err[component[8]])
					param0['local_conti_bvalue']='%s %s'%(pfit[component[9]],pfit_err[component[9]])
				elif component[1]=='poly':
					order=int(component[0])
					for j in range(0,order+1):
						param0['a%d_value'%j]='%s %s'%(pfit[component[2+j+(order+1)*3]],pfit_err[component[2+j+(order+1)*3]])
				else: print('The header do not record that type of component: %s'%component[1])
			params={0:param0,1:param1,2:param2}
#		print(datas)
		if not self.path_savefile==None:
			create_fits(datas,params,self.path_savefile)

	def waveflux_ppxf(self,path,low,up,show=True):
		hdu=fits.open(path)
		wave=hdu[1].data['wave']/(1+hdu[0].header['redshift'])
		flux=hdu[1].data['original_data']-hdu[1].data['stellar']
		err=(hdu[1].data['err_sdss']**2+hdu[1].data['err_stellar']**2)**0.5
		ind=(wave>low) & (wave<up)
		
		self.wave,self.loglam,self.flux,self.err=wave[ind],num.log(wave[ind]),flux[ind],err[ind]
		if show:
			plt.plot(self.wave,self.flux)
			plt.show()
		return ':)'
	def waveflux_dbsp(self,path,low,up,show=True):
		hdu=fits.open(path)
		wave=hdu[1].data['wave_dbsp']
		flux=hdu[1].data['flux_dbsp']-hdu[1].data['continuum']
		err=(hdu[1].data['err_dbsp']**2+hdu[1].data['conti_err']**2)**0.5
		ind=(wave>low) & (wave<up)
		self.wave,self.loglam,self.flux,self.err=wave[ind],num.log(wave[ind]),flux[ind],err[ind]
		if show:
			plt.plot(self.wave,self.flux)
			plt.show()
		return ':)'
	@staticmethod
	def F_test(path1,path2,confidence_level,weight=False,overfit=True):
		raise Exception('NOTE Ftest weight=True/or False, line-new里F-test的函数都不一致')
		hdu=fits.open(path1)
		deviation_1=hdu[0].header['deviation']
		chi2_1=hdu[0].header['chi2_min']
		if overfit:
			if chi2_1<1:
				chi2_1=1
		p1=hdu[0].header['parameter_number']
		hdu.close()

		hdu=fits.open(path2)
		deviation_2=hdu[0].header['deviation']
		chi2_2=hdu[0].header['chi2_min']
		if overfit:
			if chi2_2<1:
				chi2_2=1
		p2=hdu[0].header['parameter_number']
		

		wave=hdu[1].data['wave']
		n=len(wave)
		hdu.close()
		print(p1,p2)
		if p2>p1:
			if weight:
				F_test=(chi2_1*(n-p1)-chi2_2*(n-p2))/(p2-p1)/chi2_2 #NOTE if the best-fit for a model is chosen by minimizing the chi2, so we should use chi2 here ?
			else:
				F_test=(deviation_1*n-deviation_2*n)/(p2-p1)/(deviation_2*n/(n-p2))
			Fppf=ft.ppf(confidence_level,p2-p1,n-p2)
			if (F_test>Fppf) & (chi2_1>chi2_2):
				return path2
			else:
				return path1
		elif p1>p2:
			if weight:
				F_test=(chi2_2*(n-p2)-chi2_1*(n-p1))/(p1-p2)/chi2_1
			else:
				F_test=(deviation_2*n-deviation_1*n)/(p1-p2)/(deviation_1*n/(n-p1))
			Fppf=ft.ppf(confidence_level,p1-p2,n-p1)
			if (F_test>Fppf) & (chi2_2>chi2_1):
				return path1
			else:
				return path2
		else:
			print('The same parameter numbers????')
			return 0

		return ":)"

	def upperlimit(self,upN,freedom,lname_upper=None,lname_sigma=None,lname_wc=None,confidence_level=0.90,scale_accuracy=10000,accuracy_chi2=1000,save_results=True,show=True,Niteration=50):
		#the components gives the current components used to fit the line, an the upN record the number of which the component will be calculted the upper limit
		#the upN can be a value or a list 
		#the parameter 'Niteration' gives the maximum number of dichotomy, when setting the value as 50, after 50 times of dichotomy, the interval length of [scale_low,scale_up]
		# will be about 1./2**50 of the value scale
		# if upN==[], this function was only proper for that the flux of the upper components was zero even if it's included when  fitting
		components=copy.deepcopy(self.components)#avoid that the input components would be changed when we run this function 'upperlimit'

		pmin,linefs,lineferrs,line_sigma,yfitmin=copy.deepcopy(self.pmin),self.linefs.copy(),self.lineferrs.copy(),self.line_sigma.copy(),self.yfitmin.copy() # To avoid that we forgot to save the fitting results before do the line flux upper limit

		if upN.__class__.__name__=='list':
			upN=[float(i) for i in upN]
		else: upN=[float(upN)]

		lnames=[]
		for key in upN:
			lnames.append(self.components[key][0]+'_'+self.components[key][1]+'_'+self.components[key][2]) # what if the components was belonged to different types, like one is 'gauss' and the other is 'lorentz'
			del self.components[key]
		if len(set(lnames))>1:
			print(lnames)
			raise Exception('The components to calculate the upper limits should have same name,width and type')
		if lnames==[]:
			if lname_upper==None: raise Exception('Please give the name of lines needed to calculate the upper limit now that upN==[]')
			lnames=[lname_upper]

		#we should confirm that the components left was not tied to one of the components we deleted
		for key in self.components:
			component=self.components[key]
			if component[1]=='gauss':
				for tied_index in component[6:9]:
					if tied_index==-1:
						continue
					if tied_index.__class__==str:
						try:
							tiedindexs=self.tied_decomp(tied_index,get_index=True)
						except KeyError as e:
							raise Exception('Maybe an component to which other components tied was deleted')
						print(tiedindexs)
						for index in tiedindexs:
							if index not in self.components.keys():
								raise Exception('Maybe component No.%s to which other components  tied was deleted'%index)
						continue
					if tied_index not in self.components.keys():
						raise Exception('Maybe component No.%s to which other components  tied was deleted'%tied_index)
			elif component[1]=='lorentz':
				print('Add this part if necessary')
		
		#calculate the upper limit 
		delta_chi2=scipy.stats.chi2.isf(1-confidence_level,freedom)
		print('hahahahahah',delta_chi2)
		dchi2_accuracy=(scipy.stats.chi2.isf(1-confidence_level*(1+1./accuracy_chi2),freedom)-scipy.stats.chi2.isf(1-confidence_level*(1-1./accuracy_chi2),freedom))/2. 
		sigma,wave_peak=composite_spectra_width(lnames[0].split('_')[0]) # The composite_spectra_width function can be found in the linefit library
		if lname_sigma is not None: sigma=lname_sigma 
		if lname_wc is not None: wave_peak=lname_wc

		flux_lname=scipy.stats.norm.pdf(num.log(self.wave),num.log(wave_peak),sigma)
		lname_integrate=sum([flux_lname[i]*(self.wave[i+1]-self.wave[i]) for i in range(0,len(self.wave)-1)])
		
		if 'O3a' in lname_upper:
			flux_lname=flux_lname+1.0/2.98*scipy.stats.norm.pdf(num.log(self.wave),num.log(4960.36),sigma)*5008.22/4960.36	
		elif 'O3b' in lname_upper:
			flux_lname=flux_lname+2.98*scipy.stats.norm.pdf(num.log(self.wave),num.log(5008.22),sigma)*4960.36/5008.22


		scale_up=num.max(self.flux)/num.max(flux_lname)
#		if scale_up<0.1:
#			scale_up=0.1  #why we need set the scale_up=0.1 here?

		dchi2,N=0,0
		scale,scale_low,sign=scale_up/2.0,0.0,1
		flux,fnorm,modifyP,chi2min=self.flux.copy(),self.pmin['fnorm'],self.modifyP,self.chi2min
		self.components_initialp()
		self.modifyP=True

		#NOTE we couldn't set the scale_up-scale_low>scale/scale_accuracy as good upper limits
		#NOTE when we rerun the procedure, the results could convert from 'bad' status to 'good' status 
		#NOTE how could we avoid this and try to get the 'good' status in the only one run ?
		while (abs(dchi2-delta_chi2)>dchi2_accuracy) & ( (scale_up-scale_low)>scale/scale_accuracy): #we may also limit the minmum number of the dichotomy to arrive an accuracy of scale 
			self.flux=flux-scale*flux_lname
			self.fitting()
		#	dchi2=(self.chi2min-chi2min)
			dchi2=self.pmin['fnorm']-fnorm
			if dchi2>delta_chi2: scale_up=scale
			else: scale_low=scale
			scale=(scale_low+scale_up)/2.0 
			
			print(scale_low,scale_up,dchi2)
			
			N=N+1
			if N>Niteration:
				break
				sign=0

		if sign==0: print('Warning: After 50 iterations, the results was still not so good')
		self.modifyP=modifyP

		if save_results:
			if self.path_savefile==None:
				raise Exception('Please give the path to save the results')
			if not os.path.exists(self.path_savefile):
				raise Exception('Maybe you need to save the fitting results first')
			hdu=fits.open(self.path_savefile)
			hdu[0].header.set(lnames[0]+'_upper',scale*lname_integrate)
			hdu[0].header.set(lnames[0]+'_dchi2',dchi2)
			hdu[0].header.set(lnames[0]+'_status','good' if sign==1 else 'bad')
			hdu.writeto(self.path_savefile.replace('.fits','temp.fits'))
			hdu.close()
			os.remove(self.path_savefile)
			os.rename(self.path_savefile.replace('.fits','temp.fits'),self.path_savefile)
		if show:
			self.plot_upperlimit=True
			self.flux_upper=flux_lname*scale
			original_show,original_savefig=self.show,self.path_savefig
			self.show,self.path_savefig=True,None
			self.plot_results()
			self.show,self.path_savefig=original_show,original_savefig
		self.chi2min_upper=self.chi2min
		self.flux,self.chi2min,self.yfitmin=flux,chi2min,yfitmin
		self.components,self.pmin,self.linefs,self.lineferrs,self.line_sigma=components,pmin,linefs,lineferrs,line_sigma
		return scale*lname_integrate,sign

	def templates(self,choice,index=0,extrao3=False,o3value_tied=True,sigma_fixed=None,nii_tied=True,sii_tied=False,o3_tied=True,HaHb_tied=True,local_conti=False):
		#Maybe we can add an additionlal parameter 'broad components number' to add broad components in this function and hence we don't need to add broad components additionally
		#Maybe we can improve the paramter input by using a dictionary
		if choice=='Ha':
			#1*narrow
			self.add_gauss(index,'Ha','n')
			self.add_gauss(index+1,'nii6549','n',value_tied=index,sigma_tied=index)
			if nii_tied: self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
			else: self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index)
	#		if local_conti: lf.add_linear(index+3,'local_conti')
			return
		elif choice=='sii':
			self.add_gauss(index,'sii1','n')
			if sii_tied: self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index)
			else: self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index)
		elif choice=='Ha+sii':
			self.add_gauss(index,'Ha','n')
			self.add_gauss(index+1,'nii6549','n',value_tied=index,sigma_tied=index)
			if nii_tied: self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
			else: self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index)
			self.add_gauss(index+3,'sii1','n',value_tied=index,sigma_tied=index)
			if sii_tied: self.add_gauss(index+4,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index+3)
			else: self.add_gauss(index+4,'sii2','n',value_tied=index,sigma_tied=index)
	#		if local_conti: lf.add_linear(index+5,'local_conti')
			return 
		elif choice=='o3':
			self.add_gauss(index,'O3a','n')
			self.add_gauss(index+1,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index)
		elif choice=='o3+sii':
			self.add_gauss(index,'O3a','n')
			self.add_gauss(index+1,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index)
			if HaHb_tied:
				self.add_gauss(index+2,'sii1','n',value_tied=index,sigma_tied=index)
				self.add_gauss(index+3,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index+2)
			else:
				self.add_gauss(index+2,'sii1','n',sigma_tied=index)
				self.add_gauss(index+3,'sii2','n',sigma_tied=index,scale_tied=index+2)
		elif choice=='Hb':
			#1*narrow
			self.add_gauss(index,'Hb','n')
			if o3value_tied:
				self.add_gauss(index+1,'O3a','n',value_tied=index,sigma_tied=index)
				if o3_tied: self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
				else: self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index)
			else:
				self.add_gauss(index+1,'O3a','n',sigma_tied=index)
				if o3_tied:	self.add_gauss(index+2,'O3b','n',value_tied=index+1,sigma_tied=index,scale_tied=index+1)
				else: self.add_gauss(index+2,'O3b','n',value_tied=index+1,sigma_tied=index)
	#		if local_conti: lf.add_linear(index+3,'local_conti')
			if extrao3:
				self.add_gauss(index+3,'O3a','n',sigma_limits=[60,1500],value_limits=[-25,25])
				self.add_gauss(index+4,'O3b','n',value_tied=index+3,sigma_tied=index+3,scale_tied=index+3,sigma_limits=[60,1500],value_limits=[-25,25])
			return
		elif choice=='FeX':
			self.add_gauss(index,'OI6301','n')
			self.add_gauss(index+1,'OI6364','n',value_tied=index,sigma_tied=index,scale_tied=index)
			self.add_gauss(index+2,'FeX','n',value_tied=index,sigma_limits=[60,1500])
			return 
		elif choice=='OI':
			self.add_gauss(index,'OI6301','n')
			self.add_gauss(index+1,'OI6364','n',value_tied=index,sigma_tied=index,scale_tied=index)
			return 
		elif choice in ['FeVII3759','FeV4071','FeVII5160','FeXIV5304','FeVII5722','FeVII6087','FeXI7894','HeII','NIII']:
			self.add_gauss(index,choice,'n',sigma_limits=[60,1500])
		elif choice=='Ha+Hb':
			if sigma_fixed==None:
				self.add_gauss(index,'Hb','n')
			else:
				self.add_gauss(index,'Hb','n',sigma=sigma_fixed,sigma_fixed=True)
			if o3value_tied:
				self.add_gauss(index+1,'O3a','n',value_tied=index,sigma_tied=index)
				if o3_tied:	self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
				else: self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index)
			else:
				self.add_gauss(index+1,'O3a','n',sigma_tied=index)
				if o3_tied:	self.add_gauss(index+2,'O3b','n',value_tied=index+1,sigma_tied=index,scale_tied=index+1)
				else: self.add_gauss(index+2,'O3b','n',value_tied=index+1,sigma_tied=index)
			if HaHb_tied:
				self.add_gauss(index+3,'Ha','n',value_tied=index,sigma_tied=index)
				self.add_gauss(index+4,'nii6549','n',value_tied=index,sigma_tied=index)
				if nii_tied: self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+4)
				else: self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index)
				self.add_gauss(index+6,'sii1','n',value_tied=index,sigma_tied=index)
				if sii_tied: self.add_gauss(index+7,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index+6)
				else: self.add_gauss(index+7,'sii2','n',value_tied=index,sigma_tied=index)
			else:
				self.add_gauss(index+3,'Ha','n',sigma_tied=index)
				self.add_gauss(index+4,'nii6549','n',value_tied=index+3,sigma_tied=index)
				if nii_tied: self.add_gauss(index+5,'nii6585','n',value_tied=index+3,scale_tied=index+4,sigma_tied=index)
				else: self.add_gauss(index+5,'nii6585','n',value_tied=index+3,sigma_tied=index)
				self.add_gauss(index+6,'sii1','n',value_tied=index+3,sigma_tied=index)
				if sii_tied: self.add_gauss(index+7,'sii2','n',value_tied=index+3,scale_tied=index+6,sigma_tied=index)
				else: self.add_gauss(index+7,'sii2','n',value_tied=index+3,sigma_tied=index)				
	#		if local_conti=True: self.add_linear(index+8,'local_conti')
			if extrao3:
				self.add_gauss(index+8,'O3a','n',sigma_limits=[60,1500],value_limits=[-25,25])
				self.add_gauss(index+9,'O3b','n',value_tied=index+8,sigma_tied=index+8,scale_tied=index+8,sigma_limits=[60,1500],value_limits=[-25,25])
		else: raise Exception('No such choice of templates: %s'%choice)		

	def templates2(self,choice,index=0,ref_index=None,sigma_fixed=None,o3value_tied=True,nii_tied=True,sii_tied=False,o3_tied=True,HaHb_tied=True,local_conti=False):
		#NOTE  in this  templates, the all the sigma, value, scale were tied to the first narrow components to make all the profile of the whole narrow componet for a line was same to each other 
		if choice=='Ha+sii':
			self.add_gauss(             index  ,'Ha'     ,'n'                                  ,scale_tied='sca%s/sca%s*sca%s'%(index+3,ref_index+3,ref_index+0))
			self.add_gauss(             index+1,'nii6549','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+3,ref_index+3,ref_index+1))
			if nii_tied: self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
			else:        self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+3,ref_index+3,ref_index+2))
			self.add_gauss(             index+3,'sii1'   ,'n',value_tied=index,sigma_tied=index)
			if sii_tied: self.add_gauss(index+4,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied=index+3)
			else:        self.add_gauss(index+4,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+3,ref_index+3,ref_index+4))
	#		if local_conti: lf.add_linear(index+5,'local_conti')
			return 
		elif choice=='sii':
			self.add_gauss(             index  ,'sii1','n')
			if sii_tied: self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index)
			else:        self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index,ref_index,ref_index+1))
		elif choice=='Hb':
			#1*narrow
			self.add_gauss(                index  ,'Hb' ,'n'                                  ,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index))
			if o3value_tied:
				#NOTE when the first double o3 lines was tied together, the results for o3value_tied=True would be the same that is False
				#     but when the first o3 group was not tied with each other, then the two cases would be different 
				self.add_gauss(            index+1,'O3a','n',value_tied=index,sigma_tied=index)
				if o3_tied: self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index+1)
				else:       self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+2))
			else:
				self.add_gauss(            index+1,'O3a','n',value_tied='val%s+val%s-val%s'%(ref_index+1,index,ref_index),sigma_tied=index)
				if o3_tied: self.add_gauss(index+2,'O3b','n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index,scale_tied=index+1)
				else: self.add_gauss(      index+2,'O3b','n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+2))				
	#		if local_conti: lf.add_linear(index+3,'local_conti')
			return

		elif choice=='Ha+Hb':
			if sigma_fixed==None: self.add_gauss(index  ,'Hb'     ,'n'                                   ,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index))
			else:                 self.add_gauss(index  ,'Hb'     ,'n',sigma=sigma_fixed,sigma_fixed=True,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index))
			if o3value_tied:
				self.add_gauss(                  index+1,'O3a'    ,'n',value_tied=index,sigma_tied=index)
				if o3_tied:	      self.add_gauss(index+2,'O3b'    ,'n',value_tied=index,sigma_tied=index,scale_tied=index+1)
				else:             self.add_gauss(index+2,'O3b'    ,'n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+2))
			else:
				self.add_gauss(                  index+1,'O3a'    ,'n',value_tied='val%s+val%s-val%s'%(ref_index+1,index,ref_index),sigma_tied=index)
				if o3_tied:	      self.add_gauss(index+2,'O3b'    ,'n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index,scale_tied=index+1)
				else:             self.add_gauss(index+2,'O3b'    ,'n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+2))
			if HaHb_tied:
				self.add_gauss(                  index+3,'Ha'     ,'n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+3))
				self.add_gauss(                  index+4,'nii6549','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+4))
				if nii_tied:      self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+4)
				else:             self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+5))
				self.add_gauss(                  index+6,'sii1'   ,'n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+6))
				if sii_tied:      self.add_gauss(index+7,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied=index+6)
				else:             self.add_gauss(index+7,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+7))
			else:
			#	self.add_gauss(index+3,'Ha','n',value_tied='val%s+val%s-val%s'%(ref_index+3,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+3))
			#	self.add_gauss(index+4,'nii6549','n',value_tied='val%s+val%s-val%s'%(ref_index+4,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+4))
			#	if nii_tied: self.add_gauss(index+5,'nii6585','n',value_tied='val%s+val%s-val%s'%(ref_index+5,index,ref_index),sigma_tied=index,scale_tied=index+4)
			#	else: self.add_gauss(index+5,'nii6585','n',value_tied='val%s+val%s-val%s'%(ref_index+5,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+5))
			#	self.add_gauss(index+6,'sii1','n',value_tied='val%s+val%s-val%s'%(ref_index+6,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+6))
			#	if sii_tied: self.add_gauss(index+7,'sii2','n',value_tied='val%s+val%s-val%s'%(ref_index+7,index,ref_index),sigma_tied=index,scale_tied=index+6)
			#	else: self.add_gauss(index+7,'sii2','n',value_tied='val%s+val%s-val%s'%(ref_index+7,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+7))
	#		if local_conti=True: self.add_linear(index+8,'local_conti')
				self.add_gauss(                 index+3,'Ha'     ,'n',value_tied='val%s+val%s-val%s'%(ref_index+3,index,ref_index),sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+3))
				self.add_gauss(                 index+4,'nii6549','n',value_tied=index+3 ,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+4))
				if nii_tied:     self.add_gauss(index+5,'nii6585','n',value_tied=index+3,sigma_tied=index,scale_tied=index+4)
				else:            self.add_gauss(index+5,'nii6585','n',value_tied=index+3,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+5))
				self.add_gauss(                 index+6,'sii1'   ,'n',value_tied=index+3,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+6))
				if sii_tied:     self.add_gauss(index+7,'sii2'   ,'n',value_tied=index+3,sigma_tied=index,scale_tied=index+6)
				else:            self.add_gauss(index+7,'sii2'   ,'n',value_tied=index+3,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index+1,ref_index+1,ref_index+7))

			return 
		else: raise Exception('No such choice of templates: %s'%choice)	
	def templates3(self,choice,index=0,ref_index=None,redshift_fixed=None,sigma_fixed=None,o3value_tied=True,nii_tied=True,sii_tied=False,o3_tied=True,HaHb_tied=True,local_conti=False):
		#Maybe we can improve the paramter input by using a dictionary
		#NOTE In this templates, a second group indepent from the first group was added, and only a fixed redshift or sigma was input as initial conditions 
		#NOTE It seems that, in the line_new.py, we prefer to use this templates for the second group, check this and also the templates here 
		sigma_limits=[60,2500.0]
		value_limits=[-25,25]
		if choice=='o3':
			self.add_gauss(index  ,'O3a','n'                                                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			self.add_gauss(index+1,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index,sigma_limits=sigma_limits,value_limits=value_limits)
		elif choice=='o3+sii':
			self.add_gauss(index  ,'O3a','n'                                                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			self.add_gauss(index+1,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index,sigma_limits=sigma_limits,value_limits=value_limits)
			if HaHb_tied:
				self.add_gauss(index+2,'sii1','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(index+3,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index+2,sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				self.add_gauss(index+2,'sii1','n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(index+3,'sii2','n',value_tied=index+2                                          ,sigma_tied=index,scale_tied=index+2,sigma_limits=sigma_limits,value_limits=value_limits)			
		elif choice=='sii':
			self.add_gauss(index,'sii1','n')
			if sii_tied: self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index,scale_tied=index)
			else:        self.add_gauss(index+1,'sii2','n',value_tied=index,sigma_tied=index,scale_tied='sca%s/sca%s*sca%s'%(index,ref_index,ref_index+1))
		elif choice=='Hb':
			#1*narrow
			if sigma_fixed==None:
				if redshift_fixed==None:self.add_gauss(index,'Hb','n'                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                   self.add_gauss(index,'Hb','n',value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				if redshift_fixed==None:self.add_gauss(index,'Hb','n',sigma=sigma_fixed,sigma_fixed=True                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                   self.add_gauss(index,'Hb','n',sigma=sigma_fixed,sigma_fixed=True,value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			if o3value_tied:
				self.add_gauss(            index+1,'O3a','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if o3_tied:	self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index+1,sigma_limits=sigma_limits,value_limits=value_limits)
				else:       self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				self.add_gauss(            index+1,'O3a','n',value_tied='val%s+val%s-val%s'%(ref_index+1,index,ref_index),sigma_tied=index                    ,sigma_limits=sigma_limits,value_limits=value_limits)
				if o3_tied:	self.add_gauss(index+2,'O3b','n',value_tied=index+1                                          ,sigma_tied=index,scale_tied=index+1 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:       self.add_gauss(index+2,'O3b','n',value_tied=index+1                                          ,sigma_tied=index                    ,sigma_limits=sigma_limits,value_limits=value_limits)
	#		if local_conti: lf.add_linear(index+3,'local_conti')
			return
		elif choice=='Ha+sii':
			if sigma_fixed==None:
				if redshift_fixed==None: self.add_gauss(index,'Ha','n'                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                    self.add_gauss(index,'Ha','n',value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				if redshift_fixed==None: self.add_gauss(index,'Ha','n',sigma=sigma_fixed,sigma_fixed=True                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                    self.add_gauss(index,'Ha','n',sigma=sigma_fixed,sigma_fixed=True,value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			self.add_gauss(            index+1,'nii6549','n',value_tied=index,sigma_tied=index,sigma_limits=sigma_limits,value_limits=value_limits)
			
			if nii_tied:self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+1,sigma_limits=sigma_limits,value_limits=value_limits)
			else:       self.add_gauss(index+2,'nii6585','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			self.add_gauss(            index+3,'sii1'   ,'n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			if sii_tied:self.add_gauss(index+4,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied=index+3,sigma_limits=sigma_limits,value_limits=value_limits)
			else:       self.add_gauss(index+4,'sii2'   ,'n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
		elif choice=='Ha+Hb':
			if sigma_fixed==None:
				if redshift_fixed==None:self.add_gauss(index,'Hb','n'                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                   self.add_gauss(index,'Hb','n',value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				if redshift_fixed==None:self.add_gauss(index,'Hb','n',sigma=sigma_fixed,sigma_fixed=True                                                 ,sigma_limits=sigma_limits,value_limits=value_limits)
				else:                   self.add_gauss(index,'Hb','n',sigma=sigma_fixed,sigma_fixed=True,value_tied='val%s+%s'%(ref_index,redshift_fixed),sigma_limits=sigma_limits,value_limits=value_limits)
			if o3value_tied:
				self.add_gauss(           index+1,'O3a','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if o3_tied:self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index,scale_tied=index+1,sigma_limits=sigma_limits,value_limits=value_limits)
				else:      self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				#NOTE the results here is the same as when o3value_tied=True? 
				self.add_gauss(           index+1,'O3a','n',value_tied='val%s+val%s-val%s'%(ref_index+1,index,ref_index),sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if o3_tied:self.add_gauss(index+2,'O3b','n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index,scale_tied=index+1,sigma_limits=sigma_limits,value_limits=value_limits)
				else:      self.add_gauss(index+2,'O3b','n',value_tied='val%s+val%s-val%s'%(ref_index+2,index,ref_index),sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			if HaHb_tied:
				self.add_gauss(             index+3,'Ha'     ,'n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(             index+4,'nii6549','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if nii_tied: self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index,scale_tied=index+4,sigma_limits=sigma_limits,value_limits=value_limits)
				else:        self.add_gauss(index+5,'nii6585','n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(             index+6,'sii1'   ,'n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if sii_tied: self.add_gauss(index+7,'sii2'   ,'n',value_tied=index,sigma_tied=index,scale_tied=index+6,sigma_limits=sigma_limits,value_limits=value_limits)
				else:        self.add_gauss(index+7,'sii2'   ,'n',value_tied=index,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
			else:
				self.add_gauss(             index+3,'Ha'     ,'n',value_tied='val%s+val%s-val%s'%(ref_index+3,index,ref_index),sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(             index+4,'nii6549','n',value_tied=index+3                                          ,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if nii_tied: self.add_gauss(index+5,'nii6585','n',value_tied=index+3                                          ,sigma_tied=index,scale_tied=index+4,sigma_limits=sigma_limits,value_limits=value_limits)
				else:  self.add_gauss(      index+5,'nii6585','n',value_tied=index+3                                          ,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				self.add_gauss(             index+6,'sii1'   ,'n',value_tied=index+3                                          ,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)
				if sii_tied: self.add_gauss(index+7,'sii2'   ,'n',value_tied=index+3                                          ,sigma_tied=index,scale_tied=index+6,sigma_limits=sigma_limits,value_limits=value_limits)
				else:        self.add_gauss(index+7,'sii2'   ,'n',value_tied=index+3                                          ,sigma_tied=index                   ,sigma_limits=sigma_limits,value_limits=value_limits)				
	#		if local_conti=True: self.add_linear(index+8,'local_conti')
		
			return 
		else: raise Exception('No such choice of templates: %s'%choice)
	







#	def line_fluxes(self,linename,n_or_b,serialNs=None):
#		integrate_flux=0
#		if self.reconstruct==True:
#			if not serialNs==None:
#				for serialn in serialNs:
#					if not component[0]==linename:
#						raise Exception('The component guided by the parameter serialN was not that of %s'%linename)
#					if component[1]=='gauss':
#						value,sigma=component[3],component[4]
#						print('The length is %s'%len(component))
#						scale,scale_err=component[5],component[17]
#						fluxlam=scale*scipy.stats.norm.pdf(self.loglam,value,sigma)
#						fluxsum=sum([ flux[x]*(wave[x]-wave[x-1]) for x in range(1,len(wave))])
#						integrate_flux=integrate_flux+fluxsum
#					elif component[1]=='lorentz':
#						print('add this part if necessary')
#					else:
#						raise Exception('No such components type %s for emission line'%component[1])
#				return integrate_flux
#			for key in self.components.keys():
#				component=self.components[key] # If there are two components for the same line, can we simple calculated the error of the total fluxes 
#				#using the error propagation formula?
#				if (component[0]==linename) & (component[1]=='gauss') & (component[2]==n_or_b):
#					value,sigma=component[3],component[4]
#					scale,scale_err=component[5],component[17]
#					fluxlam=scale*scipy.stats.norm.pdf(self.loglam,value,sigma)
#					fluxsum=sum([ flux[x]*(wave[x]-wave[x-1]) for x in range(1,len(wave))])
#					integrate_flux=integrate_flux+fluxsum					
#					pass
#				elif (component[0]==linename) & (component[1]=='lorentz') & (component[2]==n_or_b):
#					print('Add this part if neccessary!!!')
#				else:
#					raise Exception('No %s Components for Emission lines'%component[1])
#			return integrate_flux		
#		elif self.fit_done==True:
#			for key in self.components.keys():
#				if (component[0]==linename) & (component[1]=='gauss') & (component[2]==n_or_b):
#					value,sigma=self.pmin.params[component[13]],self.pmin.params[component[14]]
#					scale,scale_err=self.pmin.params[component[12]],self.pmin.perror[component[12]]
#					fluxlam=scale*scipy.stats.norm.pdf(self.loglam,value,sigma)
#					fluxsum=sum([ flux[x]*(wave[x]-wave[x-1]) for x in range(1,len(wave))])
#					integrate_flux=integrate_flux+fluxsum						
#			pass
#			return integrate_flux
#		elif not self.path_reconstruct==None:
#			self.re_construct()
#			for key in self.components.keys():
#				pass
#			pass
#		else:
#			raise Exception('No fitting results for calculating the emission lines fluxes')
#		return 
#	
'''
'''

'''
		write for J0120
		elif choice=='Ha+Hb':
			self.add_gauss(index,'Hb','n')
			self.add_gauss(index+1,'O3a','n',value_tied=index,sigma=sigma_fixed,sigma_fixed=True)
			self.add_gauss(index+2,'O3b','n',value_tied=index,sigma_tied=index+1,scale_tied=index+1)
			self.add_gauss(index+3,'Ha','n',sigma_tied=index)
			self.add_gauss(index+4,'nii6549','n',sigma_tied=index+1)
			self.add_gauss(index+5,'nii6585','n',value_tied=index+4,scale_tied=index+4,sigma_tied=index+1)
			self.add_gauss(index+5,'nii6585','n',value_tied=index+4,sigma_tied=index+1)
			self.add_gauss(index+6,'sii1','n',value_tied=index+4,sigma_tied=index+1)
			self.add_gauss(index+7,'sii2','n',value_tied=index+4,scale_tied=index+6,sigma_tied=index+1)	
			return 	
'''
