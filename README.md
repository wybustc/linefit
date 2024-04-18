# linefit
A flexible emission lines fitting procedure 

# How to use 
first initiate the instance 'linefit' 
```
lf=linefit(wave,flux,ferr,path_savefile=path_savefile, path_savefig)
# wave,flux, ferr are the wavelength,flux, and error the emission lines spectra
# path_savefile, the path to save the best-fit results
# path_savefig,  the path to save the best-fit figure 
```
Then construct the fitting function that will be used to fitting the emission lines spectra 
```
#Here is an example for Ha region
lf.add_gauss(1, 'Ha', 'n') #add a narrow gauss function representing the narrow component of Ha, 1 is the serial number of the component, 'Ha' is the name of the component, 'n' means this is a narrow Gauss component
lf.add_gauss(2, 'nii6583', 'n',sigma_tied=1, value_tied=1)#similar to the above, but representing for the narrow component of nii6583, sigma_tied=1 means fixing the sigma of this component same as the one of the component whose serial number is 1. value_tied=1, means fixing the value (i.e. the line center) same as the component '1'
lf.add_gauss(3, 'nii6549', 'n',sigma_tied=1, value_tied=1, scale_tied=2)# scaled_tied=2 means that the scale of the Gauss component was tied to the component '2', i.e., 'nii6583'. The procedure will search the library and found the theoretical ratio for the doublets .
lf.add_gauss(4, 'sii2', 'n', sigma_tied=1, value_tied=1)
lf.add_gauss(5, 'sii1', 'n', sigma_tied=1, value_tied=1)
lf.add_gauss(6, 'Ha', 'b')# 'b' means this is a broad component
lf.add_poly(7, order=1) # Add an 1-order polynominal to correct for local continuum fitting residual. This is optional.
```
In fact, when constructing the fitting function for ordinary line groups, and you can choose to use the templates in the procedure if you do not have special demands 
```
#Also for Ha region
lf.templates('Ha+sii')# This can reach the same functionality as the above sentences 
```
Finally, used the function constructed above to fit the emission lines 
```
lf.running() 
```
