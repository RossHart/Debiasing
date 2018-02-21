from astropy.table import Table, column
import math
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def z_to_dist(z):
    return cosmo.luminosity_distance(z)


def dist_to_z(D):
    return Distance.compute_z(D,cosmo)


def mag_to_Mag(mag,z):
    D = z_to_dist(z)
    Mag = mag - 5*(np.log10(D/(u.pc))-1)
    return Mag


def Mag_to_mag(Mag,z):
    D = z_to_dist(z)
    mag = Mag + 5*(np.log10(D/(u.pc))-1)
    return mag
  
  
def Mag_to_z(Mag,mag=17):
    D = 10**((mag-Mag+5)/5)*(u.pc)
    z = dist_to_z(D)
    return z
 
    
def Mag_to_flux_density(Mag):
    S = 3631*10**(Mag/-2.5)*u.Jy # AB -> flux density
    L = S*(4*math.pi)*(10*u.pc)**2 # absolute magnitude = 10pc
    return L.to(u.erg/u.s/u.Hz)


def wavelength_to_frequency(wavelength):
    c = const.c
    frequency = (c/wavelength).to(u.Hz)
    return frequency


def Mag_to_lum(Mag,wavelength):
    frequency = wavelength_to_frequency(wavelength*(u.Angstrom))
    L_density = Mag_to_flux_density(Mag)
    L = L_density*frequency
    return L


def lum_to_solar(L):
    L_solar = 3.828e33*(u.erg/u.s)
    logLsun = np.log10(L/L_solar)
    return logLsun


def solar_to_lum(logLsun):
    L_solar = 3.828e33*(u.erg/u.s)
    L = 10**logLsun*L_solar
    return L


def lum_to_Mag(L,wavelength):
    c = const.c
    frequency = wavelength_to_frequency(wavelength*(u.Angstrom))
    L_density = (L/frequency).to(u.erg/u.s/u.Hz)
    S = (L_density/(4*math.pi*(10*u.pc)**2)).to(u.Jy)
    Mag = -2.5*np.log10(S/(3631*(u.Jy)))
    return Mag