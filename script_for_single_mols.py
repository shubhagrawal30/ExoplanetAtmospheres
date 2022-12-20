from petitRADTRANS import nat_cst as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from petitRADTRANS import Radtrans
from petitRADTRANS.retrieval import cloud_cond
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances

WLEN = [2.3, 2.5]
data_source = 'kpic'
lbl_downsample = 6  # so we get R~100000

# CHANGE PATH - where to save the plot
outpath = '/scr3/jxuan/kpic_analysis/prt_out/diagnostics/'

# CHANGE PATH - load Sonora PT profile; Teff = 2000, logg=4.5. 
sonora_file = '/scr3/jxuan/kpic_analysis/models/chem0.0_profile/t2000g316nc_m0.0.cmp'
so_df = pd.read_csv(sonora_file, delim_whitespace=True)
sonora_pt = np.array( [ so_df['TEMP'], so_df['P(BARS)'] ] )
# Pressure grid is from Sonora
pressure_grid = sonora_pt[1]
temp = sonora_pt[0]

# only H2O and CO for simplicity
highres_species = ['CO_main_iso_highres', 'H2O_main_iso_highres']
continuum_opacities_list = ['H2-H2', 'H2-He', 'H-']
input_cloud_species_list = np.copy(cloud_species_list)

# cloudy with scattering, following Molliere2020 for HR 8799 e
rt_object = Radtrans(line_species=highres_species,
                    rayleigh_species=['H2', 'He'],
                    continuum_opacities=continuum_opacities_list,
                    wlen_bords_micron=WLEN, mode='lbl', do_scat_emis=True,
                    lbl_opacity_sampling=lbl_downsample)

# Create the RT arrays of appropriate lengths
rt_object.setup_opa_structure(pressure_grid)

mass_mjup = 20
rad_rjup = 1.25
mass_cgs = mass_mjup*nc.m_jup
radius_cgs = rad_rjup*nc.r_jup
grav = mass_cgs*nc.G / radius_cgs**2

# Set the abundances
ab_dict = {}
ab_dict['C_O'] = 0.6
ab_dict['C_H'] = 0.0

fs = 24
fig, ax = plt.subplots(figsize=(20,10))
for molecule in ['H2O', 'CO_main']: 
    CO_val = ab_dict['C_O']
    C_H_val = ab_dict['C_H']

    # simply call pRT function
    abundances = {}
    COs = CO_val * np.ones_like(pressure_grid)
    C_Hs = C_H_val * np.ones_like(pressure_grid)
    abundances_interp = interpol_abundances(COs, C_Hs, temp, pressure_grid)
    MMW = abundances_interp['MMW']
    # make sure we have correct abundances names
    for species in rt_object.line_species:
        if molecule in species:
            abundances[species] = abundances_interp[species.split('_')[0]]
        # set abundances of other species to 0
        else:
            abundances[species] = np.zeros(len(pressure_grid))
                
    abundances['H2'] = abundances_interp['H2']
    abundances['He'] = abundances_interp['He']
    abundances['H'] = abundances_interp['H']
    abundances['H-'] = abundances_interp['H-']
    abundances['e-'] = abundances_interp['e-']

    # convert logKzz to Kzz, and give as array
    Kzz = np.ones_like(temp)*1e1**logKzz
          
    # compute the emission spectrum
    rt_object.calc_flux(temp, abundances, grav, MMW)

    # grab the flux and wvl, and contribution function
    wvl_microns = nc.c / rt_object.freq / 1e-4  # cm to microns

    # flux from Hz^-1 to micron^-1, then erg/s/cm^2 to W/m^2
    flux_unit_fac = (nc.c / wvl_microns ** 2 * 1e4) * 1e-3
    flux_SI_surface = rt_object.flux * flux_unit_fac
    
    # plot spectum of only 1 molecule
    ax.plot(wvl_microns, flux_SI_surface, alpha=0.6, linewidth=4, label=molecule)

ax.set_xlabel('Wavelength (microns)', fontsize=fs)
ax.set_ylabel('Outgoing Flux (W/m^2)', fontsize=fs)
ax.tick_params(labelsize=fs)
ax.legend(fontsize=fs)
plt.savefig(outpath + 'fsed1_diagnostic_h2o_co.png', dpi=100,  bbox_inches='tight')
