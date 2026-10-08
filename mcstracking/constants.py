#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2016-11-29T13:36:38-10:00
# Email:  dab8@hawaii.edu
# Last modified by:   daniel
# Last modified time: 2016-11-29T13:36:38-10:00
#
# @Project@ All projects
# Version: 1.0
# Description: Files containing frequently used constants
#
# Dependencies: None
#
# Files:
#
#####################################################################
"""



class const:
    Rd = 287.04 #gas constant air (dry)
    Rv = 461.5  #gas constant vapor
    cp = 7.*Rd/2.
    cv = 718.
    epsilon_gamma = 0.62197
    es_base_bolton = 611.2 #Pa
    es_Abolton = 17.67 #
    es_Bbolton = 243.5 # degC
    es_base_tetens = 6.1078
    es_Atetens_vapor = 7.5
    es_Btetens_vapor = 237.3
    es_Atetens_ice = 9.5
    es_Btetens_ice = 265.5
    g = 9.81
    p1000mb = 100000.
    pconst = 10000.
    gamma = 0.0065
    rcp = Rd/cp
    tkelvin = 273.15
    missingval = 1.e+20
    kappa = (cp-cv)/cp
    L= 2.501e6 # latent heat of vaporization
    #Constants used to calcualte moist adiabatic lapse rate
    #Formula 3.16 in Rogers&Yau (from Geir Arne Waagboe ;http://code.google.com/p/pywrfplot/)
    a = 2./7.
    b=epsilon_gamma*L*L/(Rd*cp)
    c = a*L/Rd
    secinday = 86400 # Number of seconds in a day
    earth_radius=6371000 #Earth radius in m
    SB_sigma = 5.6704e-8 #Stefan–Boltzmann constant W/m2·K
