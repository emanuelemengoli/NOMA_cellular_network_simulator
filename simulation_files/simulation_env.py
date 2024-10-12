DISTANCE_METRIC = 'km'
POWER_METRIC = 'dBm'

earth_radius_km = 6371.0 #km

DATA_SOURCE_LINK = 'https://files.data.gouv.fr/arcep_donnees/mobile/sites/2022_T3/2022_T3_sites_Metropole.csv'

#nbss = 30 "simulation_datasets/Paris_BSs_subset_km.csv"
# ORIGIN = (48.83700932160592, 2.254146303457459)

# NET_WIDTH = 2.5 #km
# NET_HEIGHT = 2.45 #km

# BS_MAX_RANGE = 0.45 #km sqrt(1/(n_bss / area_km2))
#------------------------

#nbss = 30 "simulation_datasets/Paris_BSs_subset_km_30bs_reparametrized.csv"
# ORIGIN = (48.8383583040148, 2.2561957586436474)

# NET_WIDTH = 2.447 #km

# NET_HEIGHT = 2.4 #km

# BS_MAX_RANGE = 0.45 #km 

#-----------------------
#nbss = 12 -> "simulation_datasets/Paris_BSs_subset_km2.csv" 
ORIGIN = (48.839509321605924, 2.2541463716478995) 

NET_WIDTH = 2.4 #km

NET_HEIGHT = 1.1 #km

BS_MAX_RANGE = 0.47 #km

#-----------------------

SITE_TYPE = 'site_4g'


GEO_AREA = 'Paris'

TILE_SIZE = 25/1000  #'km'

##Constants

W = 20*(10**6) #(M)hz Channel Bandwidth

N = -125 #dBm/Hz background noise

L_ref = 120 #dBm/1km

ALPHA = 3.76 #lin-scale

BS_P_TX = 46 #dBm

EPS_BORDER = 0.2 #km


def simulation_params():
    # Get all globals defined in this module
    for name, val in globals().items():
        if not name.startswith('__') and not callable(val):
            print(f"{name}: {val}")

