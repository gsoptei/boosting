import os
import pyreadr as pr
from scipy.stats import uniform, randint

# Define variables
WORK_DIR = os.getcwd()
DATA_FILE = pr.read_r('/home/gergeli/Asztal/Suli/Lisbon/Datasets/ppe_qu_20231010.R')
# MODEL = '/home/gergeli/jupyter_dir/machineLearning/Modeling/xgboost_test.json'
FILE_PATH = '/home/gergeli/jupyter_dir/machineLearning/Modeling/'
COLUMNS_TO_DROP = ['OBJECTID',
                   'X',
                   'Y',
                   'AI',
                   'BIO1', 
                   'BIO10', 
                   'BIO11', 
                   'BIO12', 
                   'BIO13', 
                   'BIO14', 
                   'BIO15', 
                   'BIO16', 
                   'BIO17', 
                   'BIO18', 
                   'BIO19', 
                   'BIO2', 
                   'BIO3', 
                   'BIO4', 
                   'BIO5', 
                   'BIO6', 
                   'BIO7', 
                   'BIO8', 
                   'BIO9',
                   'CMI_MAX', 
                   'CMI_MEAN', 
                   'CMI_MIN', 
                   'CMI_RANGE', 
                   'FCF', 
                   'FGD', 
                   'NGD0', 
                   'SFCWIND_RANGE',
                   'AMPL', 
                   'EOSD', 
                   'EOSV', 
                   'LENGTH', 
                   'LSLOPE', 
                   'MAXD', 
                   'MAXV', 
                   'MINV', 
                   'RSLOPE', 
                   'SOSD', 
                   'SOSV', 
                   'TPROD',
                   'pp_tot',
                   'ev_bs',
                   'p_ve',
                   'ss_runoff',
                   'ev_t',
                   'surf_sol_rad',
                   'surf_therm_rad',
                   'dew_temp',
                   'temp',
                   'soil_temp_l1',
                   'soil_temp_l2',
                   'vol_soil_water_l1',
                   'vol_soil_water_l2'
]
# Feature labels for plotting -- better readibility
# Matching number of labels and columns or else plotting won't work
LABELS = ['pt_dem',
          'pt_slope',
          'pt_psr',          
          'bio_01',
          'bio_02',
          'bio_03',
          'bio_04',
          'bio_05',
          'bio_06',
          'bio_07',
          'bio_08',
          'bio_09',
          'bio_10',
          'bio_11',
          'bio_12',
          'bio_13',
          'bio_14',
          'bio_15',
          'bio_16',
          'bio_17',
          'bio_18',
          'bio_19',
          'AWC',
          'Bulk_dens',
          'Clay',
          'Coarse',
          'Sand',
          'Silt',
          'CaCO3',
          'CEC',
          'CN',
          'K',
          'N',
          'pH_CaCl',
          'pH_H2O',
          'pH_H2O_CaCl',
          'P',
          'Area',
          'Perc_Evergreen',
          'Perc_Deciduous',
          'Perc_Unknown',
          'Dens_Ed_repair',
          'Dens_Vacant',
          'Pop_density',
          'Pop_dens_0_14',
          'Pop_dens_15_24',
          'Pop_dens_25_64',
          'Pop_dens_65more']
         
# Parameter thresholds for hyperparameter search
PARAMS = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4),
}

# Create list of models
MODELS = []
# Save predictions
PREDICTIONS = []

ITERATIONS = [5, 10, 100, 200, 400]

CV = [5, 10]