#%% INPUT FILE

sheet_start_month = 9
sheet_start_year = 2022

#%% OPTIM / MODEL PARAMETERS

time_limit_minutes = 12

weekend_cost = 5
flexible_request_cost = 100
malus_no_consult = 50

#%% BALANCE  & FORBID

forbid_single_day_shifts = False
max_we_in_a_row = 2
breaks = dict({7: [2, 2, 2, 2, 2]})  # 10: [3, 3, 3, 3, 3]
nb_max_consecutive_days = [5, 5, 5, 5, 5]

nb_j_off_default = 5
threshold_prorata_holidays = 9
tolerance = dict({'echo_pay': 5, 'hospit_days': 4, 'flat_rate_days': 10, 'single_we': 2, 'we_days': 5,
                  'echo_percentage': 1000})

#%% JOB NAMES

hospit_solo_name = 'Hospit'
consult_name = 'Consult'
echo_name = 'Echo'
endo_name = 'Endo'
hospit_echo_name = 'Hospit+E'

default_cycle_start_days = {0, 3}
pay = {hospit_echo_name: 2 / 3, echo_name: 1}

lousy_sequence = [consult_name, hospit_solo_name]
jobs_exceptions = {'Esther': {hospit_echo_name: [0, 2, 3, 4, 5]}}
#jobs_exceptions = {}


shift_mort_length = 4
shift_mort_name = consult_name

#%% LOCATION

olivier_sheet_id = '1e-b3kbXg8_xdyO6vtlhIiYXokBZSrrCawRZnvg5K8wk'
calendar_sheet_range = 'Input Logiciel!A1:ZZ10'
stats_sheet_range = 'Statistiques!A1:K7'
sandbox_sheet_range = 'Sandbox!A1:K7'
files_directory = 'excel/'
excel_files_suffix = '.xlsx'

