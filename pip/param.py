olivier_sheet_id = '1e-b3kbXg8_xdyO6vtlhIiYXokBZSrrCawRZnvg5K8wk'
calendar_sheet_range = 'Input Logiciel!A1:AZ10'
stats_sheet_range = 'Statistiques!A1:K7'
sandbox_sheet_range = 'Sandbox!A1:K7'
cwd = '/Users/julie/PycharmProjects/misc/pip/'
files_directory = 'excel/'
files_suffix = '_UCVet.xlsx'
jobs_path = cwd + "jobs.csv"

hospit_solo_name = 'Hospit'
consult_name = 'Consult'
echo_name = 'Echo'
endo_name = 'Endo'
hospit_echo_name = 'Hospit+E'

pay = {hospit_echo_name: 2 / 3, echo_name: 1}
max_we_in_a_row = 2
breaks = dict({7: [2, 2, 2, 2, 2]})  # 10: [3, 3, 3, 3, 3]
nb_max_consecutive_days = [5, 5, 5, 5, 5]
forbid_single_day_shifts = True
default_cycle_start_days = {0, 3}

shift_mort_name = 'Consult'
shift_mort_length = 4

lousy_sequence = ['Consult', 'Hospit']
nb_j_off_default = 5
threshold_prorata_holidays = 7

tolerance = dict({'echo_pay': 2, 'hospit_days': 3, 'flat_rate_days': 3, 'single_we': 1, 'we_days': 3})
