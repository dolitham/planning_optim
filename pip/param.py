olivier_sheet_id = '1e-b3kbXg8_xdyO6vtlhIiYXokBZSrrCawRZnvg5K8wk'
cwd = '/Users/julie/PycharmProjects/misc/pip/'

# folder_path = "/Users/julie/Desktop/ucvet/ucvet/"
# holidays_path = folder_path + "Sheet 1-Holidays.csv"
jobs_path = cwd + "Sheet 1-Jobs.csv"

hospit_solo_name = 'Hospit'
consult_name = 'Consult'
echo_name = 'Echo'
endo_name = 'Endo'
hospit_echo_name = 'Hospit+E'

pay = {hospit_echo_name: 2 / 3, echo_name: 1}
max_we_in_a_row = 2
breaks = dict({7: [2, 2, 2, 2, 2]})  #, 10: [3, 3, 3, 3, 3]})

nb_max_consecutive_days = [5, 5, 5, 5, 5]

hospit_days = dict({0: 2, 3: 3})