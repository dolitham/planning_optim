import itertools
from datetime import datetime
from datetime import timedelta
import humanize
import locale
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import xlsxwriter
import openpyxl

########################
# PEOPLE PARAMETERS
########################

#min_each_everyday_job = 11

min_echo_job = 6
pay = {'Hospit+E': 2 / 3, 'Echo': 1}
print('Nb jours minimum echo / personne :', min_echo_job)

########################
# PREVIOUS PLANNING PARAMETERS
########################

last_day_hospit_person = 1
current_shift = [0, 2, 1, 0, 1]
delta_weekend_days = [2, 0, 2, 2, 2]
delta_flat_rate_days = [0, 2, 0, 0, 0]
delta_hospit_days = [0, 0, 0, 0, 0]
nb_max_consecutive_days = [4, 4, 4, 4, 4]
current_weekend_streak = [0, 0, 0, 0, 0]
#max_flat_rate_days = [np.inf, 22, np.inf, np.inf, np.inf]
max_flat_rate_days = [np.inf, np.inf, np.inf, np.inf, np.inf]

########################
# CLINIC PARAMETERS
########################

weekend_prefix = {'Sam ', 'Dim '}
always_regroup_weekends = [True, True, True, True, True]

break_length = 2
nb_rolling_days = 7

break_length_bis = 3
nb_rolling_days_bis = 11

consult_job = 1
echo_job = 2
hospit_echo_job = 3
hospit_solo_job = 0
hospit_jobs = {hospit_echo_job, hospit_solo_job}
hospit_days = dict({0: 2, 3: 3})
new_everyday_jobs = [{consult_job}, {hospit_solo_job, hospit_echo_job}]

#shift_mort_start_dow = 4
shift_mort_length = 4
shift_mort_job = 1

########################
# OPTIM PARAMETERS
########################

time_limit = 300
weekend_cost = 10


locale.setlocale(locale.LC_TIME, "fr_FR")
pd.set_option("max_columns", 10)
pd.options.display.float_format = '€{:,.2f}'.format

folder_path = "/Users/julie/Desktop/ucvet/ucvet/"
holidays_path = folder_path + "Sheet 1-Holidays.csv"
jobs_path = folder_path + "Sheet 1-Jobs.csv"
suffix_time = datetime.now().strftime('%H%M%S')
result_path = folder_path + suffix_time + "_people"
planning_path = folder_path + suffix_time + "_clinic"
info_path = folder_path + suffix_time + "_info"

param_holidays = pd.read_csv(holidays_path, parse_dates=[0], delimiter=";", index_col=0) \
    .fillna(0).astype(int).replace({0: 1, 1: 0})
param_jobs = pd.read_csv(jobs_path, delimiter=";", index_col=0)
flat_rate_jobs = list(itertools.chain.from_iterable(new_everyday_jobs))
echo_weekdays = hospit_days.keys()
total_days = len(param_holidays.index)
people = len(param_holidays.columns)
jobs = len(param_jobs.index)
weekend_days_map = param_holidays.index.map(lambda x: x.dayofweek) >= 5

echo_pay_goal_each = total_days * pay['Hospit+E'] / 4 + min_echo_job * (pay['Echo'] - pay['Hospit+E'])
people_names = param_jobs.columns


def get_dow(date):
    return param_holidays.index[date].dayofweek


def is_weekend_day(date_str):
    return any(date_str.startswith(pref) for pref in weekend_prefix)


def get_flat_rate_days_goal(person):
    goal = (param_holidays.shape[0] * 2 + sum(delta_flat_rate_days)) / 5 - delta_flat_rate_days[person]
    mini = int(goal)
    maxi = int(np.ceil(goal))
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_weekend_days_goal(person):
    goal = (sum(weekend_days_map) * 2 + sum(delta_weekend_days)) / 5 - delta_weekend_days[person]
    mini = int(2 * (goal // 2))
    maxi = mini + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_hospit_days_goal(person):
    goal = (param_holidays.shape[0] + sum(delta_hospit_days)) / 5 - delta_hospit_days[person]
    mini = int(goal) - 1
    maxi = int(np.ceil(goal)) + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


model = pyo.ConcreteModel()
model.index_days = pyo.Set(initialize=range(total_days))
model.index_people = pyo.Set(initialize=range(people))
model.index_jobs = pyo.Set(initialize=range(jobs))
model.indexes = pyo.Set(initialize=model.index_days * model.index_people * model.index_jobs)

model.x = pyo.Var(model.indexes, within=pyo.NonNegativeIntegers, initialize=0)

model.one_person_per_job = pyo.ConstraintList()
for day in model.index_days:
    for job in model.index_jobs:
        model.one_person_per_job.add(
            sum(model.x[day, person, job] for person in model.index_people) <= 1
        )


model.holidays = pyo.ConstraintList()
for day in model.index_days:
    for person in model.index_people:
        model.holidays.add(
            sum(model.x[day, person, job] for job in model.index_jobs) <= param_holidays.iloc[day, person])


model.qualifications = pyo.ConstraintList()
for person in model.index_people:
    for job in model.index_jobs:
        if param_jobs.iloc[job, person] == 0:
            for day in model.index_days:
                model.qualifications.add(model.x[day, person, job] == 0)


model.consecutive_days = pyo.ConstraintList()
for person in model.index_people:
    for day0 in range(total_days - nb_max_consecutive_days[person]):
        model.consecutive_days.add(
            sum(sum(model.x[day, person, job]
                    for job in model.index_jobs)
                for day in range(day0, day0 + nb_max_consecutive_days[person] + 1))
            <= nb_max_consecutive_days[person]
        )

model.breaks = pyo.ConstraintList()
for day0 in range(total_days - nb_rolling_days):
    for person in model.index_people:
        model.breaks.add(
            sum(sum(model.x[day, person, job]
                for job in model.index_jobs)
                for day in range(day0, day0 + nb_rolling_days))
            <= nb_rolling_days - break_length
        )

for day0 in range(total_days - nb_rolling_days_bis):
    for person in model.index_people:
        model.breaks.add(
            sum(sum(model.x[day, person, job]
                for job in model.index_jobs)
                for day in range(day0, day0 + nb_rolling_days_bis))
            <= nb_rolling_days_bis - break_length_bis
        )


max_we_row = 2
model.weekends_in_a_row = pyo.ConstraintList()
for day0 in range(total_days - max_we_row*7 - 1):
    if param_holidays.index[day0].dayofweek == 5:
        for person in model.index_people:
            model.weekends_in_a_row.add(
                sum(sum(model.x[day0 + 7*i, person, job] for job in model.index_jobs) for i in range(max_we_row+1))
                <= max_we_row
            )


model.force_shifts = pyo.ConstraintList()
# esther hospit du 2 au 5
for day in [0, 1, 2, 3]:
    this_person = 3
    model.force_shifts.add(
        sum(model.x[day, this_person, hospit_job] for hospit_job in hospit_jobs) == 1
    )
# estelle consult 2 au 3
for day in [0, 1]:
    this_person = 4
    this_job = 1
    model.force_shifts.add(
        model.x[day, this_person, this_job] == 1
    )
# olivier consult 4 et 5
for day in [2, 3]:
    this_person = 0
    this_job = 1
    model.force_shifts.add(
        model.x[day, this_person, this_job] == 1
    )


model.occupy_everyday_positions = pyo.ConstraintList()
for day in model.index_days:
    for everyday_job_set in new_everyday_jobs:
        model.occupy_everyday_positions.add(
            sum(model.x[day, person, everyday_job] for person in model.index_people for everyday_job in
                everyday_job_set) == 1
        )

"""
# if someone starts cycle on day1, someone needs to do resulting job the same day
model.start_cycle = pyo.ConstraintList()
for day0 in range(days - 1):
    for jm in model.index_people:
        # if jm starts cycle job on day0+1, jm_start_cycle_day1 = 2
        jm_start_cycle_day1 = model.x[day0 + 1, jm, cycle_start_job] + sum(
            model.x[day0, person, cycle_start_job] for person in model.index_people if person != jm)
        someone_resulting_day1 = sum(model.x[day0 + 1, person, resulting_job] for person in model.index_people)
        model.start_cycle.add(
            2 * jm_start_cycle_day1 - someone_resulting_day1 <= 3
        )
"""

model.echo = pyo.ConstraintList()
for day in model.index_days:
    if param_holidays.index[day].dayofweek in echo_weekdays:
        model.echo.add(
            sum(model.x[day, person, echo_job] for person in model.index_people) == 1
        )

    model.echo.add(
        sum(2 * model.x[day, person, hospit_echo_job] + model.x[day, person, hospit_solo_job] + model.x[
            day, person, echo_job]
            for person in model.index_people
            )
        == 2
    )
"""        
# if pierre is in hospit_job, someone needs to do echo_job
for day in model.index_days:
    model.echo.add(
        model.x[day, pierre_index, hospit_job] <= sum(model.x[day, person, echo_job] for person in model.index_people)
    )
"""

model.cycle_hospit = pyo.ConstraintList()
for day in model.index_days:
    for first_day, nb_consecutive_days in hospit_days.items():
        if get_dow(day) == first_day:
            for person in model.index_people:
                for same_cycle_day in range(day + 1, min(day + 1 + nb_consecutive_days, total_days)):
                    model.cycle_hospit.add(
                        sum(model.x[day, person, hospit_job] for hospit_job in hospit_jobs)
                        ==
                        sum(model.x[same_cycle_day, person, hospit_job] for hospit_job in hospit_jobs)
                    )

"""
model.nb_days_each = pyo.ConstraintList()
for person in model.index_people:
    for everyday_job_set in new_everyday_jobs:
        model.nb_days_each.add(
            sum(model.x[day, person, same_job] for day in model.index_days for same_job in everyday_job_set) >= 4
        )
"""

model.balance_flat_rate_days = pyo.ConstraintList()
print('JOURS FORFAIT')
for person in model.index_people:
    mini_flat_days, maxi_flat_days = get_flat_rate_days_goal(person)
    maxi_flat_days = min(maxi_flat_days, max_flat_rate_days[person])
    model.balance_flat_rate_days.add(
        mini_flat_days <= sum(sum(model.x[day, person, job] for job in flat_rate_jobs) for day in model.index_days)
    )
    model.balance_flat_rate_days.add(
        maxi_flat_days >= sum(sum(model.x[day, person, job] for job in flat_rate_jobs) for day in model.index_days)
    )
print('\n')

model.balance_hospit_days = pyo.ConstraintList()
print('JOURS CHENIL')
for person in model.index_people:
    mini_hospit_days, maxi_hospit_days = get_hospit_days_goal(person)
    model.balance_hospit_days.add(
        mini_hospit_days <= sum(sum(model.x[day, person, job] for job in hospit_jobs) for day in model.index_days)
    )
    model.balance_hospit_days.add(
        maxi_hospit_days >= sum(sum(model.x[day, person, job] for job in hospit_jobs) for day in model.index_days)
    )
print('\n')


model.balance_weekend_days = pyo.ConstraintList()
weekend_days_indexes = np.where(weekend_days_map)[0]
print('JOURS WEEKEND')
for person in model.index_people:
    min_weekend_days, max_weekend_days = get_weekend_days_goal(person)
    model.balance_weekend_days.add(
        sum(sum(model.x[day, person, job] for job in model.index_jobs) for day in
            weekend_days_indexes) <= max_weekend_days
    )
    model.balance_weekend_days.add(
        sum(sum(model.x[day, person, job] for job in model.index_jobs) for day in
            weekend_days_indexes) >= min_weekend_days
    )
print('\n')

model.no_echo_on_weekends = pyo.ConstraintList()
for person in model.index_people:
    for day in weekend_days_indexes:
        model.no_echo_on_weekends.add(
            model.x[day, person, echo_job] <= model.x[day, 1, hospit_solo_job]
        )


model.balance_echo_days = pyo.ConstraintList()
people_echo_indexes = np.where(param_jobs.iloc[2] > 0)[0]
for person in people_echo_indexes:
    model.balance_echo_days.add(
        sum(model.x[day, person, echo_job] for day in model.index_days) >= min_echo_job
    )

model.balance_echo_pay = pyo.ConstraintList()
people_echo_indexes = np.where(param_jobs.iloc[echo_job] > 0)[0]
print('SALAIRES ECHO')
for person in people_echo_indexes:
    print(people_names[person], '[' +"{:.1f}".format(echo_pay_goal_each) + ', ' + "{:.1f}".format(echo_pay_goal_each + 1.5*pay['Echo']) + ']', end =' | ')
    model.balance_echo_pay.add(
        sum(model.x[day, person, echo_job] * pay['Echo']
            + model.x[day, person, hospit_echo_job] * pay['Hospit+E']
            for day in model.index_days)
        >=  echo_pay_goal_each
    )


    model.balance_echo_pay.add(
        sum(model.x[day, person, echo_job] * pay['Echo']
            + model.x[day, person, hospit_echo_job] * pay['Hospit+E']
            for day in model.index_days)
        <= echo_pay_goal_each + 1.5*pay['Echo']
    )
print('\n')

model.no_shift_de_la_mort = pyo.ConstraintList()
for day in model.index_days:
    #if get_dow(day) == shift_mort_start_dow:
    for person in model.index_people:
        model.no_shift_de_la_mort.add(
            sum(model.x[day, person, shift_mort_job] for day in
                range(day, min(day + shift_mort_length, total_days)))
            <= shift_mort_length - 1
        )

model.only_screw_full_weekends = pyo.ConstraintList()
for day in model.index_days:
    if get_dow(day) == 5:
        for person in model.index_people:
            if always_regroup_weekends[person] & (day < total_days - 1):
                #model.only_screw_full_weekends.add(
                #    sum(model.x[day, person, job] for job in model.index_jobs) == sum(
                #        model.x[day + 1, person, job] for job in model.index_jobs)
                #)
                for job in model.index_jobs:
                    model.only_screw_full_weekends.add(
                        model.x[day, person, job] == model.x[day + 1, person, job]
                    )

model.interface_cycle_hospit = pyo.ConstraintList()
dow = param_holidays.index[0].dayofweek
if dow not in hospit_days.keys():
    nb_hospit_days_same_person = [st + le - (dow - 1) for st, le in hospit_days.items() if st <= dow <= st + le][0]
    for day in range(nb_hospit_days_same_person):
        model.interface_cycle_hospit.add(
            sum(model.x[day, last_day_hospit_person, hospit_job] for hospit_job in hospit_jobs) == 1
        )

model.interface_max_shift = pyo.ConstraintList()
for person in model.index_people:
    nb_consecutive_days_left = nb_max_consecutive_days[person] - current_shift[person]
    model.interface_max_shift.add(
        sum(sum(model.x[day, person, job]
                for job in model.index_jobs)
            for day in range(0, nb_consecutive_days_left + 1))
        <= nb_consecutive_days_left
    )

"""
model.no_single_day_shifts = pyo.ConstraintList()
for person in model.index_people:
    for day0 in range(days - 2):
        model.no_single_day_shifts.add(
            sum(model.x[day0, person, job] for job in model.index_jobs) 
            + sum(model.x[day0 + 2, person, job] for job in model.index_jobs)
            >= sum(model.x[day0 + 1, person, job] for job in model.index_jobs)
        )
"""


def nb_person_shifts_weekend(m):
    return sum(sum(m.x[my_day, person, my_job] for my_day in m.index_days for my_job in m.index_jobs if
                   param_holidays.index[0].dayofweek >= 5) for person in m.index_people)


def obj_min_staff(m):
    return (weekend_cost - 1) * nb_person_shifts_weekend(m) + pyo.summation(
        m.x)


model.obj1 = pyo.Objective(rule=obj_min_staff, sense=pyo.minimize)


def transform_solved_model_into_result(m):
    my_result = param_holidays.copy().replace(1, 0).replace(0, '')
    my_planning = pd.DataFrame(index=param_holidays.index, columns=param_jobs.index)
    for v in m.component_data_objects(pyo.Var):
        my_day, person, my_job = [int(u) for u in str(v)[2:-1].split(',')]
        if v.value > 0:
            my_result.iloc[my_day, person] = my_result.iloc[my_day, person] + param_jobs.index[my_job]
            my_planning.iloc[my_day, my_job] = people_names[person]

    my_planning.fillna('', inplace=True)
    my_planning['Hospit'] = my_planning['Hospit'].str.cat(my_planning['Hospit+E'])
    my_planning = my_planning.drop(labels='Hospit+E', axis='columns')
    my_result = my_result.where((my_result == 'Echo').sum(axis=1) == 1, my_result.replace('Hospit', 'Hospit+E'))
    return my_result, my_planning


def run_with_time_limit(seconds):
    solver = pyo.SolverFactory('glpk')
    solved = solver.solve(model, timelimit=seconds)
    my_result, my_planning = transform_solved_model_into_result(model)
    return my_result, my_planning, solved


def print_time_limit_info(seconds):
    print('TIME LIMIT :', humanize.naturaldelta(timedelta(seconds=seconds)))
    now = datetime.now()
    print('STARTED AT', now)
    return now


def print_solved_time_info(time):
    now = datetime.now()
    print('SOLVED AT', now)
    print('RUNTIME :', humanize.naturaldelta(now - time))


start_time = print_time_limit_info(time_limit)
result, planning, opt = run_with_time_limit(time_limit)
print('STATUS', str(opt.solver.termination_condition))
print_solved_time_info(start_time)

shifts_info = pd.DataFrame(columns=people_names, index=param_jobs.index).fillna(0)
for name in people_names:
    nb_shifts = result[name].value_counts()
    for job in set(result[name].value_counts().index) & set(param_jobs.index):
        shifts_info.loc[job, name] = nb_shifts[job]
    shifts_info.loc['Jours weekend travaillés', name] = (result[name][weekend_days_map] != '').sum()
    shifts_info.loc['Jours off', name] = (result[name] == '').sum()
    shifts_info.loc['Jours on', name] = (result[name] != '').sum()
shifts_info.loc['Jours forfait', :] = shifts_info.loc['Hospit', :] + shifts_info.loc['Consult', :] \
                                      + shifts_info.loc['Hospit+E', :]
shifts_info.loc['equiv Jours echo', :] = (pay['Echo'] * shifts_info.loc['Echo', :]
                                          + pay['Hospit+E'] * shifts_info.loc['Hospit+E', :])
shifts_info.loc['Hospit', :] = shifts_info.loc['Hospit+E', :] + shifts_info.loc['Hospit', :]
shifts_info.drop(labels='Hospit+E', axis="rows", inplace=True)
for index, row in shifts_info.iterrows():
    if index == 'equiv Jours echo':
        shifts_info.loc[index, :] = shifts_info.loc[index, :].map('€{:,.2f}'.format)
    else:
        shifts_info.loc[index, :] = shifts_info.loc[index, :].astype(int).astype(str)


def color_weekend(val):
    color = 'background-color: #c4c4c4' if any(val.startswith(weekend_day) for weekend_day in weekend_prefix) else ''
    return color


def color_off(val):
    color = 'background-color: #A0DE98' if val == 'off' else ''
    return color  # 'color: %s' % color


def color_people(val):
    colors = dict({"Esther": "#65B7FF",
                   "Gaël": "#FFC665",
                   "Olivier": "#8DD797",
                   "Marine": "#FF6565",
                   "Estelle": "#FFEF65"
                   })
    if val in colors.keys():
        return 'background-color: ' + colors[val]
    return ''


def export_colored_df(df, path, with_dates=True, with_colors=True):
    if with_dates:
        df.reset_index(inplace=True)
        df['Days'] = df['Days'].dt.strftime('%a %d %b')

    df_html = df.style
    if with_colors:
        df_html = df_html.applymap(color_weekend).hide_index().applymap(color_off).applymap(color_people)

    text_file = open(path + '.html', "w")
    text_file.write('<meta charset="UTF-8">' + df_html.render())
    text_file.close()


# export results
for person_name in param_holidays.columns:
    result[person_name] = result[person_name].str.cat(param_holidays.replace(0, 'off').replace(1, '')[person_name])
#export_colored_df(result, result_path)
#export_colored_df(planning, planning_path)
#export_colored_df(shifts_info, info_path, with_dates=False, with_colors=False)


colors = dict({"Esther": "#65B7FF",
               "Gaël": "#FFC665",
               "Olivier": "#8DD797",
               "Marine": "#FF6565",
               "Estelle": "#FFEF65",
               "off": "#A0DE98",
               'Hospit': "#F8CED3",
               "Hospit+E": "#D1C2CE",
               "Consult": "#FCECE0",
               "Echo": "#DBDAD9"})

def highlight(df):
    weekend_color = 'background-color: {}'.format('#c4c4c4')
    df_attributes = pd.DataFrame('', index=df.index, columns=df.columns)
    if 'Days' in df.columns:
        df_attributes['Days'] = pd.DataFrame(np.where(df['Days'].str.startswith('Dim '), weekend_color, '')) \
        + pd.DataFrame(np.where(df['Days'].str.startswith('Sam '), weekend_color, ''))
    for name, color in colors.items():
        attr = 'background-color: {}'.format(color)
        df_attributes = df_attributes + pd.DataFrame(np.where(df ==name, attr, ""), index= df.index, columns=df.columns)
    return df_attributes


def export_as_excel(df, writer, sheet_name, index=False, with_dates=True):
    if with_dates:
        df.reset_index(inplace=True)
        df['Days'] = df['Days'].dt.strftime('%a %d %b')
    df.style.apply(highlight, axis=None).to_excel(writer, engine="openpyxl", index=index, sheet_name=sheet_name)

with pd.ExcelWriter(folder_path + suffix_time + '_UCVet.xlsx') as writer:
    export_as_excel(result, writer, 'People', )
    export_as_excel(planning, writer, 'Clinic')
    export_as_excel(shifts_info, writer, 'Info', index=True, with_dates=False)
