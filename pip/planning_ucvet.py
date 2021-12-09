#%% IMPORTS + DISPLAY SETTING

import itertools
from datetime import datetime
from datetime import timedelta
import humanize
import locale
import pyomo.environ as pyo
import pandas as pd
import numpy as np

locale.setlocale(locale.LC_TIME, "fr_FR")
pd.set_option("max_columns", 10)
pd.options.display.float_format = '€{:,.2f}'.format

#%%  FILES READING + DIRECT INFERENCES

folder_path = "/Users/julie/Desktop/ucvet/ucvet/"
holidays_path = folder_path + "Sheet 1-Holidays.csv"
jobs_path = folder_path + "Sheet 1-Jobs.csv"

param_holidays = pd.read_csv(holidays_path, parse_dates=[0], delimiter=";", index_col=0).isnull()*1
param_jobs = pd.read_csv(jobs_path, delimiter=";", index_col=0)

people_names = list(param_jobs.columns)
job_names = list(param_jobs.index)
total_days = len(param_holidays.index)
people = len(param_holidays.columns)
weekend_days_map = param_holidays.index.map(lambda x: x.dayofweek) >= 5
jobs = len(job_names)

#%% OUTPUT FILENAMES

suffix_time = datetime.now().strftime('%H%M%S')
result_path = folder_path + suffix_time + "_people"
planning_path = folder_path + suffix_time + "_clinic"
info_path = folder_path + suffix_time + "_info"

#%% JOBS NAMES & PARAMETERS

hospit_solo_name = 'Hospit'
consult_name = 'Consult'
echo_name = 'Echo'
endo_name = 'Endo'
hospit_echo_name = 'Hospit+E'

hospit_days = dict({0: 2, 3: 3})
echo_weekdays = hospit_days.keys()

hospit_solo_job = job_names.index(hospit_solo_name)
consult_job = job_names.index(consult_name)
echo_job = job_names.index(echo_name)
hospit_echo_job = job_names.index(hospit_echo_name)
do_endo = endo_name in job_names
echo_people = np.where(param_jobs.iloc[echo_job] > 0)[0]

hospit_jobs = {hospit_echo_job, hospit_solo_job}
new_everyday_jobs = [{consult_job}, {hospit_solo_job, hospit_echo_job}]
flat_rate_jobs = list(itertools.chain.from_iterable(new_everyday_jobs))

#%% OPTIM / MODEL PARAMETERS

time_limit = 60*5
weekend_cost = 100

#%%  SHIFT PARAMETERS

#min_echo_job = 6
max_delta_echo_pay = 1.5
pay = {hospit_echo_name: 2 / 3, echo_name: 1}
max_we_in_a_row = 2
breaks = dict({7: [2, 2, 2, 2, 2], 11: [3, 3, 3, 3, 3]})

#%%  PREVIOUS PLANNING PARAMETERS

last_day_hospit_person = 1
current_shift = [0, 0, 0, 0, 0]
delta_weekend_days = [0, 0, 0, 0, 0]
#delta_flat_rate_days = [0, 0, 0, 0, 0]
#delta_hospit_days = [0, 0, 0, 0, 0]
nb_max_consecutive_days = [4, 4, 4, 4, 4]
current_weekend_streak = [0, 0, 0, 0, 0]

#%%  CLINIC PARAMETERS

shift_mort_length = 4
shift_mort_job = consult_job
nb_j_off_default = 5
seuil_prorata_vacances = 10

#%% LOGS

global output
output = ''


def my_print(text):
    global output
    line = ' '.join(str(u) for u in text) + '\n'
    output += line
    print(line, end='')

#%% TIME MANAGEMENT

def get_dow(date):
    return param_holidays.index[date].dayofweek

#%% SETTING SHIFT TARGETS

availability = param_holidays.sum()
availability[availability >= total_days - seuil_prorata_vacances] = total_days - nb_j_off_default
target_flat_rate = availability/sum(availability)*total_days*2
target_hospit = availability/sum(availability)*total_days


availability_echo = param_holidays.sum()
availability_echo.drop(index="Marine", inplace=True)
min_j_echo = 2*(availability_echo >= total_days - seuil_prorata_vacances)
total_salaire_echo = total_days*pay['Hospit+E'] + sum(min_j_echo)*(pay['Echo'] - pay['Hospit+E'])
echo_goal = availability_echo/sum(availability_echo)*total_salaire_echo


def get_weekend_days_goal(person):
    goal = (sum(weekend_days_map) * 2 + sum(delta_weekend_days)) / 5 - delta_weekend_days[person]
    mini = int(2 * (goal // 2))
    maxi = mini + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_flat_rate_days_goal(person):
    goal = target_flat_rate.loc[people_names[person]]
    mini = int(goal) - 1
    maxi = mini + 1 + 2 #TODO REMOVE THIS
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_hospit_days_goal(person):
    goal = target_hospit.loc[people_names[person]]
    mini = int(goal) - 1
    maxi = mini + 1 + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_echo_pay_goal(person):
    goal = echo_goal.loc[people_names[person]]
    mini = int(goal) - 1
    maxi = mini + 1 + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


#%% MODEL INIT

model = pyo.ConcreteModel()
model.index_days = pyo.Set(initialize=range(total_days))
model.index_people = pyo.Set(initialize=range(people))
model.index_jobs = pyo.Set(initialize=range(jobs))
model.indexes = pyo.Set(initialize=model.index_days * model.index_people * model.index_jobs)

model.x = pyo.Var(model.indexes, within=pyo.NonNegativeIntegers, initialize=0)

#%%

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
for nb_rolling_days, break_lengths in breaks.items():
    for day0 in range(total_days - nb_rolling_days):
        for person in model.index_people:
            model.breaks.add(
                sum(sum(model.x[day, person, job]
                        for job in model.index_jobs)
                    for day in range(day0, day0 + nb_rolling_days))
                <= nb_rolling_days - break_lengths[person]
            )

model.weekends_in_a_row = pyo.ConstraintList()
for person in model.index_people:
    for day0 in range(total_days - max_we_in_a_row * 7 - 1):
        if param_holidays.index[day0].dayofweek == 5:
            model.weekends_in_a_row.add(
                sum(sum(model.x[day0 + 7 * i, person, job] for job in model.index_jobs) for i in
                    range(max_we_in_a_row + 1))
                <= max_we_in_a_row
            )


model.occupy_everyday_positions = pyo.ConstraintList()
for day in model.index_days:
    for everyday_job_set in new_everyday_jobs:
        model.occupy_everyday_positions.add(
            sum(model.x[day, person, everyday_job] for person in model.index_people for everyday_job in
                everyday_job_set) == 1
        )


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

#%% ENDO
if do_endo:
    endo_job = job_names.index(endo_name)
    endo_person = np.where(param_jobs.iloc[endo_job] > 0)[0][0]
    model.endo = pyo.ConstraintList()
    for day0 in range(total_days):
        if param_holidays.index[day0].dayofweek == 0:
            if sum(param_holidays.iloc[day0:day0 + 5, endo_person]) >= 4:
                model.endo.add(
                    sum(model.x[day, endo_person, endo_job]
                        for day in range(day0, day0 + 5))
                    == 1
                )
#%% CONSTRAINTS - SHIFT GOALS

model.balance_flat_rate_days = pyo.ConstraintList()
my_print(['JOURS FORFAIT'])
for person in model.index_people:
    mini_flat_days, maxi_flat_days = get_flat_rate_days_goal(person)
    model.balance_flat_rate_days.add(
        mini_flat_days <= sum(sum(model.x[day, person, job] for job in flat_rate_jobs) for day in model.index_days)
    )
    model.balance_flat_rate_days.add(
        maxi_flat_days >= sum(sum(model.x[day, person, job] for job in flat_rate_jobs) for day in model.index_days)
    )
my_print([''])

model.balance_hospit_days = pyo.ConstraintList()
my_print(['JOURS CHENIL'])
for person in model.index_people:
    mini_hospit_days, maxi_hospit_days = get_hospit_days_goal(person)
    model.balance_hospit_days.add(
        mini_hospit_days
        <= sum(sum(model.x[day, person, job] for job in hospit_jobs) for day in model.index_days)
    )
    model.balance_hospit_days.add(
        sum(sum(model.x[day, person, job] for job in hospit_jobs) for day in model.index_days)
        <= maxi_hospit_days
    )
my_print([''])

model.balance_weekend_days = pyo.ConstraintList()
weekend_days_indexes = np.where(weekend_days_map)[0]
my_print(['JOURS WEEKEND'])
for person in model.index_people:
    min_weekend_days, max_weekend_days = get_weekend_days_goal(person)
    model.balance_weekend_days.add(
        sum(sum(model.x[day, person, job] for job in model.index_jobs) for day in weekend_days_indexes)
        <= max_weekend_days
    )
    model.balance_weekend_days.add(
        min_weekend_days
        <= sum(sum(model.x[day, person, job] for job in model.index_jobs) for day in weekend_days_indexes)
    )
my_print([''])

#%% ECHO

model.no_echo_on_weekends = pyo.ConstraintList()
for person in model.index_people:
    for day in weekend_days_indexes:
        model.no_echo_on_weekends.add(
            model.x[day, person, echo_job] <= model.x[day, 1, hospit_solo_job]
        )

model.balance_echo_pay = pyo.ConstraintList()
my_print(['SALAIRES ECHO'])
for person in echo_people:
    mini_echo_pay, maxi_echo_pay = get_echo_pay_goal(person)
    model.balance_echo_pay.add(
        mini_echo_pay <= sum(
            model.x[day, person, echo_job] * pay[echo_name] + model.x[day, person, hospit_echo_job] * pay[hospit_echo_name]
            for day in model.index_days)
    )

    model.balance_echo_pay.add(
        sum(model.x[day, person, echo_job] * pay[echo_name] + model.x[day, person, hospit_echo_job] * pay[hospit_echo_name]
            for day in model.index_days)
        <= maxi_echo_pay
    )
my_print([''])

#%%

model.no_shift_de_la_mort = pyo.ConstraintList()
for day in model.index_days:
    for person in model.index_people:
        model.no_shift_de_la_mort.add(
            sum(model.x[day, person, shift_mort_job] for day in
                range(day, min(day + shift_mort_length, total_days)))
            <= shift_mort_length - 1
        )

model.only_screw_full_weekends = pyo.ConstraintList()
for day in model.index_days:
    if get_dow(day) == 5 & (day < total_days - 1):
        for person in model.index_people:
            for job in model.index_jobs:
                model.only_screw_full_weekends.add(
                    model.x[day, person, job] == model.x[day + 1, person, job]
                )

model.interface_cycle_hospit = pyo.ConstraintList()
dow = param_holidays.index[0].dayofweek
if dow not in hospit_days.keys():
    nb_hospit_days_same_person = [start + length - (dow - 1) for start, length in hospit_days.items()
                                  if start <= dow <= start + length][0]
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
    for day0 in range(total_days - 2):
        model.no_single_day_shifts.add(
            sum(model.x[day0, person, job] for job in model.index_jobs) 
            + sum(model.x[day0 + 2, person, job] for job in model.index_jobs)
            >= sum(model.x[day0 + 1, person, job] for job in model.index_jobs)
        )
"""

#%% OBJECTIVE

def nb_person_shifts_weekend(m):
    return sum(sum(m.x[my_day, person, my_job] for my_day in m.index_days for my_job in m.index_jobs if
                   param_holidays.index[0].dayofweek >= 5) for person in m.index_people)


def obj_min_staff(m):
    return (weekend_cost - 1) * nb_person_shifts_weekend(m) + pyo.summation(
        m.x)

#%% HANDLING

def transform_solved_model_into_result(m):
    my_result = param_holidays.copy().replace(1, 0).replace(0, '')
    my_planning = pd.DataFrame(index=param_holidays.index, columns=job_names)
    for v in m.component_data_objects(pyo.Var):
        my_day, person, my_job = [int(u) for u in str(v)[2:-1].split(',')]
        if v.value > 0:
            my_result.iloc[my_day, person] = job_names[my_job]
            my_planning.iloc[my_day, my_job] = people_names[person]

    my_planning.fillna('', inplace=True)
    my_planning[hospit_solo_name] = my_planning[hospit_solo_name].str.cat(my_planning[hospit_echo_name])
    my_planning = my_planning.drop(labels=hospit_echo_name, axis='columns')
    my_result = my_result.where((my_result == echo_name).sum(axis=1) == 1, my_result.replace(hospit_solo_name, hospit_echo_name))
    return my_result, my_planning


def run_with_time_limit(seconds):
    solver = pyo.SolverFactory('glpk')
    solved = solver.solve(model, timelimit=seconds)
    my_result, my_planning = transform_solved_model_into_result(model)
    return my_result, my_planning, solved


def print_time_limit_info(seconds):
    my_print(['TIME LIMIT :', humanize.naturaldelta(timedelta(seconds=seconds))])
    now = datetime.now()
    my_print(['STARTED AT', now])
    return now


def print_solved_time_info(time):
    now = datetime.now()
    my_print(['SOLVED AT', now])
    my_print(['RUNTIME :', humanize.naturaldelta(now - time)])


model.obj1 = pyo.Objective(rule=obj_min_staff, sense=pyo.minimize)
start_time = print_time_limit_info(time_limit)
result, planning, opt = run_with_time_limit(time_limit)
my_print(['STATUS', str(opt.solver.termination_condition)])
print_solved_time_info(start_time)

shifts_info = pd.DataFrame(columns=people_names, index=job_names).fillna(0)
for name in people_names:
    nb_shifts = result[name].value_counts()
    for job in set(result[name].value_counts().index) & set(job_names):
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

for person_name in param_holidays.columns:
    result[person_name] = result[person_name].str.cat(param_holidays.replace(0, 'off').replace(1, '')[person_name])

colors = dict({"Esther": "#65B7FF",
               "Gaël": "#FFC665",
               "Olivier": "#8DD797",
               "Marine": "#FF6565",
               "Estelle": "#FFEF65",
               "off": "#A0DE98",
               hospit_solo_name: "#C8E4F4",
               hospit_echo_name: '#B9CBED',
               consult_name: "#FDD5E8",
               echo_name: "#C2C3EF",
               endo_name: "#F7E0CA",
               "": "#FFFFFF"})

"""
def highlight(df):
    weekend_color = 'background-color: {}'.format('#c4c4c4')
    df_attributes = pd.DataFrame('', index=df.index, columns=df.columns)
    if 'Days' in df.columns:

        for prefix in weekend_prefix:
            df_attributes['Days'] = df_attributes['Days'] + 
            pd.DataFrame(np.where(df['Days'].str.startswith(prefix), weekend_color, ''))
    for name, color in colors.items():
        attr = 'background-color: {}'.format(color)
        df_attributes = df_attributes + pd.DataFrame(np.where(df ==name, attr, ""), index= df.index, columns=df.columns)
    return df_attributes

def export_as_excel(df, writer, sheet_name, index=False, with_dates=True):
    #if 'Days' in df.columns:
        #df['Days'] = df['Days'].dt.strftime('%a %d %b')
    df = df.style.apply(highlight, axis=None)
    df.to_excel(writer, engine="openpyxl", index=index, sheet_name=sheet_name)

with pd.ExcelWriter(folder_path + suffix_time + '_UCVet.xlsx') as writer:
    export_as_excel(planning, writer, 'Clinic', with_dates=False)
    export_as_excel(result, writer, 'People', with_dates=False)
    export_as_excel(shifts_info, writer, 'Info', index=True, with_dates=False)
    
"""