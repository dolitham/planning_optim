# %% IMPORTS + DISPLAY SETTING

import itertools
from datetime import datetime
from datetime import timedelta
import humanize
import locale
import pyomo.environ as pyo
import pandas as pd
import numpy as np

from google_sheet_reader import get_holiday_requests
from param import jobs_path, hospit_days, hospit_solo_name, consult_name, echo_name, hospit_echo_name, endo_name, \
    nb_max_consecutive_days, pay, max_we_in_a_row, breaks
from excel_writer import write_excel

locale.setlocale(locale.LC_TIME, "fr_FR")
pd.set_option("max_columns", 10)
pd.options.display.float_format = '€{:,.2f}'.format

# %%  PARAM FILES

month = 1
year = 2022

#%% FILES READING + DIRECT INFERENCES
# param_holidays = pd.read_csv(holidays_path, parse_dates=[0], delimiter=";", index_col=0).isnull() * 1
param_jobs = pd.read_csv(jobs_path, delimiter=";", index_col=0)
param_requests = get_holiday_requests(month, year).loc[:, param_jobs.columns]
param_holidays = param_requests.replace(2, 1)
param_flexible = param_requests.replace(2, 0)

people_names = list(param_jobs.columns)
job_names = list(param_jobs.index)
total_days = len(param_holidays.index)
people = len(param_holidays.columns)
weekend_days_map = param_holidays.index.map(lambda x: x.dayofweek) >= 5
jobs = len(job_names)

# %% JOBS NAMES & PARAMETERS

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

# %% OPTIM / MODEL PARAMETERS

time_limit = 60
weekend_cost = 10
flexible_request_cost = 100

# %%  PREVIOUS PLANNING PARAMETERS

last_day_hospit_person = 3
current_shift = [0, 0, 3, 0, 4]
delta_weekend_days = [4, 0, 2, 4, 6]
# delta_flat_rate_days = [0, 0, 0, 0, 0]
# delta_hospit_days = [0, 0, 0, 0, 0]
current_weekend_streak = [0, 0, 1, 1, 0]

# %%  CLINIC PARAMETERS

shift_mort_length = 4
shift_mort_job = consult_job
nb_j_off_default = 5
threshold_prorata_holidays = 20

# %% LOGS

global output
output = ''


def my_print(text):
    global output
    line = ' '.join(str(u) for u in text) + '\n'
    output += line
    print(line, end='')


# %% TIME MANAGEMENT

def get_dow(date):
    return param_holidays.index[date].dayofweek


# %% SETTING SHIFT TARGETS

availability = param_holidays.sum()
availability[availability >= total_days - threshold_prorata_holidays] = total_days - nb_j_off_default
target_flat_rate = availability / sum(availability) * total_days * 2
target_hospit = availability / sum(availability) * total_days

availability_echo = param_holidays.sum()[echo_people]
availability_echo = availability_echo/2 + sum(availability_echo)/(2*len(availability_echo))

min_j_echo = 2 * (availability_echo >= total_days - threshold_prorata_holidays)
total_echo_pay = total_days * pay[hospit_echo_name] + sum(min_j_echo) * (pay[echo_name] - pay[hospit_echo_name])
echo_goal = availability_echo / sum(availability_echo) * total_echo_pay


def get_weekend_days_goal(person):
    goal = (sum(weekend_days_map) * 2 + sum(delta_weekend_days)) / 5 - delta_weekend_days[person]
    mini = int(2 * (goal // 2)) - 1
    maxi = mini + 2 + 2
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_flat_rate_days_goal(person):
    goal = target_flat_rate.loc[people_names[person]]
    mini = int(goal) - 1
    maxi = mini + 1 + 2  # TODO REMOVE THIS
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


def get_hospit_days_goal(person):
    goal = target_hospit.loc[people_names[person]]
    mini = int(goal) - 2
    maxi = mini + 1 + 3
    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


print('echo relax')


def get_echo_pay_goal(person):
    goal = echo_goal.loc[people_names[person]]
    mini = int(goal) - 1
    maxi = mini + 1 + 2  # TODO remove this

    print(people_names[person], "{:.1f} >".format(goal), '[' + str(mini) + ', ' + str(maxi) + ']', end=' | ')
    return mini, maxi


# %% MODEL INIT

model = pyo.ConcreteModel()
model.index_days = pyo.Set(initialize=range(total_days))
model.index_people = pyo.Set(initialize=range(people))
model.index_jobs = pyo.Set(initialize=range(jobs))
model.indexes = pyo.Set(initialize=model.index_days * model.index_people * model.index_jobs)

model.x = pyo.Var(model.indexes, within=pyo.NonNegativeIntegers, initialize=0)

# %%

model.force_shift = pyo.ConstraintList()

#  model.force_shift.add(model.x[15, 3, hospit_solo_job] == 1)

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
    for day0 in range(total_days - nb_rolling_days + 1):
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
        if not(everyday_job_set == {1} and param_holidays.index[day].day in [9, 16]):
            model.occupy_everyday_positions.add(
                sum(model.x[day, person, everyday_job] for person in model.index_people for everyday_job in
                    everyday_job_set) == 1
            )

abort_echo_days = [17, 20]

model.echo = pyo.ConstraintList()
for day in model.index_days:
    if param_holidays.index[day].dayofweek in echo_weekdays:
        if param_holidays.index[day].day not in abort_echo_days:
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
model.noel = pyo.ConstraintList()
for day in model.index_days:
    if param_holidays.index[day].day - 1 in abort_echo_days:
        print(param_holidays.index[day])
        for person in model.index_people:
            model.noel.add(
                sum(model.x[day, person, job] for job in model.index_jobs) == sum(
                    model.x[day + 1, person, job] for job in model.index_jobs)
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

# %% ENDO
if do_endo:
    endo_job = job_names.index(endo_name)
    endo_person = np.where(param_jobs.iloc[endo_job] > 0)[0][0]
    model.endo = pyo.ConstraintList()
    for day0 in range(total_days):
        if param_holidays.index[day0].dayofweek == 0 and param_holidays.index[day0].day not in abort_echo_days:
            if sum(param_holidays.iloc[day0:day0 + 5, endo_person]) >= 4:
                model.endo.add(
                    sum(model.x[day, endo_person, endo_job]
                        for day in range(day0, day0 + 5))
                    == 1
                )
# %% CONSTRAINTS - SHIFT GOALS / FAIRNESS

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

model.balance_echo_pay = pyo.ConstraintList()
my_print(['SALAIRES ECHO'])
for person in echo_people:
    mini_echo_pay, maxi_echo_pay = get_echo_pay_goal(person)
    model.balance_echo_pay.add(
        mini_echo_pay <= sum(
            model.x[day, person, echo_job] * pay[echo_name] + model.x[day, person, hospit_echo_job] * pay[
                hospit_echo_name]
            for day in model.index_days)
    )

    model.balance_echo_pay.add(
        sum(model.x[day, person, echo_job] * pay[echo_name] + model.x[day, person, hospit_echo_job] * pay[
            hospit_echo_name]
            for day in model.index_days)
        <= maxi_echo_pay
    )
my_print([''])

# %% ECHO

model.no_echo_on_weekends = pyo.ConstraintList()
for person in model.index_people:
    for day in weekend_days_indexes:
        model.no_echo_on_weekends.add(
            model.x[day, person, echo_job] <=
            sum(model.x[day, person, hospit_solo_job] for person in model.index_people)
        )

# %% NO HEAVY SCHEDULES + INTERFACE

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
    if (get_dow(day) == 5) & (day < total_days - 1):
        for person in model.index_people:
            for job in model.index_jobs:
                model.only_screw_full_weekends.add(
                    model.x[day, person, job] >= model.x[day + 1, person, job]
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


# %% OBJECTIVE

def nb_person_shifts_weekend(m):
    return sum(sum(m.x[my_day, person, my_job] for my_day in m.index_days for my_job in m.index_jobs if
                   param_holidays.index[0].dayofweek >= 5) for person in m.index_people)


def obj_respect_flexible_min_we(m):
    return (weekend_cost - 1) * nb_person_shifts_weekend(m) + pyo.summation(m.x) + nb_flexible_ignored(m)


def nb_flexible_ignored(m):
    return flexible_request_cost * sum(
        sum(m.x[my_day, my_person, my_job] for my_job in m.index_jobs) * param_flexible.iloc[my_day, person]
        for my_person in m.index_people for my_day in m.index_days
    )


# %% HANDLING

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
    my_result = my_result.where((my_result == echo_name).sum(axis=1) == 1,
                                my_result.replace(hospit_solo_name, hospit_echo_name))
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


#%% ACTUAL CODE

model.obj1 = pyo.Objective(rule=obj_respect_flexible_min_we, sense=pyo.minimize)
start_time = print_time_limit_info(time_limit)
result, planning, opt = run_with_time_limit(time_limit)
my_print(['STATUS', str(opt.solver.termination_condition)])
print_solved_time_info(start_time)

for person_name in param_holidays.columns:
    result[person_name] = result[person_name].str.cat(param_requests.replace(0, 'off')
                                                      .replace(1, '!').replace(2, '')[person_name])

if len(set(planning['Hospit'])):
    write_excel(planning, result)
