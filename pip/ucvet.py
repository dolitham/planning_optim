# %% IMPORTS + DISPLAY SETTING

import itertools
from datetime import datetime
from datetime import timedelta
import humanize
import locale
import pyomo.environ as pyo
import pandas as pd
import numpy as np

from reader import get_calendar_input, get_current_stats, read_param_jobs
from param import *
from writer import my_print, save_planning

locale.setlocale(locale.LC_TIME, "fr_FR")
pd.set_option("max_columns", 10)
pd.options.display.float_format = '€{:,.2f}'.format

# %% FILES READING + DIRECT INFERENCES

param_jobs = read_param_jobs()
people_names = list(param_jobs.columns)
job_names = list(param_jobs.index)

holiday_requests, index_abort_consult, index_abort_echo, hospit_cycles, cycles_auto = \
    get_calendar_input(sheet_start_month, sheet_start_year)
my_print('DEBUT CYCLES HOSPIT & ECHO : ', ['custom', 'auto'][cycles_auto] + '\n')
my_print(list(pd.Series(holiday_requests.index)[hospit_cycles.keys()].dt.strftime('%a %d/%m || ')))

holiday_requests = holiday_requests.loc[:, people_names]
param_holidays = holiday_requests.replace(2, 1)
param_flexible = holiday_requests.replace(2, 0)
total_days = len(param_holidays.index)

weekend_days_map = param_holidays.index.map(lambda x: x.dayofweek) >= 5
sunday_days_map = param_holidays.index.map(lambda x: x.dayofweek) == 6
weekend_days_indexes = np.where(weekend_days_map)[0]
sunday_days_indexes = np.where(sunday_days_map)[0]
single_day_weekend_indexes = list(set([i + 1 for i in index_abort_consult if i > 0]) & set(weekend_days_indexes) |
                                  set([i - 1 for i in index_abort_consult if i > 0]) & set(weekend_days_indexes))

shift_mort_job = job_names.index(shift_mort_name)
lousy_sequence_jobs = [job_names.index(job_name) for job_name in lousy_sequence]

# %% JOBS NAMES & PARAMETERS

hospit_solo_job = job_names.index(hospit_solo_name)
consult_job = job_names.index(consult_name)
echo_job = job_names.index(echo_name)
hospit_echo_job = job_names.index(hospit_echo_name)
do_endo = endo_name in job_names
echo_people = np.where(param_jobs.iloc[echo_job] > 0)[0]

hospit_jobs = {hospit_echo_job, hospit_solo_job}
new_everyday_jobs = [{consult_job}, {hospit_solo_job, hospit_echo_job}]
flat_rate_jobs = list(itertools.chain.from_iterable(new_everyday_jobs))

# %%  PREVIOUS PLANNING PARAMETERS

stats_so_far = get_current_stats(people_names)

current_shift = list(stats_so_far['Current shift'])
weekend_days = list(stats_so_far['Jours WE'])
single_weekend_shifts = list(stats_so_far['Single day WE'])
hospit_shifts = list(stats_so_far['Hospit'])
current_weekend_streak = list(stats_so_far['Current WE streak'])

delta_weekend_days = [d - min(weekend_days) for d in weekend_days]
delta_single_weekend_days = [round(d - np.mean(single_weekend_shifts), 2) for d in single_weekend_shifts]
delta_hospit_days = [round(d - np.mean(hospit_shifts), 2) for d in hospit_shifts]


# %% LOGS FUNCTIONS

def goal_print(the_person, goal, mini, maxi):
    my_print([people_names[the_person], "{:.1f} >".format(goal),
              '[' + str(mini) + ', ' + str(maxi) + ']'], end_char=' || ')


def my_series_print(s):
    for the_person in s.index:
        my_print([the_person, str(s[the_person])], end_char=' || ')
    my_print('')


def get_dow(date):
    return param_holidays.index[date].dayofweek


# %% SETTING SHIFT TARGETS

def get_availability(holidays):
    avail = pd.Series(index=holidays.columns, dtype=int)
    for name in holidays.columns:
        for g in itertools.groupby(list(holidays[name])):
            duration = len(list(g[1]))
            avail[name] += duration * (g[0] == 1 or duration < threshold_prorata_holidays)
    return avail


availability = get_availability(param_holidays)
availability[availability > total_days - threshold_prorata_holidays] = total_days - nb_j_off_default
target_flat_rate = availability / sum(availability) * total_days * 2
target_hospit = availability / sum(availability) * total_days

availability_echo = param_holidays.sum()[echo_people]
availability_echo = availability_echo / 2 + sum(availability_echo) / (2 * len(availability_echo))
percentage_echo = (100 * availability_echo / sum(availability_echo)).round(0).astype(int)

min_j_echo = 1 * (availability_echo >= total_days - threshold_prorata_holidays)
total_echo_pay = total_days * pay[hospit_echo_name] + sum(min_j_echo) * (pay[echo_name] - pay[hospit_echo_name])
echo_goal = availability_echo / sum(availability_echo) * total_echo_pay

# %% LOG STATE
my_print(['BREAKS (nb de jours glissants : nb de jours off par personne)\n', breaks])
my_print(['MAX WE D\'AFFILÉE :', max_we_in_a_row])
my_print(['MAX JOURS CONSÉCUTIFS :', nb_max_consecutive_days])
my_print(['SINGLE DAY SHIFTS INTERDITS :', forbid_single_day_shifts])
my_print(['ENDOSCOPIE :', do_endo])
my_print('')
my_print(['FORFAIT ECHO', pay])
my_print(['CONSULT ANNULÉES'] + list(pd.Series(param_holidays.index[index_abort_consult]).dt.strftime('%a %d %B')))
my_print(['ECHO ANNULÉES'] + list(pd.Series(param_holidays.index[index_abort_echo]).dt.strftime('%a %d %B')))
my_print(['SHIFT INTERDIT :', shift_mort_name, 'PENDANT', str(shift_mort_length), 'JOURS'])
my_print(['SHIFT INTERDIT :', 'SUCCESSION'] + lousy_sequence)
my_print(['FOURCHETTE TOLERANCE\n', tolerance])
my_print('')
my_print(['DELTA WEEKEND DAYS :', delta_weekend_days])
my_print(['DELTA SINGLE WEEKEND DAYS :', delta_single_weekend_days])
my_print(['DELTA HOSPIT DAYS :', delta_hospit_days])
my_print(['AVAILABILITY - SEUIL JOURS CONTIGUS DE VACANCES POUR PRORATA :', threshold_prorata_holidays])
my_print(['AVAILABILITY - NB JOURS RETIRES PAR DÉFAUT SI PAS DE VACS :', nb_j_off_default])
my_print(['AVAILABILITY'], end_char=' : ')
my_series_print(availability)


# %% GOAL FUNCTIONS


def get_weekend_days_goal(the_person):
    goal = (sum(weekend_days_map) * 2 - len(index_abort_consult) + sum(delta_weekend_days)) / 5 - delta_weekend_days[
        the_person]
    mini = int(goal) - 1
    maxi = mini + tolerance['we_days']
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


def get_single_weekend_days_goal(the_person):
    goal = len(index_abort_consult) / 5 - delta_single_weekend_days[the_person]
    mini = int(goal)
    maxi = max(mini + tolerance['single_we'], 0)
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


def get_flat_rate_days_goal(the_person):
    goal = target_flat_rate.loc[people_names[the_person]]
    mini = int(goal) - 1
    maxi = mini + tolerance['flat_rate_days']
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


def get_hospit_days_goal(the_person):
    goal = target_hospit.loc[people_names[the_person]] - delta_hospit_days[the_person]
    mini = int(goal) - 1
    maxi = mini + tolerance['hospit_days']
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


def get_echo_pay_goal(the_person):
    goal = echo_goal.loc[people_names[the_person]]
    mini = round(goal - tolerance['echo_pay'] / 2, 1)
    maxi = mini + tolerance['echo_pay']
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


def get_echo_percentage_goal(the_person):
    goal = percentage_echo.loc[people_names[the_person]]
    mini = round(goal - tolerance['echo_percentage'] / 2, 1)
    maxi = mini + tolerance['echo_percentage']
    goal_print(the_person, goal, mini, maxi)
    return mini, maxi


# %% MODEL INIT

model = pyo.ConcreteModel()
model.index_days = pyo.Set(initialize=range(total_days))
model.index_people = pyo.Set(initialize=range(len(people_names)))
model.index_jobs = pyo.Set(initialize=range(len(job_names)))
model.indexes = pyo.Set(initialize=model.index_days * model.index_people * model.index_jobs)

model.x = pyo.Var(model.indexes, within=pyo.NonNegativeIntegers, initialize=0)

# %% STAFF FILL POSITIONS

model.one_person_per_job = pyo.ConstraintList()
for day in model.index_days:
    for job in model.index_jobs:
        model.one_person_per_job.add(sum(model.x[day, person, job] for person in model.index_people) <= 1)

model.qualifications = pyo.ConstraintList()
for person in model.index_people:
    for job in model.index_jobs:
        if param_jobs.iloc[job, person] == 0:
            for day in model.index_days:
                model.qualifications.add(model.x[day, person, job] == 0)

model.occupy_everyday_positions = pyo.ConstraintList()
for day in model.index_days:
    for everyday_job_set in new_everyday_jobs:
        if not (everyday_job_set == {1} and day in index_abort_consult):
            model.occupy_everyday_positions.add(
                sum(model.x[day, person, everyday_job] for person in model.index_people for everyday_job in
                    everyday_job_set) == 1
            )
        else:
            model.occupy_everyday_positions.add(
                sum(model.x[day, person, everyday_job] for person in model.index_people for everyday_job in
                    everyday_job_set) == 0
            )
# %% BREAKS & HOLIDAYS

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

model.holidays = pyo.ConstraintList()
for day in model.index_days:
    for person in model.index_people:
        model.holidays.add(
            sum(model.x[day, person, job] for job in model.index_jobs) <= param_holidays.iloc[day, person])

model.weekends_in_a_row = pyo.ConstraintList()
for person in model.index_people:
    for day0 in range(total_days - max_we_in_a_row * 7 - 1):
        if param_holidays.index[day0].dayofweek == 5:
            model.weekends_in_a_row.add(
                sum(sum(model.x[day0 + 7 * i, person, job] for job in model.index_jobs) for i in
                    range(max_we_in_a_row + 1))
                <= max_we_in_a_row
            )

# %% ENDO
if do_endo:
    endo_job = job_names.index(endo_name)
    endo_person = np.where(param_jobs.iloc[endo_job] > 0)[0][0]
    model.endo = pyo.ConstraintList()
    for day0 in range(total_days):
        if param_holidays.index[day0].dayofweek == 0 and day0 not in index_abort_echo:
            if sum(param_holidays.iloc[day0:day0 + 5, endo_person]) >= 4:
                model.endo.add(
                    sum(model.x[day, endo_person, endo_job]
                        for day in range(day0, day0 + 5))
                    == 1
                )

    for day in weekend_days_indexes:
        model.endo.add(
            model.x[day, endo_person, endo_job] == 0
        )
# %% ECHO
model.echo = pyo.ConstraintList()
no_echo_person = np.where(param_jobs.iloc[hospit_echo_job] == 0)[0][0]
for day in model.index_days:
    if day in hospit_cycles.keys() and day not in index_abort_echo:
        model.echo.add(
            sum(model.x[day, person, echo_job] for person in model.index_people) == 1
        )
    else:
        model.echo.add(
            sum(model.x[day, person, echo_job] for person in model.index_people) <=
            model.x[day, no_echo_person, hospit_solo_job]
        )

    if day in sunday_days_indexes:
        model.echo.add(
            sum(model.x[day, person, echo_job] for person in model.index_people) == 0
        )
    else:
        model.echo.add(
            sum(2 * model.x[day, person, hospit_echo_job] + model.x[day, person, hospit_solo_job] + model.x[
                day, person, echo_job]
                for person in model.index_people
                )
            == 2
        )

# %% CYCLE HOSPIT

model.cycle_hospit = pyo.ConstraintList()
for first_day, nb_consecutive_days in hospit_cycles.items():
    for person in model.index_people:
        for same_cycle_day in range(first_day + 1, min(first_day + nb_consecutive_days, total_days)):
            model.cycle_hospit.add(
                sum(model.x[first_day, person, hospit_job] for hospit_job in hospit_jobs)
                ==
                sum(model.x[same_cycle_day, person, hospit_job] for hospit_job in hospit_jobs)
            )

# %% CONSTRAINTS - SHIFT GOALS / FAIRNESS

model.balance_flat_rate_days = pyo.ConstraintList()
my_print(['OBJECTIF JOURS FORFAIT'])
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
my_print(['OBJECTIF JOURS CHENIL'])
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
my_print(['OBJECTIF JOURS WEEKEND'])
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

model.balance_single_weekend_days = pyo.ConstraintList()
my_print(['OBJECTIF JOURS WEEKEND SINGLE'])
for person in model.index_people:
    min_single_weekend_days, max_single_weekend_days = get_single_weekend_days_goal(person)
    model.balance_single_weekend_days.add(
        sum(sum(model.x[day - 1, person, job] for job in model.index_jobs) for day in single_day_weekend_indexes)
        <= max_single_weekend_days
    )
    model.balance_weekend_days.add(
        min_single_weekend_days
        <= sum(sum(model.x[day - 1, person, job] for job in model.index_jobs) for day in single_day_weekend_indexes)
    )
my_print([''])

model.balance_echo_pay = pyo.ConstraintList()
my_print(['OBJECTIFS SALAIRES ECHO'])
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

"""
model.balance_echo_percentage = pyo.ConstraintList()
my_print(['OBJECTIFS POURCENTAGES ECHO'])
for person in echo_people:
    mini_echo_percentage, maxi_echo_percentage = get_echo_percentage_goal(person)
    model.balance_echo_percentage.add(
        mini_echo_percentage / 100 * (
                pay[echo_name] * sum(model.x[day, person, echo_job] for day in model.index_days) +
                pay[hospit_echo_name] * sum(model.x[day, person, hospit_echo_job] for day in model.index_days)
        )
        <=
        pay[echo_name] * sum(
            model.x[day, the_person, echo_job] for day in model.index_days for the_person in model.index_people) +
        pay[hospit_echo_name] * sum(
            model.x[day, the_person, hospit_echo_job] for day in model.index_days for the_person in model.index_people)
    )

    model.balance_echo_percentage.add(
        pay[echo_name] * sum(
            model.x[day, the_person, echo_job] for day in model.index_days for the_person in model.index_people) +
        pay[hospit_echo_name] * sum(
            model.x[day, the_person, hospit_echo_job] for day in model.index_days for the_person in model.index_people)
        <=
        maxi_echo_percentage / 100 * (
                pay[echo_name] * sum(model.x[day, person, echo_job] for day in model.index_days) +
                pay[hospit_echo_name] * sum(model.x[day, person, hospit_echo_job] for day in model.index_days)
        )
    )

my_print([''])
"""
# %% NO ECHO ON WEEKENDS

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

model.no_lousy_sequence = pyo.ConstraintList()
for person in model.index_people:
    for day0 in range(total_days - 1):
        model.no_lousy_sequence.add(
            model.x[day0, person, lousy_sequence_jobs[0]] + model.x[day0 + 1, person, lousy_sequence_jobs[1]] <= 1
        )

model.only_screw_full_weekends = pyo.ConstraintList()
for day in model.index_days:
    if (get_dow(day) == 5) & (day < total_days - 1):
        for person in model.index_people:
            for job in model.index_jobs:
                model.only_screw_full_weekends.add(
                    model.x[day, person, job] >= model.x[day + 1, person, job]
                )

# model.interface_cycle_hospit = pyo.ConstraintList()
# dow = param_holidays.index[0].dayofweek
# if dow not in hospit_days.keys():
#     nb_hospit_days_same_person = [start + length - (dow - 1) for start, length in hospit_days.items()
#                                   if start <= dow <= start + length][0]
#     for day in range(nb_hospit_days_same_person):
#         model.interface_cycle_hospit.add(
#             sum(model.x[day, last_day_hospit_person, hospit_job] for hospit_job in hospit_jobs) == 1
#         )

model.interface_max_shift = pyo.ConstraintList()
for person in model.index_people:
    nb_consecutive_days_left = nb_max_consecutive_days[person] - current_shift[person]
    model.interface_max_shift.add(
        sum(sum(model.x[day, person, job]
                for job in model.index_jobs)
            for day in range(0, nb_consecutive_days_left + 1))
        <= nb_consecutive_days_left
    )

if forbid_single_day_shifts:
    model.no_single_day_shifts = pyo.ConstraintList()
    for person in model.index_people:
        for day0 in range(total_days - 2):
            model.no_single_day_shifts.add(
                sum(model.x[day0, person, job] for job in model.index_jobs)
                + sum(model.x[day0 + 2, person, job] for job in model.index_jobs)
                >= sum(model.x[day0 + 1, person, job] for job in model.index_jobs)
            )

        if current_shift[person] == 0:
            model.no_single_day_shifts.add(
                sum(model.x[1, person, job] for job in model.index_jobs)
                >= sum(model.x[0, person, job] for job in model.index_jobs)
            )


# %% OBJECTIVE

def obj_respect_flexible_min_we(m):
    return cost_we_shifts(m) + pyo.summation(m.x) + cost_flexible_ignored(m) + cost_marine_hospit_no_consult(m)


def cost_we_shifts(m):
    return (weekend_cost - 1) * sum(sum(m.x[my_day, the_person, my_job]
                                        for my_day in m.index_days
                                        for my_job in m.index_jobs
                                        if param_holidays.index[0].dayofweek >= 5) for the_person in m.index_people)


def cost_marine_hospit_no_consult(m):
    return malus_no_consult * sum(m.x[my_day, endo_person, my_job] for my_day in single_day_weekend_indexes
                                  for my_job in hospit_jobs)


def cost_flexible_ignored(m):
    return flexible_request_cost * sum(
        sum(m.x[my_day, my_person, my_job] for my_job in m.index_jobs) * param_flexible.iloc[my_day, my_person]
        for my_person in m.index_people for my_day in m.index_days
    )


model.obj1 = pyo.Objective(rule=obj_respect_flexible_min_we, sense=pyo.minimize)


# %% HANDLING

def transform_solved_model_into_result(m):
    my_result = param_holidays.copy().replace(1, 0).replace(0, '')
    my_planning = pd.DataFrame(index=param_holidays.index, columns=job_names)
    for v in m.component_data_objects(pyo.Var):
        my_day, my_person, my_job = [int(u) for u in str(v)[2:-1].split(',')]
        if v.value > 0:
            my_result.iloc[my_day, my_person] = job_names[my_job]
            my_planning.iloc[my_day, my_job] = people_names[my_person]

    my_planning.fillna('', inplace=True)
    my_planning[hospit_solo_name] = my_planning[hospit_solo_name].str.cat(my_planning[hospit_echo_name])
    my_planning = my_planning.drop(labels=hospit_echo_name, axis='columns')
    my_result = my_result.where((my_result == echo_name).sum(axis=1) == 1,
                                my_result.replace(hospit_solo_name, hospit_echo_name))
    my_result.loc[my_result.loc[:, 'Marine'] == 'Hospit+E', 'Marine'] = 'Hospit'

    for person_name in param_holidays.columns:
        my_result[person_name] = my_result[person_name].str.cat(holiday_requests.replace(0, 'off')
                                                                .replace(1, '!').replace(2, '')[person_name])
    return my_result, my_planning


def run_with_time_limit(seconds, send_email=True):
    start_time = print_time_limit_info(seconds)
    run_id = pd.Series(param_holidays.index).dt.strftime("%B").mode()[0].title() + start_time.strftime('_%d_%H%M%S')
    my_print(['RUN ID', run_id], end_char='\n')
    solver = pyo.SolverFactory('glpk')
    solved = solver.solve(model, timelimit=seconds)
    my_result, my_planning = transform_solved_model_into_result(model)
    my_print(['STATUS', str(solved.solver.termination_condition)])
    print_solved_time_info(start_time)
    if send_email:
        save_planning(my_planning, my_result, stats_so_far, run_id)
    return my_result, my_planning, solved


def print_time_limit_info(seconds):
    my_print(['\nTIME LIMIT :', humanize.naturaldelta(timedelta(seconds=seconds))])
    now = datetime.now()
    my_print(['STARTED AT', now])
    return now


def print_solved_time_info(time):
    now = datetime.now()
    my_print(['SOLVED AT', now])
    my_print(['RUNTIME :', humanize.naturaldelta(now - time)])


# %% RUN CODE

time_limit_seconds = int(60 * time_limit_minutes)
result, planning, opt = run_with_time_limit(time_limit_seconds, send_email=True)
