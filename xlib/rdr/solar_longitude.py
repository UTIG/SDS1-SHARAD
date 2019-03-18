__authors__ = 'Kirk Scanlan, kirk.scanlan@utexas.edu'
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Tool to estimate solar longitude from date stamps
                  in SHARAD USGEOM RDR data products'}}

# -*- coding: utf-8 -*-

def Ls(yr, mnth, dy, hr, mnt, scnd):
    """
    Tool for converting Earth date into Martian solar longitude and Martian
    year. Based on 1 Martian year lasting 668.6 sols and 1 sol lasting
    88775.245 seconds. Martian year 1 began on April 11 1955 (solar longitude
    of 0)

    Inputs
    -----------------
        yr: Earth year of query date
      mnth: Earth month of query date
        dy: Earth day of query date
        hr: Earth hour of query date [24 hour clock]
       mnt: Earth minute of query date
      scnd: Earth second of query date

    Outputs
    -----------------
      MY: Martian year
      Ls: Solar longitude
    """

    import datetime as dt
    import numpy as np
    
    # define the number of seconds between start of Martian Year 1 and the
    # query date
    MY1_start_date = dt.datetime(1955, 4, 11, 00, 00, 00)
    query_date = dt.datetime(yr, mnth, dy, hr, mnt, scnd)
    seconds_between = (query_date - MY1_start_date).total_seconds()
    
    # define the number of Martian year of the query date
    seconds_in_MY = 668.6 * 88775.245
    MY = np.ceil(seconds_between / seconds_in_MY).astype(int)
    
    # define the number of sols that have elapsed within the Martian year of
    # the query date
    elapsed_seconds = seconds_between - ((MY - 1) * seconds_in_MY)
    elapsed_sols = elapsed_seconds / 88775.245
    
    # convert the elapsed sols to a solar longitude assuming a linear
    # relationship between sols and solar longitude within each Martian month
    if elapsed_sols >= 0.0 and elapsed_sols < 61.2:
        min_Ls = 0; min_sol = 0.0; max_sol = 61.2
    elif elapsed_sols >= 61.2 and elapsed_sols < 126.6:
        min_Ls = 30; min_sol = 61.2; max_sol = 126.6
    elif elapsed_sols >= 126.6 and elapsed_sols < 193.3:
        min_Ls = 60; min_sol = 126.6; max_sol = 193.3
    elif elapsed_sols >= 193.3 and elapsed_sols < 257.8:
        min_Ls = 90; min_sol = 193.3; max_sol = 257.8
    elif elapsed_sols >= 257.8 and elapsed_sols < 317.5:
        min_Ls = 120; min_sol = 257.8; max_sol = 317.5
    elif elapsed_sols >= 317.5 and elapsed_sols < 371.9:
        min_Ls = 150; min_sol = 317.5; max_sol = 371.9
    elif elapsed_sols >= 371.9 and elapsed_sols < 421.6:
        min_Ls = 180; min_sol = 371.9; max_sol = 421.6
    elif elapsed_sols >= 421.6 and elapsed_sols < 468.5:
        min_Ls = 210; min_sol = 421.6; max_sol = 468.5
    elif elapsed_sols >= 468.5 and elapsed_sols < 514.6:
        min_Ls = 240; min_sol = 468.5; max_sol = 514.6
    elif elapsed_sols >= 514.6 and elapsed_sols < 562.0:
        min_Ls = 270; min_sol = 514.6; max_sol = 562.0
    elif elapsed_sols >= 562.0 and elapsed_sols < 612.9:
        min_Ls = 300; min_sol = 562.0; max_sol = 612.9
    elif elapsed_sols >= 612.9 and elapsed_sols < 668.6:
        min_Ls = 330; min_sol = 612.9; max_sol = 668.6
    month_length_in_sols = max_sol - min_sol
    proportion_through_month = (elapsed_sols - min_sol) / month_length_in_sols
    Ls = (30 * proportion_through_month) + min_Ls

    return MY, Ls
