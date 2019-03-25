#!/usr/bin/env python3

__authors__ = 'Kirk Scanlan, kirk.scanlan@utexas.edu'
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 20 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'Tool to estimate solar longitude from date stamps '
                  'in SHARAD USGEOM RDR data products'},
    '1.1':
        {'date': 'March 25 2019',
         'author': 'Gregory Ng, UTIG',
         'info': 'Revised structure to separate date/time parsing and astronomical '
                 'estimations. See known issues. '},
        
}

# -*- coding: utf-8 -*-


import datetime as dt
import numpy as np
# GNG TODO: support spline interpolation instead of linear?
# GNG TODO: evaluate accuracy and definitions/documentation of time standard

MY1_start_date = dt.datetime(1955, 4, 11, 00, 00, 00)
seconds_per_sol = 88775.245
# Seconds in one martian year
seconds_in_MY = 668.6 * seconds_per_sol
# Start oof J2000 epoch
dt_epoch_J2000 = dt.datetime(2000, 1, 1, 12, 0)


def UTC_datetime_to_J2000(datetime1):
    """ 
    Convert a datetime object representing a UTC time to J2000 seconds.

    Known issue: Note that UTC leap seconds are ignored, 
    so datetimes are converted to J2000 seconds as if leap seconds
    had never happened.

    Inputs
    -----------------
    datetime1: A python datetime object representing a UTC time

    Outputs
    -----------------
    jsec: J2000 seconds
    """
    # TODO: unittest
    return (datetime1 - dt_epoch_J2000).total_seconds()


def ISO8601_to_J2000(timestamp):
    """ 
    Convert a timestamp in ISO 8601 format (such as 
    "2019-01-01T06:00:01.123", or "2019-01-01T06:00:01"),
    which represents a UTC time, into J2000 seconds.

    If timestamp isn't parseable, raises ValueError

    Known issue: Note that UTC leap seconds are ignored, 
    so dates are converted to J2000 seconds as if leap seconds
    had never happened.

    Inputs
    -----------------
    timestamp: A string with ISO 8601 timestamp representing a UTC time

    Outputs
    -----------------
    jsec: J2000 seconds
    """
    # TODO: unittest
    e1 = None
    for timefmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            datetime1 = dt.datetime.strptime(timestamp,timefmt)
            return (datetime1 - dt_epoch_J2000).total_seconds()
        except ValueError as e:
            e1 = e
            continue
    raise e1 # raise an error if we haven't successfully parsed


# Start of Mars Year 1 in J2000 seconds
MY1_J2000 = UTC_datetime_to_J2000(MY1_start_date)

def Ls(year, month, day, hour, minute, second):
    """
    Tool for converting Earth date into Martian solar longitude and Martian
    year. Based on 1 Martian year lasting 668.6 sols and 1 sol lasting
    88775.245 seconds. Martian year 1 began on April 11 1955 (solar longitude
    of 0)

    Uses piecewise linear interpolation to calculate position within the month

    Inputs
    -----------------
      year: Earth year of query date
     month: Earth month of query date
       day: Earth day of query date
      hour: Earth hour of query date [24 hour clock]
    minute: Earth minute of query date
    second: Earth second of query date

    Outputs
    -----------------
      MY: Martian year
      Ls: Solar longitude
    """
    global MY1_start_date, seconds_in_MY, seconds_per_sol
    
    # define the number of seconds between start of Martian Year 1 and the
    # query date
    query_date = dt.datetime(year, month, day, hour, minute, second)
    seconds_between = (query_date - MY1_start_date).total_seconds()
    
    # define the number of Martian year of the query date
    MY = np.ceil(seconds_between / seconds_in_MY).astype(int)
    
    # define the number of sols that have elapsed within the Martian year of
    # the query date
    elapsed_seconds = seconds_between - ((MY - 1) * seconds_in_MY)
    elapsed_sols = elapsed_seconds / seconds_per_sol
    
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
    else:
        # It isn't in range!
        assert(False)
    month_length_in_sols = max_sol - min_sol
    proportion_through_month = (elapsed_sols - min_sol) / month_length_in_sols
    Ls = (30 * proportion_through_month) + min_Ls

    return MY, Ls


def Ls_J2000(sec_J2000):
    """
    Tool for converting Earth date into Martian solar longitude and Martian
    year. Based on 1 Martian year lasting 668.6 sols and 1 sol lasting
    88775.245 seconds. Martian year 1 began on April 11 1955 (solar longitude
    of 0)

    Uses piecewise linear interpolation to calculate position within the month

    See also Ls which takes year, month, day, etc

    Inputs
    -----------------
    sec_J2000: Current time, in seconds relative to J2000

    Outputs
    -----------------
      MY: Martian year
      Ls: Solar longitude
    """
    global MY1_J2000, seconds_in_MY
    
    seconds_between = sec_J2000 - MY1_J2000
    
    # define the number of Martian year of the query date
    # GNG TODO: Should this be defined as ceiling, or floor + 1?  the 
    # difference is in what year is returned on midnight January 1.
    # MY = np.floor(seconds_between / seconds_in_MY).astype(int) + 1
    MY = np.ceil(seconds_between / seconds_in_MY).astype(int)
    
    # define the number of sols that have elapsed within the Martian year of
    # the query date
    elapsed_seconds = seconds_between - ((MY - 1) * seconds_in_MY)
    elapsed_sols = elapsed_seconds / seconds_per_sol
    
    Ls = np.interp(elapsed_sols, g_solval, g_Lsval)

    return MY, Ls




# convert the elapsed sols to a solar longitude assuming a linear
# relationship between sols and solar longitude within each Martian month
g_solval = np.array([0.0,61.2,126.6,193.3,257.8,317.5,371.9,421.6,468.5,514.6,562.0,612.9,668.6])
g_Lsval  = np.array([0.0,30.0, 60.0, 90.0,120.0,150.0,180.0,210.0,240.0,270.0,300.0,330.0,360.0])
# TODO: versions of function that allow you to calculate multiple longitudes as a vector
def Ls_new1(year, month, day, hour, minute, second):
    """
    Tool for converting Earth date into Martian solar longitude and Martian
    year. Based on 1 Martian year lasting 668.6 sols and 1 sol lasting
    88775.245 seconds. Martian year 1 began on April 11 1955 (solar longitude
    of 0)

    Uses piecewise linear interpolation to calculate position within the month

    Inputs
    -----------------
      year: Earth year of query date
     month: Earth month of query date
       day: Earth day of query date
      hour: Earth hour of query date [24 hour clock]
    minute: Earth minute of query date
    second: Earth second of query date

    Outputs
    -----------------
      MY: Martian year
      Ls: Solar longitude
    """
    global MY1_start_date, seconds_in_MY
    
    # define the number of seconds between start of Martian Year 1 and the
    # query date
    query_date = dt.datetime(year, month, day, hour, minute, second) 
    query_J2000 = UTC_datetime_to_J2000(query_date)

    return Ls_J2000(query_J2000)


def test_Ls_new1():
    # Test similarity of original algorithm and numpy-based version

    errors=[]
    for yyyy in range(100):
        yweeks = yyyy*52
        for week in range(52):
            for hour in range(0,7*24,4):
                deltat = dt.timedelta(weeks=week+yweeks, hours=hour, seconds=2.0)
                date1 = MY1_start_date + deltat
                MY1,lon1 = Ls(     date1.year, date1.month, date1.day, date1.hour, date1.minute, date1.second)
                MY2,lon2 = Ls_new1(date1.year, date1.month, date1.day, date1.hour, date1.minute, date1.second)

                deltalon=abs(lon1 - lon2) 
                errors.append(deltalon)
                try:
                    assert(deltalon < 1e-9)
                except AssertionError as e:
                    print("date",date1)
                    print("lon1",lon1)
                    print("lon2",lon2)
                    raise
        print(date1)
    errors = np.array(errors)
    print("Average error: {:0.3g} deg".format(np.mean(errors)))
    print("Std dev error: {:0.3g} deg".format(np.std(errors)))
    print("Maximum error: {:0.3g} deg".format(np.max(errors)))


def test_ISO8601():

    testcases = (
        ("2019-01-01T06:00:01.123" ,   dt.datetime(2019, 1, 1, 6, 0, 1, 123000) ),
        ("2019-01-01T06:00:01.000" ,   dt.datetime(2019, 1, 1, 6, 0, 1, 0     ) ),
        ("2019-01-01T06:00:01.1234",   dt.datetime(2019, 1, 1, 6, 0, 1,123400) ),
        ("2019-01-01T06:00:01.123456", dt.datetime(2019, 1, 1, 6, 0, 1,123456) ),
        ("2019-01-01T06:00:01" ,   dt.datetime(2019, 1, 1, 6, 0, 1, 0) ),
    )

    for datestr1, datetime1 in testcases:
        j2000a = ISO8601_to_J2000(datestr1)
        j2000b = UTC_datetime_to_J2000(datetime1)
        
        assert( abs(j2000a-j2000b) < 1e-9 )





def main():
    test_ISO8601()
    test_Ls_new1()

if __name__ == "__main__":
    # execute only if run as a script
    main()



