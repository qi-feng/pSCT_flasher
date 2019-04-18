import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("ticks")
    #sns.set_style({"axes.axisbelow": False})
    sns.set_style({"xtick.direction": "in" ,"ytick.direction": "in"})
except:
    print("sns problem")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import glob
import os
import sys
import select
import time

from subprocess import Popen, PIPE
import logging

import visa


def setup_logger(logger_name, log_file, level=logging.INFO, show_log=True):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)
    if show_log:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log_setup.addHandler(ch)


def FWHM(X, Y, negY=True, baseline=0, verbose=True):
    assert len(X) == len(Y)
    factor = 1
    if negY:
        factor = -1
    baseline = factor * baseline
    Y = factor * Y - baseline
    # plt.plot(X, Y)
    half_max = max(Y) / 2.
    indices = [x for x in range(len(Y)) if Y[x] > half_max]
    if not indices:
        return 0,factor * half_max + factor * baseline, -1,-1
    right_idx = max(indices)
    left_idx = min(indices)
    # print(min(indices), max(indices))
    # print(X[right_idx] - X[left_idx])
    # print(half_max)
    if verbose:
        print("FWHM is {}".format(factor * half_max))
    return X[right_idx] - X[
        left_idx], factor * half_max + factor * baseline, left_idx, right_idx  # return the difference (full width)


def flare_func_w_const(ts, amp, tpeak, trise, tdecay, const):
    #negative peak
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = const-amp*np.exp((ts[rise_slice]-tpeak)/trise)
    fs[decay_slice] = const-amp*np.exp(-(ts[decay_slice]-tpeak)/tdecay)
    return fs


def chisq(y_vals, y_expected, y_errs, num_params=1):
    #returns chi2, dof, red_chi2
    #  for reduced chisq test, under the assumption of Poisson counting
    #  we have lnL = const - (1/2.)*chi2
    if y_vals.shape[0] != y_expected.shape[0]:
        print("Inconsistent input sizes")
        return
    #z = (y_vals[i] - y_expected[i]) / y_errs[i]
    z = (y_vals - y_expected) / y_errs
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (y_vals.shape[0] - num_params)
    return chi2, (y_vals.shape[0] - num_params), chi2dof


def plot_tek_trace_pandas(tekname='USB0::0x0699::0x039F::C010383::INSTR', scope_resource=None,
                   dfname="pocket_pulser.csv", plot_name="pocket_pulser.png", width_file="Flasher_fwhm.csv",
                   log_file="Flasher_characterization.log",
                   data_start=50000, data_stop=65000,
                   tstart=0, tstop=1e-7, initial_params=[0.1, 8.e-8, 1.e-9, 10.e-9, 0.],
                   pot_setting=1.55e3, onLEDs=[1], verbose=True, show=False,
                   num_read=10):
    ##initial_params: amp, tpeak, trise, tdecay, const

    setup_logger("flasher_log", log_file,
                 level=logging.INFO, show_log=verbose)
    logger = logging.getLogger('flasher_log')

    # establish comm to scope DPO2014B
    if scope_resource is None:
        rm = visa.ResourceManager()
        # print(rm.list_resources())
        tek = rm.open_resource(tekname)
        print(tek.query('*IDN?'))
        tek.timeout = 100000
        tek.write('DATA:ENC ASCI')
    else:
        tek = scope_resource

    tek.write("DATa:STARt {}".format(data_start))
    tek.write("DATa:STOP {}".format(data_stop))
    print("Set data acquisition start and stop to")
    print(tek.query("DATa:STARt?"), tek.query("DATa:STOP?"))

    for i_trace in range(num_read):
        # get the trace
        vs = np.array(tek.query_ascii_values('CURV?'))

        xin = float(tek.query('wfmpre:XIN?'))
        xze = float(tek.query('wfmpre:XZE?'))
        x_start = xze + data_start * xin
        ts = np.arange(x_start, x_start + vs.shape[0] * xin, xin)

        ymult = float(tek.query("wfmpre:ymult?"))
        yoff = float(tek.query("wfmpre:yoff?"))
        yzero = float(tek.query("wfmpre:yzero?"))

        vs = (vs - yoff) * ymult + yzero

        ind = np.where((ts > tstart) & (ts < tstop))
        if verbose:
            print("xze {}, xin {}, x_start {}".format(xze, xin, x_start))
            print("ymult {}, yoff {}, yzero {}".format(ymult, yoff, yzero))
            print("Clippinng the trace between {} and {}".format(tstart, tstop))
        logger.info("xze {}, xin {}, x_start {}".format(xze, xin, x_start))
        logger.info("ymult {}, yoff {}, yzero {}".format(ymult, yoff, yzero))
        logger.info("Clippinng the trace between {} and {}".format(tstart, tstop))
        # logger.info()
        ts = ts[ind]
        vs = vs[ind]
        # plt.plot(ts[ind], vs[ind])


        # Fit the trace w exponential
        # params: amp, tpeak, trise, tdecay, const
        parLCL3, covLCL3 = curve_fit(flare_func_w_const, ts,
                                     vs, p0=initial_params,
                                     )  # ,
        # method='trf', ftol=1e-10, loss='soft_l1') #approx abs

        par_names8 = ['amplitude', 'tpeak', 'trise', 'tdecay', 'const']
        if verbose:
            for i in range(len(par_names8)):
                print("Flare exponential fit param %d (%s) = %.4g +/- %.4g" % (i, par_names8[i],
                                                                               parLCL3[i],
                                                                               np.sqrt(abs(covLCL3[i, i]))))

                # def chisq(y_vals, y_expected, y_errs, num_params=1):
                #  for reduced chisq test, under the assumption of Poisson counting
                #  we have lnL = const - (1/2.)*chi2
        chi2LC3L1, dof3L1, redchi2LC3L1 = chisq(vs,
                                                flare_func_w_const(ts, *parLCL3),
                                                1,
                                                5)
        lnL_LC3L1 = -0.5 * chi2LC3L1
        if verbose:
            print("fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC3L1, dof3L1, redchi2LC3L1))
            print("Log likelihood lnL={0}".format(lnL_LC3L1))

        # calc FWHM
        # wid, half_max, lind, rind = FWHM(ts[ind], vs[ind])
        wid, half_max, lind, rind = FWHM(ts, vs, verbose=verbose)

        # print(wid, half_max, lind, rind, ts[lind], ts[rind])
        # quality cut
        if abs(parLCL3[0]-parLCL3[4]) < 3*np.std(vs) or wid>100:
            print(abs(parLCL3[0]-parLCL3[4]),  3*np.std(vs))
            print("Failed quality cuts")
            logger.error("Failed quality cuts")
            continue

        # integrate charge:
        start_window = parLCL3[1] - 1.5 * np.e * parLCL3[2]
        stop_window = parLCL3[1] + 3 * np.e * parLCL3[3]

        charge_all = np.trapz(vs, ts)
        #print("integral all = {}".format(np.trapz(vs, ts)))

        int_slice = np.where((ts >= start_window) & (ts <= stop_window))
        charge_window=np.trapz(vs[int_slice], ts[int_slice])
        #rint("integral window = {}".format(np.trapz(vs[int_slice], ts[int_slice])))

        # save info
        df = pd.DataFrame({'T': ts, 'V': vs}, columns=['T', 'V'])
        trace_file = dfname.split('.')[0] + '_{}'.format(i_trace) + '.' + dfname.split('.')[1]
        if verbose:
            print("saving trace to file {}".format(trace_file))
        logger.info("saving trace to file {}".format(trace_file))
        # logger.info()
        df.to_csv(trace_file)


        if os.path.exists(width_file):
            df_wid = pd.read_csv(width_file)
            df_wid_new = pd.DataFrame({'LEDs': onLEDs, 'pot_value_ohm': pot_setting,
                                       'FWHM_s': wid, 'half_max_V': half_max,
                                       'fit_trise_s': parLCL3[2], 'fit_tdecay_s': parLCL3[3],
                                       'fit_peak_V': parLCL3[0], 'fit_tpeak_s': parLCL3[1],
                                       'fit_baseline_V': parLCL3[4],
                                       'integrate_window_start': start_window,'integrate_window_stop': stop_window,
                                       'integrated_charge_all': charge_all, 'integrated_charge_window': charge_window},
                                      columns=['LEDs', 'pot_value_ohm', 'FWHM_s', 'half_max_V', 'fit_trise_s',
                                               'fit_tdecay_s', 'fit_peak_V', 'fit_tpeak_s', 'fit_baseline_V',
                                               'integrate_window_start', 'integrate_window_stop',
                                               'integrated_charge_all', 'integrated_charge_window'], index=[0])
            df_full = df_wid.append(df_wid_new, ignore_index=True)
            df_full.to_csv(width_file, index=False)
        else:
            df_wid = pd.DataFrame({'LEDs': onLEDs, 'pot_value_ohm': pot_setting,
                                   'FWHM_s': wid, 'half_max_V': half_max,
                                   'fit_trise_s': parLCL3[2], 'fit_tdecay_s': parLCL3[3],
                                   'fit_peak_V': parLCL3[0], 'fit_tpeak_s': parLCL3[1],
                                   'fit_baseline_V': parLCL3[4],
                                   'integrate_window_start': start_window, 'integrate_window_stop': stop_window,
                                   'integrated_charge_all': charge_all, 'integrated_charge_window': charge_window
                                   },
                                  columns=['LEDs', 'pot_value_ohm', 'FWHM_s', 'half_max_V', 'fit_trise_s',
                                           'fit_tdecay_s', 'fit_peak_V', 'fit_tpeak_s', 'fit_baseline_V',
                                           'integrate_window_start', 'integrate_window_stop',
                                           'integrated_charge_all', 'integrated_charge_window'
                                           ], index=[0])
            df_wid.to_csv(width_file, index=False)

        # plot
        fig = plt.figure()
        plt.plot(ts, vs)
        plt.hlines(half_max, ts[lind], ts[rind], colors='b', linestyles=":", alpha=0.4)
        plt.plot(ts, flare_func_w_const(ts, *parLCL3), 'g--', alpha=0.4)

        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plot_name_ = plot_name.split('.')[0] + '_{}'.format(i_trace) + '.' + plot_name.split('.')[1]
        plt.savefig(plot_name_)
        if show:
            plt.show()

def plot_tek_trace(tekname='USB0::0x0699::0x039F::C010383::INSTR', scope_resource=None,
                   dfname="pocket_pulser.csv", plot_name="pocket_pulser.png", width_file="Flasher_fwhm.csv",
                   log_file="Flasher_characterization.log",
                   data_start=50000, data_stop=65000,
                   skip_quality_cut=False,
                   tstart=0, tstop=1e-7, initial_params=[0.1, 8.e-8, 1.e-9, 10.e-9, 0.],
                   pot_setting=1.55e3, onLEDs=[1], verbose=True, show=False,
                   num_read=10):
    ##initial_params: amp, tpeak, trise, tdecay, const

    setup_logger("flasher_log", log_file,
                 level=logging.INFO, show_log=verbose)
    logger = logging.getLogger('flasher_log')

    # establish comm to scope DPO2014B
    if scope_resource is None:
        rm = visa.ResourceManager()
        # print(rm.list_resources())
        tek = rm.open_resource(tekname)
        print(tek.query('*IDN?'))
        tek.timeout = 500000
        tek.write('DATA:ENC ASCI')
    else:
        tek = scope_resource

    tek.write("DATa:STARt {}".format(data_start))
    tek.write("DATa:STOP {}".format(data_stop))
    print("Set data acquisition start and stop to")
    print(tek.query("DATa:STARt?"), tek.query("DATa:STOP?"))
    # Assumes no changes will be made durinng the measurement
    xin = float(tek.query('wfmpre:XIN?'))
    xze = float(tek.query('wfmpre:XZE?'))
    x_start = xze + data_start * xin
    ymult = float(tek.query("wfmpre:ymult?"))
    yoff = float(tek.query("wfmpre:yoff?"))
    yzero = float(tek.query("wfmpre:yzero?"))
    if verbose:
        print("xze {}, xin {}, x_start {}".format(xze, xin, x_start))
        print("ymult {}, yoff {}, yzero {}".format(ymult, yoff, yzero))
        print("Clippinng the trace between {} and {}".format(tstart, tstop))
    logger.info("xze {}, xin {}, x_start {}".format(xze, xin, x_start))
    logger.info("ymult {}, yoff {}, yzero {}".format(ymult, yoff, yzero))
    logger.info("Clippinng the trace between {} and {}".format(tstart, tstop))

    for i_trace in range(num_read):
        # get the trace
        vs = np.array(tek.query_ascii_values('CURV?'))
        if i_trace==0:
            # Calc ts once
            ts = np.arange(x_start, x_start + vs.shape[0] * xin, xin)
            ind = np.where((ts > tstart) & (ts < tstop))
            ts = ts[ind]

        vs = (vs - yoff) * ymult + yzero
        # logger.info()
        vs = vs[ind]
        # plt.plot(ts[ind], vs[ind])


        # Fit the trace w exponential
        # params: amp, tpeak, trise, tdecay, const
        parLCL3, covLCL3 = curve_fit(flare_func_w_const, ts,
                                     vs, p0=initial_params,
                                     )  # ,
        # method='trf', ftol=1e-10, loss='soft_l1') #approx abs

        par_names8 = ['amplitude', 'tpeak', 'trise', 'tdecay', 'const']
        if verbose:
            for i in range(len(par_names8)):
                print("Flare exponential fit param %d (%s) = %.4g +/- %.4g" % (i, par_names8[i],
                                                                               parLCL3[i],
                                                                               np.sqrt(abs(covLCL3[i, i]))))

                # def chisq(y_vals, y_expected, y_errs, num_params=1):
                #  for reduced chisq test, under the assumption of Poisson counting
                #  we have lnL = const - (1/2.)*chi2


        # calc FWHM
        # wid, half_max, lind, rind = FWHM(ts[ind], vs[ind])
        wid, half_max, lind, rind = FWHM(ts, vs, verbose=verbose)

        # print(wid, half_max, lind, rind, ts[lind], ts[rind])
        # quality cut
        if abs(parLCL3[0]-parLCL3[4]) < 3*np.std(vs) or wid>100 :
            print(abs(parLCL3[0]-parLCL3[4]),  3*np.std(vs))
            print("Failed quality cuts")
            logger.error("Failed quality cuts")
            if skip_quality_cut:
                parLCL3 = initial_params
            else:
                continue

        # integrate charge:
        start_window = parLCL3[1] - 1.5 * np.e * parLCL3[2]
        stop_window = parLCL3[1] + 3 * np.e * parLCL3[3]


        charge_all = np.trapz(vs, ts)
        #print("integral all = {}".format(np.trapz(vs, ts)))

        int_slice = np.where((ts >= start_window) & (ts <= stop_window))
        charge_window=np.trapz(vs[int_slice], ts[int_slice])
        #rint("integral window = {}".format(np.trapz(vs[int_slice], ts[int_slice])))

        chi2LC3L1, dof3L1, redchi2LC3L1 = chisq(vs[int_slice],
                                                flare_func_w_const(ts[int_slice], *parLCL3),
                                                1,
                                                5)
        #lnL_LC3L1 = -0.5 * chi2LC3L1
        if verbose:
            print("fit Chisq in the integration window is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC3L1, dof3L1, redchi2LC3L1))
            #print("Log likelihood lnL={0}".format(lnL_LC3L1))

        # save info
        #df = pd.DataFrame({'T': ts, 'V': vs}, columns=['T', 'V'])
        trace_file = dfname.split('.')[0] + '_{}'.format(i_trace) + '.' + dfname.split('.')[1]
        if verbose:
            print("saving trace to file {}".format(trace_file))
        logger.info("saving trace to file {}".format(trace_file))
        # logger.info()
        with open(trace_file, 'a') as file:
            file.write("T,V\n")
            for t_, v_ in zip(ts, vs):
                file.write("{},{}\n".format(t_, v_))

        #df.to_csv(trace_file)

        f_exists=os.path.exists(width_file)
        with open(width_file, 'a') as file:
            if not f_exists:
                file.write('LEDs,pot_value_ohm,FWHM_s,half_max_V,fit_trise_s,fit_tdecay_s,fit_peak_V,fit_tpeak_s,'
                           'fit_baseline_V,fit_redchisq,integrate_window_start,integrate_window_stop,integrated_charge_all,'
                           'integrated_charge_window\n')
            file.write(",".join(map(str, [onLEDs[0], pot_setting, wid, half_max, parLCL3[2], parLCL3[3],
                           parLCL3[0],parLCL3[1],parLCL3[4],redchi2LC3L1,start_window, stop_window,
                           charge_all, charge_window])))
            file.write('\n')


        # plot
        if i_trace<5:
            fig = plt.figure()
            plt.plot(ts, vs)
            plt.hlines(half_max, ts[lind], ts[rind], colors='b', linestyles=":", alpha=0.4)
            plt.plot(ts, flare_func_w_const(ts, *parLCL3), 'g--', alpha=0.4)
            plt.vlines(start_window, 0, 2 * half_max, colors='c',
                      # linestyles=":",
                      linestyles="--",
                      alpha=0.4)
            plt.vlines(stop_window, 0, 2 * half_max, colors='c',
                      # linestyles=":",
                      linestyles="--",
                      alpha=0.4)

            plt.hlines(half_max * 2, start_window, stop_window, colors='c',
                      # linestyles=":",
                      linestyles="--",
                      alpha=0.4)

            plt.xlabel('Time [s]')
            plt.ylabel('Voltage [V]')
            plot_name_ = plot_name.split('.')[0] + '_{}'.format(i_trace) + '.' + plot_name.split('.')[1]
            plt.savefig(plot_name_)
            if show:
                plt.show()


def adjust_scope_scale(tek, LED=1, timeout=30,
                       log_file=None, use_default=False,
                       default_scales={1:0.4, 2:0.3, 3:0.4, 4:0.07, 5:0.11,
                                       6:0.25, 7:0.6, 8:0.25, 9:0.4, 10:0.23}):
    import signal
    class MyException(Exception):
        print('Timed out handled.')
        pass
    def my_timeout(signum, frame):
        "called when read times out"
        print('Timed out!')
        raise MyException
    def my_input():
        try:
            foo = input()
            return foo
        except:
            # timeout
            return

    if log_file is not None:
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
        logger = logging.getLogger('flasher_log')
    happy=False
    factor=1
    while not happy:
        #current_scale = float(tek.query("CH1:Scale?"))
        if LED in default_scales:
            def_scale = default_scales[LED]
            if factor != 1:
                tek.write("CH1:Scale {}".format(def_scale*factor))
                print("Let's set the scale of CH1 to scaled default value {} V for LED {}".format(def_scale*factor, LED))
                if log_file is not None:
                    logger.info("Let's set the scale of CH1 to scaled default value {} V for LED {}".format(def_scale*factor, LED))
            else: #if abs(current_scale-def_scale)/def_scale > 0.02:
                tek.write("CH1:Scale {}".format(def_scale))
                print("Let's set the scale of CH1 to default value {} V for LED {}".format(def_scale, LED))
                if log_file is not None:
                    logger.info("Let's set the scale of CH1 to default value {} V for LED {}".format(def_scale, LED))

        current_scale = float(tek.query("CH1:Scale?"))
        if log_file is not None:
            logger.info("Current scale of CH1 of the oscilloscope is {} V.".format(current_scale))

        if not use_default:
            print("Check waveform in the oscilloscope! \n### Enter a new scale value (in unit V) in {} sec: ".format(timeout))
            #logger.info()
            signal.signal(signal.SIGALRM, my_timeout)
            # set alarm
            signal.alarm(timeout)
            s = my_input()
        else:
            s="-"
        if s:
            signal.alarm(0)
            if s!="-":
                try:
                    new_scale = float(s)
                except ValueError:
                    print("Please enter a float!")
                    continue
                tek.write("CH1:Scale {}".format(new_scale))
                print("Setting CH1:Scale to {}".format(new_scale))
                print("Are we happy?")
                signal.signal(signal.SIGALRM, my_timeout)
                # set alarm
                signal.alarm(timeout)
                inhap = my_input()
                if inhap in ['yes', 'Yes', 'y', 'Y', ""]:
                    signal.alarm(0)
                    happy = True
                    if LED in default_scales:
                        if abs(current_scale-def_scale)/def_scale > 0.02:
                            factor = current_scale/def_scale
                    print ("Good. Continuing...")
                    if log_file is not None:
                        logger.info("For LED {}, final voltage per division is set to {} V".format(LED, new_scale))
            else:
                happy = True
        else:
            print("Oops, no input was received in {} sec.".format(timeout))
            if LED in default_scales:
                new_scale = default_scales[LED]
                print("Using default value {} V for LED {}".format(new_scale, LED))
            #elif LED in [4,5]:
            #    new_scale=current_scale * 0.1
            #    print("This is a dim LED, guessing a new value {} V.".format(new_scale))
            #elif LED==6:
            #    new_scale=current_scale * 10
            #    print("Guess we were on a dim LED just now, guessing a new value {} V.".format(new_scale))
            else:
                new_scale=current_scale
                print("Guess we set to the same value as before")
            print("Setting CH1:Scale to {} for LED {}".format(new_scale, LED))
            if log_file is not None:
                logger.info("For LED {}, final voltage per division is set to {} V".format(LED, new_scale))
            tek.write("CH1:Scale {}".format(new_scale))
            happy=True
    return


def adjust_scope_scale_reliable(tek, LED=1, timeout=30,
                       log_file=None, ):

    if log_file is not None:
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
        logger = logging.getLogger('flasher_log')
    happy=False
    while not happy:
        current_scale = float(tek.query("CH1:Scale?"))
        if log_file is not None:
            logger.info("Current scale of CH1 of the oscilloscope is {} V.".format(current_scale))

        s = input("Check waveform in the oscilloscope! \n### Enter a new scale value (in unit V) in {} sec: ".format(timeout))
        if s:
            try:
                new_scale = float(s)
            except ValueError:
                print("Please enter a float!")
                continue
            tek.write("CH1:Scale {}".format(new_scale))
            print("Setting CH1:Scale to {}".format(new_scale))
            inhap = input("Are we happy?")
            if inhap in ['yes', 'Yes', 'y', 'Y', ""]:
                happy = True
                print ("Good. Continuing...")
                if log_file is not None:
                    logger.info("For LED {}, final voltage per division is set to {} V".format(LED, new_scale))
    return


def auto_adjust_scope_scale(tek, min_scale=0.003, max_scale=5, max_trial=100, saturation_level=88,
                       log_file=None, ratio=5.3, low_ratio=2.5):
    #absolute min scale is 2 mV but cannnot operate at full bandwidth, limiting to 3 mV
    if log_file is not None:
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
        logger = logging.getLogger('flasher_log')
    i=0
    current_scale = float(tek.query("CH1:Scale?"))
    vs = np.array(tek.query_ascii_values('CURV?'))
    saturated=False
    if np.max(vs)>saturation_level:
        logger.info("Initial pulse is clipping")
        saturated=True
    #print(np.max(vs))

    ymult = float(tek.query("wfmpre:ymult?"))
    yoff = float(tek.query("wfmpre:yoff?"))
    yzero = float(tek.query("wfmpre:yzero?"))

    vs = (vs - yoff) * ymult + yzero

    if np.mean(vs) < 0:
        vmax = abs(np.min(vs))
    else:
        vmax = np.max(vs)
    #print(vmax, current_scale, (vmax/current_scale))
    while ((vmax/current_scale)<low_ratio and (current_scale>min_scale) ) or ((vmax/current_scale)>ratio+0.2) or saturated:
        if i>max_trial:
            logger.error("maximum number of trials {} for auto readjust scope reached...".format(max_trial))
            return -1
        if log_file is not None:
            logger.info("Current scale of CH1 of the oscilloscope is {} V.".format(current_scale))
            logger.info("Peak voltage is 5 times higher than division interval! readjusting scope scale")
        #current_scale = float(tek.query("CH1:Scale?"))

        if saturated:
            current_scale = current_scale*2.
        else:
            current_scale = float("{0:.3f}".format(vmax/ratio))
        #print(vmax, current_scale, (vmax / current_scale))
        if current_scale<min_scale:
            logger.info("Setting to minimum scale {}".format(min_scale))
            current_scale=min_scale
        if current_scale>max_scale:
            logger.info("Setting to maximum scale {}".format(max_scale))
            current_scale = max_scale
        tek.write("CH1:Scale {}".format(current_scale))
        logger.info("Setting CH1:Scale to {}".format(current_scale))
        i=i+1

        vs = np.array(tek.query_ascii_values('CURV?'))
        ymult = float(tek.query("wfmpre:ymult?"))
        yoff = float(tek.query("wfmpre:yoff?"))
        yzero = float(tek.query("wfmpre:yzero?"))
        #print(np.max(vs))
        if np.max(vs)>saturation_level:
            logger.info("Pulse is still clipping")
            saturated=True
        else:
            saturated=False
        vs = (vs - yoff) * ymult + yzero

        if np.mean(vs)<0:
            vmax = abs(np.min(vs))
        else:
            vmax = np.max(vs)
        current_scale = float(tek.query("CH1:Scale?"))

    logger.info("Final voltage per division is set to {} V".format(current_scale))
    return current_scale


def quick_test(data_start=50000, data_stop=65000, tstart=0.5e-7, tstop=2e-7,
               use_default=True, default_scales={1:0.11, 2:0.08, 3:0.11, 4:0.02, 5:0.03,
                                       6:0.07, 7:0.16, 8:0.07, 9:0.1, 10:0.06}):
    rm = visa.ResourceManager()
    print(rm.list_resources())
    tek = rm.open_resource('USB0::0x0699::0x039F::C010383::INSTR')
    print(tek.query('*IDN?'))

    # LEDnum="6_7_8_9_10"
    pot_val = "test"
    for LEDnum, trig in zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            ['1000000000', '0100000000', '0010000000', '0001000000', '0000100000',
                             '0000010000', '0000001000', '0000000100', '0000000010', '0000000001']):
        log_file = "flasher_fwhm_pot{}.log".format(pot_val)
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=True)
        logger = logging.getLogger('flasher_log')
        logger.info("########")
        logger.info("Setting LED pattern to {}".format(trig))
        process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if not os.path.exists("flasher_data/pot_{}/".format(pot_val)):
            os.mkdir("flasher_data/pot_{}/".format(pot_val));
        adjust_scope_scale(tek, LED=LEDnum, timeout=3, use_default=use_default, default_scales=default_scales)

        logger.info("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        print("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        plot_tek_trace(dfname="flasher_data/pot_{}/LED{}_pot{}_trace.csv".format(pot_val, LEDnum, pot_val),
                       plot_name="flasher_data/pot_{}/LED{}_pot{}_trace.png".format(pot_val, LEDnum, pot_val),
                       width_file="flasher_fwhm_pot{}.csv".format(pot_val),
                       log_file="flasher_fwhm_pot{}.log".format(pot_val),
                       data_start=data_start, data_stop=data_stop,
                       tstart=tstart, tstop=tstop, initial_params=[0.1, 7.e-8, 1.e-9, 10.e-9, 0.],
                       pot_setting=pot_val, onLEDs=[LEDnum], verbose=True,
                       num_read=1)


def pico_test(data_start=50000, data_stop=65000, tstart=0, tstop=2e-7,
              Vscale=0.02, num_read=1, pico_setting=8.0, initial_params=[0.1, 7.e-8, 1.e-9, 10.e-9, 0.]):
    rm = visa.ResourceManager()
    print(rm.list_resources())
    tek = rm.open_resource('USB0::0x0699::0x039F::C010383::INSTR')
    print(tek.query('*IDN?'))

    # LEDnum="6_7_8_9_10"

    log_file = "flasher_fwhm_pico_test2.log"
    setup_logger("flasher_log", log_file,
                 level=logging.INFO, show_log=True)
    logger = logging.getLogger('flasher_log')
    logger.info("########")
    if not os.path.exists("flasher_data/picotest2/"):
        os.mkdir("flasher_data/picotest2/")
    tek.write("CH1:Scale {}".format(Vscale))
    print("Let's set the scale of CH1 to {} V ".format(Vscale))

    plot_tek_trace(dfname="flasher_data/picotest2/pico{}_trace.csv".format(pico_setting),
                   plot_name="flasher_data/picotest2/pico{}_trace.png".format(pico_setting),
                   width_file="flasher_fwhm_picotest2.csv",
                   log_file="flasher_fwhm_picotest2.log",
                   data_start=data_start, data_stop=data_stop,
                   tstart=tstart, tstop=tstop, initial_params=initial_params,
                   pot_setting=pico_setting, onLEDs=1, verbose=True,
                   num_read=num_read)


def main_test(pot_val = "3.5k", num_traces=100, timeout=1,
              startingLED=1, starting_i=0, skip_cut=False,
              data_start=49920, data_stop=50130,
              default_scales={1: 0.11, 2: 0.08, 3: 0.11, 4: 0.02, 5: 0.03,
                              6: 0.07, 7: 0.16, 8: 0.07, 9: 0.1, 10: 0.06},
              #faint_initial_params=[0, 0, 0, 0, 0.],
              #faint_initial_params=[0.1, 8.e-8, 5.e-9, 7.e-9, 0.],
              ):
    rm = visa.ResourceManager()
    print(rm.list_resources())
    tek = rm.open_resource('USB0::0x0699::0x039F::C010383::INSTR')
    print(tek.query('*IDN?'))

    for LEDnum, trig in zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            ['1000000000', '0100000000', '0010000000', '0001000000', '0000100000',
                             '0000010000', '0000001000', '0000000100', '0000000010', '0000000001']):
        if LEDnum<startingLED:
            continue
        log_file = "flasher_fwhm_pot{}.log".format(pot_val)
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
        logger = logging.getLogger('flasher_log')
        logger.info("########")
        logger.info("Setting LED pattern to {}".format(trig))
        process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        time.sleep(2) # just do this again to make sure...
        process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if not os.path.exists("flasher_data/pot_{}/".format(pot_val)):
            os.mkdir("flasher_data/pot_{}/".format(pot_val));
        adjust_scope_scale(tek, LED=LEDnum, timeout=timeout, log_file=log_file,
                           use_default=True, default_scales=default_scales)

        logger.info("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        print("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        initial_params=[0.1, 8.e-8, 5.e-9, 7.e-9, 0.]

        #if faint_initial_params is not None:
        #    if LEDnum == 4 or LEDnum==5:
        #        initial_params = faint_initial_params
        #        skip_cut = True

        plot_tek_trace(dfname="flasher_data/pot_{}/LED{}_pot{}_trace.csv".format(pot_val, LEDnum, pot_val),
                       plot_name="flasher_data/pot_{}/LED{}_pot{}_trace.png".format(pot_val, LEDnum, pot_val),
                       width_file="flasher_fwhm_pot{}.csv".format(pot_val),
                       log_file="flasher_fwhm_pot{}.log".format(pot_val),
                       data_start=data_start, data_stop=data_stop, skip_quality_cut=skip_cut,
                       tstart=0, tstop=2e-7, initial_params=initial_params,
                       pot_setting=pot_val, onLEDs=[LEDnum], verbose=False,
                       num_read=num_traces)


def append_test(pot_val = "3.5k", num_traces=100, timeout=1,
              skip_cut=False,
              data_start=49920, data_stop=50130,
              ):
    rm = visa.ResourceManager()
    print(rm.list_resources())
    tek = rm.open_resource('USB0::0x0699::0x039F::C010383::INSTR')
    print(tek.query('*IDN?'))

    trig = input("Trigger patter you want: ")
    print("You typed {}".format(trig))
    input("Press Enter to continue...")

    LEDnum = input("On LEDs are: ")
    LEDnum=str(LEDnum)
    print("You typed {}".format(trig))
    input("Press Enter to continue...")

    log_file = "flasher_fwhm_pot{}.log".format(pot_val)
    setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
    logger = logging.getLogger('flasher_log')
    logger.info("########")
    logger.info("Setting LED pattern to {}".format(trig))
    process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    time.sleep(2) # just do this again to make sure...
    process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if not os.path.exists("flasher_data/pot_{}/".format(pot_val)):
            os.mkdir("flasher_data/pot_{}/".format(pot_val))
    adjust_scope_scale_reliable(tek, log_file=log_file,)

    logger.info("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
    print("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
    initial_params=[0.1, 8.e-8, 5.e-9, 7.e-9, 0.]


    plot_tek_trace(dfname="flasher_data/pot_{}/LED{}_pot{}_trace.csv".format(pot_val, LEDnum, pot_val),
                       plot_name="flasher_data/pot_{}/LED{}_pot{}_trace.png".format(pot_val, LEDnum, pot_val),
                       width_file="flasher_fwhm_pot{}.csv".format(pot_val),
                       log_file="flasher_fwhm_pot{}.log".format(pot_val),
                       data_start=data_start, data_stop=data_stop, skip_quality_cut=skip_cut,
                       tstart=0, tstop=2e-7, initial_params=initial_params,
                       pot_setting=pot_val, onLEDs=[LEDnum], verbose=False,
                       num_read=num_traces)

def long_test(pot_val = "3.5k", num_traces=100, timeout=1,
              startingLED=1, starting_i=0, skip_cut=False,
              data_start=49920, data_stop=50130,start_pattern=0,
              scope_setting_outfile="scope_2p5k.csv",
              default_scales={1: 0.11, 2: 0.08, 3: 0.11, 4: 0.02, 5: 0.03,
                              6: 0.07, 7: 0.16, 8: 0.07, 9: 0.1, 10: 0.06},
              faint_initial_params=[0, 0, 0, 0, 0.],
              ):
    rm = visa.ResourceManager()
    print(rm.list_resources())
    tek = rm.open_resource('USB0::0x0699::0x039F::C010383::INSTR')
    print(tek.query('*IDN?'))

    # lo:  L2 L5 L7 L9 L10 0  1  1
    # hi:  L1 L3 L4 L6 L8  1  1  1
    #sort_LED_lev=[9,10,7,5,2,1,6,4,8,3]
    #        for ii, i in enumerate([4,5,3,2,1,6,9,8,10,7]):

    """
    for trig in ['0001100000', '0011000000', '0010100000', '0101000000', '0100100000',
                 '0110000000', '1001000000', '1000100000', '1010000000', '1100000000',
                 '0000110000', '0001010000', '0010010000', '1110010000', '0000010100',
                 '0000110100', '0001110100', '0101110100', '0000011000', '0001111000',
                 '0010110100', '1011110100', '1101110100', '0000000101', '0000000011',
                 '0000001001', '0000101001', '0001101001', '0100101001', '0110101001',
                 '0000001101', '0001101101', '1000001101', '0100001101', '0000011101',
                 '0000001111', '0000011111', '1000001111', '0011101111', '0110001111',
                 '1000011111', '1000011111', '1100011111', '1011111111', '0110111111',
                 '1111111111']:
                 """
    triglist =  ["0001000000", "0000100000", "1000000000", "0100000000","0010000000",
                 "0000010000", "0000001000", "0000000100", "0000000010","0000000001",
                 "0001100000","0011000000","0101000000","0010100000","0100100000",
                 "1001000000","0110000000","1000100000","0001010000","1010000000",
                 "0000110000","1100000000","0010010000","0001100001","0010000001",
                 "0011100001","0000010100","0000110100","0000011000","0001110100",
                 "0000000101","0001111000","0000000011","1110010000","0000001001",
                 "0000101001","0001101001","0010110100","0101110100","0010001001",
                 "0100101001","0110101001","1011110100","1101110100","0000001101",
                 "0001101101",
                 "0100001101","1000001101","0000011101","0000001111","0011101111",
                 "1000001111","0110001111","0000011111","0001111111","0010011111",
                 "1000011111","0110111111","1100011111","1011111111","1111111111",
                    ]
    for trig in triglist[start_pattern:]:
        LEDnum=trig
        log_file = "flasher_fwhm_pot{}.log".format(pot_val)
        setup_logger("flasher_log", log_file,
                     level=logging.INFO, show_log=False)
        logger = logging.getLogger('flasher_log')
        logger.info("########")
        logger.info("Setting LED pattern to {}".format(trig))
        process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        time.sleep(2) # just do this again to make sure...
        process = Popen(['python2', 'LED_Flasher_TestQF.py', trig], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if not os.path.exists("flasher_data/pot_{}/".format(pot_val)):
            os.mkdir("flasher_data/pot_{}/".format(pot_val));

        #os.system('say Adjust Scope now!')
        #adjust_scope_scale_reliable(tek, log_file=log_file, )
        #current_scale = float(tek.query("CH1:Scale?"))
        #os.system('say Auto adjusting scope now!')
        logger.info('say Auto adjusting scope now!')
        current_scale = auto_adjust_scope_scale(tek, min_scale=0.003, max_trial=100,
                                log_file=log_file, ratio=5.4, low_ratio=3.8)
        if current_scale<0.003:
            logger.error('This should never happen, auto adjusted to a too small scale!')
            exit()
        with open(scope_setting_outfile,'a') as file:
            #file.write("T,V\n")
            file.write("{},{}\n".format(trig.zfill(10), current_scale))
        logger.info("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        print("Starting characterization LED pattern {}, pot {}".format(LEDnum, pot_val))
        initial_params=[0.1, 8.e-8, 5.e-9, 7.e-9, 0.]



        plot_tek_trace(dfname="flasher_data/pot_{}/LED{}_pot{}_trace.csv".format(pot_val, LEDnum, pot_val),
                       plot_name="flasher_data/pot_{}/LED{}_pot{}_trace.png".format(pot_val, LEDnum, pot_val),
                       width_file="flasher_fwhm_pot{}.csv".format(pot_val),
                       log_file="flasher_fwhm_pot{}.log".format(pot_val),
                       data_start=data_start, data_stop=data_stop, skip_quality_cut=skip_cut,
                       tstart=0, tstop=2e-7, initial_params=initial_params,
                       pot_setting=pot_val, onLEDs=[LEDnum], verbose=False,
                       num_read=num_traces)


if __name__ == "__main__":
    #3.5k diffuser
    #quick_test(data_start=49000, data_stop=62000, tstart=0, tstop=2e-7, default_scales={1:0.03, 2:0.02, 3:0.015, 4:0.016, 5:0.02,
    #                                   6:0.017, 7:0.03, 8:0.035, 9:0.04, 10:0.04})
    # 4k diffuser
    #quick_test(data_start=49000, data_stop=62000, tstart=0, tstop=2e-7,
    #           default_scales={1: 0.035, 2: 0.025, 3: 0.02, 4: 0.02, 5: 0.023,
    #                           6: 0.02, 7: 0.035, 8: 0.035, 9: 0.04, 10: 0.04})

    #main_test(pot_val="3p5k_diffuser", num_traces=100, timeout=3, default_scales={1:0.11, 2:0.08, 3:0.11, 4:0.02, 5:0.03,
    #                                   6:0.07, 7:0.16, 8:0.07, 9:0.1, 10:0.06})
    #3.5k with diffuser
    #main_test(pot_val="3p5k_diffuser", num_traces=100, timeout=1, default_scales={1:0.03, 2:0.02, 3:0.015, 4:0.016, 5:0.02,
    #                                   6:0.017, 7:0.03, 8:0.035, 9:0.04, 10:0.04})
    #4.1k w diff new setup Feb 11
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #                      default_scales={1: 0.075, 2: 0.055, 3: 0.05, 4: 0.045, 5: 0.05,
    #                                      6: 0.07, 7: 0.1, 8: 0.085, 9: 0.085, 10: 0.085})

    #pico_test(data_start=49000, data_stop=50130, tstart=0, tstop=2e-7,
    #          Vscale=0.09, num_read=500, pico_setting="9p1", initial_params=[0.1, 9.e-8, 2.e-9, 6.e-9, 0.])
    #3.5k with diffuser
    #main_test(pot_val="4k_diffuser", num_traces=100, timeout=1,
    #          default_scales={1: 0.035, 2: 0.025, 3: 0.02, 4: 0.02, 5: 0.023,
    #                          6: 0.025, 7: 0.035, 8: 0.035, 9: 0.04, 10: 0.04})
    #4.1k with diffuser new setup Feb 11
    #main_test(pot_val="4p1k_diffuser_new_setup_DC_power_source_Feb12", num_traces=1000, timeout=1,startingLED=3,
    #          default_scales={1: 0.075, 2: 0.055, 3: 0.05, 4: 0.05, 5: 0.05,
    #                          6: 0.07, 7: 0.1, 8: 0.085, 9: 0.085, 10: 0.085})

    # 3p75k/4k with diffuser new setup Feb 16 and 18
    #quick_test(tstart=0, tstop=2e-7,
    #                     data_start=49920, data_stop=50130,
    #      default_scales={1: 0.075, 2: 0.055, 3: 0.05, 4: 0.045, 5: 0.05,
    #                          6: 0.07, 7: 0.1, 8: 0.085, 9: 0.085, 10: 0.085})

    #3p75k/4.0k with diffuser new setup Feb 16
    #main_test(pot_val="3p75k_Feb18", num_traces=1000, timeout=1,startingLED=1,
    #main_test(pot_val="4p0k_Feb18", num_traces=1000, timeout=1, startingLED=1,
    #                    data_start=49920, data_stop=50130,
    #          default_scales={1: 0.075, 2: 0.055, 3: 0.05, 4: 0.045, 5: 0.05,
    #                          6: 0.07, 7: 0.1, 8: 0.085, 9: 0.085, 10: 0.085})
    # 4,25/4.5k with diffuser new setup Feb 16 / 18
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #                                 default_scales={1: 0.09, 2: 0.065, 3: 0.065, 4: 0.06, 5: 0.065,
    #                                                 6: 0.085, 7: 0.11, 8: 0.09, 9: 0.09, 10: 0.09})

    #main_test(pot_val="4p5k_Feb18", num_traces=1000, timeout=1, startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales = {1: 0.09, 2: 0.065, 3: 0.065, 4: 0.06, 5: 0.065,
    #                  6: 0.085, 7: 0.11, 8: 0.09, 9: 0.09, 10: 0.09})

    # 4.75k/5.0k with diffuser new setup Feb 16 18
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #                                 default_scales={1: 0.1, 2: 0.075, 3: 0.075, 4: 0.07, 5: 0.075,
    #                                                 6: 0.095, 7: 0.12, 8: 0.1, 9: 0.1, 10: 0.1})

    #main_test(pot_val="5p0k_Feb16", num_traces=1000, timeout=1, startingLED=1,
    #main_test(pot_val="4p75k_Feb18", num_traces=1000, timeout=1, startingLED=1,
    #main_test(pot_val="5p0k_Feb18", num_traces=1000, timeout=1, startingLED=1,
    #                    data_start=49920, data_stop=50130,
    #          default_scales={1: 0.1, 2: 0.075, 3: 0.075, 4: 0.07, 5: 0.075,
    #                          6: 0.095, 7: 0.12, 8: 0.1, 9: 0.1, 10: 0.1})

    # 5.25k/5.5k with diffuser new setup Feb 16 / 18
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #            default_scales={1: 0.11, 2: 0.085, 3: 0.085, 4: 0.08, 5: 0.085,
    #                            6: 0.1, 7: 0.135, 8: 0.11, 9: 0.11, 10: 0.11})

    #main_test(pot_val="5p25k_Feb19", num_traces=1000, timeout=1, startingLED=1,
    #main_test(pot_val="5p5k_Feb19", num_traces=1000, timeout=1, startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.11, 2: 0.085, 3: 0.085, 4: 0.08, 5: 0.085,
    #                          6: 0.1, 7: 0.135, 8: 0.11, 9: 0.11, 10: 0.11})

    # 5.75/6.0k with diffuser new setup Feb 16/19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #            default_scales={1: 0.12, 2: 0.095, 3: 0.095, 4: 0.09, 5: 0.095,
    #                            6: 0.11, 7: 0.15, 8: 0.12, 9: 0.12, 10: 0.12})

    #main_test(pot_val="6p0k_Feb16", num_traces=1000, timeout=1, startingLED=1,
    #main_test(pot_val="5p75k_Feb19", num_traces=1000, timeout=1, startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.12, 2: 0.095, 3: 0.095, 4: 0.09, 5: 0.095,
    #                          6: 0.11, 7: 0.15, 8: 0.12, 9: 0.12, 10: 0.12})

    #3.5k with diffuser new setup Feb 16
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.05, 2: 0.035, 3: 0.03, 4: 0.02, 5: 0.03,
    #                          6: 0.05, 7: 0.06, 8: 0.06, 9: 0.06, 10: 0.06})

    # 3.5k with diffuser new setup Feb 16
    #main_test(pot_val="3p5k_Feb16", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.05, 2: 0.035, 3: 0.03, 4: 0.02, 5: 0.03,
    #                          6: 0.05, 7: 0.06, 8: 0.06, 9: 0.06, 10: 0.06})

    # 3.25k with diffuser new setup Feb 18
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.05, 2: 0.035, 3: 0.03, 4: 0.02, 5: 0.03,
    #                          6: 0.05, 7: 0.06, 8: 0.06, 9: 0.06, 10: 0.06})

    # 3.25k annd 3.5k with diffuser new setup Feb 18
    #main_test(pot_val="3p5k_Feb18", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.05, 2: 0.035, 3: 0.03, 4: 0.02, 5: 0.03,
    #                          6: 0.05, 7: 0.06, 8: 0.06, 9: 0.06, 10: 0.06})


    #3.0k with diffuser new setup Feb 16
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.03, 2: 0.02, 3: 0.02, 4: 0.008, 5: 0.016,
    #                          6: 0.04, 7: 0.04, 8: 0.04, 9: 0.04, 10: 0.04})

    # 3.0k with diffuser new setup Feb 16
    #main_test(pot_val="3p0k_Feb18", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.03, 2: 0.02, 3: 0.02, 4: 0.008, 5: 0.016,
    #                          6: 0.04, 7: 0.04, 8: 0.04, 9: 0.04, 10: 0.04})
    #2.5k with diffuser new setup Feb 16
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.018, 2: 0.01, 3: 0.01, 4: 0.004, 5: 0.004,
    #                          6: 0.022, 7: 0.024, 8: 0.024, 9: 0.024, 10: 0.024})

    # 2.5k with diffuser new setup Feb 16
    #main_test(pot_val="2p5k_Feb16", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.018, 2: 0.01, 3: 0.01, 4: 0.004, 5: 0.004,
    #                          6: 0.022, 7: 0.024, 8: 0.024, 9: 0.024, 10: 0.024})

    #2.75k with diffuser new setup Feb 16 and feb 18
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.022, 2: 0.014, 3: 0.014, 4: 0.004, 5: 0.008,
    #                          6: 0.03, 7: 0.03, 8: 0.03, 9: 0.03, 10: 0.03})

    # 2.75k with diffuser new setup Feb 16 and 18
    #main_test(pot_val="2p75k_Feb18", num_traces=500, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.022, 2: 0.014, 3: 0.014, 4: 0.004, 5: 0.008,
    #                          6: 0.03, 7: 0.03, 8: 0.03, 9: 0.03, 10: 0.03})

    #2.25k with diffuser new setup Feb 19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.012, 2: 0.006, 3: 0.006, 4: 0.003, 5: 0.003,
    #                          6: 0.015, 7: 0.018, 8: 0.018, 9: 0.018, 10: 0.018})

    # 2.25k with diffuser new setup Feb 19
    #main_test(pot_val="2p25k_Feb19", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.012, 2: 0.006, 3: 0.006, 4: 0.003, 5: 0.003,
    #                          6: 0.015, 7: 0.018, 8: 0.018, 9: 0.018, 10: 0.018})

    #2.0k with diffuser new setup Feb 19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.01, 2: 0.004, 3: 0.004, 4: 0.003, 5: 0.003,
    #                          6: 0.012, 7: 0.012, 8: 0.012, 9: 0.012, 10: 0.012})

    # 2.0k with diffuser new setup Feb 19
    #main_test(pot_val="2p0k_Feb19", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.01, 2: 0.004, 3: 0.004, 4: 0.003, 5: 0.003,
    #                          6: 0.012, 7: 0.012, 8: 0.012, 9: 0.012, 10: 0.012})

    #1.75k with diffuser new setup Feb 19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.004, 2: 0.003, 3: 0.003, 4: 0.003, 5: 0.003,
    #                          6: 0.008, 7: 0.008, 8: 0.008, 9: 0.008, 10: 0.008})

    # 1.75k with diffuser new setup Feb 19
    #main_test(pot_val="1p75k_Feb19", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.004, 2: 0.003, 3: 0.003, 4: 0.003, 5: 0.003,
    #                          6: 0.008, 7: 0.008, 8: 0.008, 9: 0.008, 10: 0.008})

    #1.5k with diffuser new setup Feb 19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.003, 2: 0.003, 3: 0.003, 4: 0.003, 5: 0.003,
    #                          6: 0.003, 7: 0.003, 8: 0.003, 9: 0.003, 10: 0.003})

    # 1.5k with diffuser new setup Feb 19
    #main_test(pot_val="1p5k_Feb19", num_traces=1000, timeout=1,startingLED=4,
    #          data_start=49920, data_stop=50130, skip_cut=True,
    #          default_scales={1: 0.003, 2: 0.003, 3: 0.003, 4: 0.003, 5: 0.003,
    #                          6: 0.003, 7: 0.003, 8: 0.003, 9: 0.003, 10: 0.003})

    #2.5k with diffuser new setup Feb 19
    #quick_test(data_start=49920, data_stop=50130, tstart=0, tstop=2e-7,
    #          default_scales={1: 0.018, 2: 0.01, 3: 0.01, 4: 0.003, 5: 0.004,
    #                          6: 0.022, 7: 0.024, 8: 0.024, 9: 0.024, 10: 0.024})

    # 2.5k with diffuser new setup Feb 19
    #main_test(pot_val="2p5k_Feb19", num_traces=1000, timeout=1,startingLED=1,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.018, 2: 0.01, 3: 0.01, 4: 0.003, 5: 0.004,
    #                          6: 0.022, 7: 0.024, 8: 0.024, 9: 0.024, 10: 0.024})

    #append_test(pot_val="2p5k_Feb19_append", num_traces=500, timeout=30,
    #            skip_cut=False,
    #            data_start=49920, data_stop=50130,
    #            )
    #long_test(pot_val="2p5k_Feb19_long", num_traces=500, timeout=30,
    #            skip_cut=False,
    #            data_start=49920, data_stop=50130,
    #            )
    #Feb 25
    #main_test(pot_val="2p5k_Feb25", num_traces=1000, timeout=1,startingLED=4,
    #          data_start=49920, data_stop=50130,
    #          default_scales={1: 0.018, 2: 0.01, 3: 0.01, 4: 0.003, 5: 0.004,
    #                          6: 0.022, 7: 0.024, 8: 0.024, 9: 0.024, 10: 0.024})
    #long_test(pot_val="2p5k_Feb25_long", num_traces=500, timeout=30,
    #long_test(pot_val="2p5k_Feb25_long", num_traces=500, timeout=30,
    #                    skip_cut=False, start_pattern=14,
    #          scope_setting_outfile="scope_2p5k_auto.csv",
    #                      data_start=49920, data_stop=50130,
    #                      )
    # Feb 28 openned up the aperture
    #long_test(pot_val="2p5k_Mar4_long", num_traces=500, timeout=30,
    #          skip_cut=False, start_pattern=0,
    #          scope_setting_outfile="scope_2p5k_auto_Mar4_36mm2.csv",
    #          data_start=49920, data_stop=50130,
    #          )

    # Apr 18:
    long_test(pot_val="2p5k_Apr18", num_traces=100, timeout=30,
              skip_cut=False, start_pattern=0,
              scope_setting_outfile="scope_2p5k_Apr18_36mm2.csv",
              data_start=49920, data_stop=50130,
              )

    # Feb  16/19/25 test, distance 91 cm
    os.system('say done!')