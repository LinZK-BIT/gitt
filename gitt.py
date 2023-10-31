# python3.11
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d, corrcoef
from math import ceil
import sys
import tkinter
from tkinter import filedialog


def load_data(filename: str):
    """
    Data example:
    工步类型 | 时间(h) | 电压(V) | 电流(A) | 比容量(mAh/g)
    搁置      | 0.0       | 0.787   | 0.102   | 0.00
    """
    data = pd.read_excel(filename, sheet_name=0)
    return data


def pick_discharge_data(data, min_time: float, max_time: float):
    dis_data = data[data['工步类型'] == '恒流放电']
    dis_data = dis_data[dis_data['时间(h)'] < max_time]
    dis_data = dis_data[dis_data['时间(h)'] > min_time]
    return dis_data


def pick_charge_data(data, min_time: float, max_time: float):
    dis_data = data[data['工步类型'] == '恒流充电']
    dis_data = dis_data[dis_data['时间(h)'] < max_time]
    dis_data = dis_data[dis_data['时间(h)'] > min_time]
    return dis_data


def get_dEs(data):
    Es1 = data['电压(V)'].iloc[0]
    Es2 = data['电压(V)'].iloc[-1]
    return Es2 - Es1


def fit(dis_time, disp_volt, h1):
    sqrt_dis_time = np.sqrt(dis_time)
    slope, intercept = polyfit(sqrt_dis_time, disp_volt, 1)
    r2 = corrcoef(sqrt_dis_time, disp_volt,)[0, 1]
    r2 = r2 ** 2
    y_ = poly1d((slope, intercept))
    # print(f"{slope=:.4e}, {intercept=:.4e}, {r2=:.4f}")
    h1.scatter(sqrt_dis_time, disp_volt)
    h1.plot(sqrt_dis_time, y_(sqrt_dis_time), 'r--')
    h1.set_title(f"{slope=:.4e}, {r2=:.4f}")
    return slope, intercept, r2


def auto_fit_discharge(filename, min_time, max_time, ax1, ax2):
    data = load_data(filename)
    dis_data = pick_discharge_data(data, min_time, max_time)
    dis_time = dis_data['时间(h)']*3600
    tau = data[data['工步类型'] == '恒流放电']['时间(h)'].max()*3600
    voltage = dis_data['电压(V)']
    capacity = dis_data['比容量(mAh/g)'].iloc[0]
    iR_drop = data['电压(V)'].iloc[1] - data['电压(V)'].iloc[0]
    R = iR_drop / dis_data['电流(A)'].iloc[2]
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    Es = get_dEs(data)
    ax2.plot(data['时间(h)']*3600, data['电压(V)'])
    if r2 < 0.8:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!!!"
    elif r2 < 0.9:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!!"
    elif r2 < 0.95:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!"
    else:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}"
    print(info)
    return info, capacity, slope, Es, tau, R, inter, r2


def auto_fit_charge(filename, min_time, max_time, ax1, ax2):
    data = load_data(filename)
    dis_data = pick_charge_data(data, min_time, max_time)
    dis_time = dis_data['时间(h)']*3600
    tau = data[data['工步类型'] == '恒流充电']['时间(h)'].max()*3600
    iR_drop = data['电压(V)'].iloc[1] - data['电压(V)'].iloc[0]
    voltage = dis_data['电压(V)']
    R = iR_drop / dis_data['电流(A)'].iloc[2]
    # print(dis_data.head)
    capacity = dis_data['比容量(mAh/g)'].iloc[0]
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    Es = get_dEs(data)
    ax2.plot(data['时间(h)']*3600, data['电压(V)'])
    if r2 < 0.8:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!!!"
    elif r2 < 0.9:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!!"
    elif r2 < 0.95:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}\t!"
    else:
        info = f"{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{inter:.6e}\t{r2:.6f}"
    print(info)
    return info, capacity, slope, Es, tau, R, inter, r2


def auto_seperate_discharge_relaxation(data, dir):
    start_index_discharge = data[data['工步类型'] == '恒流放电'][data['时间(h)'] == 0]
    start_index_discharge = np.array(start_index_discharge.index)
    k = 1
    for i in range(len(start_index_discharge)):
        try:
            start = start_index_discharge[i] - 1
            end = start_index_discharge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/discharge'+str(k)+'.xlsx'
            temp.to_excel(outfile)
            k += 1
            print(f'save to file: {outfile}. {start}:{end}, len={end-start}')
        except:
            continue
    return start_index_discharge


def auto_seperate_charge_relaxation(data, dir):
    start_index_charge = data[data['工步类型'] == '恒流充电'][data['时间(h)'] == 0]
    start_index_charge = np.array(start_index_charge.index)
    k = 1
    for i in range(len(start_index_charge)):
        try:
            start = start_index_charge[i] - 1
            end = start_index_charge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/charge'+str(k)+'.xlsx'
            temp.to_excel(outfile)
            k += 1
            print(f'save to file: {outfile}. {start}:{end}, len={end-start}')
        except:
            continue
    return start_index_charge


def divide_data(filename, dir, sheet_name):
    try:
        import os
        os.mkdir(dir)
        print(f"Creating directory {dir}")
    except:
        pass
    # * read gitt data
    print(f"Reading data from file: {filename}\nThis may take minutes...")
    data_list = pd.read_excel(filename, sheet_name=sheet_name)
    print(f"concat dataframe of sheet_name {sheet_name}...")
    data = pd.concat(data_list.values(), ignore_index=True)
    # * seperate massive data into many smaller files
    # * so that GITT fitting can be handeled more easily
    print("Auto seperate discharge relaxation loops...")
    idx_discharge = auto_seperate_discharge_relaxation(data, dir)
    print("Auto seperate charge relaxation loops...")
    idx_charge = auto_seperate_charge_relaxation(data, dir)
    print(f"There are {len(idx_charge)} of charge files and {len(idx_discharge)} of discharge files.")
    

def func_D(capacity, slope, Es, tau, R):
    # this function should return Diffusity in cm^2/s
    # OR other any function: func(capacity, slope, Es, tau, *args)
    return 4/np.pi * (R/(3*tau) * Es / slope)**2

def discharge_process(dir, fignum, s1, args, func_D=func_D, savefig=True):
    # process discharge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("Discharge " + dir)
    print("capacity\tslope\tdEs\tinter\tR2")
    with open(dir+'_discharge.txt', 'w') as f:
        f.write(
            "D(cm^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tInter\tR2\n")
    with open(dir+'_discharge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/discharge' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, inter, r2 = auto_fit_discharge(
                    filename, 4/3600, 60/3600, ax1, ax2)
                D = func_D(capacity, slope, Es, tau, *args)
                info = f"{D:.6e}\t{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{tau:.3f}\t{Rohm:.4f}\t{inter:.6e}\t{r2:.6f}\n"
                f.write(info)
                ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
                ax1.set_ylabel(r"$E\rm(t)\ (V)$")
            except Exception as e:
                print(f"{i} is failed!")
                print(e)
    if savefig:
        fig1.savefig(dir+'_discharge.png')
    

def charge_process(dir, fignum, s1, args, func_D=func_D, savefig=True):
    # process charge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("Charge " + dir)
    print("capacity\tslope\tdEs\tinter\tR2")
    with open(dir+'_charge.txt', 'w') as f:
        f.write(
            "D(cm^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tInter\tR2\n")
    with open(dir+'_charge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/charge' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, inter, r2 = auto_fit_charge(
                    filename, 4/3600, 60/3600, ax1, ax2)
                D = func_D(capacity, slope, Es, tau, *args)
                info = f"{D:.6e}\t{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{tau:.3f}\t{Rohm:.4f}\t{inter:.6e}\t{r2:.6f}\n"
                f.write(info)
                ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
                ax1.set_ylabel(r"$E\rm(t)\ (V)$")
            except Exception as e:
                print(f"{i} is failed!")
                print(e)
    if savefig:
        fig1.savefig(dir+'_charge.png')


if __name__ == '__main__': 
    # ! Step 1: divide data into smaller xlsx files
    filename = './整理GITT(v8.0)-NCM-CNT-153.33mg.xlsx'
    dir = 'test-GITT-NCM'
    # filename = filedialog.askopenfilename()
    # dir = filedialog.askdirectory()
    # divide_data(filename, dir, sheet_name=['Sheet1', 'Sheet2'])

    # ! Fitting discharge data
    # discharge_process(dir, fignum=73, s1=4, args=(1e-6*100,), func_D=func_D, savefig=False)

    # ! Fitting charge data
    # charge_process(dir, fignum=70, s1=4, args=(1e-6*100,), func_D=func_D, savefig=False)

# D = 4/pi * (n_M*V_M/(A*tau) * dEs / slope)**2
# D = 4/pi * (R/(3*tau) * dEs / slope)**2
