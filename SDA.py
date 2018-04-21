# -*- coding: utf-8 -*-
"""
@author: Ali
"""

import sys
import time

import os
import re
import numpy as np
from scipy.optimize import fmin
from datetime import datetime

import sda_main_form
from sda_calibration_form import Ui_frmSDACalibration
from about_form import Ui_frmAbout

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox#, QLabel#, QPushButton, QToolTip
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as L2

global cal_mypath, cal_black_body_curve, cal_wl_fbb, cal_spect_cond, cal_srf # delete cal_srf
global analyze_mypath, fig_spec


class SDAMain(QMainWindow, sda_main_form.Ui_frmSDAMain):
    
###############################
#
#       Calibration
#
###############################
    def openWindow(self,parent=sda_main_form.Ui_frmSDAMain):
        self.hide()
        self.window.show()
        
        self.ui.btnCal_ProcessCalibration.clicked.connect(self.process_calibrate_spectrometer)        
        self.ui.btnCal_LoadFiles.clicked.connect(self.Load_Calibration_Data)
        self.ui.btnCal_Cancel.clicked.connect(self.Close_Window)

    def Load_Files(self):
        mypath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        return mypath

    def Black_Body_Curve(self,temp,wavelength,wl_scale, wl_value, plot_flag):  
        c1=3.741772e-25    
        c2=0.01438777
        wavelength_meters = wavelength/1e9
        cal_black_body_curve = c1 / ( np.pi * wavelength **5 * ( np.exp(c2/(wavelength_meters*temp)) - 1 ) )
        
        if wl_scale !=0:
            temp_index = [i for i, x in enumerate(wavelength >= (wl_value)) if x]
            cal_black_body_curve = cal_black_body_curve / cal_black_body_curve[temp_index[0] * wl_scale]
        return cal_black_body_curve

    def Spectrum_to_Black_Body_Curve_Compare(self,T):
        global cal_wl_fbb, cal_spect_cond
        c1=3.741772E-25
        c2=0.01438777
        L = c1 / ( np.pi * cal_wl_fbb ** 5 * ( np.exp(c2 / cal_wl_fbb / T) - 1 ) )
        L = L / max(L)   
        scale = np.linalg.lstsq(np.array(L).reshape(-1,1), np.array(cal_spect_cond).reshape(-1,1))[0] # GOOD
        L = np.linalg.norm(cal_spect_cond - scale*L) # GOOD
        return L

    def Fit_to_Black_Body(self,T0,wavelength,spectrum):
        global cal_wl_fbb, cal_spect_cond
        cal_wl_fbb = wavelength/1e9
        cal_spect_cond = spectrum / max(spectrum)
        Temperature = fmin(self.Spectrum_to_Black_Body_Curve_Compare, T0, xtol=1, maxfun=5000, maxiter=10000, disp=0)
        return Temperature

    def process_calibrate_spectrometer(self):
        txtLampName = self.ui.txtCal_LampName.text()
        txtDarkName = self.ui.txtCal_DarkName.text()
        txtMinWL = self.ui.txtCal_MinCFWL.text()
        txtMaxWL = self.ui.txtCal_MaxCFWL.text()
        txtRefTemp = self.ui.txtCal_RefTemp.text()
        txtCalCheck = self.ui.txtCal_CalibrationCheck.text()
        
        global cal_srf, cal_mypath #delete cal_srf
        
#        lamp_dir_n = [d for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtLampName) and x.endswith(".txt")] # Lamp Name user input
        lamp_file_n = [x for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtLampName) and x.endswith(".txt")] # Dark Name user input ... for all 
        lamp_dirfile_n = [os.path.join(d, x) for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtLampName) and x.endswith(".txt")]
        
#        dark_dir_n = [d for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtDarkName) and x.endswith(".txt")]
        dark_file_n = [x for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtDarkName) and x.endswith(".txt")]
        dark_dirfile_n = [os.path.join(d, x) for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith(txtDarkName) and x.endswith(".txt")]
        print(lamp_file_n)
        print(dark_file_n)
        
        # Read temperatures
        lamp_temps = [int(re.search(r'Lamp (.+?) ', i).group(1)) for i in lamp_file_n if re.search(r'Lamp (.+?) ', i)]
        dark_temps = [int(re.search(r'Dark (.+?) ', i).group(1)) for i in dark_file_n if re.search(r'Dark (.+?) ', i)]
        
        lamp_header = list(np.zeros(len(lamp_dirfile_n),dtype=np.int))
        lamp_data = {'set_temp' : [],
                     'contents' : [],
                     'spectrum' : [],
                     'int_time' : [],
                     'spec_bkgd' : [],
                     'spec_unity' : [],
                     'signal' : [],
                     'fit_temp' : []}
        
        dark_header = list(np.zeros(len(dark_dirfile_n),dtype=np.int))
        dark_data = {'set_temp' : [],
                     'contents' : [],
                     'spectrum' : [],
                     'int_time' : []}
        
        for k in range(0,len(lamp_temps)):
            with open(lamp_dirfile_n[k]) as fp:
                lamp_header[k] = fp.readlines()[1:13]
            lamp_header[k] = [i.rstrip('\n') for i in lamp_header[k]][1:]    
            lamp_data['set_temp'].append(lamp_temps[k])
            lamp_data['contents'].append(np.loadtxt(lamp_dirfile_n[k], delimiter='\t', skiprows=14, unpack=True))
            lamp_data['spectrum'].append(lamp_data['contents'][k][1])
            lamp_data['int_time'].append(float(lamp_header[k][4][24:]))
            
            with open(dark_dirfile_n[k]) as fp:
                dark_header[k] = fp.readlines()[1:13]
            dark_header[k] = [i.rstrip('\n') for i in dark_header[k]][1:]
            dark_data['set_temp'].append(dark_temps[k])
            dark_data['contents'].append(np.loadtxt(dark_dirfile_n[k], delimiter='\t', skiprows=14, unpack=True))
            dark_data['spectrum'].append(dark_data['contents'][k][1])
            dark_data['int_time'].append(float(dark_header[k][4][24:]))    
            lamp_data['spec_bkgd'].append(lamp_data['spectrum'][k] - dark_data['spectrum'][k])
        

        cal_wavelength = lamp_data['contents'][0][0]
        
#        wavelength_match_value = 750
        index = [i for i, x in enumerate(cal_wavelength >= 750) if x]
        
        for k in range(0,len(lamp_temps)):
            lamp_data['spec_unity'].append(lamp_data['spec_bkgd'][k]/np.mean(lamp_data['spec_bkgd'][k][ index[0]-2 : index[0] + 3]))
        
        
        index = [i for i, x in enumerate(cal_wavelength >= 750) if x]
        
        index_ref = [i for i, x in enumerate(np.array(lamp_data['set_temp']) == int(txtRefTemp)) if x][0] # reference temperature to text box (lamp temp)
        bb_temp = lamp_data['set_temp'][index_ref] # reference temperature to text box (lamp temp)
        
        bb_curve = self.Black_Body_Curve(bb_temp+273, cal_wavelength, 1, 750, 0)
        
        cal_srf = (lamp_data['spec_bkgd'][index_ref])/bb_curve # reference temperature to text box (lamp temp)
        
        cal_wl_min = float(txtMinWL) # get from input box
        cal_wl_max = float(txtMaxWL) # get from input box
        cal_fit_range = [i for i in [i for i, x in enumerate(cal_wavelength > cal_wl_min) if x] if i in [i for i, x in enumerate(cal_wavelength < cal_wl_max) if x]]
        
        
        for k in range(0,len(lamp_data['set_temp'])):
            lamp_data['signal'].append(lamp_data['spec_bkgd'][k]/cal_srf)
            lamp_data['fit_temp'].append(self.Fit_to_Black_Body(2000, cal_wavelength[cal_fit_range], 
                     lamp_data['signal'][k][cal_fit_range]) - 273) # 2000 is a seed value
        
        cal_set_temp_array = [np.array([i]) for i in lamp_data['set_temp']]
        temp_errors = np.subtract(lamp_data['fit_temp'],cal_set_temp_array)
        
        index_to_plot = [i for i, x in enumerate(np.array(lamp_temps) == int(txtCalCheck) ) if x][0] # grab from calibration check
        bb_temp_ex = lamp_data['set_temp'][index_to_plot]
        bbcurve_1_ex = self.Black_Body_Curve(bb_temp_ex+273, cal_wavelength, 1, 750, 0)
        bbcurve_2_ex = self.Black_Body_Curve(lamp_data['fit_temp'][index_to_plot]+273, cal_wavelength, 1, 750, 0)
        
        
        lamp_signal = lamp_data['signal'][index_to_plot][cal_fit_range]/lamp_data['signal'][index_to_plot][index[0]]
        lamp_bbcurve1 = bbcurve_1_ex[cal_fit_range]
        lamp_bbcurve2 = bbcurve_2_ex[cal_fit_range]
            
        fig_cal = plt.figure(figsize=(11,6.5)) ## figure it out
        fig_cal.suptitle("Calibration", fontsize=12, x = .53, y = .98)

        plt_cal_curve_fit = fig_cal.add_subplot(221)
        cf1 = plt_cal_curve_fit.plot(cal_wavelength, lamp_data['spec_unity'][index_ref], 'b', linewidth = 1) 
        cf2 = plt_cal_curve_fit.plot(cal_wavelength, bb_curve, 'g', linewidth = 1)
        plt_cal_curve_fit.set_title('Lamp Spectrum (Background Subtracted) & Black Body Curve at ' + str(int(txtRefTemp)) + ' C', fontsize=10)# add reference temperature
        plt_cal_curve_fit.set_xlabel('Wavelength / nm', fontsize=8)
        plt_cal_curve_fit.set_ylabel('Arbitrary Units', fontsize=8)
        plt_cal_curve_fit.legend(['Lamp Spectrum' , 'Black Body Curve'])
        
        plt_cal_temp_errors = fig_cal.add_subplot(222)
        plt_cal_temp_errors.plot(lamp_data['set_temp'] , temp_errors.tolist(), 'm', linewidth = 1)
        plt_cal_temp_errors.set_title("Temperature Fit Errors", fontsize=10)
        plt_cal_temp_errors.set_xlabel('Lamp Set Temperature (C)', fontsize=8)
        plt_cal_temp_errors.set_ylabel('Temperature Error (C)', fontsize=8)
        
        plt_cal_srf = fig_cal.add_subplot(223)
        plt_cal_srf.plot(cal_wavelength, cal_srf, 'b', linewidth = 1)
        plt_cal_srf.set_title("SRF", fontsize=10)
        plt_cal_srf.set_xlabel('Wavelength / nm', fontsize=8)
        plt_cal_srf.set_ylabel('Arbitrary Units', fontsize=8)
        
        plt_cal_signal_bbcurve = fig_cal.add_subplot(224)
        sb1 = plt_cal_signal_bbcurve.plot(cal_wavelength[cal_fit_range], lamp_signal, '-r', linewidth = 1) 
        sb2 = plt_cal_signal_bbcurve.plot(cal_wavelength[cal_fit_range], lamp_bbcurve1, '-g', linewidth = 1)
        sb3 = plt_cal_signal_bbcurve.plot(cal_wavelength[cal_fit_range], lamp_bbcurve2, '-b', linewidth = 1)
        plt_cal_signal_bbcurve.set_title("Lamp Signal and Black Body Curve at " + str(bb_temp_ex) + ' C', fontsize=10)# add calibration check temperature
        plt_cal_signal_bbcurve.set_xlabel('Wavelength / nm', fontsize=8)
        plt_cal_signal_bbcurve.set_ylabel('Arbitrary Units', fontsize=8)
        plt_cal_signal_bbcurve.legend(['Lamp Spectrum' , 'Black Body Curve: ' + str(bb_temp_ex) + ' C', 'Black Body Curve (Best Fit) : ' + str(int(round(lamp_data['fit_temp'][index_to_plot][0]))) + ' C'])# declare temperatures after @
        
        fig_cal.tight_layout(rect=[0, 0, 1, .96], w_pad=3, h_pad=2)
        plt.show()
        
        
    def Save_and_Close(self):
        global cal_srf
        if self.ui.txtCal_SRFName.text() != "":
            self.window.close()
            self.show()
            self.lblCalibrationStatus.setText("Calibrated")
            self.lblCalibrationStatus.setStyleSheet('color: darkgreen')
            self.lblCalibrationStatus.repaint()
            self.Save_SRF_File()
        else:
            self.lblCalibrationStatus.setText('Not Calibrated (No SRF Found)')
            self.lblCalibrationStatus.setStyleSheet('color: darkred') #darkgreen
            self.lblCalibrationStatus.repaint()
            self.window.close()
            self.show()
        
    def Close_Window(self):
        self.window.close()
        self.show()
        
    def Reset_Main(self):
        global cal_mypath, cal_black_body_curve, cal_wl_fbb, cal_spect_cond, cal_srf # delete cal_srf
        global analyze_mypath, fig_spec
        global analyze_data_dir_n, analyze_data_file_n, analyze_data_dirfile_n
        
        if 'cal_black_body_curve' in globals(): del cal_black_body_curve
        if 'cal_wl_fbb' in globals(): del cal_wl_fbb
        if 'cal_spect_cond' in globals(): del cal_spect_cond
        if 'analyze_mypath' in globals(): del analyze_mypath
        if 'fig_spec' in globals(): del fig_spec
        if 'analyze_data_dir_n' in globals(): del analyze_data_dir_n
        if 'analyze_data_file_n' in globals(): del analyze_data_file_n
        if 'analyze_data_dirfile_n' in globals(): del analyze_data_dirfile_n
        self.txtAna_DataFiles.setText('No files loaded')
        self.txtAna_MinFraction.setText(str(0.25))
        self.txtAna_MinADWL.setText(str(600))
        self.txtAna_MaxADWL.setText(str(800))
        self.txtAna_PlotWavelength.setText(str(750))
        self.cmbAna_PlotDataFile.clear()
        
    def Load_Calibration_Data(self):
        global lamp_dir_n, lamp_file_n, lamp_dirfile_n, dark_dir_n, dark_file_n, dark_dirfile_n, cal_mypath
        cal_mypath = self.Load_Files()
        if cal_mypath:
            if 'lamp_dir_n' in globals(): del lamp_dir_n
            if 'lamp_file_n' in globals(): del lamp_file_n
            if 'lamp_dirfile_n' in globals(): del lamp_dirfile_n
            
            if 'dark_dir_n' in globals(): del dark_dir_n
            if 'dark_file_n' in globals(): del dark_file_n
            if 'dark_dirfile_n' in globals(): del dark_dirfile_n
            
            lamp_dir_n = [d for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("L") and x.endswith(".txt")] # Lamp Name user input
            lamp_file_n = [x for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("L") and x.endswith(".txt")] # Dark Name user input ... for all 
            lamp_dirfile_n = [os.path.join(d, x) for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("L") and x.endswith(".txt")]
            
            dark_dir_n = [d for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("D") and x.endswith(".txt")]
            dark_file_n = [x for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("D") and x.endswith(".txt")]
            dark_dirfile_n = [os.path.join(d, x) for d, dirs, files in os.walk(cal_mypath) for x in files if x.startswith("D") and x.endswith(".txt")]
            
            lamp_temps = [int(re.search(r'Lamp (.+?) ', i).group(1)) for i in lamp_file_n if re.search(r'Lamp (.+?) ', i)]
            dark_temps = [int(re.search(r'Dark (.+?) ', i).group(1)) for i in dark_file_n if re.search(r'Dark (.+?) ', i)]
            
            if dark_temps == lamp_temps and dark_temps != [] : # success
                print(lamp_temps)                
            else:   # fail
                QMessageBox.warning(self, "Lamp and Dark Files Not Equal", "Please verify that files have matching Lamp and Dark pairs.")
                    
    def Load_Spectrometer_Data(self):
        global analyze_data_dir_n, analyze_data_file_n, analyze_data_dirfile_n
        analyze_mypath = self.Load_Files()
        if analyze_mypath:
            # Extract all file and directory strings
            analyze_data_file_name_start = 'Z'
            analyze_data_file_name_end = '.txt'
            if 'analyze_data_dir_n' in globals(): del analyze_data_dir_n
            if 'analyze_data_file_n' in globals(): del analyze_data_file_n
            if 'analyze_data_dirfile_n' in globals(): del analyze_data_dirfile_n
            analyze_data_dir_n = [d for d, dirs, files in os.walk(analyze_mypath) for x in files if x.startswith(analyze_data_file_name_start) and x.endswith(analyze_data_file_name_end)]
            analyze_data_file_n = [x for d, dirs, files in os.walk(analyze_mypath) for x in files if x.startswith(analyze_data_file_name_start) and x.endswith(analyze_data_file_name_end)]
            analyze_data_dirfile_n = [os.path.join(d, x) for d, dirs, files in os.walk(analyze_mypath) for x in files if x.startswith(analyze_data_file_name_start) and x.endswith(analyze_data_file_name_end)]        
            
            self.txtAna_DataFiles.setText('No files loaded')                    
            self.cmbAna_PlotDataFile.clear()                          
            
            if analyze_data_dir_n:
                self.txtAna_DataFiles.setText(str(len(analyze_data_file_n)) + ' files found')                    
                # Extract all serial numbers
                serial_num = [str(re.search(r'_(.+)_', i).group(1)) for i in analyze_data_file_n if re.search(r'_(.+)_', i)]
                # Extract all z
                z = [str(re.search(r' (.+)_'+ serial_num[0], i).group(1)) for i in analyze_data_file_n if re.search(r' (.+)_', i)]
            
                for k in range(0,len(z)):
                    self.cmbAna_PlotDataFile.addItem(z[k])
        
    def process_analyze_spectrometer_data(self):
        global cal_srf, fig_spec#delete cal_srf
        
        cal_status_txt = self.lblCalibrationStatus.text()
        if self.txtAna_DataFiles.text() != 'No files loaded' and self.lblCalibrationStatus.text() != 'Not Calibrated (No SRF Found)':
            
#            cal_status_txt = self.lblCalibrationStatus.text()
            
            self.lblCalibrationStatus.setStyleSheet('color: blue')
            self.lblCalibrationStatus.repaint()
            
            self.lblCalibrationStatus.setText('Acquiring Parameters...')
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.lblCalibrationStatus.repaint()
            time.sleep(.3)
            
            analyze_min_fraction = float(self.txtAna_MinFraction.text()) # user
            analyze_wl_min = float(self.txtAna_MinADWL.text()) #600 # user
            analyze_wl_max = float(self.txtAna_MaxADWL.text()) #800 #user 
            upi = self.cmbAna_PlotDataFile.currentIndex() #user plot index
            analyze_srf = cal_srf#srf_file_location

            self.lblCalibrationStatus.setText('Pre-Allocating Data...')
            self.lblCalibrationStatus.repaint()
            time.sleep(.2)
            # Pre-allocate Dictionary
            analyze_data = {'name' : [],
                         'date' : [],
                         'z' : [],
                         'int_time' : [],
                         'trigger_mode' : [],
                         'date_stamp' : [],
                         'time_vector' : [],
                         'num_scans' : [],
                         'num_pixels' : [],
                         'data' : [],
                         'intensities' : [],
                         'wavelength' : [],
                         'spec_bkgd' : [],
                         'spectra' : [],
                         'num_spectra' : [],
                         'signal' : [],
                         'time_spectra' : [],
                         'num_layers' : [],
                         'int_wl_srf' : [],
                         'temperature' : []} 
            header = list(np.zeros(len(analyze_data_file_n),dtype=np.int))
    
            # Extract all serial numbers
            serial_num = [str(re.search(r'_(.+)_', i).group(1)) for i in analyze_data_file_n if re.search(r'_(.+)_', i)]
            # Extract all z
            z = [str(re.search(r' (.+)_'+ serial_num[0], i).group(1)) for i in analyze_data_file_n if re.search(r' (.+)_', i)]
            
            self.lblCalibrationStatus.setText('Extracting Data...')
            self.lblCalibrationStatus.repaint()
            for k in range(0,len(analyze_data_dirfile_n)):
                analyze_data['name'].append(analyze_data_file_n[k]) # Extract File Names
                with open(analyze_data_dirfile_n[k]) as fp:
                    header[k] = fp.readlines()[1:13] # extract header data for all files
                header[k] = [i.rstrip('\n') for i in header[k]][1:] # remove \n text from header
                analyze_data['z'].append(z[k]) # append all z to dict analyze_data
                analyze_data['int_time'].append(float(header[k][4][24:])) # extract and append all int_time to dict analyze_data
                analyze_data['trigger_mode'].append(int(header[k][3][14:])) # extract and append all trigger_mode to dict analyze_data
                analyze_data['date_stamp'].append(np.loadtxt(analyze_data_dirfile_n[k], delimiter="\t", usecols=(0), dtype=object, skiprows=15, unpack=True))
                analyze_data['date'].append(header[k][0][6:])
                fmt = '%Y-%m-%d %H:%M:%S.%f'
                d0 = datetime.strptime(analyze_data['date_stamp'][k][0],fmt)
                analyze_data['time_vector'].append( [(datetime.strptime(date,fmt) - d0).total_seconds() for ind, date in enumerate(analyze_data['date_stamp'][k])])
                analyze_data['num_scans'].append(len(analyze_data['date_stamp'][k]))
                analyze_data['num_pixels'].append(int(header[k][10][30:]))
                analyze_data['data'].append(np.loadtxt(analyze_data_dirfile_n[k], delimiter='\t', skiprows=14, usecols=(range(2,analyze_data['num_pixels'][k]+2)), unpack=True))
                analyze_data['intensities'].append(np.rot90(analyze_data['data'][k][:,1:])[::-1])
                analyze_data['wavelength'].append(analyze_data['data'][k][:,0])
            
            analyze_data.pop('data', None)
            analyze_wavelength = analyze_data['wavelength'][0]
            
            analyze_fit_range = [i for i in [i for i, x in enumerate(analyze_data['wavelength'][0] > analyze_wl_min) if x] if i in [i for i, x in enumerate(analyze_data['wavelength'][0] < analyze_wl_max) if x]]
            r_wl = [i for i in [i for i, x in enumerate(analyze_data['wavelength'][0] > 725) if x] if i in [i for i, x in enumerate(analyze_data['wavelength'][0] < 775) if x]]
            
            self.lblCalibrationStatus.setText('Processing Data...')
            self.lblCalibrationStatus.repaint()
            for k in range(0,len(analyze_data_dirfile_n)):
            #     Determine which spectra are in each file
            #     It is the one with minimum intensity averaged by wavelength about 750 nm
            #     Wavelength regions of HSTV bands
                
                int_750_scans = np.mean(analyze_data['intensities'][k][:,r_wl],1).tolist() # take average for columns (mean 2)
                # find min and max
                min_750 = np.min(int_750_scans)
                index_min_750 = int_750_scans.index(min_750)
                max_750 = np.max(int_750_scans)
            
                if min_750 > 3000 and max_750 > min_750 + 100: # 200 does not work; criteria not met for if function
                    analyze_data['spec_bkgd'].append(analyze_data['intensities'][k][index_min_750])
                    threshold = analyze_min_fraction * (max_750 - min_750) + min_750
                    select = [i for i, x in enumerate(int_750_scans >= threshold) if x] 
                    analyze_data['spectra'].append(analyze_data['intensities'][k][select] - analyze_data['spec_bkgd'][k])
                    analyze_data['num_spectra'].append(np.size(analyze_data['spectra'][k],0))
                    
                    analyze_data['signal'].append([])
                    for kk_spectra in range(0,analyze_data['num_spectra'][k]):
                        analyze_data['signal'][k].append(analyze_data['spectra'][k][kk_spectra]/analyze_srf)
                    analyze_data['signal'][k] = np.rot90(analyze_data['signal'][k])[::-1]            
                    analyze_data['time_spectra'].append([analyze_data['time_vector'][k][i] for i in select])
                    large_gaps = np.diff( [np.array( analyze_data['time_spectra'] [k] )] ) > 8
                    analyze_data['num_layers'].append(np.sum(large_gaps)+1)              
                    analyze_data['int_wl_srf'].append( np.mean(analyze_data['signal'][k][r_wl],0))      
                    analyze_data['temperature'].append([])
                    for mm in range(0,analyze_data['num_spectra'][k]):
                        analyze_data['temperature'][k].append(self.Fit_to_Black_Body(2000,np.transpose(analyze_data['wavelength'][0][analyze_fit_range]),np.transpose(analyze_data['signal'][k][analyze_fit_range,mm]))-273)
                else:
                    analyze_data['num_spectra'][k] = 0
                    analyze_data['num_layers'][k] = 0  
            
            self.lblCalibrationStatus.setText('Preparing Plots...')    
            self.lblCalibrationStatus.repaint()
            tzt = list(np.zeros((5,sum(analyze_data['num_spectra'])),dtype=np.float))
            pointer = 0
            time0 = 0
            for k in range(0,len(analyze_data_dirfile_n)):
                if analyze_data['num_spectra'] != 0:
                    for ns in range(0,analyze_data['num_spectra'][k]):
                        tzt[0][pointer] = analyze_data['time_spectra'][k][ns] + time0
                        tzt[1][pointer] = float(analyze_data['z'][k])
                        tzt[2][pointer] = analyze_data['int_time'][k]
                        tzt[3][pointer] = analyze_data['trigger_mode'][k]
                        tzt[4][pointer] = analyze_data['temperature'][k][ns]
                        pointer = pointer + 1
                    time0 = time0 + analyze_data['time_spectra'][k][ns]
            
            signal_srf_scaled = list(np.zeros((analyze_data['num_spectra'][upi],len(analyze_wavelength)),dtype=np.float))
            
            for index_signal in range(0,analyze_data['num_spectra'][upi]):
                signal_srf_scaled[index_signal] = (analyze_data['signal'][upi][:,index_signal]) / (analyze_data['signal'][upi][[i for i, x in enumerate(analyze_data['wavelength'][0] > 750) if x][0],index_signal])
            
            fig_spec = plt.figure(figsize=(11,7))
            fig_spec.suptitle("Z Starting at " + analyze_data['z'][upi] , fontsize=16, x = .7, y = .98)
            
            plt_spec_strip_chart = fig_spec.add_subplot(231)
            plt_spec_strip_chart.plot(tzt[0],tzt[4], '-b',tzt[0],tzt[4], '.k', linewidth = 1) 
            plt_spec_strip_chart.set_title('Strip Chart of Temperature vs Time', fontsize=10)
            plt_spec_strip_chart.set_xlabel('Time (s)', fontsize=8)
            plt_spec_strip_chart.set_ylabel('Temperature (C)', fontsize=8)
            plt_spec_strip_chart.grid()
            
            plt_spec_temp_hist = fig_spec.add_subplot(234)
            plt_spec_temp_hist.set_axisbelow(True)
            plt_spec_temp_hist.hist(tzt[4], color = 'lightskyblue', histtype='bar',ec='black')
            plt_spec_temp_hist.set_title('Temperature Histogram', fontsize=10)
            plt_spec_temp_hist.set_xlabel('Temperature (C)', fontsize=8)
            plt_spec_temp_hist.set_ylabel('Counts', fontsize=8)
            plt_spec_temp_hist.yaxis.grid(True)
            
            plt_spec_bkgd_subtracted_spectra = fig_spec.add_subplot(232)
            plt_spec_bkgd_subtracted_spectra.plot(analyze_wavelength, np.rot90(analyze_data['spectra'][upi])[::-1]) 
            plt_spec_bkgd_subtracted_spectra.set_title('Background Subtracted Spectra', fontsize=10)
            plt_spec_bkgd_subtracted_spectra.set_xlabel('Wavelength (nm)', fontsize=8)
            plt_spec_bkgd_subtracted_spectra.set_ylabel('Intensity', fontsize=8)
            x1,x2,y1,y2 = plt_spec_bkgd_subtracted_spectra.axis()
            plt_spec_bkgd_subtracted_spectra.axis((400,1000,y1,y2))
            plt_spec_bkgd_subtracted_spectra.grid()
            
            plt_spec_intensities_srf_c = fig_spec.add_subplot(233)
            plt_spec_intensities_srf_c.plot(analyze_data['time_spectra'][upi], analyze_data['int_wl_srf'][upi]/analyze_data['time_spectra'][upi][0], 'g', linewidth = 1)
            plt_spec_intensities_srf_c.set_title('Intensities (SRF Corrected)', fontsize=10)
            plt_spec_intensities_srf_c.set_xlabel('Time (s)', fontsize=8)
            plt_spec_intensities_srf_c.set_ylabel('SRF Corrected Intensity', fontsize=8)
            plt_spec_intensities_srf_c.legend(['750 nm'])
            plt_spec_intensities_srf_c.grid()
            
            plt_spec_signal_spectra = fig_spec.add_subplot(235)
            plt_spec_signal_spectra_sp1 = plt_spec_signal_spectra.plot(analyze_wavelength, np.rot90(signal_srf_scaled)[::-1])
            plt_spec_signal_spectra_sp2 = plt_spec_signal_spectra.plot([analyze_wl_min,analyze_wl_min],[0,3], 'r:',[analyze_wl_max,analyze_wl_max],[0,3], 'r:',linewidth = 3)
            plt_spec_signal_spectra.set_title('Signal Spectra', fontsize=10)
            plt_spec_signal_spectra.set_xlabel('Wavelength (nm)', fontsize=8)
            plt_spec_signal_spectra.set_ylabel('SRF Corrected Intensity', fontsize=8)
            plt_spec_signal_spectra.axis((500.00,930.00,0,3))
            plt_spec_signal_spectra.grid()
            
            plt_spec_est_temp = fig_spec.add_subplot(236)
            plt_spec_est_temp.plot(analyze_data['time_spectra'][upi], analyze_data['temperature'][upi], '-ko', linewidth = 1) 
            plt_spec_est_temp.set_title('Estimated Temperature', fontsize=10)
            plt_spec_est_temp.set_xlabel('Time (s)', fontsize=8)
            plt_spec_est_temp.set_ylabel('Temperature (C)', fontsize=8)
            plt_spec_est_temp.grid()
            
            l1 = L2([.335, .335], [.985, 0], color = 'k', linewidth = 2, transform=fig_spec.transFigure, figure=fig_spec) #transform=fig.transFigure
            fig_spec.lines.extend([l1])
            fig_spec.canvas.draw()
            fig_spec.tight_layout(rect=[0, 0, 1, .96], w_pad=3, h_pad=2)
            fig_spec.show()
            
            QApplication.restoreOverrideCursor()
            self.lblCalibrationStatus.setText(cal_status_txt)
            self.lblCalibrationStatus.setStyleSheet('color: darkgreen')
            self.lblCalibrationStatus.repaint()
        elif self.txtAna_DataFiles.text() == 'No files loaded' and self.lblCalibrationStatus.text() != 'Not Calibrated (No SRF Found)':
            self.lblCalibrationStatus.setText("Please Load Spectrometer Data")
            self.lblCalibrationStatus.setStyleSheet('color: darkred')
            self.lblCalibrationStatus.repaint()
            time.sleep(1)
            self.lblCalibrationStatus.setText(cal_status_txt)
            self.lblCalibrationStatus.setStyleSheet('color: darkgreen')
            self.lblCalibrationStatus.repaint()
            
    def Analyze_View_Plots(self):
        if 'fig_spec' in globals(): fig_spec.show()
        
    def Load_SRF_File(self):
        global cal_srf
        srf_name_dir, _ = QFileDialog.getOpenFileName(self,"Select a SRF File", "","SRF Files (*.csv)")
        if srf_name_dir:
            if 'cal_srf' in globals(): del cal_srf
            cal_srf = np.loadtxt(srf_name_dir,delimiter=",")   
            self.lblCalibrationStatus.setText("Calibrated (" + srf_name_dir[len(srf_name_dir)- srf_name_dir[::-1].find('/'):] + ")")
            self.lblCalibrationStatus.setStyleSheet('color: darkgreen')
            
    def Save_SRF_File(self):
        global cal_srf
        if 'cal_srf' in globals():
            txtSRFName = self.ui.txtCal_SRFName.text()
            print(txtSRFName)
            if txtSRFName != '':
                odir = os.getcwd()
                os.chdir(cal_mypath)
                print(cal_mypath)
                np.savetxt(txtSRFName + '.csv', cal_srf)
                os.chdir(odir)
                
#    def Clear_Calibration(self):
#        self.ui.
        
    def Show_About(self):
        self.about.show()
        
    def __init__(self, parent=None):
        super(SDAMain, self).__init__(parent)
        self.setupUi(self)
        
        # calibration form settings
        self.window = QMainWindow()
        self.ui = Ui_frmSDACalibration()
        self.ui.setupUi(self.window)
        
        # calibration form settings
        self.about = QMainWindow()
        self.ui_about = Ui_frmAbout()
        self.ui_about.setupUi(self.about)
        
        self.ui.btnCal_Done.clicked.connect(self.Save_and_Close)
        
        self.btnAbout.clicked.connect(self.Show_About)
        self.btnCalibrateSpectrometerLink.clicked.connect(self.openWindow)
        
        self.btnAnalyzeData.clicked.connect(self.process_analyze_spectrometer_data)
        self.btnAna_ViewPlots.clicked.connect(self.Analyze_View_Plots)
        self.btnAna_LoadData.clicked.connect(self.Load_Spectrometer_Data)
        self.btnAna_LoadSRF.clicked.connect(self.Load_SRF_File)
        self.btnReset.clicked.connect(self.Reset_Main)
        
        self.show()
        
        self.lblCalibrationStatus.setText('Not Calibrated (No SRF Found)')
        self.lblCalibrationStatus.setStyleSheet('color: darkred') #darkgreen
        self.lblCalibrationStatus.repaint()

def refresh():
    QApplication.processEvents()
    
    
def main():
    app = QApplication(sys.argv)
    form = SDAMain()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()