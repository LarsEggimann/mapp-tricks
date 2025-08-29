# 📦 X-ray Spectrometer Efficiency Calibration and Efficiency Query

This package is used to calibrate the X-ray spectrometer and query the final efficiency.

Developed by Samuel Dominique Juillerat, modified by Lars Eggimann.

------------------------------------------------------------------------

## 🚀 Features

-   Modification of the `.csv` peak information files with the `.mca` spectrum files for the calibration in [modification-files_Xray-spectrometer.py](./code/modification-files_Xray-spectrometer.py)
-   Calibration with the modified files in [calibration_Xray-spectrometer.py](./code/calibration_Xray-spectrometer.py)
-   Querying the efficiency for a certain energy in [efficiency_Xray-spectrometer.py](./code/efficiency_Xray-spectrometer.py)

## 📂 Project Structure

```         
.
├─ code/                                                        # folder with all scripts
│     ├─ modification-files_Xray-spectrometer.py                # script for modification
│     ├─ calibration_Xray-spectrometer.py                       # script for calibration
│     ├─ efficiency_Xray-spectrometer.py                        # script for efficiency query
│     └─ utils/                                                 # folder with helper functions
│           ├─ functions_geometrical_factor.py
│           └─ functions_normalized_efficiency_detector.py
│
├─ files/                                             # folder with all necessary default data
│     ├─ attenuation/                                 # attenuation coefficients
│     ├─ decay data nuclides/                         # decay data of the nuclides
│     ├─ SDD Efficiency files/                        # computed default efficiency curves
│     └─ Source-data_Xray-spectrometer.csv            # data sheet for calibration sources
│
├─ raw files/                                         # folder for the raw files
│     ├─ raw mca files/                               # folder of all .mca files
│     └─ raw peak files/                              # raw peak files of InterSpec
│           ├─ unread/                                # folder for files to modifiy
│           └─ read/                                  # folder for files after modification
│                 └─ Gain3-398_PeakingTime1us/        # unique setup name
│                       ├─ level 00/                  # folder of all files for level 00
│                       :                             #                               ::
│                       └─ level 14/                  # folder of all files for level 14
│                 
├─ modified peak files/                     # folder for modified files
│     └─ Gain3-398_PeakingTime1us/          # unique setup name
│           ├─ level 00/                    # folder of all files for level 00
│           :                               #                               ::
│           └─ level 14/                    # folder of all files for level 14
│
├─ result files calibration/                # folder for results of calibration
│     └─ Gain3-398_PeakingTime1us/          # unique setup name
│           ├─ level 00/                    # folder of all files for level 00
│           :                               #                               ::
│           └─ level 14/                    # folder of all files for level 14
│
├─ efficiency plots/                        # folder for final efficiency plots
│
└─ README.md                                # ReadMe file
```

## ▶️ Usage

### 📐 Calibration

The following steps are necessary in order to perform a new calibration.

1.  You are using a calibration source with known data and those data need to be added to the [Source-data_Xray-spectrometer.csv](./files/Source-data_Xray-spectrometer.csv) if it is not already existing.\
    You also have to define a source name (e.g. `133Ba-source1`) which will be important for the whole process. Don't use the character `_` in the source name! If your source contains more than one nuclide you have to add a new source name for each nuclide.\
    \
    Notice: If your source is embedded in plastic you can add the thickness of the plastic the photons have to pass and the code scales the peak areas according to the calculated attenuation. For this calculation Polyethylene is used with a density of $\rho = (0.935 \pm =0.1)\ \mathrm{g/cm^3}$ .\
2.  You need to save your `.mca` spectrum files with the source name and the corresponding level on which is was measured (e.g. `133Ba-source1_level01_OTHER-PARAMETERS.mca`) in the folder `raw files/raw mca files/`.\
3.  Go to InterSpec and choose your nuclide of the measured source under 'Reference Photopeaks'. You can then start selecting your peaks of the spectrum and those should automatically be assigned to the corresponding photo peak, which is important.\
    \
    If there is more than one relevant photo peak in your selected peak, but InterSpec only fitted one peak, then please note bullet point number 6 of this list!\
    \
    Notice: Not assigned peaks will cause an error later!\
4.  In InterSpec, go to 'Peak Manager' and download the CSV-file of your selected peaks. The name of the CSV-file should automatically have the correct name (e.g. `peaks_133Ba-source1_level01_OTHER-PARAMETERS.CSV`).\
    \
    Choose a **new unique** setup name (e.g. `GainX-XXX_PeakingTimeYY-Yus`) and create a new folder with this name in `raw files/raw peak files/unread/` and then create sub-folders for the measured levels in this folder (e.g. `level XX` ).\
    \
    You then have to insert your downloaded CSV-files into the specific level folders of your setup name.\
    It should look like:

```         
:
├─ raw files/                                         
│     ├─ raw mca files/                               
│     └─ raw peak files/                             
│           ├─ unread/                                
│           │      └─ GainX-XXX_PeakingTimeYY-Yus/       
│           │            ├─ level 00/                  # folder of all files for level 00
│           │            :                             #                               ::
│           │            └─ level 14/                  # folder of all files for level 14
│           └─ read/
:                 :
```

5.  If you used a source with a new nuclide you also have to create a `.csv` with the decay energies and the corresponding branching ratios. Get those data from [NNDC](https://www.nndc.bnl.gov){.uri} and for the formatting take a look at the already existing files. It can simply be copied from the webpage and the 'decay energy' and the 'branching ratio' columns must be kept.\
    \
    Your new file has to be saved in `files/decay data nuclides/` and the file name has to start with the nuclide name (e.g. `133Ba_decay-data.csv` ).\

6.  If there is more than one photo peak in your fitted peak area, you have to add the missing photo peak(s) manually to the downloaded CSV-file. In the column 'Photopeak_Energy' you have to add the value(s) of the missed photo peak to the existing one. The only restriction is that all photo peaks have to be listed in the decay data file (bullet point number 5).\
    \
    Here's an example: InterSpec recognized a a photo peak at 34.99 keV, but there is another relevant photo peak at 34.92 keV which InterSpec missed. Adjust in the column 'Photopeak_Energy' the listed photo peak of `34.99` to `34.99;34.92` . It is important that the photo peaks are separated with a semi-colon `;` . There is no limit on the number of photo peaks which can be added.\

7.  Now you can run the python script [modification-files_Xray-spectrometer.py](./code/modification-files_Xray-spectrometer.py) . This script modifies all peak information files automatically with the corresponding `.mca`-files and creates new files in the directory `modified peak files/`, which will be used for the calibration later.\

8.  Execute the python script [calibration_Xray-spectrometer.py](./code/calibration_Xray-spectrometer.py) which will calibrate the modified files. By retaining the default settings (recommended) the script perform a calibration of all available files of the mentioned setup name and saves the parameters of the fit to a new file. This takes some time due to complex calculations in the background.\
    \
    Notice: You can adjust several setting of the script. It can be chosen if the X-rays, the gammas or both should be included in the calibration, if the script should use the default efficiency curve of Amptek (which was calculated for perpendicular incoming rays) or the individually computed one for each level, taking the attenuation of the air into account or neglecting it.\

## 🖥️ Queries

Open the python script [efficiency_Xray-spectrometer.py](./code/efficiency_Xray-spectrometer.py) and choose the parameters according to your efficiency query and let it run. Also this script will take some time due to complex calculations in the background. The code will scale the efficiency according to the stated radius.

In the end you will receive a terminal output and a plot.

Notice: The curve of the plot is for a point source! So if your radius is non-negligible the point of the requested energy will not be on this line anymore since it got scaled to a source with the stated radius!
