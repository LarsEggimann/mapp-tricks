from mapp_tricks.peakfit import PeakFitter

# calib = XRayCalibration(
#     level=1,
# )

# test = calib.evaluate_efficiency_at(
#     ufloat(10, 0.1)
#     )

# fig = calib.get_plot()


# from mapp_tricks.spectrometer_calibrations import HPGeCalibration

# calib = HPGeCalibration(level=1, with_aluminum_foil=True)

# e1 = 670.32
# e2 = 962.84

# eff1 = calib.evaluate_efficiency_at_energy(e1)
# eff2 = calib.evaluate_efficiency_at_energy(e2)

# print(f"Efficiency at {e1} keV: {eff1:uS}")
# print(f"Efficiency at {e2} keV: {eff2:uS}")

# calib.print_summary()
# calib.plot_fit()


# pf = PeakFitter()

# res = pf.process_folder(
#     folder_path="/home/lars/Downloads/23-07-2026/15min spectra",
#     energy_range=(662, 682),
# )

# test new peakfit parser

from mapp_tricks.peakfit import parse_spectrum_file


from_cnf = parse_spectrum_file('./data/spectra_test_data/original_cnf_file.cnf')
from_cnfconv = parse_spectrum_file('./data/spectra_test_data/cnfconv_converted_file.txt')
from_interspect = parse_spectrum_file('./data/spectra_test_data/InterSpec_converted_file.txt')

# store each of with .write_to_file() to a new file and compare the outputs

from_cnf.write_to_file('./data/spectra_test_data/output_from_cnf.txt')
from_cnfconv.write_to_file('./data/spectra_test_data/output_from_cnfconv.txt')
from_interspect.write_to_file('./data/spectra_test_data/output_from_interspect.txt')