import os

class GammaPeakAndTargetInfo:
    def __init__(self, name, peak_energy_range, half_life, branching_ratio, target_mass, molar_mass, isotopic_abundance, n_sto=1, collimator_area=None):
        """
        Initialize the isotope and target information.
        """
        self.name = name
        self.peak_energy_range = peak_energy_range
        self.half_life = half_life  # in seconds
        self.branching_ratio = branching_ratio
        self.target_mass = target_mass
        self.molar_mass = molar_mass
        self.isotopic_abundance = isotopic_abundance
        self.n_sto = n_sto # stoichiometric coefficient
        self.collimator_area = collimator_area

class DataSource:
    def __init__(self, data_folder, spectra_file, orbitos_file, spectra_level=None, spectra_with_aluminum_foil=False):
        self.data_folder = data_folder
        self.spectra_file = spectra_file
        self.spectra_level = spectra_level
        self.spectra_with_aluminum_foil = spectra_with_aluminum_foil
        self.spectra_path = os.path.join(data_folder, spectra_file)
        self.orbitos_file = orbitos_file
        self.orbitos_path = os.path.join(data_folder, orbitos_file)
