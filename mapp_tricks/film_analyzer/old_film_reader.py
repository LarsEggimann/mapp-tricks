#!/usr/bin/env python3

import cv2
import numpy as np
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import plotly.express as px # type: ignore
from pathlib import Path
from scipy.signal import convolve2d # type: ignore
import matplotlib.pyplot as plt
from skimage import io
from .plotly_defaults import MyPlotlyDefaults 
from .helpers import get_e, get_n
from uncertainties import ufloat
import uncertainties
import uncertainties.unumpy as unp
import uncertainties.umath as umath
mps = MyPlotlyDefaults()

class FilmScanner:
    def __init__(self, scan_file: Path, calibration_name="EBT3_new_METAS_ImageJwRGB", radius=60, dpi=400, max_dose=20, dose_correction_factor_lamination_layer=ufloat(1,0.01), dose_correction_factor_relative_efficiency=ufloat(1,0.01)):
        self.scan_file = scan_file
        self.radius = radius
        self.dpi = dpi
        self.max_dose = max_dose

        self.image = None
        self.image_grey_pixel_value = None
        self.center = None
        self.fig = None
        self.results = None

        self.horizontal_profile = None
        self.vertical_profile = None
        self.heatmap = None
        self.x_coords = None
        self.y_coords = None

        self.film_calibration_data = self.define_film_calibration_data()
        self.selected_film_calibration = self.film_calibration_data[calibration_name]

        self.dose_correction_factor_lamination_layer = dose_correction_factor_lamination_layer
        self.dose_correction_factor_relative_efficiency = dose_correction_factor_relative_efficiency



    
    def find_dark_rectangle_center(self, crop_margin=0.2):

        gray = self.image.copy()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # corp the image to 60% width and 60% height relative to the center
        width = gray.shape[1]
        height = gray.shape[0]
        corp_center_x = width // 2
        corp_center_y = height // 2
        crop_width = int(width * 0.8)
        corp_height = int(height * 0.8)
        gray = gray[corp_center_y - corp_height // 2:corp_center_y + corp_height // 2, corp_center_x - crop_width // 2:corp_center_x + crop_width // 2]

        # where gray is 255 set it to mean
        gray[gray >= 255] = 162

        downsample_factor = 0.5

        gray = cv2.resize(gray, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA) # pylint: disable=no-member

        
        # create kernel (2d matrix) for top left corner with 30% of the image size
        kernel_size = (int(gray.shape[0] * 0.3), int(gray.shape[1] * 0.3))

        kernel_top_left = np.ones(kernel_size)
        kernel_top_right = kernel_top_left.copy()
        kernel_bottom_right = kernel_top_left.copy()
        kernel_bottom_left = kernel_top_left.copy()

        # middle of the kernel in x
        m = int(kernel_top_left.shape[0] * 0.5)
        # middle of the kernel in y
        n = int(kernel_top_left.shape[1] * 0.5)

        # top left: set the bottom right corner to -1
        kernel_top_left[:m, :n] = -1
        # top right: set the bottom left corner to -1
        kernel_top_right[:m, n:] = -1
        # bottom right: set the top left corner to -1
        kernel_bottom_right[m:, n:] = -1
        # bottom left: set the top right corner to -1
        kernel_bottom_left[m:, :n] = -1

        def show_convolution(conv, max_coords):
            f = px.imshow(conv)
            f.add_trace(go.Scatter(x=[max_coords[1]], y=[max_coords[0]], mode='markers', marker=dict(size=10, color='red')))
            f.show()

        conv_top_left = convolve2d(gray, kernel_top_left, mode='same', fillvalue=0)
        coords_top_left = np.unravel_index(np.argmax(conv_top_left), conv_top_left.shape)

        conv_top_right = convolve2d(gray, kernel_top_right, mode='same')
        coords_top_right = np.unravel_index(np.argmax(conv_top_right), conv_top_right.shape)
        
        conv_bottom_right = convolve2d(gray, kernel_bottom_right, mode='same')
        coords_bottom_right = np.unravel_index(np.argmax(conv_bottom_right), conv_bottom_right.shape)

        conv_bottom_left = convolve2d(gray, kernel_bottom_left, mode='same')
        coords_bottom_left = np.unravel_index(np.argmax(conv_bottom_left), conv_bottom_left.shape)

        x_coords = [coords_top_left[1], coords_top_right[1], coords_bottom_right[1], coords_bottom_left[1]]
        y_coords = [coords_top_left[0], coords_top_right[0], coords_bottom_right[0], coords_bottom_left[0]]
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        if (False):
            f = px.imshow(gray)
            f.show()

            show_convolution(conv_top_left, coords_top_left)
            show_convolution(conv_top_right, coords_top_right)
            show_convolution(conv_bottom_right, coords_bottom_right)
            show_convolution(conv_bottom_left, coords_bottom_left)

            plt.imshow(gray, cmap='gray')
            plt.scatter(x_coords, y_coords, c='r')
            plt.show()

        # self.create_plot_with_conv_image(self.image, kernel_top_left, conv_top_left, coords_top_left).show()

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))


        # convolution = convolve2d(gray, gray, mode='same', boundary='wrap')
        # coords = np.unravel_index(np.argmax(convolution), convolution.shape)
        # show_convolution(convolution, coords)

        # adjust for downsample factor and corp
        center_x = int(center_x / downsample_factor)
        center_x += corp_center_x - crop_width // 2

        center_y = int(center_y / downsample_factor)
        center_y += corp_center_y - corp_height // 2

        return center_x, center_y


    def extract_circular_region(self):
        mask = np.zeros(self.image_grey_pixel_value.shape[:2], dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1) # pylint: disable=no-member
        masked = cv2.bitwise_and(self.image_grey_pixel_value, self.image_grey_pixel_value, mask=mask) # pylint: disable=no-member
        return masked

    def create_heatmap(self, square_region_override=None):
        # Extract the circular region
        circular_region = self.extract_circular_region()
        
        # Create a bounding box for the circular region
        x_start, y_start = max(0, self.center[0] - self.radius), max(0, self.center[1] - self.radius)
        x_end, y_end = min(self.image_grey_pixel_value.shape[1], self.center[0] + self.radius + 1), min(self.image_grey_pixel_value.shape[0], self.center[1] + self.radius + 1)
        
        # Extract the square region containing the circle
        square_region = circular_region[y_start:y_end, x_start:x_end]

        if square_region_override:
            x_size, y_size = square_region_override[0], square_region_override[1]
            x_start, y_start = max(0, self.center[0] - x_size // 2), max(0, self.center[1] - y_size // 2)
            x_end, y_end = self.center[0] + x_size // 2 + 1, self.center[1] + y_size // 2 + 1
            self.square_region = self.image_grey_pixel_value[y_start:y_end, x_start:x_end]
        
        # Create the circular mask
        y, x = np.ogrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        mask = x*x + y*y <= self.radius*self.radius
        
        # Crop the mask to match the square region
        mask = mask[:square_region.shape[0], :square_region.shape[1]]
        
        # Apply the mask
        masked_values = np.where(mask, square_region, np.nan)
        
        # Create coordinates for the heatmap
        x_coords = np.linspace(-self.radius/self.dpi*25.4, self.radius/self.dpi*25.4, masked_values.shape[1])
        y_coords = np.linspace(self.radius/self.dpi*25.4, -self.radius/self.dpi*25.4, masked_values.shape[0])
        
        return masked_values, x_coords, y_coords

    def create_plot_with_conv_image(self, image, kernel, conv, max_coords):
        # Create subplots
        size= 300
        fig = make_subplots(rows=1, cols=3, column_widths=np.repeat(size, 3).tolist(), row_heights=np.repeat(size, 1).tolist(), subplot_titles=("Original Image", "Kernel", "Convolution"))
        
        # Plot 1: Original image with circle
        fig.add_trace(go.Image(z=image), row=1, col=1)
        fig.update_yaxes(title_text="y [px]", title_font=mps.font, col=1)
        fig.update_xaxes(title_text="x [px]", title_font=mps.font, row=1, col=1)
        
        # Plot 2: Kernel
        # mirror the kernel along the y axis to match the image
        kernel = np.flip(kernel, axis=1)
        fig.add_trace(go.Heatmap(z=kernel, colorscale='Greys', showscale=False), row=1, col=2)
        # fig.update_yaxes(title_text="y [px]", title_font=mps.font, col=2)
        fig.update_xaxes(title_text="x [px]", title_font=mps.font, row=1, col=2)


        # Plot 3: Convolution
        # flip along the x axis to match the image
        conv = np.flip(conv, axis=0)
        # consider coordinates are flipped
        max_coords = (conv.shape[0] - max_coords[0], max_coords[1])
        fig.add_trace(go.Heatmap(z=conv, colorscale='Viridis', colorbar=dict(title='convolution value g',  ypad=0, titlefont=mps.font)), row=1, col=3)
        # fig.update_yaxes(title_text="y [px]", title_font=mps.font, col=3)
        fig.update_xaxes(title_text="x [px]", title_font=mps.font, row=1, col=3)
        # add title for colorbar
        fig.update_coloraxes(colorbar=dict(title='test'))

        fig.add_trace(go.Scatter(x=[max_coords[1]], y=[max_coords[0]], mode='markers', marker=dict(size=10, color='red'), showlegend=False), row=1, col=3)

        # Update layout
        fig.update_layout(height=300, width=900, font=mps.font, margin=mps.margin)
        
        fig.write_image('show_conv.pdf')

        return fig
    
    def get_square_region_after_processing(self):
        return self.inv_green_saunders(self.square_region, *self.selected_film_calibration["pars"])

    def create_plots(self, square_region_override=None):
        # Create subplots
        fig = make_subplots(rows=1, cols=3, column_widths=np.repeat(300, 3).tolist(), row_heights=np.repeat(120, 1).tolist())
        
        fig.layout.update(
            xaxis=dict(domain=[0.0, 0.28]),
            xaxis2=dict(domain=[0.3, 0.58]),
            xaxis3=dict(domain=[0.70, 1.0]) # leave some space for the colorbar of the haetmap
        )



        # Plot 1: Original image with circle
        if square_region_override:
            x_size, y_size = square_region_override[0], square_region_override[1]
            cv2.rectangle(self.image, (self.center[0] - x_size // 2, self.center[1] - y_size // 2), (self.center[0] + x_size // 2, self.center[1] + y_size // 2), (255, 0, 0), 2) # pylint: disable=no-member
        else:
            cv2.circle(self.image, self.center, self.radius, (255, 0, 0), 2) # pylint: disable=no-member
            
        fig.add_trace(go.Image(z=self.image), row=1, col=1)
        
        # Plot 2: Heatmap
        masked_values, self.x_coords, self.y_coords = self.create_heatmap(square_region_override)
        
        # if self.selected_film_calibration == self.film_calibration_data["EBT3_new_METAS_ImageJwRGB"]:

        #     # Do = ufloat(5.19, 0.25)
        #     # PVmin = ufloat(8.45, 3.07)
        #     # PVmax = ufloat(167.12, 0.41)
        #     # beta = ufloat(-0.87, 0.02)

        #     dose_values = self.inv_green_saunders(masked_values, Do, PVmin, PVmax, beta)
        # else:
        dose_values = self.inv_green_saunders(masked_values, *self.selected_film_calibration["pars"])

        dose_values = dose_values * self.dose_correction_factor_lamination_layer / self.dose_correction_factor_relative_efficiency
        
        # Set values above max dose to zero
        dose_nominal_values = unp.nominal_values(dose_values)

        # set values above max dose to zero
        dose_values[dose_nominal_values > self.max_dose] = 0

        # extract the non zero values
        non_zero_values = dose_values[dose_nominal_values > 0]

        dose_nominal_values = unp.nominal_values(dose_values)
        self.heatmap = dose_nominal_values

        dose_mean = np.sum(non_zero_values) / len(non_zero_values)

        self.dose_mean = dose_mean
        self.dose_std = dose_mean
        error_in_percent = self.dose_std / self.dose_mean * 100

        print(f"Mean: {self.dose_mean:.5f} Gy, error: {self.dose_std:.5f}, error in percent: {error_in_percent:.10f}%")

        
        fig.add_trace(
            go.Heatmap(
                z=dose_nominal_values,
                x=self.x_coords,
                y=self.y_coords,
                colorscale='Viridis',
                colorbar=dict(
                    x=0.58,  # Adjust the horizontal position (relative to the figure width)
                    xanchor='left',  # Align the left edge of the color bar at the specified `x`
                )
            ),
            row=1, col=2
        )
        
        # Plot 3: Horizontal profile through dose values
        self.horizontal_profile = dose_nominal_values[dose_nominal_values.shape[0] // 2, :]
        # delete zero values
        self.horizontal_profile = self.horizontal_profile[self.horizontal_profile != 0]
        fig.add_trace(go.Scatter(x=self.x_coords, y=self.horizontal_profile, mode='lines', name='horizontal', line=dict(width=1.2), showlegend=True), row=1, col=3)

        # vertical profile
        self.vertical_profile = dose_nominal_values[:, dose_nominal_values.shape[1] // 2]
        # delete zero values
        self.vertical_profile = self.vertical_profile[self.vertical_profile != 0]
        fig.add_trace(go.Scatter(x=self.y_coords, y=self.vertical_profile, mode='lines', name='vertical profile',  line=dict(width=1.2), zorder=-2), row=1, col=3)

        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=90,
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        ))

        # Update layout
        fig.update_xaxes(title_text="x [mm]", title_font=mps.font, row=1, col=2)
        fig.update_yaxes(title_text="y [mm]", title_font=mps.font, row=1, col=2)
        fig.update_xaxes(title_text="x,y [mm]", title_font=mps.font, row=1, col=3)
        fig.update_yaxes(title_text="Dose [Gy]", title_font=mps.font, row=1, col=3)

        fig.update_layout(
            annotations=[
                dict(
                    text="Original Image",
                    x=0.1,
                    y=1.11,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=mps.font
                ),
                dict(
                    text=f"Dose Heatmap, mean: {self.dose_mean:.3f} ± {self.dose_std:.3f} Gy",
                    x=0.4440,  # Center of the second subplot
                    y=1.11,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=mps.font
                )
            ]
        )

        fig.update_layout(height=300, width=1000, font=mps.font, margin=mps.margin)
        
        return fig

    def process_film_scan(self, center_override=None, square_region_override=None) -> go.Figure:
        
        # Read the image, according to pixel value grey scale conversion from paper
        self.image = io.imread(str(self.scan_file))
        self.image_grey_pixel_value = np.dot(np.array(self.image[..., :3]), [0.2989, 0.5870, 0.1140])
        # test = cv2.imread(str(self.scan_file), cv2.IMREAD_GRAYSCALE)

        if self.image_grey_pixel_value is None:
            raise ValueError(f"Could not read image file: {self.scan_file}")
        
        # Find the center of the dark rectangle
        if center_override is not None:
            self.center = center_override
        else:
            self.center = self.find_dark_rectangle_center()

        
        
        # Create the plots (using your original plotting code)
        self.fig = self.create_plots(square_region_override)

        self.results = {
            "center": self.center,
            "dose_mean": self.dose_mean,
            "dose_std": self.dose_std,
        }
        
        return self.fig
    
    def calculate_lamination_layer_correction_factor(self, center_override, size_override, show_plots=False, double_layer_side='left', left_range=None, right_ragne=None, show_debug_plots=False) -> ufloat:
        fig = self.process_film_scan(center_override, size_override)
        if show_plots:
            fig.add_vline(x=center_override[0], line_color='red', line_width=1, row=1, col=1)
            fig.add_hline(y=center_override[1], line_color='red', line_width=1, row=1, col=1)
            # fig.write_image('v1_lamination_layer_difficulties.pdf')
            fig.show()

        # delete zero values
        # horizontal_profile = film_scanner.horizontal_profile
        heatmap = self.get_square_region_after_processing()
        fig = px.imshow(heatmap, aspect='auto')
        if show_debug_plots:
            fig.show()

        horizontal_profile = heatmap.mean(axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.x_coords,
            y=horizontal_profile,
            mode='lines',
            name='Horizontal Dose Profile'
        ))

        # take the left and the right side of the profile and calculate the mean
        middle_index = len(horizontal_profile) // 2
        middle = self.x_coords[middle_index]

        if (left_range is None) and (right_ragne is None):
            # draw a line at the middle
            fig.add_vline(x=middle, line_dash='dash', line_color='black')
            left_side = horizontal_profile[:middle_index]
            right_side = horizontal_profile[middle_index:]
        else:
            # left and right range are in mm, find the colosest indices in the x_coords for both ranges
            left_range_indices = np.where((self.x_coords >= left_range[0]) & (self.x_coords <= left_range[1]))
            right_range_indices = np.where((self.x_coords >= right_ragne[0]) & (self.x_coords <= right_ragne[1]))
            left_side = horizontal_profile[left_range_indices]
            right_side = horizontal_profile[right_range_indices]

            fig.add_vrect(x0=left_range[0], x1=left_range[1], fillcolor="green", opacity=0.2, layer="below", line_width=0)

            fig.add_vrect(x0=right_ragne[0], x1=right_ragne[1], fillcolor="blue", opacity=0.2, layer="below", line_width=0)


        left_side = ufloat(left_side.mean(), left_side.std())
        right_side = ufloat(right_side.mean(), right_side.std())

        # print(f'mean left: {left_side.nominal_value} Gy, mean right: {right_side.nominal_value} Gy')

        # due to the lamination layer we overestimate the dose
        # the proton looses energy in the lamination, the proton gets slower and looses more energy within the active layer (bragg peak)
        lamination_layer_correction = None
        if double_layer_side == 'left':
            lamination_layer_correction = right_side / left_side
        elif double_layer_side == 'right':
            lamination_layer_correction = left_side / right_side
        # accoring to gaussian error propagation the standard deviation is
        # print(f'lamination layer correction: {lamination_layer_correction.nominal_value} ± {lamination_layer_correction.std_dev} Gy')

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis_title='x [mm]',
            yaxis_title='Dose [Gy]',
        )

        fig.update_layout(
            width=mps.width,
            height=mps.height,
            font=mps.font
        )

        if show_plots:
            pwd = Path(__file__).parent
            # fig.write_image(pwd / '../../../latex/images/s3_lamination_layer_profile.pdf')
            fig.show()

        return lamination_layer_correction
    
    def calc_LET(self, surface_energy):   #surface_energy in MeV
        a = 4.1e5   #keV/um
        b = 2.88    #1/MeV    
        c = 22.5    #keV/um
        d = 0.142   #1/MeV
        return a*umath.exp(-b*surface_energy) + c*umath.exp(-d*surface_energy) #keV/um

    def calc_relative_efficiency(self, LET):
        A = ufloat(0.0117, 0.0016)
        B = ufloat(1.01, 0.04)
        return 1-A*(LET**B) #, (1-A*(LET**B))*(LET**B)*np.sqrt(err_A**2+(A*np.log(B)*err_B)**2))

    def get_relative_efficiency(self, beam_energy):
        let = self.calc_LET(beam_energy)
        return self.calc_relative_efficiency(let)
    

    def green_saunders(self, dose, Do, PVmin, PVmax, beta):
        return np.where(
            dose == 0, 1e20, PVmin + (PVmax - PVmin) / (1 + (Do / dose) ** beta)
        )

    def inv_green_saunders(self, pixel_value, Do, PVmin, PVmax, beta):
        # if (type(PVmin) == uncertainties.core.Variable) or (type(PVmax) == uncertainties.core.Variable):
        #     if (pixel_value < PVmin.n).any() or (pixel_value > PVmax.n).any():
        #         raise ValueError(f"Pixel value outside calibration range, pixel_value={pixel_value}")
        val = Do * ((pixel_value - PVmin) / (PVmax - pixel_value)) ** (1 / beta)
        return np.nan_to_num(val)
    
    def get_film_calibration_data_keys(self):
        return list(self.film_calibration_data.keys())

    def define_film_calibration_data(self):
        return {
            "EBT3_old_Bologna": {
                "pars": np.array([4.7956, 45.5929, 159.0524, -0.9054]),
                "calib_str": "\tOld EBT-3\n\tExp. 02.03.2023\n\tLot 030220102\n\tCalibration at Bologna with GammaCell\n\tdd.mm.2023",
            },
            "EBT3_new_Linac_ImageJ": {
                "pars": np.array([5.62, 16.82, 159.27, -0.86]),
                "calib_str": "\tNew EBT-3\n\tExp. 02.10.2024\n\tLot 10032202\n\tCalibration at Inselspital with Clinical Linac (ImageJ non-weighted Pixel Values)\n\t25.09.2023",
            },
            "EBT3_new_Linac_Py": {
                "pars": np.array([4.99, 11.73, 168.79, -0.88]),
                "calib_str": "\tNew EBT-3\n\tExp. 02.10.2024\n\tLot 10032202\n\tCalibration at Inselspital with Clinical Linac (Python weighted Pixel Values)\n\t25.09.2023",
            },
            "EBT3_new_METAS_ImageJwRGB": {
                "pars": np.array([5.19, 8.40, 167.13, -0.87]),
                "calib_str": "\tNew EBT-3\n\tExp. 02.10.2024\n\tLot 10032202\n\tCalibration at METAS with 60-Co source (ImageJ weighted Pixel Values)\n\t07.11.2023",
            },
            "EBT3_old_METAS_ImageJwRGB": {
                "pars": np.array([5.22, 16.46, 155.37, -0.87]),
                "calib_str": "\tOld EBT-3\n\tExp. 02.03.2023\n\tLot 030220102\n\tCalibration at METAS with 60-Co source (ImageJ weighted Pixel Values)\n\t07.11.2023",
            },
            "EBT3_new_combinedLiMet_ImageJwRGB": {
                "pars": np.array([5.09, 10.13, 167.96, -0.88]),
                "calib_str": "\tNew EBT-3\n\tExp. 02.10.2024\n\tLot 10032202\n\tFitted to METAS and Linac data combined (ImageJ weighted Pixel Values)\n\t15.11.2023",
            },
            "HDV2": {
                "pars": np.array([144.58, 47.81, 211.03, -0.85]),
                "calib_str": "\tNew HD-V2\n\tExp. __\n\tLot ___\n\tCalibration at ISOF Bologna with GammaCell (ImageJ weighted Pixel Values)\n\t19.12.2023",
            },
        }

