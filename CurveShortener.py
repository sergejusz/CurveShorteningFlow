import math
import cv2
import numpy as np
from scipy import signal
from scipy import interpolate
import geometry as geom
import image_operations
import singularity_areas_detection as singular

class CurveShortener():
    
    def __init__(self):
        """
        Creates curve shortening flow object.
        """
        self.callBack = None
        self.callBackObj = None
        self.max_iterations = 1
        # set True to preserve length of curve
        self.preserve_area = False
        self.is_circle = False
        self.save_additional_info = False
        self.window_length = 5
        self.poly_order = 2
        self.number_of_smooth = 1
        self.max_iterations_without_downsampling = 10
        self.check_density = 20
        self.density_threshold = 5 # percents


    def setMaxIterations(self, iterations):
        """
        Sets max. number of iterations.
        :param iterations: max. number of iterations.
        """
        self.max_iterations = iterations

    def setCallBack(self, callBackFcn, obj=None):
        """
        Sets function that is called at each iteration.
        :param callBackFcn: callback function object.
        :param obj: object containing callback specific information.
        """
        self.callBack = callBackFcn
        self.callBackObj = obj
       
    def save_list(self, dataList, prefix="save_values_"):
        """
        Saves list values to text file.
        :param dataList: list of values..
        :param prefix: prefix for file name.
        """
        filePath = prefix + ".txt"
        with open(filePath, mode='w', encoding='UTF-8') as output:
            for v in dataList:
                print(v, file=output)
            output.close()

    def set_preserve_area(self):
        """
        Sets area preservation flag.
        """
        self.preserve_area = True
        
    def set_save_additional_info(self):
        """
        Applies curve shortening flow operation to given curve,
        """
        self.save_additional_info = True

    def has_big_deviation_step(self, curve):
        """
        Detects if variance of lengths of curve segments is bigger than threshold.
        :param curve: curve data.
        :return True if variance of lengths of curve segments is bigger than threshold.
        """
        if geom.get_curve_size(curve):
            return False
        length_list = geom.get_curve_steps(curve)
        min_length = min(length_list)
        max_length = max(length_list)
        dl = (100.0*(max_length - min_length))/max_length
        #print("dL=", dl, " minL=", min_length, " maxL=", max_length)
        return dl > 5.0

    def run(self, curve):
        """
        Applies curve shortening flow to given curve.
        :param curve: curve data.
        """
        time_step = 1.0
        curvature_integral = geom.get_curvature_over_curve(curve, geom.get_curvature(curve))
        if curvature_integral < 0:
            curve = np.flip(curve, axis=1)

        curve = geom.shift_curve(curve, geom.get_curve_size(curve) // 2)
        
        curve = geom.resample_by_interpolation(curve)
        curvature_ratio_history = []
        arclen_history = []
        # curve at previous step
        prev_curve = geom.get_empty_curve()
        # number of points per curve length at the beginning
        density_initial = geom.get_curve_size(curve)/geom.get_curve_length(curve)
        # variable to count number of iterations after downsampling
        counter = 0
        iteration = 0
        finished = False
        while not finished:
            if self.has_big_deviation_step(curve):
                # curve points are distributed not evenly - should be resampled
                curve = geom.resample_by_interpolation(curve)
            
            curve = geom.smoothen_with_compensation_curve(curve, w=5, po=2, iterations=1)
            # check number of points per arc length
            if iteration % self.check_density == 0:
                curve_length = geom.get_curve_length(curve)
                density_now = geom.get_curve_size(curve) / curve_length
                # calculate relative density deviation
                density_deviation = ((density_now - density_initial) / density_now) * 100.0
                if density_deviation >= self.density_threshold:
                    old_num = geom.get_curve_size(curve)
                    new_num = int(density_initial * curve_length)
                    curve = geom.resample_by_interpolation(curve, n=new_num)
                    print(f"Resampled from {old_num} to {new_num}")

            curve_length_array = geom.get_curve_length_list(curve)
            curve_length = geom.get_curve_length_from_list(curve_length_array)
            curvature = geom.get_curvature(curve, w=self.window_length, po=self.poly_order)

            # if number of iterations without downsampling is big enough
            if counter == self.max_iterations_without_downsampling and time_step < 1.0:
                time_step = min(2.0 * time_step, 1.0)
                if time_step < 1.0:
                    counter = 0
                print(f"time_step increased to {time_step}")

            if self.save_additional_info:
                arclen_history.append(curve_length)
                max_curv = max(curvature)
                if max_curv != 0.0:
                    curvature_ratio_history.append(min(curvature)/max_curv)

            step_changed = False
            # detect and handle singularities
            if geom.get_curve_size(prev_curve) > 0:
                singular_groups = singular.detect(curvature)
                if len(singular_groups) > 0:
                    print("iter=", iteration, " Singular groups : ", len(singular_groups))
                    if self.save_additional_info:
                        self.save_list(curvature, prefix="singular_curvature_"+str(iteration))

                    curve = prev_curve.copy()
                    curve_length_array = geom.get_curve_length_list(curve)
                    curve_length = geom.get_curve_length_from_list(curve_length_array)
                    time_step = time_step * 0.5
                    step_changed = True
                    print(f"time_step decreased to {time_step}")
                    curvature = geom.get_curvature(curve, w=self.window_length, po=self.poly_order)
                    counter = 0
                    
            # user supplied callback function is called if set
            if self.callBack is not None:
                finished = self.callBack(curve, curvature, curve_length, iteration, self.is_circle, self.callBackObj)
            else:
                if self.max_iterations is not None:
                    finished = iteration >= self.max_iterations

            if not step_changed:
                prev_curve = curve.copy()

            curve = self.get_next_curve(curve, curvature, curve_length, time_step)

            iteration += 1
            counter += 1

        if self.save_additional_info:
            self.save_list(arclen_history, prefix="arclen_"+str(iteration))
            self.save_list(curvature_ratio_history, prefix="curvature_ratio_history_"+str(iteration))

    def get_next_curve(self, curve : np.ndarray, curvature : np.ndarray, curve_length : float, time_step : float) -> np.ndarray :
        """
        Applies single curve shortening flow operation to given curve,
        :param curve: curve data.
        :param curvature: curvature array.
        :param curve_length: length of curve.
        :return curve after shortening flow applied.
        """
        a = (2.0*np.pi)/curve_length if self.preserve_area else 0.0

        if not self.is_circle:
            self.is_circle = geom.is_circle(curve)

        return np.subtract(curve, np.multiply(geom.get_normal_unit_field(curve), np.multiply(np.subtract(curvature, a), time_step)))

