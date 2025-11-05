import math
import cv2
import numpy as np
from scipy import signal
from scipy import interpolate
import geometry as geom
import image_operations
import curve_operations as curve_ops
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
        self.use_lsq_resample = False
        self.save_additional_info = False
        self.window_length = 5
        self.poly_order = 2
        self.number_of_smooth = 1
        self.max_iterations_without_downsampling = 10


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
        
    def set_use_lsq_resample(self):
        """
        Sets usage of LSQ for resampling.
        Otherwise interpolation is used.
        """
        self.use_lsq_resample = True
        
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
        if curve_ops.get_curve_size(curve):
            return False
        length_list = geom.get_curve_steps(curve)
        min_length = min(length_list)
        max_length = max(length_list)
        dl = (100.0*(max_length - min_length))/max_length
        #print("dL=", dl, " minL=", min_length, " maxL=", max_length)
        return dl > 5.0


    def get_density_for_singular_part(self, parts, curve_length_list):
        """
        Calculates number of points per length for given parts of curve (singular part of curve).
        :param parts: list of pairs of positions in curve that contain singular part of curve.
        :param curve_length_list: list of lengths of curve segments.
        :return curve points density
        """
        length = 0.0
        count = 0
        for part in parts:
            length += geom.get_part_curve_length_from_list(curve_length_list, part[0], part[1])
            count += part[1] - part[0] + 1
        return count/length


    def get_density_for_regular_part(self, parts, curve, curve_length_list):
        """
        Calculates number of points per length for given parts of curve (regular part of curve).
        :param parts: list of pairs of positions in curve that contain singular part of curve.
        :param curve: curve data.
        :param curve_length_list: list of lengths of curve segments
        :return curve points density
        """
        length = 0.0
        count = 0
        for part in parts:
            length += geom.get_part_curve_length_from_list(curve_length_list, part[0], part[1])
            count += part[1] - part[0] + 1
        return (curve_ops.get_curve_size(curve) - count)/(geom.get_curve_length_from_list(curve_length_list) - length)


    def run(self, curve):
        """
        Applies curve shortening flow to given curve.
        :param curve: curve data.
        """
        curvature_integral = geom.get_curvature_over_curve(curve, geom.get_curvature(curve))
        #print(curvature_integral)
        if curvature_integral < 0:
            curve = np.flip(curve, axis=1)

        curve = geom.shift_curve(curve, curve_ops.get_curve_size(curve) // 2)
        
        curve = geom.resample_by_lsq(curve)
        curvature_ratio_history = []
        arclen_history = []
        accumulated_curves = []
        # curve at previous step
        prev_curve = curve_ops.get_empty_curve()
        # initial number of points per curve length
        num_points_per_length = curve_ops.get_curve_size(curve)/geom.get_curve_length(curve)
        # variable to count number of iterations after downsampling
        counter = 0
        iteration = 0
        finished = False
        while not finished:
            if self.has_big_deviation_step(curve):
                # curve points are distributed not evenly - should be resampled
                if self.use_lsq_resample:
                    curve = geom.resample_by_lsq(curve)
                else:
                    curve = geom.resample_by_interpolation(curve)
            
            curve = geom.smoothen_with_compensation_curve(curve, w=5, po=2, iterations=1)

            curve_length_array = geom.get_curve_length_list(curve)
            curve_length = geom.get_curve_length_from_list(curve_length_array)
            curvature = geom.get_curvature(curve, w=self.window_length, po=self.poly_order)
            # if number of iterations without downsampling is big enough
            if counter == self.max_iterations_without_downsampling:
                new_num = int(num_points_per_length * curve_length)
                if curve_ops.get_curve_size(curve) + 1 < new_num:
                    curve = geom.resample_by_interpolation(curve, n=new_num)
                    curve_length_array = geom.get_curve_length_list(curve)
                    curve_length = geom.get_curve_length_from_list(curve_length_array)
                    curvature = geom.get_curvature(curve, w=self.window_length, po=self.poly_order)

            
            if self.save_additional_info:
                arclen_history.append(curve_length)
                max_curv = max(curvature)
                if max_curv != 0.0:
                    curvature_ratio_history.append(min(curvature)/max_curv)
            
            # detect and handle singularities
            if curve_ops.get_curve_size(prev_curve) > 0:
                singular_groups = singular.detect(curvature)
                if len(singular_groups) > 0:
                    if self.save_additional_info:
                        self.save_list(curvature, prefix="singular_curvature_"+str(iteration))

                    density_of_singular_part = self.get_density_for_singular_part(singular_groups, curve_length_array)
                    
                    curve = prev_curve.copy()
                    curve_length_array = geom.get_curve_length_list(curve)
                    curve_length = geom.get_curve_length_from_list(curve_length_array)
                    
                    density_of_regular_part = self.get_density_for_regular_part(singular_groups, curve, curve_length_array)
                    
                    new_num = int((density_of_regular_part*curve_ops.get_curve_size(curve))/density_of_singular_part)
                    # resample for new_num points
                    curve = geom.resample_by_interpolation(curve, n=new_num)
                    curve_length_array = geom.get_curve_length_list(curve)
                    curve_length = geom.get_curve_length_from_list(curve_length_array)
                    curvature = geom.get_curvature(curve, w=self.window_length, po=self.poly_order)
                    counter = 0
                    
            # user supplied callback function is called if set
            if self.callBack is not None:
                finished = self.callBack(curve, curvature, curve_length, iteration, self.is_circle, self.callBackObj)
            else:
                if self.max_iterations is not None:
                    finished = iteration >= self.max_iterations

            prev_curve = curve.copy()
            #s0 = geom.get_convex_curve_square(curve)

            curve = self.get_next_curve(curve, curvature, curve_length)

            #s1 = geom.get_convex_curve_square(curve)
            #print(curve_ops.get_curve_size(curve), " --> ", curve_length)
            #if s0 > s1:
            #    print("s0=", "{:.5f}".format(s0), " > s1=", "{:.5f}".format(s1), "s0=", (100.0*math.fabs(s0-s1))/s0 )
            #else:
            #    print("s0=", "{:.5f}".format(s0), " <= s1=", "{:.5f}".format(s1), "s0=", (100.0 * math.fabs(s0 - s1)) / s0)

            iteration += 1
            counter += 1

        if self.save_additional_info:
            self.save_list(arclen_history, prefix="arclen_"+str(iteration))
            self.save_list(curvature_ratio_history, prefix="curvature_ratio_history_"+str(iteration))

    def get_next_curve(self, curve : np.ndarray, curvature : np.ndarray, curve_length : float) -> np.ndarray :
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

        return np.subtract(curve, np.multiply(geom.get_normal_unit_field(curve), np.subtract(curvature, a)))

