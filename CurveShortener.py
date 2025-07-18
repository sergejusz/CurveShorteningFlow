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
        self.iter = 0
        self.callBack = None
        self.callBackObj = None
        self.max_iterations = None
        # set True to preserve length of curve
        self.preserve_curve_length = False
        self.is_circle = False
        self.use_lsq_resample = False
        self.save_additional_info = False
        self.window_length = 5
        self.poly_order = 2
        self.number_of_smooth = 1


    def setMaxIterations(self, iterations):
        self.max_iterations = iterations

    def setCallBack(self, callBackFcn, obj=None):
        self.callBack = callBackFcn
        self.callBackObj = obj
       
    def save_list(self, dataList, prefix="save_values_"):
        filePath = prefix + ".txt"
        with open(filePath, mode='w', encoding='UTF-8') as output:
            for v in dataList:
                print(v, file=output)
            output.close()

    def set_preserve_curve_length(self):
        self.preserve_curve_length = True
        
    def set_use_lsq_resample(self):
        self.use_lsq_resample = True
        
    def set_save_additional_info(self):
        self.save_additional_info = True

    def has_big_deviation_step(self, curve):
        if curve_ops.get_curve_size(curve):
            return False
        length_list = geom.get_curve_steps(curve)
        min_length = min(length_list)
        max_length = max(length_list)
        dl = (100.0*(max_length - min_length))/max_length
        #print("dL=", dl, " minL=", min_length, " maxL=", max_length)
        return dl > 5.0


    def get_density_for_singular_part(self, parts, curve_length_list):
        length = 0.0
        count = 0
        for part in parts:
            length += geom.get_part_curve_length_from_list(curve_length_list, part[0], part[1])
            count += part[1] - part[0] + 1
        return count/length


    def get_density_for_regular_part(self, parts, curve, curve_length_list):
        s = 0.0
        count = 0
        for part in parts:
            s += geom.get_part_curve_length_from_list(curve_length_list, part[0], part[1])
            count += part[1] - part[0] + 1
        return (curve_ops.get_curve_size(curve) - count)/(geom.get_curve_length_from_list(curve_length_list) - s)


    def smoothen_curve(self, curve):
        # get data that we need for compensation of shrinking effect of smoothing
        # for that we use average distance from curve points to the center of a curve
        cx, cy = geom.get_curve_center(curve)
        r0 = geom.get_mean_distances_to_point(cx, cy, curve)
        
        #perform smoothing of curve using Savitzky-Golay
        curve = geom.smoothen_curve(curve, self.window_length, self.poly_order, self.number_of_smooth)
        # we use homothety transformation to compensate that
        # curve slightly shrinks after smoothing
        r1 = geom.get_mean_distances_to_point(cx, cy, curve)
        return geom.homothety_transform(curve, cx, cy, r0/r1)


    def run(self, curve):
        
        curvature_integral = geom.get_curvature_over_curve(curve, geom.get_curvature(curve))
        #print(curvature_integral)
        if curvature_integral < 0:
            curve = np.flip(curve, axis=1)

        curve = geom.shift_curve(curve, curve_ops.get_curve_size(curve) // 2)
        
        curve = geom.resample_by_lsq(curve)
        curvature_ratio_history = []
        arclen_history = []
        # curve at previous step
        prev_curve = curve_ops.get_empty_curve()
        prev_curve_length = 0.0

        iter = 0
        finished = False
        while not finished:
            if self.has_big_deviation_step(curve):
                # curve points are distributed not evenly - should be resampled
                if self.use_lsq_resample:
                    curve = geom.resample_by_lsq(curve)
                else:
                    curve = geom.resample_by_interpolation(curve)
            
            curve = self.smoothen_curve(curve)

            curve_length_array = geom.get_curve_length_list(curve)
            curve_length = geom.get_curve_length_from_list(curve_length_array)
            curvature = geom.get_curvature(curve, w=window_length, po=poly_order)
            
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
                        self.save_list(curvature, prefix="singular_curvature_"+str(iter))

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
                    curvature = geom.get_curvature(curve, w=window_length, po=poly_order)
                    
            # user supplied callback function is called if set
            if self.callBack is not None:
                finished = self.callBack(curve, curvature, iter, self.is_circle, self.callBackObj)
            else:
                if self.max_iterations is not None:
                    finished = iter >= self.max_iterations

            prev_curve = curve.copy()
            curve = self.get_next_curve(curve, curvature, curve_length)
            iter += 1

        if self.save_additional_info:
            self.save_list(arclen_history, prefix="arclen_"+str(iter))
            self.save_list(curvature_ratio_history, prefix="curvature_ratio_history_"+str(iter))

    
    def get_next_curve(self, curve, curvature, curve_length):
        a = 0.0
        if self.preserve_curve_length:
            a = 2.0*np.pi/curve_length

        is_circle = geom.is_circle(curve)
        if is_circle != self.is_circle:
            self.is_circle = is_circle

        return np.subtract(curve, np.multiply(geom.get_normal_unit_field(curve), np.subtract(curvature, a)))

