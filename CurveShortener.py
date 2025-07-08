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

    def save_curve(self, curve, prefix="save_curve_"):
        filePath = prefix + ".txt"
        with open(filePath, mode='w', encoding='UTF-8') as output:
            for p in curve:
                print(p[0], ",", p[1], file=output)
            output.close()

    def set_preserve_curve_length(self):
        self.preserve_curve_length = True
        
    def set_use_lsq_resample(self):
        self.use_lsq_resample = True
        
    def set_save_additional_info(self):
        self.save_additional_info = True

    def has_big_deviation_step(self, curve):
        if len(curve) == 0:
            return False
        length_list = geom.get_curve_steps(curve)
        min_length = min(length_list)
        max_length = max(length_list)
        dl = (100.0*(max_length - min_length))/max_length
        #print("dL=", dl, " minL=", min_length, " maxL=", max_length)
        return dl > 5.0


    def run(self, curve):
        
        curvature_integral = geom.get_curvature_over_curve(curve, geom.get_curvature(curve))
        print(curvature_integral)
        if curvature_integral < 0:
            curve = np.flip(curve, axis=1)

        curve = geom.shift_curve(curve, curve_ops.get_curve_size(curve) // 2)
        
        window_length = 5
        poly_order = 2
        number_of_smooth = 1
        curve = geom.resample_by_lsq(curve)
        density_initial = geom.get_curve_length(curve)/curve_ops.get_curve_size(curve)
        number_of_smooth = 1
        curvature_ratio_history = []
        arclen_history = []
        single_upsampling = False
        # curve at previous step
        prev_curve = curve_ops.get_empty_curve()

        iter = 0
        finished = False
        while not finished:
            uniform_sampling_needed = self.has_big_deviation_step(curve)
            if uniform_sampling_needed:
                # curve points are distributed not evenly - should be rearranged 
                if self.use_lsq_resample:
                    curve = geom.resample_by_lsq(curve, n=len(curve))
                else:
                    curve = geom.resample_by_interpolation(curve)
            
            if self.is_circle:
                cx, cy = geom.get_curve_center(curve)
                r0 = geom.get_mean_distances_to_point(cx, cy, curve)
            
            #perform smoothing of curve using Savitzky-Golay
            smoothed_curve = geom.smoothen_curve(curve, window_length, poly_order, number_of_smooth)
            # if curve has circle shape we use homothety transformation to compensate 
            # curve slightly shrinks after smoothing
            if self.is_circle:
                r1 = geom.get_mean_distances_to_point(cx, cy, smoothed_curve)
                smoothed_curve = geom.homothety_transform(smoothed_curve, cx, cy, r0/r1)

            curvature = geom.get_curvature(smoothed_curve, w=window_length, po=poly_order)
            
            if self.save_additional_info:
                arclen_history.append(geom.get_curve_length(curve))
                max_curv = max(curvature)
                if max_curv != 0.0:
                    curvature_ratio_history.append(min(curvature)/max_curv)
            
            # detect and handle singularities
            singular_groups = singular.detect(curvature)
            if len(singular_groups) > 0:
                if self.save_additional_info:
                    self.save_list(curvature, prefix="curvature_"+str(iter))
                length_of_singular_part = 0
                count_of_singular_part = 0
                for group in singular_groups:
                    s = geom.get_part_curve_length(smoothed_curve, group[0], group[1])
                    length_of_singular_part += s
                    n = group[1] - group[0] + 1
                    #print("n=", n, " s=", s, " density=", s/n)
                    count_of_singular_part += n
                
                #print("singular Part n=", count_of_singular_part, " l=", length_of_singular_part, " l/n=", length_of_singular_part/count_of_singular_part)
                
                if curve_ops.get_curve_size(prev_curve) > 0:
                    smoothed_curve = prev_curve.copy()
                
                length_of_regular_part = geom.get_excl_curve_length(smoothed_curve, singular_groups)
                count_of_regular_part = curve_ops.get_curve_size(smoothed_curve) - count_of_singular_part
                #print("regular Part n=", count_of_regular_part, " l=", length_of_regular_part, " l/n=", length_of_regular_part/count_of_regular_part)
                density_of_singular_part = length_of_singular_part/count_of_singular_part
                density_of_regular_part = length_of_regular_part/count_of_regular_part
                if density_of_singular_part < density_of_regular_part:
                    new_num = int((density_of_singular_part*curve_ops.get_curve_size(smoothed_curve))/density_of_regular_part)
                    #print("Resampling: num=", curve_ops.get_curve_size(smoothed_curve), " new_num=", new_num)
                    curve = geom.resample_by_lsq(smoothed_curve, n=new_num)
                    curvature = geom.get_curvature(curve, w=window_length, po=poly_order)
                    #print("After resampling number of points=", curve_ops.get_curve_size(curve))
                    if self.callBack is not None:
                        finished = self.callBack(curve, curvature, iter, self.is_circle, self.callBackObj)
                    prev_curve = curve.copy()
                else:
                    print("density_singular=", density_of_singular_part, " density_regular=", density_of_regular_part)
                    return
                iter += 1
                continue
                    
            # user supplied callback function is called if set
            if self.callBack is not None:
                finished = self.callBack(smoothed_curve, curvature, iter, self.is_circle, self.callBackObj)
            else:
                if self.max_iterations is not None:
                    finished = iter >= self.max_iterations

            prev_curve = curve.copy()
            curve = self.get_next_curve(smoothed_curve, curvature)
            iter += 1

        if self.save_additional_info:
            self.save_list(arclen_history, prefix="arclen_"+str(iter))
            self.save_list(curvature_ratio_history, prefix="curvature_ratio_history_"+str(iter))

    
    def get_next_curve(self, curve, curvature):
        a = 0.0
        if self.preserve_curve_length:
            l  = geom.get_curve_length(curve)
            a = 2.0*np.pi/l

        is_circle = geom.is_circle(curve)
        if is_circle != self.is_circle:
            self.is_circle = is_circle

        normal_unit_field = geom.get_normal_unit_field(curve)
        return np.subtract(curve, np.multiply(normal_unit_field, np.subtract(curvature, a)))

    def get_next_curve2(self, curve, curvature):
        a = 0.0
        if self.preserve_curve_length:
            l  = geom.get_curve_length(curve)
            a = 2.0*np.pi/l

        is_circle = geom.is_circle(curve)
        if is_circle != self.is_circle:
            self.is_circle = is_circle

        normal_unit_field = geom.get_normal_unit_field(curve)
        next_curve = []        
        for i in range(0, len(curve)):
            p = curve[i]
            n = normal_unit_field[i]
            x = p[0] - (curvature[i]-a)*n[0]
            y = p[1] - (curvature[i]-a)*n[1]
            next_curve.append((x, y))
        return next_curve
