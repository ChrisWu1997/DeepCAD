import numpy as np
import matplotlib.lines as lines
import matplotlib.patches as patches
from .math_utils import rads_to_degs, angle_from_vector_to_x
from .macro import *


# FIXME: these two functions can be treated as static method
def construct_curve_from_dict(stat):
    if stat['type'] == "Line3D":
        return Line.from_dict(stat)
    elif stat['type'] == "Circle3D":
        return Circle.from_dict(stat)
    elif stat['type'] == "Arc3D":
        return Arc.from_dict(stat)
    else:
        raise NotImplementedError("curve type not supported yet: {}".format(stat['type']))


def construct_curve_from_vector(vec, start_point, is_numerical=True):
    type = vec[0]
    if type == LINE_IDX:
        return Line.from_vector(vec, start_point, is_numerical=is_numerical)
    elif type == CIRCLE_IDX:
        return Circle.from_vector(vec, start_point, is_numerical=is_numerical)
    elif type == ARC_IDX:
        res = Arc.from_vector(vec, start_point, is_numerical=is_numerical)
        if res is None: # for visualization purpose, replace illed arc with line
            return Line.from_vector(vec, start_point, is_numerical=is_numerical)
        return res
    else:
        raise NotImplementedError("curve type not supported yet: command idx {}".format(vec[0]))


#######################  base  #######################
class CurveBase(object):
    """Base class for curve. All types of curves shall inherit from this."""
    def __init__(self):
        pass

    @staticmethod
    def from_dict(stat):
        """construct curve from json data"""
        raise NotImplementedError

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        """construct curve from vector representation"""
        raise NotImplementedError

    @property
    def bbox(self):
        """compute bounding box of the curve"""
        raise NotImplementedError

    def direction(self, from_start=True):
        """return a vector indicating the curve direction"""
        raise NotImplementedError

    def transform(self, translate, scale):
        """linear transformation"""
        raise NotImplementedError

    def flip(self, axis):
        """flip the curve about axis"""
        raise NotImplementedError

    def reverse(self):
        """reverse the curve direction"""
        raise NotImplementedError

    def numericalize(self, n=256):
        """quantize curve parameters into integers"""
        raise NotImplementedError

    def to_vector(self):
        """represent curve using a vector. see macro.py"""
        raise NotImplementedError

    def draw(self, ax, color):
        """draw the curve using matplotlib"""
        raise NotImplementedError

    def sample_points(self, n=32):
        """uniformly sample points from the curve"""
        raise NotImplementedError


####################### curves #######################
class Line(CurveBase):
    def __init__(self, start_point, end_point):
        super(Line, self).__init__()
        self.start_point = start_point
        self.end_point = end_point

    def __str__(self):
        return "Line: start({}), end({})".format(self.start_point.round(4), self.end_point.round(4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Line3D"
        start_point = np.array([stat['start_point']['x'],
                                stat['start_point']['y']])
        end_point = np.array([stat['end_point']['x'],
                              stat['end_point']['y']])
        return Line(start_point, end_point)

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        return Line(start_point, vec[1:3])

    @property
    def bbox(self):
        points = np.stack([self.start_point, self.end_point], axis=0)
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    def direction(self, from_start=True):
        return self.end_point - self.start_point

    def transform(self, translate, scale):
        self.start_point = (self.start_point + translate) * scale
        self.end_point = (self.end_point + translate) * scale

    def flip(self, axis):
        if axis == 'x':
            self.start_point[1], self.end_point[1] = -self.start_point[1], -self.end_point[1]
        elif axis == 'y':
            self.start_point[0], self.end_point[0] = -self.start_point[0], -self.end_point[0]
        elif axis == 'xy':
            self.start_point = self.start_point * -1
            self.end_point = self.end_point * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        self.start_point, self.end_point = self.end_point, self.start_point

    def numericalize(self, n=256):
        self.start_point = self.start_point.round().clip(min=0, max=n-1).astype(np.int)
        self.end_point = self.end_point.round().clip(min=0, max=n-1).astype(np.int)

    def to_vector(self):
        vec = [LINE_IDX, self.end_point[0], self.end_point[1]]
        return np.array(vec + [PAD_VAL] * (1 + N_ARGS - len(vec)))

    def draw(self, ax, color):
        xdata = [self.start_point[0], self.end_point[0]]
        ydata = [self.start_point[1], self.end_point[1]]
        l1 = lines.Line2D(xdata, ydata, lw=1, color=color, axes=ax)
        ax.add_line(l1)
        ax.plot(self.start_point[0], self.start_point[1], 'ok', color=color)
        # ax.plot(self.end_point[0], self.end_point[1], 'ok')

    def sample_points(self, n=32):
        return np.linspace(self.start_point, self.end_point, num=n)


class Arc(CurveBase):
    def __init__(self, start_point, end_point, center, radius,
                 normal=None, start_angle=None, end_angle=None, ref_vec=None):
        super(Arc, self).__init__()
        self.start_point = start_point
        self.end_point = end_point
        self.center = center
        self.radius = radius
        self.normal = normal
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ref_vec = ref_vec
        self.mid_point = self.get_mid_point()

    def __str__(self):
        return "Arc: start({}), end({}), mid({})".format(self.start_point.round(4), self.end_point.round(4),
                                                         self.mid_point.round(4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Arc3D"
        start_point = np.array([stat['start_point']['x'],
                                stat['start_point']['y']])
        end_point = np.array([stat['end_point']['x'],
                              stat['end_point']['y']])
        center = np.array([stat['center_point']['x'],
                           stat['center_point']['y']])
        radius = stat['radius']
        normal = np.array([stat['normal']['x'],
                           stat['normal']['y'],
                           stat['normal']['z']])
        start_angle = stat['start_angle']
        end_angle = stat['end_angle']
        ref_vec = np.array([stat['reference_vector']['x'],
                            stat['reference_vector']['y']])
        return Arc(start_point, end_point, center, radius, normal, start_angle, end_angle, ref_vec)

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        end_point = vec[1:3]
        sweep_angle = vec[3] / 256 * 2 * np.pi if is_numerical else vec[3]
        clock_sign = vec[4]
        s2e_vec = end_point - start_point
        if np.linalg.norm(s2e_vec) == 0:
            return None
        radius = (np.linalg.norm(s2e_vec) / 2) / np.sin(sweep_angle / 2)
        s2e_mid = (start_point + end_point) / 2
        vertical = np.cross(s2e_vec, [0, 0, 1])[:2]
        vertical = vertical / np.linalg.norm(vertical)
        if clock_sign == 0:
            vertical = -vertical
        center_point = s2e_mid - vertical * (radius * np.cos(sweep_angle / 2))

        start_angle = 0
        end_angle = sweep_angle
        if clock_sign == 0:
            ref_vec = end_point - center_point
        else:
            ref_vec = start_point - center_point
        ref_vec = ref_vec / np.linalg.norm(ref_vec)

        return Arc(start_point, end_point, center_point, radius,
                   start_angle=start_angle, end_angle=end_angle, ref_vec=ref_vec)

    def get_angles_counterclockwise(self, eps=1e-8):
        c2s_vec = (self.start_point - self.center) / (np.linalg.norm(self.start_point - self.center) + eps)
        c2m_vec = (self.mid_point - self.center) / (np.linalg.norm(self.mid_point - self.center) + eps)
        c2e_vec = (self.end_point - self.center) / (np.linalg.norm(self.end_point - self.center) + eps)
        angle_s, angle_m, angle_e = angle_from_vector_to_x(c2s_vec), angle_from_vector_to_x(c2m_vec), \
                                    angle_from_vector_to_x(c2e_vec)
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s
        return angle_s, angle_e

    @property
    def bbox(self):
        points = [self.start_point, self.end_point]
        angle_s, angle_e = self.get_angles_counterclockwise()
        if angle_s < 0 < angle_e:
            points.append(np.array([self.center[0] + self.radius, self.center[1]]))
        if angle_s < np.pi / 2 < angle_e or angle_s < -np.pi / 2 * 3 < angle_e:
            points.append(np.array([self.center[0], self.center[1] + self.radius]))
        if angle_s < np.pi < angle_e or angle_s < -np.pi < angle_e:
            points.append(np.array([self.center[0] - self.radius, self.center[1]]))
        if angle_s < np.pi / 2 * 3 < angle_e or angle_s < -np.pi/2 < angle_e:
            points.append(np.array([self.center[0], self.center[1] - self.radius]))
        points = np.stack(points, axis=0)
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    def direction(self, from_start=True):
        if from_start:
            return self.mid_point - self.start_point
        else:
            return self.end_point - self.mid_point

    @property
    def clock_sign(self):
        """get a boolean sign indicating whether the arc is on top of s->e """
        s2e = self.end_point - self.start_point
        s2m = self.mid_point - self.start_point
        sign = np.cross(s2m, s2e) >= 0 # counter-clockwise
        return sign

    def get_mid_point(self):
        mid_angle = (self.start_angle + self.end_angle) / 2
        rot_mat = np.array([[np.cos(mid_angle), -np.sin(mid_angle)],
                            [np.sin(mid_angle), np.cos(mid_angle)]])
        mid_vec = rot_mat @ self.ref_vec
        return self.center + mid_vec * self.radius

    def transform(self, translate, scale):
        self.start_point = (self.start_point + translate) * scale
        self.mid_point = (self.mid_point + translate) * scale
        self.end_point = (self.end_point + translate) * scale
        self.center = (self.center + translate) * scale
        if isinstance(scale * 1.0, float):
            self.radius = abs(self.radius * scale)

    def flip(self, axis):
        if axis == 'x':
            self.transform(0, np.array([1, -1]))
            new_ref_vec_angle = angle_from_vector_to_x(self.ref_vec) + self.end_angle - self.start_angle
            self.ref_vec = np.array([np.cos(new_ref_vec_angle), -np.sin(new_ref_vec_angle)])
        elif axis == 'y':
            self.transform(0, np.array([-1, 1]))
            new_ref_vec_angle = angle_from_vector_to_x(self.ref_vec) + self.end_angle - self.start_angle
            self.ref_vec = np.array([-np.cos(new_ref_vec_angle), np.sin(new_ref_vec_angle)])
        elif axis == 'xy':
            self.transform(0, -1)
            self.ref_vec = self.ref_vec * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        self.start_point, self.end_point = self.end_point, self.start_point

    def numericalize(self, n=256):
        self.start_point = self.start_point.round().clip(min=0, max=n-1).astype(np.int)
        self.mid_point = self.mid_point.round().clip(min=0, max=n-1).astype(np.int)
        self.end_point = self.end_point.round().clip(min=0, max=n-1).astype(np.int)
        self.center = self.center.round().clip(min=0, max=n-1).astype(np.int)
        tmp = np.array([self.start_angle, self.end_angle])
        self.start_angle, self.end_angle = (tmp / (2 * np.pi) * n).round().clip(
                                            min=0, max=n-1).astype(np.int)

    def to_vector(self):
        sweep_angle = max(abs(self.start_angle - self.end_angle), 1)
        return np.array([ARC_IDX, self.end_point[0], self.end_point[1], sweep_angle, int(self.clock_sign), PAD_VAL,
                         *[PAD_VAL] * N_ARGS_EXT])

    def draw(self, ax, color):
        ref_vec_angle = rads_to_degs(angle_from_vector_to_x(self.ref_vec))
        start_angle = rads_to_degs(self.start_angle)
        end_angle = rads_to_degs(self.end_angle)
        diameter = 2.0 * self.radius
        ap = patches.Arc(
            (self.center[0], self.center[1]),
            diameter,
            diameter,
            angle=ref_vec_angle,
            theta1=start_angle,
            theta2=end_angle,
            lw=1,
            color=color
        )
        ax.add_patch(ap)
        ax.plot(self.start_point[0], self.start_point[1], 'ok', color=color)
        # ax.plot(self.center[0], self.center[1], 'ok', color=color)
        ax.plot(self.mid_point[0], self.mid_point[1], 'ok', color=color)
        # ax.plot(self.end_point[0], self.end_point[1], 'ok')

    def sample_points(self, n=32):
        c2s_vec = (self.start_point - self.center) / np.linalg.norm(self.start_point - self.center)
        c2m_vec = (self.mid_point - self.center) / np.linalg.norm(self.mid_point - self.center)
        c2e_vec = (self.end_point - self.center) / np.linalg.norm(self.end_point - self.center)
        angle_s, angle_m, angle_e = angle_from_vector_to_x(c2s_vec), angle_from_vector_to_x(c2m_vec), \
                                    angle_from_vector_to_x(c2e_vec)
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s

        angles = np.linspace(angle_s, angle_e, num=n)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.radius + self.center[np.newaxis]
        return points


class Circle(CurveBase):
    def __init__(self, center, radius, normal=None):
        super(Circle, self).__init__()
        self.center = center
        self.radius = radius
        self.normal = normal

    def __str__(self):
        return "Circle: center({}), radius({})".format(self.center.round(4), round(self.radius, 4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Circle3D"
        center = np.array([stat['center_point']['x'],
                           stat['center_point']['y']])
        radius = stat['radius']
        normal = np.array([stat['normal']['x'],
                           stat['normal']['y'],
                           stat['normal']['z']])
        return Circle(center, radius, normal)

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        return Circle(vec[1:3], vec[5])

    @property
    def bbox(self):
        return np.stack([self.center - self.radius, self.center + self.radius], axis=0)

    def direction(self, from_start=True):
        return self.center - self.start_point

    @property
    def start_point(self):
        return np.array([self.center[0] - self.radius, self.center[1]])

    @property
    def end_point(self):
        return np.array([self.center[0] + self.radius, self.center[1]])

    def transform(self, translate, scale):
        self.center = (self.center + translate) * scale
        self.radius = self.radius * scale

    def flip(self, axis):
        if axis == 'x':
            self.center[1] = -self.center[1]
        elif axis == 'y':
            self.center[0] = -self.center[0]
        elif axis == 'xy':
            self.center = self.center * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        pass

    def numericalize(self, n=256):
        self.center = self.center.round().clip(min=0, max=n-1).astype(np.int)
        self.radius = np.round(self.radius).clip(min=1, max=n-1).astype(np.int)

    def to_vector(self):
        vec = [CIRCLE_IDX, self.center[0], self.center[1], PAD_VAL, PAD_VAL, self.radius]
        return np.array(vec + [PAD_VAL] * (1 + N_ARGS - len(vec)))

    def draw(self, ax, color):
        ap = patches.Circle((self.center[0], self.center[1]), self.radius,
                            lw=1, fill=None, color=color)
        ax.add_patch(ap)
        ax.plot(self.center[0], self.center[1], 'ok')

    def sample_points(self, n=32):
        angles = np.linspace(0, np.pi * 2, num=n, endpoint=False)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.radius + self.center[np.newaxis]
        return points
