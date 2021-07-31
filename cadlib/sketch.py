import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .curves import *
from .macro import *


##########################   base  ###########################
class SketchBase(object):
    """Base class for sketch (a collection of curves). """
    def __init__(self, children, reorder=True):
        self.children = children

        if reorder:
            self.reorder()

    @staticmethod
    def from_dict(stat):
        """construct sketch from json data

        Args:
            stat (dict): dict from json data
        """
        raise NotImplementedError

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        """construct sketch from vector representation

        Args:
            vec (np.array): (seq_len, n_args)
            start_point (np.array): (2, ). If none, implicitly defined as the last end point.
        """
        raise NotImplementedError

    def reorder(self):
        """rearrange the curves to follow counter-clockwise direction"""
        raise NotImplementedError

    @property
    def start_point(self):
        return self.children[0].start_point

    @property
    def end_point(self):
        return self.children[-1].end_point

    @property
    def bbox(self):
        """compute bounding box (min/max points) of the sketch"""
        all_points = np.concatenate([child.bbox for child in self.children], axis=0)
        return np.stack([np.min(all_points, axis=0), np.max(all_points, axis=0)], axis=0)

    @property
    def bbox_size(self):
        """compute bounding box size (max of height and width)"""
        bbox_min, bbox_max = self.bbox[0], self.bbox[1]
        bbox_size = np.max(np.abs(np.concatenate([bbox_max - self.start_point, bbox_min - self.start_point])))
        return bbox_size

    @property
    def global_trans(self):
        """start point + sketch size (bbox_size)"""
        return np.concatenate([self.start_point, np.array([self.bbox_size])])

    def transform(self, translate, scale):
        """linear transformation"""
        for child in self.children:
            child.transform(translate, scale)

    def flip(self, axis):
        for child in self.children:
            child.flip(axis)
        self.reorder()

    def numericalize(self, n=256):
        """quantize curve parameters into integers"""
        for child in self.children:
            child.numericalize(n)

    def normalize(self, size=256):
        """normalize within the given size, with start_point in the middle center"""
        cur_size = self.bbox_size
        scale = (size / 2 * NORM_FACTOR - 1) / cur_size # prevent potential overflow if data augmentation applied
        self.transform(-self.start_point, scale)
        self.transform(np.array((size / 2, size / 2)), 1)

    def denormalize(self, bbox_size, size=256):
        """inverse procedure of normalize method"""
        scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
        self.transform(-np.array((size / 2, size / 2)), scale)

    def to_vector(self):
        """convert to vector representation"""
        raise NotImplementedError

    def draw(self, ax):
        """draw sketch on matplotlib ax"""
        raise NotImplementedError

    def to_image(self):
        """convert to image"""
        fig, ax = plt.subplots()
        self.draw(ax)
        ax.axis('equal')
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return X

    def sample_points(self, n=32):
        """uniformly sample points from the sketch"""
        raise NotImplementedError


####################### loop & profile #######################
class Loop(SketchBase):
    """Sketch loop, a sequence of connected curves."""
    @staticmethod
    def from_dict(stat):
        all_curves = [construct_curve_from_dict(item) for item in stat['profile_curves']]
        this_loop = Loop(all_curves)
        this_loop.is_outer = stat['is_outer']
        return this_loop

    def __str__(self):
        s = "Loop:"
        for curve in self.children:
            s += "\n      -" + str(curve)
        return s

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_curves = []
        if start_point is None:
            # FIXME: explicit for loop can be avoided here
            for i in range(vec.shape[0]):
                if vec[i][0] == EOS_IDX:
                    start_point = vec[i - 1][1:3]
                    break
        for i in range(vec.shape[0]):
            type = vec[i][0]
            if type == SOL_IDX:
                continue
            elif type == EOS_IDX:
                break
            else:
                curve = construct_curve_from_vector(vec[i], start_point, is_numerical=is_numerical)
                start_point = vec[i][1:3] # current curve's end_point serves as next curve's start_point
            all_curves.append(curve)
        return Loop(all_curves)

    def reorder(self):
        """reorder by starting left most and counter-clockwise"""
        if len(self.children) <= 1:
            return

        start_curve_idx = -1
        sx, sy = 10000, 10000

        # correct start-end point order
        if np.allclose(self.children[0].start_point, self.children[1].start_point) or \
            np.allclose(self.children[0].start_point, self.children[1].end_point):
            self.children[0].reverse()

        # correct start-end point order and find left-most point
        for i, curve in enumerate(self.children):
            if i < len(self.children) - 1 and np.allclose(curve.end_point, self.children[i + 1].end_point):
                self.children[i + 1].reverse()
            if round(curve.start_point[0], 6) < round(sx, 6) or \
                    (round(curve.start_point[0], 6) == round(sx, 6) and round(curve.start_point[1], 6) < round(sy, 6)):
                start_curve_idx = i
                sx, sy = curve.start_point

        self.children = self.children[start_curve_idx:] + self.children[:start_curve_idx]

        # ensure mostly counter-clock wise
        if isinstance(self.children[0], Circle) or isinstance(self.children[-1], Circle): # FIXME: hard-coded
            return
        start_vec = self.children[0].direction()
        end_vec = self.children[-1].direction(from_start=False)
        if np.cross(end_vec, start_vec) <= 0:
            for curve in self.children:
                curve.reverse()
            self.children.reverse()

    def to_vector(self, max_len=None, add_sol=True, add_eos=True):
        loop_vec = np.stack([curve.to_vector() for curve in self.children], axis=0)
        if add_sol:
            loop_vec = np.concatenate([SOL_VEC[np.newaxis], loop_vec], axis=0)
        if add_eos:
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
        if max_len is None:
            return loop_vec

        if loop_vec.shape[0] > max_len:
            return None
        elif loop_vec.shape[0] < max_len:
            pad_vec = np.tile(EOS_VEC, max_len - loop_vec.shape[0]).reshape((-1, len(EOS_VEC)))
            loop_vec = np.concatenate([loop_vec, pad_vec], axis=0) # (max_len, 1 + N_ARGS)
        return loop_vec

    def draw(self, ax):
        colors = ['red', 'blue', 'green', 'brown', 'pink', 'yellow', 'purple', 'black'] * 10
        for i, curve in enumerate(self.children):
            curve.draw(ax, colors[i])

    def sample_points(self, n=32):
        points = np.stack([curve.sample_points(n) for curve in self.children], axis=0) # (n_curves, n, 2)
        return points


class Profile(SketchBase):
    """Sketch profile，a closed region formed by one or more loops. 
    The outer-most loop is placed at first."""
    @staticmethod
    def from_dict(stat):
        all_loops = [Loop.from_dict(item) for item in stat['loops']]
        return Profile(all_loops)

    def __str__(self):
        s = "Profile:"
        for loop in self.children:
            s += "\n    -" + str(loop)
        return s

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_loops = []
        command = vec[:, 0]
        end_idx = command.tolist().index(EOS_IDX)
        indices = np.where(command[:end_idx] == SOL_IDX)[0].tolist() + [end_idx]
        for i in range(len(indices) - 1):
            loop_vec = vec[indices[i]:indices[i + 1]]
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
            if loop_vec[0][0] == SOL_IDX and loop_vec[1][0] not in [SOL_IDX, EOS_IDX]:
                all_loops.append(Loop.from_vector(loop_vec, is_numerical=is_numerical))
        return Profile(all_loops)

    def reorder(self):
        if len(self.children) <= 1:
            return
        all_loops_bbox_min = np.stack([loop.bbox[0] for loop in self.children], axis=0).round(6)
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        self.children = [self.children[i] for i in ind]

    def draw(self, ax):
        for i, loop in enumerate(self.children):
            loop.draw(ax)
            ax.text(loop.start_point[0], loop.start_point[1], str(i))

    def to_vector(self, max_n_loops=None, max_len_loop=None, pad=True):
        loop_vecs = [loop.to_vector(None, add_eos=False) for loop in self.children]
        if max_n_loops is not None and len(loop_vecs) > max_n_loops:
            return None
        for vec in loop_vecs:
            if max_len_loop is not None and vec.shape[0] > max_len_loop:
                return None
        profile_vec = np.concatenate(loop_vecs, axis=0)
        profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis]], axis=0)
        if pad:
            pad_len = max_n_loops * max_len_loop - profile_vec.shape[0]
            profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return profile_vec

    def sample_points(self, n=32):
        points = np.concatenate([loop.sample_points(n) for loop in self.children], axis=0)
        return points
