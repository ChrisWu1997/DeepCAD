import numpy as np
import random
from .sketch import Profile
from .macro import *
from .math_utils import cartesian2polar, polar2cartesian, polar_parameterization, polar_parameterization_inverse


class CoordSystem(object):
    """Local coordinate system for sketch plane."""
    def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
        self.origin = origin
        self._theta = theta # 0~pi
        self._phi = phi     # -pi~pi
        self._gamma = gamma # -pi~pi
        self._y_axis = y_axis # (theta, phi)
        self.is_numerical = is_numerical

    @property
    def normal(self):
        return polar2cartesian([self._theta, self._phi])

    @property
    def x_axis(self):
        normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
        return x_axis_3d

    @property
    def y_axis(self):
        if self._y_axis is None:
            return np.cross(self.normal, self.x_axis)
        return polar2cartesian(self._y_axis)

    @staticmethod
    def from_dict(stat):
        origin = np.array([stat["origin"]["x"], stat["origin"]["y"], stat["origin"]["z"]])
        normal_3d = np.array([stat["z_axis"]["x"], stat["z_axis"]["y"], stat["z_axis"]["z"]])
        x_axis_3d = np.array([stat["x_axis"]["x"], stat["x_axis"]["y"], stat["x_axis"]["z"]])
        y_axis_3d = np.array([stat["y_axis"]["x"], stat["y_axis"]["y"], stat["y_axis"]["z"]])
        theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
        return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis_3d))

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=256):
        """NOTE: shall only be called after normalization"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
        self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int)
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = ((tmp / np.pi + 1.0) / 2 * n).round().clip(
            min=0, max=n-1).astype(np.int)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = (tmp / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])


class Extrude(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 operation, extent_type, extent_one, extent_two, sketch_pos, sketch_size):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of EXTRUDE_OPERATIONS, see macro.py
            extent_type (int): index of EXTENT_TYPE, see macro.py
            extent_one (float): extrude distance in normal direction (NOTE: it's negative in some data)
            extent_two (float): extrude distance in opposite direction
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile # normalized sketch
        self.sketch_plane = sketch_plane
        self.operation = operation
        self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two

        self.sketch_pos = sketch_pos
        self.sketch_size = sketch_size

    @staticmethod
    def from_dict(all_stat, extrude_id, sketch_dim=256):
        """construct Extrude from json data

        Args:
            all_stat (dict): all json data
            extrude_id (str): entity ID for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.

        Returns:
            list: one or more Extrude instances
        """
        extrude_entity = all_stat["entities"][extrude_id]
        assert extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"

        all_skets = []
        n = len(extrude_entity["profiles"])
        for i in range(len(extrude_entity["profiles"])):
            sket_id, profile_id = extrude_entity["profiles"][i]["sketch"], extrude_entity["profiles"][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])
            # normalize profile
            point = sket_profile.start_point
            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(sketch_dim)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

        operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
        extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
        extent_one = extrude_entity["extent_one"]["distance"]["value"]
        extent_two = 0.0
        if extrude_entity["extent_type"] == "TwoSidesFeatureExtentType":
            extent_two = extrude_entity["extent_two"]["distance"]["value"]

        if operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation"):
            all_operations = [operation] + [EXTRUDE_OPERATIONS.index("JoinFeatureOperation")] * (n - 1)
        else:
            all_operations = [operation] * n

        return [Extrude(all_skets[i][0], all_skets[i][1], all_operations[i], extent_type, extent_one, extent_two,
                        all_skets[i][2], all_skets[i][3]) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        assert vec[-1][0] == EXT_IDX and vec[0][0] == SOL_IDX
        profile_vec = np.concatenate([vec[:-1], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)
        ext_vec = vec[-1][-N_ARGS_EXT:]

        sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_size = ext_vec[N_ARGS_PLANE + N_ARGS_TRANS - 1]
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        ext_param = ext_vec[-N_ARGS_EXT_PARAM:]

        res = Extrude(profile, sket_plane, int(ext_param[2]), int(ext_param[3]), ext_param[0], ext_param[1],
                      sket_pos, sket_size)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Extrude pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(4), self.sketch_size.round(4))
        s += "\n  -operation:{}, type:{}, extent_one:{}, extent_two:{}".format(
            self.operation, self.extent_type, self.extent_one.round(4), self.extent_two.round(4))
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.extent_one *= scale
        self.extent_two *= scale
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=256):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        assert -2.0 <= self.extent_one <= 2.0 and -2.0 <= self.extent_two <= 2.0
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        self.extent_one = ((self.extent_one + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int) 
        self.extent_two = ((self.extent_two + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int) 
        self.operation = int(self.operation)
        self.extent_type = int(self.extent_type)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int) 
        self.sketch_size = (self.sketch_size / 2 * n).round().clip(min=0, max=n-1).astype(np.int) 

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.extent_one = self.extent_one / n * 2 - 1.0
        self.extent_two = self.extent_two / n * 2 - 1.0
        self.sketch_plane.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2

        self.operation = self.operation
        self.extent_type = self.extent_type

    def flip_sketch(self, axis):
        self.profile.flip(axis)
        self.profile.normalize()

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        ext_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size] + \
                    [self.extent_one, self.extent_two, self.operation, self.extent_type]
        ext_vec = np.array([EXT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param])
        vec = np.concatenate([profile_vec[:-1], ext_vec[np.newaxis], profile_vec[-1:]], axis=0) # NOTE: last one is EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec


class CADSequence(object):
    """A CAD modeling sequence, a series of extrude operations."""
    def __init__(self, extrude_seq, bbox=None):
        self.seq = extrude_seq
        self.bbox = bbox

    @staticmethod
    def from_dict(all_stat):
        """construct CADSequence from json data"""
        seq = []
        for item in all_stat["sequence"]:
            if item["type"] == "ExtrudeFeature":
                extrude_ops = Extrude.from_dict(all_stat, item["entity"])
                seq.extend(extrude_ops)
        bbox_info = all_stat["properties"]["bounding_box"]
        max_point = np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
        min_point = np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
        bbox = np.stack([max_point, min_point], axis=0)
        return CADSequence(seq, bbox)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        commands = vec[:, 0]
        ext_indices = [-1] + np.where(commands == EXT_IDX)[0].tolist()
        ext_seq = []
        for i in range(len(ext_indices) - 1):
            start, end = ext_indices[i], ext_indices[i + 1]
            ext_seq.append(Extrude.from_vector(vec[start+1:end+1], is_numerical, n))
        cad_seq = CADSequence(ext_seq)
        return cad_seq

    def __str__(self):
        res = ""
        for i, ext in enumerate(self.seq):
            res += "({})".format(i) + str(ext) + "\n"
        return res

    def to_vector(self, max_n_ext=10, max_n_loops=6, max_len_loop=15, max_total_len=60, pad=False):
        if len(self.seq) > max_n_ext:
            return None
        vec_seq = []
        for item in self.seq:
            vec = item.to_vector(max_n_loops, max_len_loop, pad=False)
            if vec is None:
                return None
            vec = vec[:-1] # last one is EOS, removed
            vec_seq.append(vec)

        vec_seq = np.concatenate(vec_seq, axis=0)
        vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis]], axis=0)

        # add EOS padding
        if pad and vec_seq.shape[0] < max_total_len:
            pad_len = max_total_len - vec_seq.shape[0]
            vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        return vec_seq

    def transform(self, translation, scale):
        """linear transformation"""
        for item in self.seq:
            item.transform(translation, scale)

    def normalize(self, size=1.0):
        """(1)normalize the shape into unit cube (-1~1). """
        scale = size * NORM_FACTOR / np.max(np.abs(self.bbox))
        self.transform(0.0, scale)

    def numericalize(self, n=256):
        for item in self.seq:
            item.numericalize(n)

    def flip_sketch(self, axis):
        for item in self.seq:
            item.flip_sketch(axis)

    def random_transform(self):
        for item in self.seq:
            # random transform sketch
            scale = random.uniform(0.8, 1.2)
            item.profile.transform(-np.array([128, 128]), scale)
            translate = np.array([random.randint(-5, 5), random.randint(-5, 5)], dtype=np.int) + 128
            item.profile.transform(translate, 1)

            # random transform and scale extrusion
            t = 0.05
            translate = np.array([random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t)])
            scale = random.uniform(0.8, 1.2)
            # item.sketch_plane.transform(translate, scale)
            item.sketch_pos = (item.sketch_pos + translate) * scale
            item.extent_one *= random.uniform(0.8, 1.2)
            item.extent_two *= random.uniform(0.8, 1.2)

    def random_flip_sketch(self):
        for item in self.seq:
            flip_idx = random.randint(0, 3)
            if flip_idx > 0:
                item.flip_sketch(['x', 'y', 'xy'][flip_idx - 1])
