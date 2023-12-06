"""
Root System Library

A library of methods for synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import List, Tuple

# Internal
from root_system_lib.plot import visualise_roots as plot_roots
from root_system_lib.spatial import make_homogenous, transform

##########################################################################################################
### Library
##########################################################################################################

class RootNode:
    """
    A root node within the root map. 
    """
    def __init__(self, root_map, order, organ_id, organ_origin, num_segs, length_range, fixed_seg_length, 
            length_reduction, base_diameter, diameter_reduction, apex_diam, rng) -> None:
        self.root_map = root_map
        self.order = order
        self.root_type = 1
        self.organ_id = organ_id
        self.organ_origin = organ_origin
        self.children: List[RootNode] = []
        self.parent_id = 1
        self.parent_segment_index = 0
        self.parent: RootNode = None
        self.coordinates = []
        self.reset_transform()
        self.num_segs = num_segs
        self.fixed_seg_length = fixed_seg_length
        self.length_reduction = length_reduction
        self.base_diameter = base_diameter
        self.diameter_reduction = diameter_reduction
        self.apex_diam = apex_diam
        self.rng = rng
        self.exclude =  False

        self.init_segment_lengths(length_range)
        self.init_diameters()

    def reset_transform(self):
        """
        Reset the 4x4 transformation matrix.

        Returns
        ---------
        self: RootNode
            The current root node.
        """
        self.transform_matrix = np.eye(4)
        return self

    def init_segment_lengths(self, length_range: Tuple[float]) -> None: 
        """
        Initialise an array of root segment lengths.

        Parameters
        ---------
        length_range: tuple (2,)
            The lower and upper bound of possible root segment lengths.
        """
        # Root length decreases as order increases for second order roots
        if self.order > 1:
            min_length, max_length = length_range * self.length_reduction ** (self.order - 2)
        else:
            min_length, max_length = length_range

        self.r_length = self.rng.uniform(min_length, max_length)
        if self.fixed_seg_length:
            seg_length = self.r_length / self.num_segs
            self.segment_lengths = np.repeat(seg_length, self.num_segs)
        else:
            # Uniformly distributed varying segment lengths
            samples = self.rng.uniform(0, 1, self.num_segs) 
            self.segment_lengths = samples / samples.max(axis = 0) * self.r_length
            # Segment lengths follow Dirichlet distribution using the below approach.
            # Any literature on this?
            # self.segment_lengths = rng.dirichlet(np.ones(num_segs)) * self.r_length

    def init_diameters(self) -> None:
        """
        Initialise the diameters of the root node.
        """
        # Root diameter decreases as order increases  
        base_diameter = self.base_diameter *  self.diameter_reduction ** (self.order - 1) 
        # Apply linear interpolation between start and end segment diameters and add random noise
        # Sort random noise from largest to smallest as the root should become progressively become thinner
        diameter_diff = abs(base_diameter - self.apex_diam) * 0.05
        diameter_noise = -np.sort(-self.rng.uniform(-diameter_diff, diameter_diff, self.num_segs))
        self.diameters = np.interp(range(1, self.num_segs + 1), [1, self.num_segs], [base_diameter, self.apex_diam]) + diameter_noise 

    def update_transform(self, roll = 0, pitch = 0, yaw = 0, translation = [0, 0, 0], reflect = [1, 1, 1, 1], scale = [1, 1, 1, 1]):
        """
        Updates the transformation matrix.

        Parameters
        --------------
        roll: float
            Epsilon parameter. 
        pitch: array (float)  
            Observed data.
        yaw: array (float)
            Simulated data.
        translation: float (3,)
            The translation transform.
        reflect: float (3,)
            The reflection transform.
        scale: float (3,)
            The scale transform.
        """
        transformation_matrix = transform(roll = roll, pitch = pitch, yaw = yaw, translation = translation, reflect = reflect, scale = scale)
        self.transform_matrix = self.transform_matrix @ transformation_matrix 
        
    def cascading_update_transform(self, roll = 0, pitch = 0, yaw = 0, translation = [0, 0, 0], reflect = [1, 1, 1, 1], scale = [1, 1, 1, 1]):
        """
        Updates the transformation matrix for child root nodes.

        Parameters
        --------------
        roll: float
            Epsilon parameter. 
        pitch: array (float)  
            Observed data.
        yaw: array (float)
            Simulated data.
        translation: float (3,)
            The translation transform.
        reflect: float (3,)
            The reflection transform.
        scale: float (3,)
            The scale transform.
        """
        self.update_transform(roll = roll, pitch = pitch, yaw = yaw, translation = translation, reflect = reflect, scale = scale)
        for child in self.children:
            child.cascading_update_transform(roll = roll, pitch = pitch, yaw = yaw, translation = translation, reflect = reflect, scale = scale)

    def transform(self):
        """
        Apply the transformation matrix to the root system coordinates.
        """
        # Convert to four dimensional homogenous coordinates
        ones_matrix = np.ones((len(self.coordinates), 1))
        homogenous_coordinates = np.hstack((self.coordinates, ones_matrix)).T
        # 3x4 transformation matrix 
        transformed_coordinates = self.transform_matrix[:-1] @ homogenous_coordinates
        self.coordinates = transformed_coordinates.T
        return self.reset_transform()

    def cascading_transform(self):
        """
        Apply the transformation matrix to the root system coordinates for child nodes.
        """
        self.transform()
        for child in self.children:
            child.cascading_transform()

    def sample_segments(self, floor = 0.15, ceiling = 0.85) -> np.ndarray:
        """
        Sample the indices of the root segments of a root.

        Parameters
        --------------
        floor: float
            The lower bound for the segment indices to sample from.
        ceiling: float
            The upper bound for the segment indices to sample from.

        Returns
        ---------
        indices : array   
            The sampled indices.
        """
        n_children = len(self.children)
        n_segments = len(self.coordinates)
        if n_children == 0 or n_segments == 0:
            return []
        
        coordinate_indices = np.arange(int(floor * n_segments), int(ceiling * n_segments))
        replace = n_children > len(coordinate_indices)
        sampled_indices = self.rng.choice(coordinate_indices, size = n_children, replace = replace)
        return sampled_indices
        
    def assign_children_to_segments(self, floor = 0.1, ceiling = 0.9):
        """
        Assign child root nodes to sampled root segments.

        Parameters
        --------------
        floor: float
            The lower bound for the segment indices to sample from.
        ceiling: float
            The upper bound for the segment indices to sample from.
        """
        sampled_indices = self.sample_segments(floor, ceiling)
        for segment, child in enumerate(self.children):
            child.parent_segment_index = sampled_indices[segment]

    def get_parent_coordinates(self):
        """
        Get the coordinates of the parent root.

        Returns
        --------------
        parent: float (3,)
            The parent coordinates.
        """
        return self.parent.coordinates[self.parent_segment_index]

    def get_parent_origin(self):
        """
        Get the origin of the parent root.

        Returns
        --------------
        origin: float (3,)
            The parent origin.
        """
        return self.get_parent_coordinates() + self.parent.coordinates[0]

    def get_local_origin(self):
        """
        Get the origin of the current root.

        Returns
        --------------
        origin: float (3,)
            The origin.
        """
        return self.coordinates[0]

    def get_apex(self):
        """
        Get the coordinates of the apex of the current root.

        Returns
        --------------
        apex: float (3,)
            The apex.
        """
        return self.coordinates[-1]

    def to_origin(self):
        """
        Translate the root node to the world origin.
        """
        self.update_transform(translation = -self.get_local_origin())

    def cascading_to_origin(self):
        """
        Translate all child nodes to the world origin.
        """
        self.cascading_update_transform(translation = -self.get_local_origin())

    def exclude_self(self):
        """
        Exclude the current root node from any validation and data exporting.
        """
        self.exclude = True

    def cascading_exclude(self):
        """
        Exclude child root node from any validation and data exporting.
        """
        self.exclude_self()
        for child in self.children:
            child.cascading_exclude()

    def construct_root(self, vary) -> np.ndarray:
        root = [np.repeat(self.organ_origin, 3)]

        for segment in range(0, self.num_segs - 1):
            segment_length = self.segment_lengths[segment]
            current_coord = root[segment]
            coord = np.array([np.repeat(segment_length, 3)])
            y_rotate = self.rng.uniform(-vary, vary)
            z_rotate = self.rng.uniform(-vary, vary)
            # Add a small amount of random noise to coordinates to prevent identical values (for importing into GroIMP) 
            noise = self.rng.uniform(1e-4, 1e-3)
            homogenous_coordinates = make_homogenous(coord)
            transformation_matrix = transform(pitch = y_rotate, yaw = z_rotate, translation = current_coord + noise)
            transformed_coordinates = transformation_matrix[:-1] @ homogenous_coordinates
            coord = transformed_coordinates.T[0]
            root.append(coord)

        root = np.array(root) 
        root = root / root.max(axis = 0) * self.r_length
        # Let l be the total root length
        # Perform reflection along z axis.
        # Initial root coordinates are (l, l, -l) from the organ origin
        root[:, 2] *= -1
        self.coordinates = root

    def add_parent(self, parent):
        self.parent = parent
        self.parent_id = parent.organ_id

    def incorporate_attributes_for_children(self):
        for child in self.children:
            child.inherit()

    def inherit(self):
        # Secondary roots that are closer to base are larger.
        # Secondary roots that are closer to apex are smaller.
        parent = self.parent
        parent_segment_index = self.parent_segment_index
        if parent_segment_index == 0: # No divide by zero error
            parent_segment_index = 1

        self.diameters += parent.diameters[parent_segment_index] * 0.1
        seg_length_increase = sum(parent.segment_lengths) * 0.1 / parent_segment_index / self.num_segs 
        self.segment_lengths += seg_length_increase
        self.r_length += seg_length_increase * self.num_segs
    
    def validate(self, no_root_zone, pitch = 90, max_attempts = 50):
        """Perform validation for an individual root."""
        if self.exclude:
            return

        def __transform(**kwargs):
            """Translate to world origin. Apply transform. Translate back to local origin."""
            local_origin = self.get_local_origin()
            self.cascading_to_origin()
            self.cascading_transform()
            self.cascading_update_transform(**kwargs)
            self.cascading_transform()
            self.cascading_update_transform(translation = local_origin)
            self.cascading_transform()

        coin_flip = self.rng.binomial(1, 0.5)
        if coin_flip == 1:
            pitch *= -1
            
        # Apply gravitropism
        iter_count = 0
        while self.get_apex()[2] > self.get_local_origin()[2]:
            if iter_count > max_attempts:
                return self.cascading_exclude()

            __transform(pitch = pitch)
            iter_count += 1

        # Lateral roots must be below no root zone
        iter_count = 0
        while np.any(self.coordinates[:, 2] > no_root_zone):
            if iter_count > max_attempts:
                return self.cascading_exclude()

            __transform(pitch = pitch)
            iter_count += 1

        # Remove any detached/floating secondary roots
        if self.order > 1:
            local_origin = np.around(self.get_local_origin())
            parent_coordinates = np.around(self.get_parent_coordinates())
            if np.any(np.not_equal(local_origin, parent_coordinates)):
                return self.cascading_exclude()

    def to_rows(self, round_num = 4):
        rows = []
        if self.exclude:
            return rows
                
        root_map = self.root_map
        self.id = root_map.id + 1

        diameters = np.round(self.diameters * root_map.scale_factor * 5, round_num)
        segment_lengths = np.round(self.segment_lengths * root_map.scale_factor, round_num)
        coordinates = np.round(self.coordinates * root_map.scale_factor, round_num)

        for segment, coordinate in enumerate(coordinates):
            root_map.id += 1
            x, y, z = coordinate

            # Determine parent row value.        
            if segment  == 0:
                if self.order > 1:
                    parent_root = self.parent.id + self.parent_segment_index - 1
                else:
                    parent_root = root_map.origin_id
            else:
                parent_root = root_map.id - 1

            row = { 
                "id": self.root_map.id,
                "plant_id": root_map.plant,
                "organ_id" : self.organ_id,
                "order": self.order * -1,
                "root_type": self.root_type,
                "segment_rank" : segment + 1,
                "parent": parent_root,
                "coordinates" : f"{x} {y} {z}",
                "diameter" : diameters[segment],
                "length" : segment_lengths[segment],
                "x": x,
                "y": y,
                "z": z
            }
            rows.append(row)
        return rows
            
class RootNodeMap:
    def __init__(self, max_order, organ_origin, num_segs, length_range, fixed_seg_length, 
        length_reduction, base_diameter, diameter_reduction, apex_diam, scale_factor, rng) -> None:
        self.set_id()
        self.plant = 1
        self.map = {}
        self.max_order = max_order
        self.organ_origin = organ_origin
        self.num_segs = num_segs
        self.length_range = length_range
        self.fixed_seg_length = fixed_seg_length
        self.length_reduction = length_reduction
        self.base_diameter = base_diameter
        self.diameter_reduction = diameter_reduction
        self.apex_diam = apex_diam
        self.scale_factor = scale_factor
        self.rng = rng

    def set_id(self, id = 0):
        self.id = id
        self.origin_id = self.id

    def init_plant(self, plant):
        self.plant = plant
        self.map[plant] = {}
        for order in range(1, self.max_order + 1):
            self.map[plant][order] = {}

    def get_origin_row(self):
        return { 
            "id": self.id,
            "plant_id": self.plant,
            "organ_id" : 0,
            "order": 0,
            "root_type": 0,
            "segment_rank" : 0,
            "parent": -1,
            "coordinates" : '0 0 0',
            "diameter" : 0, 
            "length" : 0,
            "x": 0,
            "y": 0,
            "z": 0,
        }

    def add(self, order, organ_id) -> RootNode:
        node = RootNode(self, order, organ_id, self.organ_origin, self.num_segs, self.length_range, self.fixed_seg_length, 
            self.length_reduction, self.base_diameter, self.diameter_reduction, self.apex_diam, self.rng)
        self.map[self.plant][order][organ_id] = node
        return node

    def add_child(self, order, organ_id, parent) -> RootNode:
        node = self.add(order, organ_id)
        node.add_parent(parent)
        parent.children.append(node)
        return node

    def get(self, order, node_id) -> RootNode:
        return self.map[self.plant][order][node_id]

    def get_order(self, order):
        return self.map[self.plant][order]
    
    def get_order_map(self):
        return self.map[self.plant]

    def get_root_ids(self, order):
        return np.array(list(self.map[self.plant][order].keys()))

    def sample_roots(self, order, size = None):
        root_ids = self.get_root_ids(order)
        n_roots = len(root_ids)
        if n_roots == 0:
            return []
        
        root_samples = self.rng.choice(root_ids, size = size, replace = False)
        return root_samples

    def cascading_transform(self):
        for primary_root in self.get_order(1).values():
            primary_root.cascading_transform()

    def cascading_update_transform(self, roll = 0, pitch = 0, yaw = 0, translation = [0, 0, 0]):
        for primary_root in self.get_order(1).values():
            primary_root.cascading_update_transform(roll = roll, pitch = pitch, yaw = yaw, translation = translation)

    def plot_roots(self):
        root_map = deepcopy(self) # Prevent stateful alterations to roots.
        df = root_map.to_dataframe()
        plot_roots(df, thickness = root_map.max_order)

    def construct_roots(self, root_var: int, proot_num: Tuple[int], sroot_num: Tuple[int], 
        snum_growth, sroot_length: float, froot_threshold: float, r_ratio: float, visualise_roots = False) -> None:
        """Construct the initial roots."""
        order = 1
        organ_id = 0
        out_root_num, in_root_num = proot_num
        sroot_num_min, sroot_num_max = sroot_num

        # Generate primary and secondary roots
        ## Primary
        for _ in range(out_root_num + in_root_num):
            organ_id += 1
            node = self.add(order, organ_id)
            node.construct_root(root_var)

        if visualise_roots:
            self.plot_roots()

        self.length_range = sroot_length
        ## Secondary 
        for order in range(2, self.max_order + 1):
            parent_roots = self.get_order(order - 1)
            snum_growth *= (order - 2) 
            if sroot_num_min > sroot_num_max:
                sroot_num_min, sroot_num_max = sroot_num_max, sroot_num_min
                
            for parent_root in parent_roots.values():
                # Number of secondary roots increases as order increases.
                n_roots = self.rng.integers(sroot_num_min, sroot_num_max, endpoint = True) 
                n_roots = int(n_roots * 1 + snum_growth)   
                for _ in range(n_roots):
                    organ_id += 1
                    node = self.add_child(order, organ_id, parent_root)

        for primary_root in self.get_order(1).values():
            primary_root.assign_children_to_segments() 
            # Secondary roots that are closer to base are larger (diameter and length).
            # Secondary roots that are closer to apex are smaller (diameter and length).
            primary_root.incorporate_attributes_for_children()

        for order in range(2, self.max_order + 1):
            for secondary_root in self.get_order(order).values():
                secondary_root.construct_root(root_var)
                secondary_root.assign_children_to_segments() 
                secondary_root.incorporate_attributes_for_children()

        self.assign_root_types(froot_threshold)

        if visualise_roots:
            self.plot_roots()

    def assign_root_types(self, froot_threshold):
        for order in range(2, self.max_order + 1):
            # total_diameters = []
            for secondary_root in self.get_order(order).values():
                first_diam = secondary_root.diameters[0]
                if first_diam > froot_threshold:
                    secondary_root.root_type = 1
                else:
                    secondary_root.root_type = 2

                # total_diameter = np.sum(secondary_root.diameters)
                # total_diameters.append(total_diameter)
            
            # sorted_diameter_indxs = np.argsort(total_diameters)
            # indx_split = round(len(sorted_diameter_indxs) * r_ratio)
            # fine_root_indxs = set(sorted_diameter_indxs[:indx_split])
            # for i, secondary_root in enumerate(self.get_order(order).values()):
            #     if i in fine_root_indxs:
            #         secondary_root.root_type = 2
            #     else:
            #         secondary_root.root_type = 1

    def position_secondary_roots(self, visualise_roots = False) -> None:
        """Rotate and translate secondary roots."""
        # Rotate secondary roots into conical shape around primary roots
        # Specifies the variance in angles of higher order roots with respect to their parent root
        ## Secondary
        for order in range(2, self.max_order + 1):
            for secondary_root in self.get_order(order).values():
                yaw = self.rng.uniform(-30, 240) 
                pitch, roll = self.rng.uniform(60, 110, 2) # 45 deg => Nearly parallel to parent. 120 deg => Nearly perpendicular to parent          
                secondary_root.update_transform(yaw = yaw)
                secondary_root.update_transform(pitch = pitch, roll = -roll)
                secondary_root.transform()
                secondary_root.update_transform(pitch = -45, roll = 35)
                secondary_root.transform()

        if visualise_roots:
            self.plot_roots()

        # ## Secondary    
        for order in range(self.max_order, 1, -1):
            for secondary_root in self.get_order(order).values():
                secondary_root.cascading_update_transform(translation = secondary_root.get_parent_origin())
                secondary_root.cascading_transform()

        if visualise_roots:
            self.plot_roots()

    def position_primary_roots(self, proot_num: Tuple[int], origin_noise_range: Tuple[float], visualise_roots = False) -> pd.DataFrame:
        """Rotate and translate primary roots."""
        # Random assignment of primary roots as either outer or inner roots, then reposition accordingly
        ## Primary
        order = 1
        out_root_num, in_root_num = proot_num
        primary_root_ids = np.unique(self.get_root_ids(1))
        outer_root_ids = np.unique(self.sample_roots(1, out_root_num))
        inner_root_ids = primary_root_ids[~np.isin(primary_root_ids, outer_root_ids)]

        def get_yaw(root_num: int):
            yaw_base = 360 / root_num
            return yaw_base, yaw_base * 0.05, yaw_base

        ### Outer
        yaw_base, yaw_noise_base, yaw = get_yaw(out_root_num)
        for outer_root_id in outer_root_ids:
            outer_root = self.get(order, outer_root_id)
            pitch = self.rng.uniform(-20, -15)
            outer_root.cascading_update_transform(pitch = pitch, yaw = yaw)
            yaw += yaw_base + self.rng.uniform(-yaw_noise_base, yaw_noise_base)
        ### Inner
        yaw_base, yaw_noise_base, yaw = get_yaw(in_root_num)
        for inner_root_id in inner_root_ids:
            inner_root = self.get(order, inner_root_id)
            pitch = self.rng.uniform(0, 45)
            inner_root.cascading_update_transform(pitch = pitch, yaw = yaw)
            yaw += yaw_base + self.rng.uniform(-yaw_noise_base, yaw_noise_base)
        ### Both
        origin_noise_min, origin_noise_max = origin_noise_range
        for primary_root in self.get_order(1).values():
            # Add small amount of noise to local origin  
            origin_noise = -self.rng.uniform(origin_noise_min, origin_noise_max, 3)
            primary_root.cascading_update_transform(translation = origin_noise)
            primary_root.cascading_transform()

        if visualise_roots:
            self.plot_roots()

    def validate(self, no_root_zone, pitch = 90, max_attempts = 50):
        order_map = self.get_order_map()
        for order in order_map.values():
            for roots in order.values():
                roots.validate(no_root_zone, pitch, max_attempts)

    def to_rows(self, round_num = 4):
        rows = []
        current_plant = self.plant
        for plant in range(1, current_plant + 1):
            self.set_id(1)
            self.plant = plant
            order_map = self.get_order_map()
            rows += [self.get_origin_row()]
            for order in order_map.values():
                for roots in order.values():
                    rows += roots.to_rows(round_num)
                            
        self.set_id()
        return rows

    def to_dataframe(self, round_num = 4) -> pd.DataFrame:
        rows = self.to_rows(round_num)
        df = pd.DataFrame(rows) 
        return df
