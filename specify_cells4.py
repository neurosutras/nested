__author__ = 'Aaron D. Milstein'
import btmorph  # must be found in system $PYTHONPATH
from function_lib import *
from neuron import h  # must be found in system $PYTHONPATH

# Includes modification of an early version of SWC_neuron.py by Daniele Linaro.
# Includes an extension of BtMorph, created by Ben Torben-Nielsen and modified by Daniele Linaro.

# SWC files must use this nonstandard convention to exploit trunk and tuft categorization
swc_types = [soma_type, axon_type, basal_type, apical_type, trunk_type, tuft_type] = [1, 2, 3, 4, 5, 6]
sec_types = ['soma', 'axon_hill', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine_neck', 'spine_head']
swc_type_enumerator = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, 'trunk': 5, 'tuft': 6}
syn_category_enumerator = {'excitatory': 0, 'inhibitory': 1, 'neuromodulatory': 2}

verbose = False  # Turn on for text reporting during model initialization and simulation


# -------Wrapper for converting SWC --> BtMorph --> Skeleton morphology in NEURON hoc------------------------


class HocCell(object):
    """
    Base class for a NEURON cell, including convenient tree structure and metadata. Subclass for each specific cell
    type to create rules for synaptic mechanisms.
    """

    def __init__(self, morph_file_path=None, mech_file_path=None, gid=0, existing_hoc_cell=None, neuroH5_dict=None):
        """
        :param morph_file_path: str : path to .swc file containing morphology
        :param mech_file_path: str : path to .pkl file specifying cable parameters and membrane mechanisms
        :param gid: int
        :param existing_hoc_cell: :class: 'h.hocObject' : instance of a cell template class already built in hoc
        :param neuroH5_dict: dict : read from a parallelized .hdf5 file
        """
        self._gid = gid
        self.tree = btmorph.STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.index = 0  # Keep track of number of nodes
        self._node_dict = {'soma': [], 'axon': [], 'basal': [], 'trunk': [], 'apical': [], 'tuft': [], 'spine': []}
        self.mech_file_path = mech_file_path
        # Refer to function_lib for description of structure of mechanism dictionary. loads from .yaml or
        # default_mech_dict in function_lib
        self.mech_dict = self.load_mech_dict(mech_file_path)
        self.morph_file_path = morph_file_path
        self.existing_hoc_cell = existing_hoc_cell
        self.neuroH5_dict = neuroH5_dict
        self.random = np.random.RandomState()
        self.spike_detector = None

    def load_morphology(self, preserve_3d=False):
        """
        Loads morphology from files provided during initialization. self.tree.root must already be defined, for example
        by creating a standard soma by running self.make_standard_soma_and_axon
        """
        if not self.morph_file_path is None:
            self.load_morphology_from_swc(preserve_3d)
            # Membrane mechanisms must be reinitialized whenever cable properties (Ra, cm) or spatial resolution (nseg)
            # changes.
        elif not self.existing_hoc_cell is None:
            self.load_morphology_from_hoc()
        elif not self.neuroH5_dict is None:
            self.load_morphology_from_neuroH5(preserve_3d)
        if self.axon:
            self.init_spike_detector()

    def make_standard_soma_and_axon(self, soma_length=16., soma_diam=9., ais_length=30., axon_length=500.):
        """
        This method implements a standardized soma and axon:
        The soma consists of two cylindrical hoc sections of equal length and diameter, connected (0) to (0).
        The basal dendritic tree is connected to soma[1](1), and the apical tree is connected to soma[0](1).
        The axon is attached to soma[0](0), and consists of three sequential hoc sections:
            1) axon[0] : a tapered cylindrical 'axon hillock' section connected to soma[0](0)
            2) axon[1] : a tapered cylindrical 'axon initial segment' section connected to axon[0](1)
            3) axon[2] : a cylindrical 'axon' section connected to axon[1](1)
        :param soma_length: float
        :param soma_diam: float
        :param ais_length: float
        :param axon_length: float
        """
        for index in xrange(2):
            node = self.make_section('soma')
            node.sec.L = soma_length / 2.
            node.sec.diam = soma_diam
            node.append_layer(0)
            self._init_cable(node)  # consults the mech_dict to initialize Ra, cm, and nseg
        self.tree.root = self.soma[0]
        self.soma[1].connect(self.soma[0], 0, 0)
        for index in xrange(3):
            self.make_section('axon')
            self.axon[index].append_layer(0)
        self.axon[0].type = 'axon_hill'
        self.axon[0].sec.L = 10.
        self.axon[0].set_diam_bounds(3., 2.)  # stores the diameter boundaries for a tapered cylindrical section
        self.axon[1].type = 'ais'
        self.axon[1].sec.L = ais_length
        self.axon[1].set_diam_bounds(2., 0.5)  # (2., 1.)
        self.axon[2].sec.L = axon_length
        self.axon[2].sec.diam = 0.5  # 1.
        self.axon[0].connect(self.soma[0], 0, 0)
        self.axon[1].connect(self.axon[0], 1, 0)
        self.axon[2].connect(self.axon[1], 1, 0)
        for node in self.axon:
            self._init_cable(node)

    def load_morphology_from_swc(self, preserve_3d=False):
        """
        This method reads from an .swc file to build an STree2 comprised of SHocNode nodes associated with hoc sections,
        connects the hoc sections, and initializes various parameters: Ra, cm, L, diam, nseg
        :param preserve_3d: bool
        """
        raw_tree = btmorph.STree2()  # import the full tree from an SWC file
        raw_tree.read_SWC_tree_from_file(self.morph_file_path, types=range(10))
        self.clean_swc_diams(raw_tree.root)
        for child in raw_tree.root.children:
            if preserve_3d:
                self.make_3d(child, self.tree.root)
            else:
                self.make_skeleton(child, self.tree.root)

    def load_morphology_from_neuroH5(self, preserve_3d=False):
        """
        This method reads from a dictionary (extracted from a parallelized .hdf5 file) to build an STree2 comprised of
        SHocNode nodes associated with hoc sections, connects the hoc sections, and initializes various parameters: Ra,
        cm, L, diam, nseg
        :param preserve_3d: bool
        """
        raw_tree = self.read_neuroH5_from_dict()
        self.clean_swc_diams(raw_tree.root)
        for child in raw_tree.root.children:
            if preserve_3d:
                self.make_3d(child, self.tree.root)
            else:
                self.make_skeleton(child, self.tree.root)

    def read_neuroH5_from_dict(self):
        """
        Populate a 'btmorph.STree2' object with 'btmorph.SNode2' objects containing standard swc content, as well as
        additional metadata from a dictionary read from neuroH5.
        :return: :class:'STree2'
        """
        raw_tree = btmorph.STree2()
        for i in xrange(len(self.neuroH5_dict['x'])):
            swc_type = self.neuroH5_dict['swc_type'][i]
            x = self.neuroH5_dict['x'][i]
            y = self.neuroH5_dict['y'][i]
            z = self.neuroH5_dict['z'][i]
            radius = self.neuroH5_dict['radius'][i]
            parent_index = self.neuroH5_dict['parent'][i]
            swc_point = btmorph.P3D2(np.array([x, y, z]), radius, swc_type)
            raw_node = btmorph.SNode2(i)
            raw_node.content = {'p3d': swc_point}
            if 'layer' in self.neuroH5_dict:
                layer = self.neuroH5_dict['layer'][i]
                raw_node.content['layer'] = layer
            if 'section' in self.neuroH5_dict:
                sec_index = self.neuroH5_dict['section'][i]
                raw_node.content['section'] = sec_index
            if parent_index == -1:
                raw_tree.root = raw_node
            else:
                raw_tree.add_node_with_parent(raw_node, raw_tree.get_node_with_index(parent_index))

        return raw_tree

    def get_mismatched_neuroH5_sections(self):
        """
        Sanity check that the section indexes contained in the morphology specified in a neuroH5_dict are the same
        indexes in the section lists of this cell object. Either returns a dictionary of the number of points in each
        section, to compare imported and exported, or returns None.

        :return: dict
        """
        mismatched = {'imported': np.array([len(section) for section in
                                            self.neuroH5_dict['section_topology']['nodes'].itervalues()]),
                      'exported': np.array([node.sec.n3d() for node in self.apical])}
        if not np.all(mismatched['imported'] == mismatched['exported']):
            return mismatched
        else:
            return None

    def clean_swc_diams(self, parent):
        """
        Some .swc files that specify unbranched dendrites with high spatial density of spheres have user errors in the
        value for the radius. This method does some reality checking and fixes very small spurious radii.
        :param parent: :class:'SNode2'
        """
        threshold = 0.1  # radius (um)
        for child in parent.children:
            current_radius = child.content['p3d'].radius
            if current_radius < threshold:
                neighbor_radius_list = []
                neighbor_radius_list.append(parent.content['p3d'].radius)
                neighbor_radius_list.extend([grandchild.content['p3d'].radius for grandchild in child.children])
                if np.all(neighbor_radius_list > threshold):
                    child.content['p3d'].radius = np.mean(neighbor_radius_list)
                    # print 'Replacing diam at point', child.index
            self.clean_swc_diams(child)

    def load_morphology_from_hoc(self, existing_hoc_cell):
        """
        In some cases, no .swc file exists, but a morphology has already been imported into a hoc neuron model. This
        method will crawl through the tree structure of the existing cell, and duplicate it within the Python HocCell
        formalism. Each section will be recreated, and it will be stripped of any existing ion channel mechanisms or
        synapses that have been specified. Ignores axon sections, and connects dendritic segments to standardized soma
        and axon.
        :param existing_hoc_cell: :class: 'h.hocObject' : instance of a cell template class already built in hoc
        """
        soma_list = existing_hoc_cell.soma
        basal_list = existing_hoc_cell.basal
        apical_list = existing_hoc_cell.apical
        for sec in soma_list:
            for child in (child for child in sec.children() if child in basal_list or child in apical_list):
                L = child.L
                diam = child.diam
                if child in basal_list:
                    new_node = self.make_section('basal')
                    new_node.sec.L = L
                    new_node.sec.diam = diam
                    new_node.connect(self.soma[1])
                    self.convert_hoc_sections(new_node, child.children())
                elif child in apical_list:
                    new_node = self.make_section('apical')
                    new_node.sec.L = L
                    new_node.sec.diam = diam
                    new_node.connect(self.soma[0])
                    self.convert_hoc_sections(new_node, child.children())

    def convert_hoc_sections(self, parent_node, child_list):
        """

        :param parent_node: :class:'SHocNode'
        :param child_list: list of :class:'h.Section'
        """
        print "Under construction: import morphology from hoc"
        for child in child_list:
            L = child.L
            diam = child.diam
            child.push()
            parent_loc = h.parent_connection()
            child_loc = h.section_orientation()
            h.pop_section()
            new_node = self.make_section(parent_node.type)
            new_node.sec.L = L
            new_node.sec.diam = diam
            new_node.connect(parent_node, parent_loc, child_loc)
            self.convert_hoc_sections(new_node, child.children())

    def make_section(self, sec_type):
        """
        Create a new hoc section to associate with this node, and this cell, and store information about it in the
        node's content dictionary.
        :param sec_type: str
        :return node: :class:'SHocNode'
        """
        node = SHocNode(self.index)
        if self.index == 0:
            self.tree.root = node
        self.index += 1
        node.type = sec_type
        if sec_type in ['spine_head', 'spine_neck']:
            self._node_dict['spine'].append(node)
        else:
            self._node_dict[sec_type].append(node)
        node.sec = h.Section(name=node.name, cell=self)
        return node

    def make_skeleton(self, raw_node, parent, length=0., diams=None):
        """
        Following construction of soma and axon nodes of type 'SHocNode' in the tree of type 'STree2', this method
        recursively converts dendritic 'SNode2' nodes into 'SHocNode' nodes, and connects them to the appropriate
        somatic nodes. Skeletonized dendritic nodes have only one hoc section for each stretch of unbranched dendrite,
        with length equal to the sum of the lengths of the converted SNode2 nodes.
        Nodes that taper more than 0.5 um remain tapered, otherwise they are converted into untapered cylinders with
        diameter equal to the mean diameter of the the converted SNode2 nodes.
        Dendrite types that are pre-categorized as basal, apical, trunk, or tuft in the input .swc file are preserved.
        :param raw_node: :class:'SNode2'
        :param parent: :class:'SHocNode'
        :param length: int or float
        :param diams: None or (list: float)
        """
        global verbose
        dend_types = ([basal_type, apical_type, trunk_type, tuft_type], ['basal', 'apical', 'trunk', 'tuft'])
        swc = raw_node.content['p3d']
        swc_type = swc.type
        if swc_type in dend_types[0]:
            diam = swc.radius * 2.
            length += self.get_node_length_swc(raw_node)
            leaves = len(raw_node.children)
            # create a new node when encountering 1) branch points, 2) terminal ends, 3) change in swc_type,
            # 4) a duplicate node, which is used in some SWC files to indicate a change in layer, or
            # 5) a change in the layer property of a neuroH5/SNode2 content dictionary
            if (leaves > 1 or leaves == 0 or
                    (leaves == 1 and not raw_node.children[0].content['p3d'].type == swc_type) or
                    (leaves == 1 and np.all(raw_node.children[0].content['p3d'].xyz == swc.xyz)) or
                    (leaves == 1 and 'layer' in raw_node.content and raw_node.content['layer'] !=
                        raw_node.children[0].content['layer'])):
                sec_type = dend_types[1][dend_types[0].index(swc_type)]
                new_node = self.make_section(sec_type)
                new_node.sec.L = length
                if (self.tree.is_root(parent)) and (sec_type == 'basal'):
                    parent = self.soma[1]
                new_node.connect(parent)
                if 'layer' in raw_node.content:
                    new_node.append_layer(raw_node.content['layer'])
                if diams is None:
                    new_node.sec.diam = diam
                    self._init_cable(new_node)
                    if verbose:
                        print '{} [nseg: {}, diam: {}, length: {}, parent: {}]'.format(new_node.name, new_node.sec.nseg,
                                                                                       diam, length,
                                                                                       new_node.parent.name)
                else:
                    diams.append(diam)
                    if len(diams) > 2:
                        mean = np.mean(diams)
                        stdev = np.std(diams)
                        if stdev * 2. > 0.5:  # If 95% of the values are within 0.5 um, don't taper
                            new_node.set_diam_bounds(mean + stdev * 2., mean - stdev * 2.)
                            self._init_cable(new_node)
                            if verbose:
                                print '{} [nseg: {}, diam: ({}:{}), length: {}, parent: {}]'.format(new_node.name,
                                                                                                    new_node.sec.nseg,
                                                                                                    mean + stdev,
                                                                                                    mean - stdev,
                                                                                                    length,
                                                                                                    new_node.parent.name)
                        else:
                            new_node.sec.diam = mean
                            self._init_cable(new_node)
                            if verbose:
                                print '{} [nseg: {}, diam: {}, length: {}, parent: {}]'.format(new_node.name,
                                                                                               new_node.sec.nseg, mean,
                                                                                               length,
                                                                                               new_node.parent.name)
                    elif abs(diams[0] - diams[1]) > 0.5:
                        new_node.set_diam_bounds(diams[0], diams[1])
                        self._init_cable(new_node)
                        if verbose:
                            print '{} [diam: ({}:{}), length: {}, parent: {}]'.format(new_node.name, new_node.sec.nseg,
                                                                                      diams[0], diams[1], length,
                                                                                      new_node.parent.name)
                    else:
                        mean = np.mean(diams)
                        new_node.sec.diam = mean
                        self._init_cable(new_node)
                        if verbose:
                            print '{} [nseg: {}, diam: {}, length: {}, parent: {}]'.format(new_node.name,
                                                                                           new_node.sec.nseg, mean,
                                                                                           length, new_node.parent.name)
                # Follow all branches from this fork
                for child in raw_node.children:
                    self.make_skeleton(child, new_node)
            else:  # Follow unbranched dendrite
                if diams is None:
                    diams = [diam]
                else:
                    diams.append(diam)
                self.make_skeleton(raw_node.children[0], parent, length, diams)
        # keep climbing down tree until dendrite nodes are encountered
        else:
            for child in raw_node.children:
                self.make_skeleton(child, parent)

    def make_3d(self, raw_node, parent, current_node=None):
        """
        Following construction of soma and axon nodes of type 'SHocNode' in the tree of type 'STree2', this method
        recursively converts dendritic 'SNode2' nodes into 'SHocNode' nodes, and connects them to the appropriate
        somatic nodes. 3D dendritic nodes have only one hoc section for each stretch of unbranched dendrite,
        with length equal to the sum of the lengths of the converted SNode2 nodes. 3D locations will be preserved, and
        fine scale changes in dendritic diameter will have the resolution of nseg for voltage calculation, but will have
        the full resolution for surface area calculations (needs confirmation).
        Dendrite types that are pre-categorized as basal, apical, trunk, or tuft in the input .swc file are preserved.
        :param raw_node: :class:'SNode2'
        :param parent: :class:'SHocNode'
        :param length: int or float
        :param diams: None or (list: float)
        """
        global verbose
        dend_swc_types = [basal_type, apical_type, trunk_type, tuft_type]
        dend_sec_types = ['basal', 'apical', 'trunk', 'tuft']
        swc = raw_node.content['p3d']
        swc_type = swc.type
        if swc_type in dend_swc_types:
            sec_type = dend_sec_types[dend_swc_types.index(swc_type)]
            if current_node is None:
                if 'section' in raw_node.content and len(self._node_dict[sec_type]) != raw_node.content['section'] and \
                        verbose:
                    print 'HocCell section index %i does not match neuroH5 section index %i' % \
                          (len(self._node_dict[sec_type]), raw_node.content['section'])
                current_node = self.make_section(sec_type)
                current_node.sec.push()
                # some SWC files contain duplicate xyz points at branch points
                raw_parent_xyz = raw_node.parent.content['p3d'].xyz
                if not np.all(raw_parent_xyz == swc.xyz):
                    h.pt3dadd(raw_parent_xyz[0], raw_parent_xyz[1], raw_parent_xyz[2], swc.radius * 2.)
                    if 'layer' in raw_node.parent.content:
                        current_node.append_layer(raw_node.parent.content['layer'])
            else:
                current_node.sec.push()
            h.pt3dadd(swc.xyz[0], swc.xyz[1], swc.xyz[2], swc.radius * 2.)
            if 'layer' in raw_node.content:
                current_node.append_layer(raw_node.content['layer'])
            h.pop_section()
            leaves = len(raw_node.children)
            # create a new node when encountering 1) branch points, 2) terminal ends, 3) change in swc_type,
            # or 4) a duplicate node, which is used in some SWC files to indicate a change in layer
            if (leaves > 1 or leaves == 0 or
                    (leaves == 1 and not raw_node.children[0].content['p3d'].type == swc_type) or
                    (leaves == 1 and np.all(swc.xyz == raw_node.children[0].content['p3d'].xyz))):
                if (leaves == 1 and np.all(swc.xyz == raw_node.children[0].content['p3d'].xyz) and verbose):
                    print 'Encountered duplicate point in ', current_node.name
                if (self.tree.is_root(parent)) and (sec_type == 'basal'):
                    parent = self.soma[1]
                current_node.connect(parent)
                self._init_cable(current_node)
                if verbose:
                    print '{} [nseg: {}, diam: {}, length: {}, parent: {}]'.format(current_node.name,
                                                                                   current_node.sec.nseg,
                                                                                   current_node.sec.diam,
                                                                                   current_node.sec.L,
                                                                                   current_node.parent.name)
                # Follow all branches from this fork
                for child in raw_node.children:
                    self.make_3d(child, current_node)
            else:  # Follow unbranched dendrite
                self.make_3d(raw_node.children[0], parent, current_node)
        # keep climbing down tree until dendrite nodes are encountered
        else:
            for child in raw_node.children:
                self.make_3d(child, parent)

    def export_synapse_attributes_to_neuroH5(self):
        """
        Output a python dictionary in the format expected by neuroH5.io.write_trees_attributes()
        For all putative synapses and/or spines, specifies section indexes (relative to swc_type), swc_type,
        synapse locations (relative to section), synapse types, layer indexes, and unique synapse identifiers (relative
        to cell).
        :return: dict
        """
        syn_locs = []
        section = []
        layer = []
        syn_category = []
        swc_type = []
        syn_id = []
        for sec_type in [sec_type for sec_type in swc_type_enumerator if sec_type in self._node_dict]:
            for section_id, node in enumerate(self._node_dict[sec_type]):
                for i in xrange(len(node.synapse_attributes['syn_locs'])):
                    this_syn_category = node.synapse_attributes['syn_category'][i]
                    syn_category.append(this_syn_category)
                    this_syn_loc = node.synapse_attributes['syn_locs'][i]
                    syn_locs.append(this_syn_loc)
                    section.append(section_id)
                    layer.append(node.get_layer(this_syn_loc))
                    swc_type.append(swc_type_enumerator[sec_type])
                    this_syn_id = node.synapse_attributes['syn_id'][i]
                    syn_id.append(this_syn_id)
        return {'syn_locs': np.array(syn_locs, dtype='float32'),
                'section': np.array(section, dtype='uint32'),
                'layer': np.array(layer, dtype='uint32'),
                'syn_category': np.array(syn_category, dtype='uint32'),
                'syn_id': np.array(syn_id, dtype='uint32'),
                'swc_type': np.array(swc_type, dtype='uint32')}

    def get_nodes_of_subtype(self, sec_type):
        """
        This method searches the node dictionary for nodes of a given sec_type and returns them in a list. Used during
        specification of membrane mechanisms.
        :param sec_type: str
        :return: list of :class:'SHocNode'
        """
        if sec_type in ['axon_hill', 'ais', 'axon']:
            return [node for node in self.axon if node.type == sec_type]
        elif sec_type in ['spine_head', 'spine_neck']:
            return [node for node in self.spine if node.type == sec_type]
        else:
            return self._node_dict[sec_type]

    def load_mech_dict(self, mech_file_path=None):
        """
        This method loads the dictionary specifying membrane mechanism parameters. If a .yaml file is not provided, a
        global variable default_mech_dict from function_lib is used.
        :param mech_file_path: str
        """
        if not mech_file_path is None and os.path.isfile(mech_file_path):
            return read_from_yaml(mech_file_path)
        else:
            local_mech_dict = copy.deepcopy(default_mech_dict)
            return local_mech_dict

    def _init_cable(self, node):
        """
        If the mechanism dictionary specifies the cable properties 'Ra' or 'cm', then _modify_mechanism() properly sets
        those parameters, and reinitializes the number of segments per section. To avoid redundancy, this
        method passes _modify_mechanism() a copy of the dictionary with the spatial_res parameter removed, since this is
        consulted in setting nseg. However, if spatial_res is the only parameter being specified, it is passed to
        _modify_mechanism()
        :param node: :class:'SHocNode'
        """
        sec_type = node.type
        if sec_type in self.mech_dict and 'cable' in self.mech_dict[sec_type]:
            mech_content = copy.deepcopy(self.mech_dict[sec_type]['cable'])
            if ('Ra' in mech_content) or ('cm' in mech_content):
                if 'spatial_res' in mech_content:
                    del mech_content['spatial_res']
                self._modify_mechanism(node, 'cable', mech_content)
            elif 'spatial_res' in mech_content:
                self._modify_mechanism(node, 'cable', mech_content)
        else:
            node.init_nseg()
            node.reinit_diam()

    def reinit_mechanisms(self, reset_cable=False, from_file=False):
        """
        Once a mechanism dictionary has been loaded, and a morphology has been specified, this method traverses through
        the tree of SHocNode nodes following order of inheritance and properly sets membrane mechanism parameters,
        including gradients and inheritance of parameters from nodes along the path from root. Since cable parameters
        are set during specification of morphology, it is not necessary to immediately reinitialize these parameters
        again. However, they can be manually reinitialized with the reset_cable flag.
        :param reset_cable: bool
        :param from_file: bool
        """
        if from_file:
            self.mech_dict = self.load_mech_dict(self.mech_file_path)
        for sec_type in sec_types:
            if sec_type in self.mech_dict:
                nodes = self.get_nodes_of_subtype(sec_type)
                self._reinit_mech(nodes, reset_cable)

    def init_synaptic_mechanisms(self):
        """
        Attributes of potential synapses are stored in the synapse_mechanism_attributes dictionary within each node. Any
        time that synapse attributes are modified, this method can be called to synchronize those attributes with any
        synaptic point processes contained either within a parent section, or child spines.
        """
        for sec_type in ['soma', 'ais', 'basal', 'trunk', 'apical', 'tuft']:
            for node in self.get_nodes_of_subtype(sec_type):
                for syn in self.get_synapses(node):
                    if syn.id is not None and syn.id in node.synapse_mechanism_attributes:
                        for mech_name in (mech_name for mech_name in node.synapse_mechanism_attributes[syn.id]
                                          if mech_name in syn.targets):
                            for param_name, param_val in \
                                    node.synapse_mechanism_attributes[syn.id][mech_name].iteritems():
                                if hasattr(syn.target(mech_name), param_name):
                                    setattr(syn.target(mech_name), param_name, param_val)
                                elif hasattr(syn.netcon(mech_name), param_name):
                                    if param_name == 'weight':
                                        syn.netcon(mech_name).weight[0] = param_val
                                    else:
                                        setattr(syn.netcon(mech_name), param_name, param_val)

    def get_synapses(self, node, syn_type=None):
        """
        Returns a list of all synapse objects contained either directly in the specified node, or in attached spines.
        Can also filter by type of synaptic point process mechanism.
        :param node: :class:'SHocNode'
        :param syn_type: str
        :return: list of :class:'Synapse'
        """
        synapses = [syn for syn in node.synapses if syn_type is None or syn_type in syn.targets]
        for spine in node.spines:
            synapses.extend([syn for syn in spine.synapses if syn_type is None or syn_type in syn.targets])
        return synapses

    def sec_type_has_synapses(self, sec_type, syn_type=None):
        """
        Checks if any nodes of a given sec_type contain synapses, or spines with synapses. Can also check for a synaptic
        point process of a specific type.
        :param sec_type: str
        :param syn_type: str
        :return: boolean
        """
        for node in self.get_nodes_of_subtype(sec_type):
            if self.node_has_synapses(node, syn_type):
                return True
        return False

    def _reinit_mech(self, nodes, reset_cable=False):
        """
        Given a list of nodes, this method loops through all the mechanisms specified in the mechanism dictionary for
        the hoc section type of each node and updates their associated parameters. If the reset_cable flag is True,
        cable parameters are modified first, then the parameters for all other mechanisms are reinitialized.
        Synapse attributes are also specified in the mechanism dictionary, but one must use the method
        init_synaptic_mechanisms() after inserting synapses to synchronize the parameters of inserted synaptic point
        processes.
        :param nodes: list of :class:'SHocNode'
        :param reset_cable: bool
        """
        for node in nodes:
            sec_type = node.type
            if sec_type in self.mech_dict:
                # cable properties must be set first, as they can change nseg, which will affect insertion of membrane
                # mechanism gradients
                if ('cable' in self.mech_dict[sec_type]) and reset_cable:
                    self._init_cable(node)
                for mech_name in (mech_name for mech_name in self.mech_dict[sec_type]
                                  if not mech_name in ['cable', 'ions']):
                    self._modify_mechanism(node, mech_name, self.mech_dict[sec_type][mech_name])
                # ion-related parameters do not exist until after membrane mechanisms have been inserted
                if 'ions' in self.mech_dict[sec_type]:
                    self._modify_mechanism(node, 'ions', self.mech_dict[sec_type]['ions'])

    def reinitialize_subset_mechanisms(self, sec_type, mech_name):
        """
        During parameter optimization, it is often convenient to reinitialize all the parameters for a single mechanism
        in a subset of compartments. For example, g_pas in basal dendrites that inherit the value from the soma after
        modifying the value in the soma compartment.
        :param sec_type: str
        :param mech_name: str
        :return:
        """
        if sec_type in self.mech_dict and mech_name in self.mech_dict[sec_type]:
            for node in self.get_nodes_of_subtype(sec_type):
                self._modify_mechanism(node, mech_name, self.mech_dict[sec_type][mech_name])

    def _modify_mechanism(self, node, mech_name, mech_content):
        """
        This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
        calls self._parse_mech_content to interpret the rules and set the values for the given node.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param mech_content: dict
        """
        if mech_content is not None:
            if 'synapse' in mech_name:
                syn_category = mech_name.split(' ')[0]
                # Only specify synapse attributes if this category of synapses has been specified in this node
                if node.get_filtered_synapse_attributes(syn_category=syn_category)['syn_locs']:
                    for syn_type in mech_content:
                        if mech_content[syn_type] is not None:
                            for param_name in mech_content[syn_type]:
                                # accommodate multiple dict entries with different location constraints for a single
                                # parameter
                                if type(mech_content[syn_type][param_name]) == dict:
                                    self._parse_mech_content(node, mech_name, param_name,
                                                             mech_content[syn_type][param_name], syn_type)
                                else:
                                    for mech_content_entry in mech_content[syn_type][param_name]:
                                        self._parse_mech_content(node, mech_name, param_name, mech_content_entry,
                                                                 syn_type)
            else:
                for param_name in mech_content:
                    # accommodate multiple dict entries with different location constraints for a single parameter
                    if type(mech_content[param_name]) == dict:
                        self._parse_mech_content(node, mech_name, param_name, mech_content[param_name])
                    else:
                        for mech_content_entry in mech_content[param_name]:
                            self._parse_mech_content(node, mech_name, param_name, mech_content_entry)
        else:
            node.sec.insert(mech_name)

    def _parse_mech_content(self, node, mech_name, param_name, rules, syn_type=None):
        """
        This method loops through all the segments in a node and sets the value(s) for a single mechanism parameter by
        interpreting the rules specified in the mechanism dictionary. Properly handles ion channel gradients and
        inheritance of values from the closest segment of a specified type of section along the path from root. Also
        handles rules with distance boundaries, and rules to set synapse attributes. Gradients can be specified as
        linear, exponential, or sigmoidal. Custom functions can also be provided to specify arbitrary distributions.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param rules: dict
        :param syn_type: str
        """
        if 'synapse' in mech_name:
            if syn_type is None:
                raise Exception('Cannot set %s mechanism parameter: %s without a specified point process' %
                                (mech_name, param_name))
        # an 'origin' with no 'value' inherits a starting parameter from the origin sec_type
        # a 'value' with no 'origin' is independent of other sec_types
        # an 'origin' with a 'value' uses the origin sec_type only as a reference point for applying a
        # distance-dependent gradient
        if 'origin' in rules:
            if rules['origin'] == 'parent':
                if node.type == 'spine_head':
                    donor = node.parent.parent.parent
                elif node.type == 'spine_neck':
                    donor = node.parent.parent
                else:
                    donor = node.parent
            elif rules['origin'] == 'branch_origin':
                donor = self.get_dendrite_origin(node)
            elif rules['origin'] in sec_types:
                donor = self._get_node_along_path_to_root(node, rules['origin'])
            else:
                if 'synapse' in mech_name:
                    raise Exception('%s mechanism: %s parameter: %s cannot inherit from unknown origin: %s' %
                                    (mech_name, syn_type, param_name, rules['origin']))
                else:
                    raise Exception('Mechanism: {} parameter: {} cannot inherit from unknown origin: {}'.format(
                        mech_name, param_name, rules['origin']))
        else:
            donor = None
        if 'value' in rules:
            baseline = rules['value']
        elif donor is None:
            if 'synapse' in mech_name:
                raise Exception('Cannot set %s mechanism: %s parameter: %s without a specified origin or value' %
                                (mech_name, syn_type, param_name))
            else:
                raise Exception('Cannot set mechanism: {} parameter: {} without a specified origin or value'.format(
                    mech_name, param_name))
        else:
            if (mech_name == 'cable') and (param_name == 'spatial_res'):
                baseline = self._get_spatial_res(donor)
            elif 'synapse' in mech_name:
                baseline = self._inherit_mech_param(donor, mech_name, param_name, syn_type)
                if baseline is None:
                    raise Exception('Cannot inherit %s mechanism: %s parameter: %s from sec_type: %s' %
                                    (mech_name, syn_type, param_name, donor.type))
            else:
                baseline = self._inherit_mech_param(donor, mech_name, param_name)
        if mech_name == 'cable':  # cable properties can be inherited, but cannot be specified as gradients
            if param_name == 'spatial_res':
                node.init_nseg(baseline)
            else:
                setattr(node.sec, param_name, baseline)
                node.init_nseg(self._get_spatial_res(node))
            node.reinit_diam()
        else:
            if 'custom' in rules:
                if hasattr(self, rules['custom']['method']):
                    method_to_call = getattr(self, rules['custom']['method'])
                    method_to_call(node, mech_name, param_name, baseline, rules, syn_type, donor)
                else:
                    raise Exception('The custom method %s is not defined for this cell type.' %
                                    rules['custom']['method'])
            elif 'min_loc' in rules or 'max_loc' in rules or 'slope' in rules:
                if 'synapse' in mech_name:
                    if donor is None:
                        raise Exception('Cannot specify %s mechanism: %s parameter: %s without a provided origin' %
                                        (mech_name, syn_type, param_name))
                    else:
                        self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
                else:
                    if donor is None:
                        raise Exception('Cannot specify mechanism: %s parameter: %s without a provided origin' %
                                        (mech_name, param_name))
                    self._specify_mech_parameter(node, mech_name, param_name, baseline, rules, donor)
            elif mech_name == 'ions':
                setattr(node.sec, param_name, baseline)
            elif 'synapse' in mech_name:
                self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type)
            else:
                node.sec.insert(mech_name)
                setattr(node.sec, param_name + "_" + mech_name, baseline)

    def _specify_mech_parameter(self, node, mech_name, param_name, baseline, rules, donor=None):
        """

        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param donor: :class:'SHocNode' or None
        """
        if donor is None:
            raise Exception('Cannot specify mechanism: {} parameter: {} without a provided origin'.format(
                mech_name, param_name))
        if 'min_loc' in rules:
            min_distance = rules['min_loc']
        else:
            min_distance = None
        if 'max_loc' in rules:
            max_distance = rules['max_loc']
        else:
            max_distance = None
        min_seg_distance = self.get_distance_to_node(donor, node, 0.5 / node.sec.nseg)
        max_seg_distance = self.get_distance_to_node(donor, node, (0.5 + node.sec.nseg - 1) / node.sec.nseg)
        # if any part of the section is within the location constraints, insert the mechanism, and specify
        # the parameter at the segment level
        if (min_distance is None or max_seg_distance >= min_distance) and \
                (max_distance is None or min_seg_distance <= max_distance):
            if not mech_name == 'ions':
                node.sec.insert(mech_name)
            if min_distance is None:
                min_distance = 0.
            for seg in node.sec:
                seg_loc = self.get_distance_to_node(donor, node, seg.x)
                if seg_loc >= min_distance and (max_distance is None or seg_loc <= max_distance):
                    if 'slope' in rules:
                        seg_loc -= min_distance
                        if 'tau' in rules:
                            if 'xhalf' in rules:  # sigmoidal gradient
                                offset = baseline - rules['slope'] / (1. + np.exp(rules['xhalf'] / rules['tau']))
                                value = offset + rules['slope'] /\
                                                 (1. + np.exp((rules['xhalf'] - seg_loc) / rules['tau']))
                            else:  # exponential gradient
                                offset = baseline - rules['slope']
                                value = offset + rules['slope'] * np.exp(seg_loc / rules['tau'])
                        else:  # linear gradient
                            value = baseline + rules['slope'] * seg_loc
                        if 'min' in rules and value < rules['min']:
                            value = rules['min']
                        elif 'max' in rules and value > rules['max']:
                            value = rules['max']
                    else:
                        value = baseline
                # by default, if only some segments in a section meet the location constraints, the parameter inherits
                # the mechanism's default value. if another value is desired, it can be specified via an 'outside' key
                # in the mechanism dictionary entry
                elif 'outside' in rules:
                    value = rules['outside']
                else:
                    value = None
                if value is not None:
                    if mech_name == 'ions':
                        setattr(seg, param_name, value)
                    else:
                        setattr(getattr(seg, mech_name), param_name, value)

    def _specify_synaptic_parameter(self, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
        """
        This method interprets an entry from the mechanism dictionary to set parameters for synapse_mechanism_attributes
        contained in this node. Appropriately implements slopes and inheritances.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param syn_type: str
        :param donor: :class:'SHocNode' or None
        """
        syn_category = mech_name.split(' ')[0]
        if 'min_loc' in rules:
            min_distance = rules['min_loc']
        else:
            min_distance = 0.
        if 'max_loc' in rules:
            max_distance = rules['max_loc']
        else:
            max_distance = None
        if 'variance' in rules and rules['variance'] == 'normal':
            normal = True
        else:
            normal = False
        this_synapse_attributes = node.get_filtered_synapse_attributes(syn_category=syn_category)
        for i in xrange(len(this_synapse_attributes['syn_locs'])):
            loc = this_synapse_attributes['syn_locs'][i]
            this_syn_id = this_synapse_attributes['syn_id'][i]
            if this_syn_id not in node.synapse_mechanism_attributes:
                node.synapse_mechanism_attributes[this_syn_id] = {}
            if syn_type not in node.synapse_mechanism_attributes[this_syn_id]:
                node.synapse_mechanism_attributes[this_syn_id][syn_type] = {}
            if donor is None:
                value = baseline
            else:
                distance = self.get_distance_to_node(donor, node, loc)
                # If only some synapses in a section meet the location constraints, the synaptic parameter will
                # maintain its default value in all other locations. values for other locations must be specified
                # with an additional entry in the mechanism dictionary.
                if distance >= min_distance and (max_distance is None or distance <= max_distance):
                    if 'slope' in rules:
                        distance -= min_distance
                        if 'tau' in rules:
                            if 'xhalf' in rules:  # sigmoidal gradient
                                offset = baseline - rules['slope'] / (1. + np.exp(rules['xhalf'] / rules['tau']))
                                value = offset + rules['slope'] / (1. + np.exp((rules['xhalf'] - distance) /
                                                                               rules['tau']))
                            else:  # exponential gradient
                                offset = baseline - rules['slope']
                                value = offset + rules['slope'] * np.exp(distance / rules['tau'])
                        else:  # linear gradient
                            value = baseline + rules['slope'] * distance
                        if 'min' in rules and value < rules['min']:
                            value = rules['min']
                        elif 'max' in rules and value > rules['max']:
                            value = rules['max']
                    else:
                        value = baseline
            if normal:
                value = self.random.normal(value, value / 6.)
            node.synapse_mechanism_attributes[this_syn_id][syn_type][param_name] = value

    def init_spike_detector(self, node=None, loc=1., param='_ref_v', delay=None, weight=None, threshold=None,
                            target=None):
        """
        Converts analog voltage in the specified section to digital spike output. By default, initializes an h.NetCon
        object with voltage as a reference parameter and no target. Can later re-initialized with a target on a cell
        contained within the local processing environment.
        :param node: :class:'SHocNode'
        :param loc: float
        :param param: str
        :param delay: float
        :param weight: float
        :param threshold: float
        :param target: object that can receive spikes
        :return: :class:'h.NetCon'
        """
        if node is None:
            if self.axon:
                node = self.axon[-1]
            else:
                raise Exception('No source node specified for spike detector.')
        if self.spike_detector is not None:
            if delay is None:
                delay = self.spike_detector.delay
            if weight is None:
                weight = self.spike_detector.weight[0]
            if threshold is None:
                threshold = self.spike_detector.threshold
        else:
            if delay is None:
                delay = 0.
            if weight is None:
                weight = 1.
            if threshold is None:
                threshold = -30.
        self.spike_detector = h.NetCon(getattr(node.sec(loc), param), target, sec=node.sec)
        self.spike_detector.delay = delay
        self.spike_detector.weight[0] = weight
        self.spike_detector.threshold = threshold

    def get_dendrite_origin(self, node, parent_type=None):
        """
        This method determines the section type of the given node, and returns the node representing the primary branch
        point for the given section type. Basal and trunk sections originate at the soma, and apical and tuft dendrites
        originate at the trunk. For spines, recursively calls with parent node to identify the parent branch first.
        :param node: :class:'SHocNode'
        :return: :class:'SHocNode'
        """
        sec_type = node.type
        if sec_type in ['spine_head', 'spine_neck']:
            return self.get_dendrite_origin(node.parent, parent_type)
        elif parent_type is not None:
            return self._get_node_along_path_to_root(node.parent, parent_type)
        elif sec_type in ['basal', 'trunk', 'axon_hill', 'ais', 'axon']:
            return self._get_node_along_path_to_root(node, 'soma')
        elif sec_type in ['apical', 'tuft']:
            if self._node_dict['trunk']:
                return self._get_node_along_path_to_root(node, 'trunk')
            else:
                return self._get_node_along_path_to_root(node, 'soma')
        elif sec_type == 'soma':
            return node

    def _get_node_along_path_to_root(self, node, sec_type):
        """
        This method follows the path from the given node to the root node, and returns the first node with section type
        sec_type.
        :param node: :class:'SHocNode'
        :param sec_type: str
        :return: :class:'SHocNode'
        """
        parent = node
        while not parent is None:
            if parent in self.soma and not sec_type == 'soma':
                parent = None
            elif parent.type == sec_type:
                return parent
            else:
                parent = parent.parent
        raise Exception('The path from node: {} to root does not contain sections of type: {}'.format(node.name,
                                                                                                      sec_type))

    def _get_closest_synapse(self, node, loc, syn_type=None, downstream=True):
        """
        This method finds the closest synapse to the specified location within or downstream of the provided node. Used
        for inheritance of synaptic mechanism parameters. Can also look upstream instead. Can also find the closest
        synapse containing a synaptic point_process of a specific type.
        :param node: :class:'SHocNode'
        :param loc: float
        :param syn_type: str
        :return: :class:'Synapse'
        """

        syn_list = [syn for syn in node.synapses if syn_type is None or syn_type in syn._syn]
        for spine in node.spines:
            syn_list.extend([syn for syn in spine.synapses if syn_type is None or syn_type in syn._syn])
        if not syn_list:
            if downstream:
                for child in [child for child in node.children if child.type == node.type]:
                    target_syn = self._get_closest_synapse(child, 0., syn_type)
                    if target_syn is not None:
                        return target_syn
                return None
            elif node.parent.type == node.type:
                return self._get_closest_synapse(node.parent, 1., syn_type, downstream=False)
            else:
                return None
        else:
            min_distance = 1.
            target_syn = None
            for syn in syn_list:
                distance = abs(syn.loc - loc)
                if distance < min_distance:
                    min_distance = distance
                    target_syn = syn
            return target_syn

    def _get_closest_synapse_attribute(self, node, loc, syn_category, syn_type=None, downstream=True):
        """
        This method finds the closest synapse_attribute to the specified location within or downstream of the specified
        node. Used for inheritance of synaptic mechanism parameters. Can also look upstream instead. Can also find the
        closest synapse_attribute specifying parameters of a synaptic point_process of a specific type.
        :param node: :class:'SHocNode'
        :param loc: float
        :param syn_category: str
        :param syn_type: str
        :param downstream: bool
        :return: tuple: (:class:'SHocNode', int) : node containing synapse, syn_id
        """
        min_distance = 1.
        target_index = None
        this_synapse_attributes = node.get_filtered_synapse_attributes(syn_category=syn_category, syn_type=syn_type)
        if this_synapse_attributes['syn_locs']:
            for i in xrange(len(this_synapse_attributes['syn_locs'])):
                this_syn_loc = this_synapse_attributes['syn_locs'][i]
                distance = abs(loc - this_syn_loc)
                if distance < min_distance:
                    min_distance = distance
                    target_index = this_synapse_attributes['syn_id'][i]
            return node, target_index
        else:
            if downstream:
                for child in (child for child in node.children if child.type not in ['spine_head', 'spine_neck']):
                    target_node, target_index = self._get_closest_synapse_attribute(child, 0., syn_category, syn_type)
                    if target_index is not None:
                        return target_node, target_index
                return node, None
            elif node.parent is not None:  # stop at the root
                return self._get_closest_synapse_attribute(node.parent, 1., syn_category, syn_type, downstream)
            else:
                return node, None

    def _inherit_mech_param(self, donor, mech_name, param_name, syn_type=None):
        """
        When the mechanism dictionary specifies that a node inherit a parameter value from a donor node, this method
        returns the value of that parameter found in the section or final segment of the donor node. For synaptic
        mechanism parameters, searches for the closest synapse_attribute in the donor node. If the donor node does not
        contain synapse_mechanism_attributes due to location constraints, this method searches first child nodes, then
        nodes along the path to root.
        :param donor: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param syn_type: str
        :return: float
        """
        # accesses the last segment of the section
        loc = donor.sec.nseg / (donor.sec.nseg + 1.)
        try:
            if mech_name in ['cable', 'ions']:
                if mech_name == 'cable' and param_name == 'Ra':
                    return getattr(donor.sec, param_name)
                else:
                    return getattr(donor.sec(loc), param_name)
            elif 'synapse' in mech_name:
                # first look downstream for a nearby synapse, then upstream.
                syn_category = mech_name.split(' ')[0]
                target_node, target_index = self._get_closest_synapse_attribute(donor, 1., syn_category, syn_type,
                                                                                downstream=True)
                if target_index is None and donor.parent is not None:
                    target_node, target_index = self._get_closest_synapse_attribute(donor.parent, 1., syn_category,
                                                                                    syn_type, downstream=False)
                if target_index is not None \
                        and param_name in target_node.synapse_mechanism_attributes[target_index][syn_type]:
                    return target_node.synapse_mechanism_attributes[target_index][syn_type][param_name]
                else:
                    return None
            else:
                return getattr(getattr(donor.sec(loc), mech_name), param_name)
        except (AttributeError, NameError, KeyError):
            if syn_type is None:
                print 'Exception: Problem inheriting mechanism: {} parameter: {} from sec_type: {}'.format(
                    mech_name, param_name, donor.type)
            else:
                print 'Exception: Problem inheriting %s mechanism: %s parameter: %s from sec_type: %s' % \
                      (mech_name, syn_type, param_name, donor.type)
            raise KeyError

    def _get_spatial_res(self, node):
        """
        Checks the mechanism dictionary if the section type of this node has a specified spatial resolution factor.
        Used to scale the number of segments per section in the hoc model by a factor of an exponent of 3.
        :param node: :class:'SHocNode
        :return: int
        """
        try:  # if spatial_res has not been specified for the origin type of section, it defaults to 0
            rules = self.mech_dict[node.type]['cable']['spatial_res']
        except KeyError:
            return 0
        if 'value' in rules:
            return rules['value']
        elif 'origin' in rules:
            if rules['origin'] in sec_types:  # if this sec_type also inherits the value, continue following the path
                return self._get_spatial_res(self._get_node_along_path_to_root(node, rules['origin']))
            else:
                print 'Exception: Spatial resolution cannot be inherited from sec_type: {}'.format(rules['origin'])
                raise KeyError
        else:
            print 'Exception: Cannot set spatial resolution without a specified origin or value'
            raise KeyError

    def modify_mech_param(self, sec_type, mech_name, param_name=None, value=None, origin=None, slope=None, tau=None,
                          xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, syn_type=None,
                          variance=None, replace=True, custom=None):
        """
        Modifies or inserts new membrane mechanisms into hoc sections of type sec_type. First updates the mechanism
        dictionary, then sets the corresponding hoc parameters. This method is meant to be called manually during
        initial model specification, or during parameter optimization. For modifications to persist across simulations,
        the mechanism dictionary must be saved to a file using self.export_mech_dict() and re-imported during HocCell
        initialization.
        :param sec_type: str
        :param mech_name: str
        :param param_name: str
        :param value: float
        :param origin: str
        :param slope: float
        :param tau: float
        :param xhalf: float
        :param min: float
        :param max: float
        :param min_loc: float
        :param max_loc: float
        :param outside: float
        :param syn_type: str
        :param variance: str
        :param replace: bool
        :param custom: dict
        """
        global verbose
        if 'synapse' in mech_name:
            self._modify_synaptic_mech_param(sec_type, mech_name, param_name, value, origin, slope, tau, xhalf, min,
                                             max, min_loc, max_loc, outside, syn_type, variance, replace, custom)
            return
        backup_content = None
        mech_content = None
        if not sec_type in sec_types:
            raise Exception('Cannot specify mechanism: {} parameter: {} for unknown sec_type: {}'.format(mech_name,
                                                                                                         param_name,
                                                                                                         sec_type))
        if param_name is None:
            if mech_name in ['cable', 'ions']:
                raise Exception('No parameter specified for mechanism: {}'.format(mech_name))
        if not param_name is None:
            if value is None and origin is None:
                raise Exception('Cannot set mechanism: {} parameter: {} without a specified origin or value'.format(
                    mech_name, param_name))
            rules = {}
            if not origin is None:
                if not origin in sec_types + ['parent', 'branch_origin']:
                    raise Exception('Cannot inherit mechanism: {} parameter: {} from unknown origin: {}'.format(
                        mech_name, param_name, origin))
                else:
                    rules['origin'] = origin
            if not custom is None:
                rules['custom'] = custom
            if not value is None:
                rules['value'] = value
            if not slope is None:
                rules['slope'] = slope
            if not tau is None:
                rules['tau'] = tau
            if not xhalf is None:
                rules['xhalf'] = xhalf
            if not min is None:
                rules['min'] = min
            if not max is None:
                rules['max'] = max
            if not min_loc is None:
                rules['min_loc'] = min_loc
            if not max_loc is None:
                rules['max_loc'] = max_loc
            if not outside is None:
                rules['outside'] = outside
            # currently only implemented for synaptic parameters
            if not variance is None:
                rules['variance'] = variance
            mech_content = {param_name: rules}
        # No mechanisms have been inserted into this type of section yet
        if not sec_type in self.mech_dict:
            self.mech_dict[sec_type] = {mech_name: mech_content}
        # This mechanism has not yet been inserted into this type of section
        elif not mech_name in self.mech_dict[sec_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = mech_content
        # This mechanism has been inserted, but no parameters have been specified
        elif self.mech_dict[sec_type][mech_name] is None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = mech_content
        # This parameter has already been specified
        elif param_name is not None and param_name in self.mech_dict[sec_type][mech_name]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            # Determine whether to replace or extend the current dictionary entry.
            if replace:
                self.mech_dict[sec_type][mech_name][param_name] = rules
            elif type(self.mech_dict[sec_type][mech_name][param_name]) == dict:
                self.mech_dict[sec_type][mech_name][param_name] = [self.mech_dict[sec_type][mech_name][param_name],
                                                                   rules]
            elif type(self.mech_dict[sec_type][mech_name][param_name]) == list:
                self.mech_dict[sec_type][mech_name][param_name].append(rules)
        # This mechanism has been inserted, but this parameter has not yet been specified
        elif param_name is not None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][param_name] = rules

        try:
            nodes = self.get_nodes_of_subtype(sec_type)
            # all membrane mechanisms in sections of type sec_type must be reinitialized after changing cable properties
            if mech_name == 'cable':
                if param_name in ['Ra', 'cm', 'spatial_res']:
                    self._reinit_mech(nodes, reset_cable=True)
                else:
                    print 'Exception: Unknown cable property: {}'.format(param_name)
                    raise KeyError
            else:
                for node in nodes:
                    try:
                        self._modify_mechanism(node, mech_name, mech_content)
                    except (AttributeError, NameError, ValueError, KeyError):
                        raise KeyError
        except KeyError:
            if backup_content is None:
                del self.mech_dict[sec_type]
            else:
                self.mech_dict[sec_type] = copy.deepcopy(backup_content)
            if not param_name is None:
                raise Exception('Problem specifying mechanism: %s parameter: %s in node: %s' %
                                (mech_name, param_name, node.name))
            else:
                raise Exception('Problem specifying mechanism: %s in node: %s' %
                                (mech_name, node.name))

    def _modify_synaptic_mech_param(self, sec_type, mech_name=None, param_name=None, value=None, origin=None,
                                    slope=None, tau=None, xhalf=None, min=None, max=None, min_loc=None, max_loc=None,
                                    outside=None, syn_type=None, variance=None, replace=True, custom=None):

        """
        Attributes of synaptic point processes are stored in the synapse_mechanism_attributes dictionary of each node.
        This method first updates the mechanism dictionary, then replaces or creates synapse_mechanism_attributes in
        nodes of type sec_type. Handles special nested dictionary specification for synaptic parameters.
        :param sec_type: str
        :param mech_name: str
        :param param_name: str
        :param value: float
        :param origin: str
        :param slope: float
        :param tau: float
        :param xhalf: float
        :param min: float
        :param max: float
        :param min_loc: float
        :param max_loc: float
        :param outside: float
        :param syn_type: str
        :param variance: str
        :param replace: bool
        :param custom: dict
        """
        global verbose
        backup_content = None
        mech_content = None
        if syn_type is None:
            raise Exception('Cannot specify %s mechanism parameters without a specified type of synaptic point process.'
                            % mech_name)
        if not sec_type in sec_types:
            raise Exception('Cannot specify %s mechanism: %s parameter: %s for unknown sec_type: %s' %
                            (mech_name, syn_type, param_name, sec_type))
        if not param_name is None:
            if value is None and origin is None:
                raise Exception('Cannot set %s mechanism: %s parameter: %s without a specified origin or value' %
                                (mech_name, syn_type, param_name))
            rules = {}
            if not origin is None:
                if not origin in sec_types + ['parent', 'branch_origin']:
                    raise Exception('Cannot inherit %s mechanism: %s parameter: %s from unknown origin: %s' %
                                    (mech_name, syn_type, param_name, origin))
                else:
                    rules['origin'] = origin
            if not custom is None:
                rules['custom'] = custom
            if not value is None:
                rules['value'] = value
            if not slope is None:
                rules['slope'] = slope
            if not tau is None:
                rules['tau'] = tau
            if not xhalf is None:
                rules['xhalf'] = xhalf
            if not min is None:
                rules['min'] = min
            if not max is None:
                rules['max'] = max
            if not min_loc is None:
                rules['min_loc'] = min_loc
            if not max_loc is None:
                rules['max_loc'] = max_loc
            if not outside is None:
                rules['outside'] = outside
            if not variance is None:
                rules['variance'] = variance
            mech_content = {param_name: rules}
        # No mechanisms have been inserted into this type of section yet
        if not sec_type in self.mech_dict:
            self.mech_dict[sec_type] = {mech_name: {syn_type: mech_content}}
        # No synapse attributes have been specified in this type of section yet
        elif not mech_name in self.mech_dict[sec_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = {syn_type: mech_content}
        # This synaptic mechanism has not yet been specified in this type of section
        elif not syn_type in self.mech_dict[sec_type][mech_name]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type] = mech_content
        # This synaptic mechanism has been specified, but no parameters have been specified
        elif self.mech_dict[sec_type][mech_name][syn_type] is None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type] = mech_content
        # This parameter has already been specified.
        elif param_name is not None and param_name in self.mech_dict[sec_type][mech_name][syn_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            # Determine whether to replace or extend the current dictionary entry.
            if replace:
                self.mech_dict[sec_type][mech_name][syn_type][param_name] = rules
            elif type(self.mech_dict[sec_type][mech_name][syn_type][param_name]) == dict:
                self.mech_dict[sec_type][mech_name][syn_type][param_name] = \
                    [self.mech_dict[sec_type][mech_name][syn_type][param_name], rules]
            elif type(self.mech_dict[sec_type][mech_name][syn_type][param_name]) == list:
                self.mech_dict[sec_type][mech_name][syn_type][param_name].append(rules)
        # This synaptic mechanism has been specified, but this parameter has not yet been specified
        elif param_name is not None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type][param_name] = rules

        for node in self.get_nodes_of_subtype(sec_type):
            try:
                self._modify_mechanism(node, mech_name, {syn_type: mech_content})
            except (AttributeError, NameError, ValueError, KeyError):
                if backup_content is None:
                    del self.mech_dict[sec_type]
                else:
                    self.mech_dict[sec_type] = copy.deepcopy(backup_content)
                if param_name is not None:
                    raise Exception('Problem specifying %s mechanism: %s parameter: %s in node: %s' %
                                    (mech_name, syn_type, param_name, node.name))
                else:
                    raise Exception('Problem specifying %s mechanism: %s in node: %s' %
                                    (mech_name, syn_type, node.name))

    def export_mech_dict(self, mech_file_path=None):
        """
        Following modifications to the mechanism dictionary either during model specification or parameter optimization,
        this method stores the current mech_dict to a pickle file stamped with the date and time. This allows the
        current set of mechanism parameters to be recalled later.
        """
        if mech_file_path is None:
            mech_file_path = data_dir + 'mech_dict_' + datetime.datetime.today().strftime('%m%d%Y%H%M') + '.yaml'
        write_to_yaml(mech_file_path, self.mech_dict)
        print "Exported mechanism dictionary to " + mech_file_path

    def get_node_by_distance_to_soma(self, distance, sec_type):
        """
        Gets the first node of the given section type at least the given distance from a soma node.
        Not particularly useful, since it will always return the same node.
        :param distance: int or float
        :param sec_type: str
        :return: :class:'SHocNode'
        """
        nodes = self._node_dict[sec_type]
        for node in nodes:
            if self.get_distance_to_node(self.tree.root, node) >= distance:
                return node
        raise Exception('No node is {} um from a soma node.'.format(distance))

    def get_distance_to_node(self, root, node, loc=None):
        """
        Returns the distance from the given location on the given node to its connection with a root node.
        :param root: :class:'SHocNode'
        :param node: :class:'SHocNode'
        :param loc: float
        :return: int or float
        """
        length = 0.
        if node in self.soma:
            return length
        if not loc is None:
            length += loc * node.sec.L
        if root in self.soma:
            while not node.parent in self.soma:
                node.sec.push()
                loc = h.parent_connection()
                h.pop_section()
                node = node.parent
                length += loc * node.sec.L
        elif self.node_in_subtree(root, node):
            while not node.parent is root:
                node.sec.push()
                loc = h.parent_connection()
                h.pop_section()
                node = node.parent
                length += loc * node.sec.L
        else:
            return None  # node is not connected to root
        return length

    def node_in_subtree(self, root, node):
        """
        Checks if a node is contained within a subtree of root.
        :param root: 'class':SNode2 or SHocNode
        :param node: 'class':SNode2 or SHocNode
        :return: boolean
        """
        nodelist = []
        self.tree._gather_nodes(root, nodelist)
        if node in nodelist:
            return True
        else:
            return False

    def get_path_length_swc(self, path):
        """
        Calculates the distance between nodes given a list of SNode2 nodes connected in a path.
        :param path: list : :class:'SNode2'
        :return: int or float
        """
        distance = 0.
        for i in xrange(len(path) - 1):
            distance += np.sqrt(np.sum((path[i].content['p3d'].xyz - path[i + 1].content['p3d'].xyz) ** 2.))
        return distance

    def get_node_length_swc(self, raw_node):
        """
        Calculates the distance between the center points of an SNode2 node and its parent.
        :param raw_node: :class:'SNode2'
        :return: float
        """
        if not raw_node.parent is None:
            return np.sqrt(np.sum((raw_node.content['p3d'].xyz - raw_node.parent.content['p3d'].xyz) ** 2.))
        else:
            return 0.

    def get_branch_order(self, node):
        """
        Calculates the branch order of a SHocNode node. The order is defined as 0 for all soma, axon, and apical trunk
        dendrite nodes, but defined as 1 for basal dendrites that branch from the soma, and apical and tuft dendrites
        that branch from the trunk. Increases by 1 after each additional branch point. Makes sure not to count spines.
        :param node: :class:'SHocNode'
        :return: int
        """
        if node.type in ['soma', 'axon_hill', 'ais', 'axon']:
            return 0
        elif node.type == 'trunk':
            children = [child for child in node.parent.children if not child.type == 'spine_neck']
            if len(children) > 1 and children[0].type == 'trunk' and children[1].type == 'trunk':
                return 1
            else:
                return 0
        else:
            order = 0
            path = [branch for branch in self.tree.path_between_nodes(node, self.get_dendrite_origin(node)) if
                    not branch.type in ['soma', 'trunk']]
            for node in path:
                if self.is_terminal(node):
                    order += 1
                elif len([child for child in node.parent.children if not child.type == 'spine_neck']) > 1:
                    order += 1
                elif node.parent.type == 'trunk':
                    order += 1
            return order

    def is_terminal(self, node):
        """
        Calculates if a node is a terminal dendritic branch.
        :param node: :class:'SHocNode'
        :return: bool
        """
        if node.type in ['soma', 'axon_hill', 'ais', 'axon']:
            return False
        else:
            return not bool([child for child in node.children if not child.type == 'spine_neck'])

    def is_bifurcation(self, node, child_type):
        """
        Calculates if a node bifurcates into at least two children of specified type.
        :param node: :class:'SHocNode'
        :param child_type: string
        :return: bool
        """
        return len([child for child in node.children if child.type == child_type]) >= 2

    def set_stochastic_synapses(self, value):
        """
        This method turns stochastic filtering of presynaptic release on or off for all synapses contained in this cell.
        :param value: int in [0, 1]
        """
        for nodelist in self._node_dict.itervalues():
            for node in nodelist:
                for syn in node.synapses:
                    syn.stochastic = value

    def insert_spines(self, sec_type_list=None):
        """
        This method inserts explicit 'spine_head' and 'spine_neck' compartments at every pre-specified excitatory
        synapse location.
        :param syn_category: str
        :param sec_type_list: list of str
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                for loc in node.get_filtered_synapse_attributes(syn_category=syn_category)['syn_locs']:
                    self.insert_spine(node, loc)
        self._reinit_mech(self.spine)

    def append_synapse_attributes_by_density(self, node, density, syn_category):
        """
        Given a mean synapse density in /um, return a list of synapse locations at the specified density.
        :param node: :class:'SHocNode'
        :param density: float: mean density in /um
        :param syn_category: str
        """
        L = node.sec.L
        beta = 1. / density
        interval = self.random.exponential(beta)
        while interval < L:
            loc = interval / L
            node.append_synapse_attribute(syn_category, loc)
            interval += self.random.exponential(beta)

    def append_synapse_attributes_by_layer(self, node, density_dict, syn_category):
        """
        This method populates a node with putative synapse locations of the specified type following layer-specific
        rules for synapse density.
        TODO: Create a consistent way to specify and interpret rules and gradients in the density_dict.
        :param node: :class:'SHocNode'
        :param density_dict: dict
        :param syn_category: str
        """
        if node.get_layer() is None:
            raise Exception('Cannot specify synapse density by layer without first specifying dendritic layers.')
        distance = 0.
        x = 0.
        L = node.sec.L
        point_index = 0
        while distance <= L:
            layer = node.get_layer(x)
            while layer not in density_dict:
                while point_index < node.sec.n3d() and node.sec.arc3d(point_index) <= distance:
                    point_index += 1
                if point_index >= node.sec.n3d():
                    break
                distance = node.sec.arc3d(point_index)
                x = distance / L
                layer = node.get_layer(x)
            if layer not in density_dict:
                break
            density = density_dict[layer]
            interval = self.random.exponential(1. / density)
            distance += interval
            if distance > L:
                break
            x = distance / L
            node.append_synapse_attribute(syn_category, x)

    def insert_spine(self, node, parent_loc, child_loc=0):
        """
        Spines consist of two hoc sections: a cylindrical spine head and a cylindrical spine neck.
        :param node: :class:'SHocNode'
        :param parent_loc: float
        :param child_loc: int
        """
        neck = self.make_section('spine_neck')
        neck.connect(node, parent_loc, child_loc)
        neck.sec.L = 1.58
        neck.sec.diam = 0.077
        self._init_cable(neck)
        head = self.make_section('spine_head')
        head.connect(neck)
        node.spines.append(head)
        head.sec.L = 0.5  # open cylinder, matches surface area of sphere with diam = 0.5
        head.sec.diam = 0.5
        self._init_cable(head)

    def insert_synapses_in_spines(self, sec_type_list=None, syn_types=None, stochastic=False):
        """
        Inserts synapse of the specified type(s) in spines attached to nodes of the specified sec_types.
        :param sec_type_list: str
        :param syn_types: list of str
        :param stochastic: int
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
        if syn_types is None:
            syn_types = ['AMPA_KIN', 'NMDA_KIN5']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                for i, syn_id in enumerate(node.get_filtered_synapse_attributes(syn_category=syn_category)['syn_id']):
                    spine = node.spines[i]
                    syn = Synapse(self, spine, type_list=syn_types, stochastic=stochastic, loc=0.5, id=syn_id)
        self.init_synaptic_mechanisms()

    def insert_synapses(self, syn_category=None, syn_types=None, sec_type_list=None, stochastic=False):
        """
        Inserts synapses of specified type(s) in nodes of the specified sec_types at the pre-determined putative
        synapse locations.
        :param syn_category: str
        :param syn_types: list of str
        :param sec_type_list: list of str
        :param stochastic: int
        """
        if syn_category is None:
            syn_category = 'excitatory'
        if sec_type_list is None:
            if syn_category == 'excitatory':
                sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
            elif syn_category == 'inhibitory':
                sec_type_list = ['soma', 'ais', 'basal', 'trunk', 'apical', 'tuft']
        if syn_types is None:
            if syn_category == 'excitatory':
                syn_types = ['AMPA_KIN', 'NMDA_KIN5']
            elif syn_category == 'inhibitory':
                syn_types = ['GABA_A_KIN']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                this_synapse_attribute = node.get_filtered_synapse_attributes(syn_category=syn_category)
                for syn_id  in this_synapse_attribute['syn_id']:
                    syn = Synapse(self, node, type_list=syn_types, stochastic=stochastic, id=syn_id)
        self.init_synaptic_mechanisms()

    def correct_g_pas_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
        dendritic sections proportional to the number of excitatory synapses contained in each section.
        """
        for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
            for node in self.get_nodes_of_subtype(sec_type):
                node.correct_g_pas_for_spines()

    def correct_cm_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in all
        dendritic sections proportional to the number of excitatory synapses contained in each section.
        """
        for loop in xrange(2):
            for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
                for node in self.get_nodes_of_subtype(sec_type):
                    node.correct_cm_for_spines()
                    if loop == 0:
                        node.init_nseg()
                        node.reinit_diam()
            if loop == 0:
                self.reinit_mechanisms()

    @property
    def gid(self):
        return self._gid

    @property
    def soma(self):
        return self._node_dict['soma']

    @property
    def axon(self):
        return self._node_dict['axon']

    @property
    def basal(self):
        return self._node_dict['basal']

    @property
    def apical(self):
        return self._node_dict['apical']

    @property
    def trunk(self):
        return self._node_dict['trunk']

    @property
    def tuft(self):
        return self._node_dict['tuft']

    @property
    def spine(self):
        return self._node_dict['spine']


# ------------------------------Extend SNode2 to interact with NEURON hoc sections------------------------


class SHocNode(btmorph.btstructs2.SNode2):
    """
    Extends SNode2 with some methods for storing and retrieving additional information in the node's content
    dictionary related to running NEURON models specified in the hoc language.
    """

    def __init__(self, index=0):
        """
        :param index: int : unique node identifier
        """
        btmorph.btstructs2.SNode2.__init__(self, index)
        self.content['spines'] = []
        self.content['synapses'] = []
        self.content['synapse_attributes'] = {'syn_locs': [],
                                              'syn_category': [],
                                              'syn_id': []}
        self.content['synapse_mechanism_attributes'] = {}

    def get_sec(self):
        """
        Returns the hoc section associated with this node, stored in the node's content dictionary.
        :return: :class:'neuron.h.Section'
        """
        if 'sec' in self.content:
            return self.content['sec']
        else:
            raise Exception('This node does not yet have an associated hoc section.')

    def set_sec(self, sec):
        """
        Stores the hoc section associated with this node in the node's content dictionary.
        :param sec: :class:'neuron.h.Section'
        """
        self.content['sec'] = sec

    sec = property(get_sec, set_sec)

    def init_nseg(self, spatial_res=0):
        """
        Initializes the number of hoc segments in this node's hoc section (nseg) based on the AC length constant.
        Must be re-initialized whenever basic cable properties Ra or cm are changed. If the node is a tapered cylinder,
        it should contain at least 3 segments. The spatial resolution parameter increases the number of segments per
        section by a factor of an exponent of 3.
        :param spatial_res: int
        """
        sugg_nseg = d_lambda_nseg(self.sec)
        # print self.name, self.sec.nseg, sugg_nseg
        if not self.get_diam_bounds() is None:
            sugg_nseg = max(sugg_nseg, 3)
        sugg_nseg *= 3 ** spatial_res
        self.sec.nseg = int(sugg_nseg)

    def reinit_diam(self):
        """
        For a node associated with a hoc section that is a tapered cylinder, every time the spatial resolution
        of the section (nseg) is changed, the section diameters must be reinitialized. This method checks the
        node's content dictionary for diameter boundaries and recalibrates the hoc section associated with this node.
        """
        if not self.get_diam_bounds() is None:
            [diam1, diam2] = self.get_diam_bounds()
            h('diam(0:1)={}:{}'.format(diam1, diam2), sec=self.sec)

    def append_synapse_attribute(self, syn_category, loc):
        """

        :param syn_category: str
        :param loc: float
        """
        self.synapse_attributes['syn_locs'].append(loc)
        self.synapse_attributes['syn_category'].append(syn_category_enumerator[syn_category])
        self.synapse_attributes['syn_id'].append(len(self.synapse_attributes['syn_id']))

    def get_filtered_synapse_attributes(self, syn_category=None, syn_type=None, layer=None):
        """
        Return dictionary containing attributes for all potential synapses that meet the query criterion. syn_category
        and layer can be specified as lists for a broad search.
        :param syn_category: str or list of str
        :param syn_type: str
        :param layer: int or list of int
        :return: dict
        """
        if type(syn_category) is list:
            syn_category_set = {syn_category_enumerator[item] for item in syn_category}
        elif syn_category is not None:
            syn_category_set = {syn_category_enumerator[syn_category]}
        if type(layer) is list:
            layer_set = set(layer)
        elif layer is not None:
            layer_set = {layer}
        filtered_attributes = {'syn_locs': [], 'syn_category': [], 'layer': [], 'syn_id': []}
        for i in xrange(len(self.synapse_attributes['syn_locs'])):
            this_syn_id = self.synapse_attributes['syn_id'][i]
            this_syn_loc = self.synapse_attributes['syn_locs'][i]
            this_syn_category = self.synapse_attributes['syn_category'][i]
            if not (syn_category is None or this_syn_category in syn_category_set):
                continue
            this_layer = self.get_layer(this_syn_loc)
            if not (layer is None or this_layer in layer_set):
                continue
            if not (syn_type is None or (this_syn_id in self.synapse_mechanism_attributes
                                         and syn_type in self.synapse_mechanism_attributes[this_syn_id])):
                continue
            filtered_attributes['syn_locs'].append(this_syn_loc)
            filtered_attributes['syn_category'].append(this_syn_category)
            filtered_attributes['layer'].append(this_layer)
            filtered_attributes['syn_id'].append(this_syn_id)
        return filtered_attributes

    def correct_cm_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in this
        dendritic section proportional to the number of excitatory synapses contained in the section.
        """
        # arrived at via optimization. spine neck appears to shield dendrite from spine head contribution to membrane
        # capacitance and time constant
        cm_fraction = 0.40
        SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
        this_syn_locs = self.get_filtered_synapse_attributes(syn_category='excitatory')['syn_locs']
        if this_syn_locs:
            this_syn_locs = np.array(this_syn_locs)
            seg_width = 1. / self.sec.nseg
            for i, segment in enumerate(self.sec):
                SA_seg = segment.area()
                num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
                cm_correction_factor = (SA_seg + cm_fraction * num_spines * SA_spine) / SA_seg
                self.sec(segment.x).cm *= cm_correction_factor

    def correct_g_pas_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in this
        dendritic section proportional to the number of excitatory synapses contained in the section.
        """
        SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
        this_syn_locs = self.get_filtered_synapse_attributes(syn_category='excitatory')['syn_locs']
        if this_syn_locs:
            this_syn_locs = np.array(this_syn_locs)
            seg_width = 1. / self.sec.nseg
            for i, segment in enumerate(self.sec):
                SA_seg = segment.area()
                num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
                soma_g_pas = self.sec.cell().mech_dict['soma']['pas']['g']['value']
                gpas_correction_factor = (SA_seg * self.sec(segment.x).g_pas + num_spines * SA_spine * soma_g_pas) / \
                                         (SA_seg * self.sec(segment.x).g_pas)
                self.sec(segment.x).g_pas *= gpas_correction_factor

    def get_diam_bounds(self):
        """
        If the hoc section associated with this node is a tapered cylinder, this method returns a list containing
        the values of the diameters at the 0 and 1 ends of the section, stored in the node's content dictionary.
        Otherwise, it returns None (for non-conical cylinders).
        :return: (list: int) or None
        """
        if 'diam' in self.content:
            return self.content['diam']
        else:
            return None

    def set_diam_bounds(self, diam1, diam2):
        """
        For a node associated with a hoc section that is a tapered cylinder, this stores a list containing the values
        of the diameters at the 0 and 1 ends of the section in the node's content dictionary.
        :param diam1: int
        :param diam2: int
        """
        self.content['diam'] = [diam1, diam2]
        self.reinit_diam()

    def get_type(self):
        """
        NEURON sections are assigned a node type for convenience in order to later specify membrane mechanisms and
        properties for each type of compartment.
        :return: str
        """
        if 'type' in self.content:
            return self.content['type']
        else:
            raise Exception('This node does not yet have a defined type.')

    def set_type(self, type):
        """
        Checks that type is a string in the list of defined section types, and stores the value in the node's content
        dictionary.
        :param type: str
        """
        if type in sec_types:
            self.content['type'] = type
        else:
            raise Exception('That is not a defined type of section.')

    type = property(get_type, set_type)

    def get_layer(self, x=None):
        """
        NEURON sections can be assigned a layer type for convenience in order to later specify synaptic mechanisms and
        properties for each layer. If 3D points are used to specify cell morphology, each element in the list
        corresponds to the layer of the 3D point with the same index.
        :param x: float in [0, 1] : optional relative location in section
        :return: list or float or None
        """
        if 'layer' in self.content:
            if x is None:
                return self.content['layer']
            elif self.sec.n3d() == 0:
                return self.content['layer'][0]
            else:
                for i in xrange(self.sec.n3d()):
                    if self.sec.arc3d(i) / self.sec.L >= x:
                        return self.content['layer'][i]
        else:
            return None

    def append_layer(self, layer):
        """
        NEURON sections can be assigned a layer type for convenience in order to later specify synaptic mechanisms and
        properties for each layer. If 3D points are used to specify cell morphology, each element in the list
        corresponds to the layer of the 3D point with the same index.
        :param layer: int
        """
        if 'layer' in self.content:
            self.content['layer'].append(layer)
        else:
            self.content['layer'] = [layer]

    def connect(self, parent, ploc=1., cloc=0.):
        """
        Connects this SHocNode node to a parent node, and establishes a connection between their associated
        hoc sections.
        :param parent: :class:'SHocNode'
        :param ploc: float in [0,1] Connect to this end of the parent hoc section.
        :param cloc: float in [0,1] Connect this end of the child hoc section
        """
        self.parent = parent
        parent.add_child(self)
        self.sec.connect(parent.sec, ploc, cloc)

    @property
    def name(self):
        """
        Returns a str containing the name of the hoc section associated with this node. Consists of a type descriptor
        and an index identifier.
        :return: str
        """
        if 'type' in self.content:
            return '{0.type}{0.index}'.format(self)
        else:
            raise Exception('This node does not yet have a defined type.')

    @property
    def spines(self):
        """
        Returns a list of the spine head sections attached to the hoc section associated with this node.
        :return: list of :class:'SHocNode' of sec_type == 'spine_head'
        """
        return self.content['spines']

    @property
    def synapses(self):
        """
        Returns a list of the objects of :class:'Synapse' associated with this node.
        :return: list of hoc objects, type depends on .mod file(s) used to implement synapses
        """
        return self.content['synapses']

    @property
    def synapse_attributes(self):
        """
        synapse_attributes is a dict specifying attributes of potential synapses, including 'syn_category'
        (e.g. 'excitatory' or 'inhibitory'), 'syn_locs', and 'syn_id' (unique index within each node).
        :return: dict of list
        """
        return self.content['synapse_attributes']

    @property
    def synapse_mechanism_attributes(self):
        """
        synapse_mechanism_attributes is a nested dict specifying parameters of synaptic point processes and netcon
        objects, indexed by syn_id.
        e.g. {syn_id: {'AMPA_KIN': {'gmax': float},
                                   {'weight': float}}}
        :return: dict of dict
        """
        return self.content['synapse_mechanism_attributes']

    @property
    def connection_loc(self):
        """
        Returns the location along the parent section of the connection with this section, except if the sec_type
        is spine_head, in which case it reports the connection_loc of the spine neck.
        :return: int or float
        """
        if self.type == 'spine_head':
            self.parent.sec.push()
        else:
            self.sec.push()
        loc = h.parent_connection()
        h.pop_section()
        return loc


class Synapse(object):
    """
    The implementation in hoc of synaptic mechanisms that can be triggered is complicated. This container is an attempt
    to wrap all the objects required to deliver synaptic events to a section, and have separable synaptic mechanisms
    (e.g. GluA-Rs and GluN-Rs) respond with individually specifiable weights and kinetics.
    To make model specification and simulation implementation straightforward, synapses are not meant to be moved once
    they are initialized.
    """

    def __init__(self, cell, node, syn_types=None, stochastic=1, loc=None, id=None, delay=0., weight=1., threshold=-30.,
                 stochastic_type=None, source=None, source_node=None, source_param=None, source_loc=0.5):
        """
        A source (like a spike detector in a pre-synaptic neuron) can be specified. If not, a VecStim object is used a
        source, which can be played events at specified times using its .play method. If stochastic, all spikes are
        intercepted by a point process with release probability dynamics and its own unique and independent random
        uniform variable. If not, the specified synaptic mechanisms are connected directly to the source of spikes.
        :param cell: :class:'HocCell'
        :param node: :class:'SHoCNode'
        :param syn_types: list of str
        :param stochastic: bool
        :param loc: float
        :param id: int
        :param delay: float
        :param weight: float
        :param threshold: float
        :param stochastic_type: str
        :param source: hoc artificial cell or otherwise hoc object not associated with a section
        :param source_node: :class:'SHocNode' specifies the section containing a range variable to be used as a source
        :param source_param: str
        :param source_loc: str
        """
        self._cell = cell
        self._node = node
        self._stochastic = stochastic
        self._delay = delay
        self._weight = weight
        self._threshold = threshold
        self._targets = {}
        self.randObj = None
        self._node.synapses.append(self)
        if source_param is None:
            source_param = '_ref_v'
        if stochastic_type is None:
            self._stochastic_type = 'Pr'
        else:
            self._stochastic_type = stochastic_type
        if not source is None:
            self._source = {'object': source}
        elif not source_node is None:
            self._source = {'object': getattr(source_node.sec(source_loc), source_param), 'node': source_node}
        else:
            self._source = {'object': h.VecStim()}
        if syn_types is None:
            syn_types = ['AMPA_KIN']
        elif type(syn_types) is not list:
            syn_types = [syn_types]
        if self.stochastic:
            self._init_stochastic()
        self._id = id
        if self.id is not None and loc is None:
            loc = self.branch.synapse_attributes['syn_locs'][self.id]
        if loc is None:
            loc = 0.5
        self._loc = loc
        for syn_type in syn_types:
            syn = getattr(h, syn_type)(self.node.sec(self._loc))
            self._targets[syn_type] = {'target': syn}
            if self.stochastic:
                self._targets[syn_type]['netcon'] = h.NetCon(self.target(self._stochastic_type), syn)
                self.netcon(syn_type).delay = delay
                self.netcon(syn_type).weight[0] = weight
                self.netcon(syn_type).threshold = threshold
            else:
                self._init_netcon(syn_type)

    def _init_stochastic(self):
        """
        This method constructs and initializes a stochastic filtering mechanism that intercepts spikes delivered to this
        synapse and calculates whether or not to pass a spike to the rest of the specified synaptic mechanisms.
        """
        if self.randObj is None:  # if this synapse has never been stochastic, it needs a new random number generator
            self.randObj = h.Random()
            self.randObj.MCellRan4(self.cell.gid * 1e4 + 1, self.node.index * 1e4 + self.node.synapses.index(self) + 1)
            # a unique sequence for up to ~10,000 spikes per synapse; ~10,000 synapses per node;
            # ~4,290,000 nodes per cell; ~4,290,000 cell in a network
            self.randObj.uniform(0, 1)
        else:  # if this synapse has already been stochastic before, this restarts its random number generator
            self.randObj.seq(self.cell.gid * 1e4 + 1)
        syn = getattr(h, self._stochastic_type)(self.node.sec(self._loc))
        self._targets[self._stochastic_type] = {'target': syn}
        self._init_netcon(self._stochastic_type, delay=0.)
        self.target(self._stochastic_type).setRandObjRef(self.randObj)

    def target(self, syn_type):
        """
        Returns the hoc object for the synaptic point process of the specified type.
        :param target: str
        :return: :class:'h.HocObject'
        """
        if syn_type in self.targets:
            return self._targets[syn_type]['target']
        else:
            raise KeyError('Synapse type: {} not found at a synapse in {}'.format(syn_type, self._node.name))

    def _init_netcon(self, syn_type, delay=None, weight=None, threshold=None):
        """
        Appropriately initializes new netcon object, depending on whether the current source dictionary specifies a
        hocObject without a section, or a reference variable contained within a section.
        :param syn_type: str
        :param delay = float
        :param weight = float
        :param threshold = float
        """
        if syn_type in self.targets:
            source = self._source['object']
            if weight is None:
                weight = self._weight
            if threshold is None:
                threshold = self.threshold
            if delay is None:
                delay = self.delay
            if 'node' in self._source:
                node = self._source['node']
                self._targets[syn_type]['netcon'] = h.NetCon(source, self.target(syn_type), sec=node.sec)
            else:
                self._targets[syn_type]['netcon'] = h.NetCon(source, self.target(syn_type))
            this_netcon = self._targets[syn_type]['netcon']
            this_netcon.delay = delay
            this_netcon.weight[0] = weight
            this_netcon.threshold = threshold
        else:
            raise KeyError('Synapse type: {} not found at a synapse in {}'.format(syn_type, self._node.name))

    def netcon(self, syn_type):
        """
        Returns the hoc network connection linking the synaptic mechanism of the specified type to a source of spikes.
        :param syn_type: str
        :return: :class:'h.NetCon'
        """
        if syn_type in self.targets:
            return self._targets[syn_type]['netcon']
        else:
            raise KeyError('Synapse type: {} not found at a synapse in {}'.format(syn_type, self._node.name))

    def change_source(self, source=None, node=None, param=None, loc=0.5):
        """
        In order to change the source of a synapse from the default VecStim object to a membrane potential or artificial
        cell, all the netcon objects must be deleted and replaced with new ones. Preserves previously set values for
        delay, weight, and threshold for all synaptic mechanisms associated with this synapse.
        :param source: hoc artificial cell or otherwise hoc object not associated with a section
        :param node: :class:'SHocNode'
        :param param: str corresponding to range variable in section
        :param loc: float
        """
        netcon_dict = {}
        if param is None:
            param = '_ref_v'
        if source is None:
            if node is None:
                raise Exception('A source or reference node must be provided to establish a new synaptic connection.')
            else:
                del self._source
                self._source = {'object': getattr(node.sec(loc), param), 'node': node}
        else:
            del self._source
            self._source = {'object': source}
        if self._stochastic:
            del self._targets[self._stochastic_type]['netcon']
            delay = netcon_dict[self._stochastic_type]['delay']
            weight = netcon_dict[self._stochastic_type]['weight']
            threshold = netcon_dict[self._stochastic_type]['threshold']
            self._init_netcon(self._stochastic_type, delay=delay, weight=weight, threshold=threshold)
        else:
            for syn_type in (syn_type for syn_type in self.targets if syn_type != self._stochastic_type):
                delay = self.netcon(syn_type).delay
                weight = self.netcon(syn_type).weight[0]
                threshold = self.netcon(syn_type).threshold
                del self._targets[syn_type]['netcon']
                self._init_netcon(syn_type, delay=delay, weight=weight, threshold=threshold)

    def load_synapse_mechanism_attributes(self):
        """
        This method synchronizes this synapse with the parameters specified in the synapse_attributes and
        synapse_mechanism_attributes dictionaries contained in the parent branch.
        """
        if self.id is not None and self.id in self.branch.synapse_mechanism_attributes:
            for syn_type in self.targets:
                if syn_type in self.branch.synapse_mechanism_attributes[self.id]:
                    for param_name in self.branch.synapse_mechanism_attributes[self.id][syn_type]:
                        value = self.branch.synapse_mechanism_attributes[self.id][syn_type][param_name]
                        if hasattr(self.target(syn_type), param_name):
                            setattr(self.target(syn_type), param_name, value)
                        elif hasattr(self.netcon(syn_type), param_name):
                            if param_name == 'weight':
                                self.netcon(syn_type).weight[0] = value
                            else:
                                setattr(self.netcon(syn_type), param_name, value)

    def get_stochastic(self):
        """
        Returns the value of an internal variable indicating if this synapse has a stochastic filter for spikes.
        :return: bool
        """
        return self._stochastic

    def set_stochastic(self, value):
        """
        Turns on or off stochastic filtering of spikes, preserving delay, weight, and threshold for all synaptic
        mechanisms associated with this synapse.
        :param value: bool
        """
        if not (value == self._stochastic):
            self._stochastic = value
            if value:
                self._init_stochastic()
                for syn_type in (syn_type for syn_type in self.targets if syn_type != self._stochastic_type):
                    delay = self.netcon(syn_type).delay
                    weight = self.netcon(syn_type).weight[0]
                    threshold = self.netcon(syn_type).threshold
                    del self._targets[syn_type]['netcon']
                    self._targets[syn_type]['netcon'] = h.NetCon(self.target(self._stochastic_type),
                                                                 self.target(syn_type))
                    self.netcon(syn_type).delay = delay
                    self.netcon(syn_type).weight[0] = weight
                    self.netcon(syn_type).threshold = threshold
            else:
                for syn_type in (syn_type for syn_type in self.targets if syn_type != self._stochastic_type):
                    delay = self.netcon(syn_type).delay
                    weight = self.netcon(syn_type).weight[0]
                    threshold = self.netcon(syn_type).threshold
                    del self._targets[syn_type]['netcon']
                    self._init_netcon(syn_type, delay=delay, weight=weight, threshold=threshold)
                del self._targets[self._stochastic_type]

    stochastic = property(get_stochastic, set_stochastic)

    @property
    def targets(self):
        """
        Returns the list of synaptic point processes inserted at this synapse.
        :return: list of str
        """
        return self._targets.keys()

    def get_delay(self):
        """
        Returns the default value of the time delay (ms) between spike and activation for this synapse.
        :return: int or float
        """
        return self._delay

    def set_delay(self, value):
        """
        Changes the value of the time delay (ms) between spike and activation for all synaptic mechanisms associated
        with this synapse, except self._stochastic_type, which retains its current value until set manually.
        :param value: int or float
        """
        self._delay = value
        for syn_type in (syn_type for syn_type in self.targets if syn_type != self._stochastic_type):
            self.netcon(syn_type).delay = value

    delay = property(get_delay, set_delay)

    def get_weight(self, syn_type=None):
        """
        Returns the value of the activation weight for the specific synaptic point processes inserted at this synapse.
        If no syn_type is specified, the default value is returned.
        :param syn_type: str
        :return: float
        """
        if syn_type is not None and syn_type in self.targets:
            return self.netcon(syn_type).weight[0]
        elif syn_type is None:
            return self._weight
        else:
            return None

    def set_weight(self, value, syn_type=None):
        """
        Changes the value of the activation weight for the specific synaptic point processes inserted at this synapse.
        If no syn_type is specified, the default value is set, and the value is changed to default for all inserted
        synaptic point processes inserted at this synapse.
        :param value: float
        :param syn_type: str
        """
        if syn_type is not None and syn_type in self.targets:
            self.netcon(syn_type).weight[0] = value
        elif syn_type is None:
            self._weight = value
            if self._stochastic:
                self.netcon(self._stochastic_type).weight[0] = self._weight
            else:
                for syn_type in (syn_type for syn_type in self.targets if syn_type != self._stochastic_type):
                    self.netcon(syn_type).weight[0] = value

    def get_threshold(self):
        """
        Returns the value of the activation threshold for this synapse.
        :return: float
        """
        return self._threshold

    def set_threshold(self, value):
        """
        Changes the value of the activation threshold for this synapse.
        :param value: float
        """
        self._threshold = value
        for syn_type in self.targets:
            self.netcon(syn_type).threshold = value

    threshold = property(get_threshold, set_threshold)

    @property
    def source(self):
        """
        Returns the hocObject currently being used as a source.
        :return: :class:'hocObject'
        """
        return self._source['object']

    @property
    def cell(self):
        """
        Returns the cell containing this synapse.
        :return: :class:'HocCell'
        """
        return self._cell

    @property
    def node(self):
        """
        Returns the node containing this synapse.
        :return: :class:'SHocNode'
        """
        return self._node

    @property
    def branch(self):
        """
        Returns the branch containing this synapse.
        :return: :class:'SHocNode'
        """
        if self._node.type == 'spine_head':
            return self._node.parent.parent
        else:
            return self._node

    @property
    def loc(self):
        """
        Returns the location along the hoc section containing this synapse. For convenience, if the synapse is
        contained in a spine_head, this property method returns the location along the branch section where the
        spine_neck is connected.
        :return: int or float
        """
        if self.node.type == 'spine_head':
            self.node.parent.sec.push()
            loc = h.parent_connection()
            h.pop_section()
            return loc
        else:
            return self._loc

    @property
    def id(self):
        """

        :return: int
        """
        return self._id


class QuickSim(object):
    """
    This method is used to run a quick simulation with a set of current injections and a set of recording sites.
    Can save detailed information about the simulation to an HDF5 file after each run. Once defined, IClamp objects
    persist when using an interactive console, but not when executing standalone scripts. Therefore, the best practice
    is simply to set amp to zero to turn off current injections, or move individual IClamp processes to different
    locations rather then adding and deleting them.
    class params:
    self.stim_list:
    self.rec_list:
    """

    def __init__(self, tstop=400., cvode=True, daspk=False, dt=None, verbose=True):
        """

        :param tstop: float
        :param cvode: bool
        :param daspk: bool
        :param dt: float
        :param verbose: bool
        """
        self.rec_list = []  # list of dicts with keys for 'cell', 'node', 'loc' and 'vec': pointer to hoc Vector object.
        # Also contains keys for 'ylabel' and 'units' for recording parameters other than Vm.
        self.stim_list = []  # list of dicts with keys for 'cell', 'node', 'stim': pointer to hoc IClamp object, and
        # 'vec': recording of actual stimulus for plotting later
        self.tstop = tstop
        h.load_file('stdrun.hoc')
        h.celsius = 35.0
        h.cao0_ca_ion = 1.3
        self.cvode = h.CVode()
        self.cvode_atol = 0.01  # 0.001
        self.daspk = daspk
        self.cvode_state = cvode
        if dt is None:
            self.dt = h.dt
        else:
            self.dt = dt
        self.verbose = verbose
        self.tvec = h.Vector()
        self.tvec.record(h._ref_t, self.dt)
        self.parameters = {}

    def run(self, v_init=-65.):
        """

        :param v_init: float
        """
        start_time = time.time()
        h.tstop = self.tstop
        if not self.cvode_state:
            h.steps_per_ms = int(1. / self.dt)
            h.dt = self.dt
        h.v_init = v_init
        # h.init()
        # h.finitialize(v_init)
        # if self.cvode_state:
        #     self.cvode.re_init()
        # else:
        #     h.fcurrent()
        h.run()
        if self.verbose:
            print 'Simulation runtime: ', time.time() - start_time, ' sec'

    def append_rec(self, cell, node, loc=None, param='_ref_v', object=None, ylabel='Vm', units='mV', description=None):
        """

        :param cell: :class:'HocCell'
        :param node: :class:'SHocNode
        :param loc: float
        :param param: str
        :param object: :class:'HocObject'
        :param ylabel: str
        :param units: str
        :param description: str
        """
        rec_dict = {'cell': cell, 'node': node, 'ylabel': ylabel, 'units': units}
        if description is None:
            rec_dict['description'] = 'rec' + str(len(self.rec_list))
        elif description in (rec['description'] for rec in self.rec_list):
            rec_dict['description'] = description + str(len(self.rec_list))
        else:
            rec_dict['description'] = description
        rec_dict['vec'] = h.Vector()
        if object is None:
            if loc is None:
                loc = 0.5
            rec_dict['vec'].record(getattr(node.sec(loc), param), self.dt)
        else:
            if loc is None:
                try:
                    loc = object.get_segment().x  # this should not push the section to the hoc stack or risk overflow
                except:
                    loc = 0.5  # if the object doesn't have a .get_loc() method, default to 0.5
            if param is None:
                rec_dict['vec'].record(object, self.dt)
            else:
                rec_dict['vec'].record(getattr(object, param), self.dt)
        rec_dict['loc'] = loc
        self.rec_list.append(rec_dict)

    def get_rec(self, description):
        """
        Return the dict corresponding to the item in the rec_dict list with the specified description.
        :param description: str
        :return: dict
        """
        for rec in self.rec_list:
            if rec['description'] == description:
                return rec
        raise Exception('No recording with description %s' % description)

    def get_rec_index(self, description):
        """
        Return the index of the item in the rec_dict list with the specified description.
        :param description: str
        :return: dict
        """
        for i, rec in enumerate(self.rec_list):
            if rec['description'] == description:
                return i
        raise Exception('No recording with description %s' % description)

    def append_stim(self, cell, node, loc, amp, delay, dur, description='IClamp'):
        """

        :param cell: :class:'HocCell'
        :param node: :class:'SHocNode'
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        """
        stim_dict = {'cell': cell, 'node': node, 'description': description}
        stim_dict['stim'] = h.IClamp(node.sec(loc))
        stim_dict['stim'].amp = amp
        stim_dict['stim'].delay = delay
        stim_dict['stim'].dur = dur
        stim_dict['vec'] = h.Vector()
        stim_dict['vec'].record(stim_dict['stim']._ref_i, self.dt)
        self.stim_list.append(stim_dict)

    def modify_stim(self, index=0, node=None, loc=None, amp=None, delay=None, dur=None, description=None):
        """

        :param index: int
        :param node: class:'SHocNode'
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        """
        stim_dict = self.stim_list[index]
        if not (node is None and loc is None):
            if not node is None:
                stim_dict['node'] = node
            if loc is None:
                loc = stim_dict['stim'].get_segment().x
            stim_dict['stim'].loc(stim_dict['node'].sec(loc))
        if not amp is None:
            stim_dict['stim'].amp = amp
        if not delay is None:
            stim_dict['stim'].delay = delay
        if not dur is None:
            stim_dict['stim'].dur = dur
        if not description is None:
            stim_dict['description'] = description

    def get_stim_index(self, description):
        """
        Return the index of the item in the stim_dict list with the specified description.
        :param description: str
        :return: dict
        """
        for i, stim in enumerate(self.stim_list):
            if stim['description'] == description:
                return i
        raise Exception('No IClamp object with description: %s' % description)

    def modify_rec(self, index=0, node=None, loc=None, object=None, param='_ref_v', ylabel=None, units=None,
                   description=None):
        """

        :param index: int
        :param node: class:'SHocNode'
        :param loc: float
        :param object: class:'HocObject'
        :param param: str
        :param ylabel: str
        :param units: str
        :param description: str
        """
        rec_dict = self.rec_list[index]
        if not ylabel is None:
            rec_dict['ylabel'] = ylabel
        if not units is None:
            rec_dict['units'] = units
        if not node is None:
            rec_dict['node'] = node
        if not loc is None:
            rec_dict['loc'] = loc
        if object is None:
            rec_dict['vec'].record(getattr(rec_dict['node'].sec(rec_dict['loc']), param), self.dt)
        elif param is None:
            rec_dict['vec'].record(object, self.dt)
        else:
            rec_dict['vec'].record(getattr(object, param), self.dt)
        if not description is None:
            rec_dict['description'] = description

    def plot(self):
        """

        """
        for rec_dict in self.rec_list:
            if 'description' in rec_dict:
                description = str(rec_dict['description'])
            else:
                description = ''
            plt.plot(self.tvec, rec_dict['vec'], label=rec_dict['node'].name + '(' + str(rec_dict['loc']) + ') - ' +
                                                       description)
            plt.xlabel("Time (ms)")
            plt.ylabel(rec_dict['ylabel'] + ' (' + rec_dict['units'] + ')')
        plt.legend(loc='upper right')
        if 'description' in self.parameters:
            plt.title(self.parameters['description'])
        plt.show()
        plt.close()

    def export_to_file(self, f, simiter=None):
        """
        Extracts important parameters from the lists of stimulation and recording sites, and exports to an HDF5
        database. Arrays are saved as datasets and metadata is saved as attributes.
        :param f: :class:'h5py.File'
        :param simiter: int
        """
        start_time = time.time()
        if simiter is None:
            simiter = len(f)
        if str(simiter) not in f:
            f.create_group(str(simiter))
        f[str(simiter)].create_dataset('time', compression='gzip', compression_opts=9, data=self.tvec)
        f[str(simiter)]['time'].attrs['dt'] = self.dt
        for parameter in self.parameters:
            f[str(simiter)].attrs[parameter] = self.parameters[parameter]
        if self.stim_list:
            f[str(simiter)].create_group('stim')
            for index, stim in enumerate(self.stim_list):
                stim_out = f[str(simiter)]['stim'].create_dataset(str(index), compression='gzip', compression_opts=9,
                                                                  data=stim['vec'])
                cell = stim['cell']
                stim_out.attrs['cell'] = cell.gid
                node = stim['node']
                stim_out.attrs['index'] = node.index
                stim_out.attrs['type'] = node.type
                loc = stim['stim'].get_segment().x
                stim_out.attrs['loc'] = loc
                distance = cell.get_distance_to_node(cell.tree.root, node, loc)
                stim_out.attrs['soma_distance'] = distance
                distance = cell.get_distance_to_node(cell.get_dendrite_origin(node), node, loc)
                stim_out.attrs['branch_distance'] = distance
                stim_out.attrs['amp'] = stim['stim'].amp
                stim_out.attrs['delay'] = stim['stim'].delay
                stim_out.attrs['dur'] = stim['stim'].dur
                stim_out.attrs['description'] = stim['description']
        f[str(simiter)].create_group('rec')
        for index, rec in enumerate(self.rec_list):
            rec_out = f[str(simiter)]['rec'].create_dataset(str(index), compression='gzip', compression_opts=9,
                                                            data=rec['vec'])
            cell = rec['cell']
            rec_out.attrs['cell'] = cell.gid
            node = rec['node']
            rec_out.attrs['index'] = node.index
            rec_out.attrs['type'] = node.type
            rec_out.attrs['loc'] = rec['loc']
            distance = cell.get_distance_to_node(cell.tree.root, node, rec['loc'])
            rec_out.attrs['soma_distance'] = distance
            distance = cell.get_distance_to_node(cell.get_dendrite_origin(node), node, rec['loc'])
            is_terminal = int(cell.is_terminal(node))
            branch_order = cell.get_branch_order(node)
            rec_out.attrs['branch_distance'] = distance
            rec_out.attrs['is_terminal'] = is_terminal
            rec_out.attrs['branch_order'] = branch_order
            rec_out.attrs['ylabel'] = rec['ylabel']
            rec_out.attrs['units'] = rec['units']
            if 'description' in rec:
                rec_out.attrs['description'] = rec['description']
        if self.verbose:
            print 'Simulation ', simiter, ': exporting took: ', time.time() - start_time, ' s'

    def get_cvode_state(self):
        """

        :return bool
        """
        return bool(self.cvode.active())

    def set_cvode_state(self, state):
        """

        :param state: bool
        """
        if state:
            self.cvode.active(1)
            self.cvode.atol(self.cvode_atol)
            self.cvode.use_daspk(int(self.daspk))
        else:
            self.cvode.active(0)

    cvode_state = property(get_cvode_state, set_cvode_state)


class CA1_Pyr(HocCell):
    """

    """

    def __init__(self, morph_file_path=None, mech_file_path=None, gid=0, existing_hoc_cell=None, full_spines=True,
                 preserve_3d=True):
        """

        :param morph_file_path:
        :param mech_file_path:
        :param gid:
        :param existing_hoc_cell:
        :param full_spines:
        :param preserve_3d:
        """
        HocCell.__init__(self, morph_file_path, mech_file_path, gid, existing_hoc_cell)
        self.random.seed(self.gid)  # This cell will always have the same spine and/or synapse locations as long as
                                    # they are inserted in the same order
        # self.make_standard_soma_and_axon(soma_length=16., soma_diam=9., ais_length=30., axon_length=500.)
        self.make_standard_soma_and_axon(soma_length=14., soma_diam=9., ais_length=15., axon_length=500.)
        self.load_morphology(preserve_3d=preserve_3d)
        self.generate_excitatory_synapse_locs()
        self.generate_inhibitory_synapse_locs()
        self.reinit_mechanisms()
        if full_spines:
            self.insert_spines()
        else:
            self.correct_cm_for_spines()
            self.correct_g_pas_for_spines()

    def generate_excitatory_synapse_locs(self, sec_type_list=None):
        """
        This method populates the cell tree with putative synapse locations following synapse density information
        from Erik Bloss & Nelson Spruston. Basal dendrites have no spines until the first branch point, and a higher
        density beyond the second branch point. Trunk dendrites have no spines until the first branch point, and an
        increasing density until the tuft branch point(s). Apical dendrites have a density that varies with the
        distance from the soma of their original branch point from the trunk. Terminal tuft branches have a higher
        density than their parents.
        :param sec_type_list: list of str
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
        densities = {'trunk': {'min': 0.2418, 'max': 3.8,
                               'start': min([self.get_distance_to_node(self.tree.root, branch) for branch in
                                             self.apical]),
                               'end': max([self.get_distance_to_node(self.tree.root, branch) for branch in
                                           self.trunk])},
                     'basal': {'1': 0., '2': 0.4428, '>2': 1.891},
                     'apical': {'min': 2.273, 'max': 2.688,
                                'start': min([self.get_distance_to_node(self.tree.root, branch) for branch in
                                              self.apical]),
                                'end': max([self.get_distance_to_node(self.tree.root, branch)
                                            for branch in self.apical if self.get_branch_order(branch) == 1])},
                     'tuft': {'parent': 1.354, 'terminal': 0.7157}
                     }
        if 'basal' in sec_type_list:
            for node in self.basal:
                order = self.get_branch_order(node)
                if order == 2:
                    self.append_synapse_attributes_by_density(node, densities['basal']['2'], syn_category)
                elif order > 2:
                    self.append_synapse_attributes_by_density(node, densities['basal']['>2'], syn_category)
        if 'trunk' in sec_type_list:
            for node in self.trunk:
                distance = self.get_distance_to_node(self.tree.root, node)
                if distance >= densities['trunk']['start']:
                    slope = (densities['trunk']['max'] - densities['trunk']['min']) / \
                            (densities['trunk']['end'] - densities['trunk']['start'])
                    density = densities['trunk']['min'] + slope * (distance - densities['trunk']['start'])
                    self.append_synapse_attributes_by_density(node, density, syn_category)
        if 'apical' in sec_type_list:
            for node in self.apical:
                distance = self.get_distance_to_node(self.tree.root, self.get_dendrite_origin(node), loc=1.)
                slope = (densities['apical']['max'] - densities['apical']['min']) / \
                        (densities['apical']['end'] - densities['apical']['start'])
                density = densities['apical']['min'] + slope * (distance - densities['apical']['start'])
                self.append_synapse_attributes_by_density(node, density, syn_category)
        if 'tuft' in sec_type_list:
            for node in self.tuft:
                if self.is_terminal(node):
                    self.append_synapse_attributes_by_density(node, densities['tuft']['terminal'], syn_category)
                else:
                    self.append_synapse_attributes_by_density(node, densities['tuft']['parent'], syn_category)

    def generate_inhibitory_synapse_locs(self, sec_type_list=None):
        """

        :param sec_type_list: str
        """
        syn_category = 'inhibitory'
        if sec_type_list is None:
            sec_type_list = ['soma', 'ais', 'basal', 'trunk', 'apical', 'tuft']
        densities = {'soma': 4.375,  # 2.857,  # 4.285,
                     'ais': 0.68,  # 0.53,
                     'trunk': {'min': 0.3022, 'max': 0.0627,
                               'start': 0.,
                               'end': max([self.get_distance_to_node(self.tree.root, branch) for branch in
                                           self.trunk])},
                     'basal': {'primary': 0.3129, 'intermediate': 0.1728, 'terminal': 0.06543},
                     'apical': {'min': 0.03885, 'max': 0.04512,
                                'start': min([self.get_distance_to_node(self.tree.root, branch) for branch in
                                              self.apical]),
                                'end': max([self.get_distance_to_node(self.tree.root, branch)
                                            for branch in self.apical if self.get_branch_order(branch) == 1])},
                     'tuft': {'parent': 0.2104, 'terminal': 0.1619}
                     }
        if 'soma' in sec_type_list:
            for node in self.soma:
                self.append_synapse_attributes_by_density(node, densities['soma'], syn_category)
        if 'ais' in sec_type_list:
            for node in self.get_nodes_of_subtype('ais'):
                self.append_synapse_attributes_by_density(node, densities['ais'], syn_category)
        if 'basal' in sec_type_list:
            for node in self.basal:
                if self.is_terminal(node):
                    self.append_synapse_attributes_by_density(node, densities['basal']['terminal'], syn_category)
                else:
                    order = self.get_branch_order(node)
                    if order == 1:
                        self.append_synapse_attributes_by_density(node, densities['basal']['primary'], syn_category)
                    else:
                        self.append_synapse_attributes_by_density(node, densities['basal']['intermediate'], syn_category)
        if 'trunk' in sec_type_list:
            for node in self.trunk:
                distance = self.get_distance_to_node(self.tree.root, node)
                if distance >= densities['trunk']['start']:
                    slope = (densities['trunk']['max'] - densities['trunk']['min']) / \
                            (densities['trunk']['end'] - densities['trunk']['start'])
                    density = densities['trunk']['min'] + slope * (distance - densities['trunk']['start'])
                    self.append_synapse_attributes_by_density(node, density, syn_category)
        if 'apical' in sec_type_list:
            for node in self.apical:
                distance = self.get_distance_to_node(self.tree.root, self.get_dendrite_origin(node), loc=1.)
                slope = (densities['apical']['max'] - densities['apical']['min']) / \
                        (densities['apical']['end'] - densities['apical']['start'])
                density = densities['apical']['min'] + slope * (distance - densities['apical']['start'])
                self.append_synapse_attributes_by_density(node, density, syn_category)
        if 'tuft' in sec_type_list:
            for node in self.tuft:
                if self.is_terminal(node):
                    self.append_synapse_attributes_by_density(node, densities['tuft']['terminal'], syn_category)
                else:
                    self.append_synapse_attributes_by_density(node, densities['tuft']['parent'], syn_category)

    def zero_na(self):
        """
        Set na channel conductances to zero in all compartments. Used during parameter optimization.
        """
        for sec_type in ['soma', 'axon_hill', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft']:
            for na_type in (na_type for na_type in ['nas_kin', 'nat_kin', 'nas', 'nax'] if na_type in
                    self.mech_dict[sec_type]):
                self.modify_mech_param(sec_type, na_type, 'gbar', 0.)

    def zero_h(self):
        """
        Set ih conductances to zero in all compartments. Used during parameter optimization.
        """
        self.modify_mech_param('soma', 'h', 'ghbar', 0.)
        self.mech_dict['trunk']['h']['ghbar']['value'] = 0.
        self.mech_dict['trunk']['h']['ghbar']['slope'] = 0.
        for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
            self.reinitialize_subset_mechanisms(sec_type, 'h')

    def set_terminal_branch_na_gradient(self, gmin=0.):
        """
        This is an admittedly ad-hoc procedure to implement a linearly decreasing gradient of sodium channels in
        terminal branches that is not easily accomplished by the general procedures implementing the mechanism
        dictionary.
        """
        na_type = (na_type for na_type in ['nas_kin', 'nat_kin', 'nas', 'nax']
                   if na_type in self.mech_dict['trunk']).next()
        self.set_special_mech_param_linear_gradient(na_type, 'gbar', ['basal', 'apical', 'tuft'],
                                                    self.is_terminal, gmin)


class DG_GC(HocCell):
    """

    """

    def __init__(self, morph_file_path=None, mech_file_path=None, gid=0, existing_hoc_cell=None, neuroH5_dict=None,
                 full_spines=True, preserve_3d=True):
        """

        :param morph_file_path:
        :param mech_file_path:
        :param gid:
        :param existing_hoc_cell:
        :param full_spines:
        :param preserve_3d:
        """
        HocCell.__init__(self, morph_file_path, mech_file_path, gid, existing_hoc_cell, neuroH5_dict)
        self.random.seed(self.gid)  # This cell will always have the same spine and GABA_A synapse locations as long as
        # they are inserted in the same order
        self.make_standard_soma_and_axon(soma_length=18.6, soma_diam=10.3, ais_length=20., axon_length=1000.)
        self.load_morphology(preserve_3d=preserve_3d)
        self.generate_excitatory_synapse_locs()
        self.generate_inhibitory_synapse_locs()
        self.reinit_mechanisms()
        if full_spines:
            self.insert_spines(sec_type_list=['apical'])
        else:
            self.correct_cm_for_spines()
            self.correct_g_pas_for_spines()

    def generate_excitatory_synapse_locs(self, sec_type_list=None):
        """
        This method populates the cell tree with putative synapse locations following type and\or layer-specific rules
        for synapse density.
        :param sec_type_list: list of str
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['apical']
        densities = {}
        for sec_type in (sec_type for sec_type in sec_type_list if self.get_nodes_of_subtype(sec_type)):
            if sec_type == 'apical':
                densities[sec_type] = {'layer': {1: 3.36, 2: 2.28, 3: 2.02},
                                       'default': 2.55}  # units: synapses / um length
        if 'apical' in sec_type_list:
            for node in self.apical:
                if node.get_layer() is not None:
                    self.append_synapse_attributes_by_layer(node, densities['apical']['layer'], syn_category)
                else:
                    self.append_synapse_attributes_by_density(node, densities['apical']['default'], syn_category)

    def generate_inhibitory_synapse_locs(self, sec_type_list=None):
        """
        This method populates the cell tree with putative synapse locations following type and\or layer-specific rules
        for synapse density.
        Thind et al...Buckmaster, J. Comp. Neurol. 2010
        :param sec_type_list: list of str
        """
        syn_category = 'inhibitory'
        if sec_type_list is None:
            sec_type_list = ['soma', 'ais', 'apical']
        densities = {}  # units: synapses / um length
        densities['soma'] = {'default': 10.22}
        densities['ais'] = {'default': 0.6}
        densities['apical'] = {'default': 0.83}  # uniform density across layers matches experimental
                                                 # distribution of synapses per layer due to variable dendritic
                                                 # length
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                self.append_synapse_attributes_by_density(node, densities[sec_type]['default'], syn_category)

    def zero_na(self):
        """
        Set na channel conductances to zero in all compartments. Used during parameter optimization.
        """
        for sec_type in ['soma', 'axon_hill', 'ais', 'axon', 'apical']:
            for na_type in (na_type for na_type in ['nas_kin', 'nat_kin', 'nas', 'nax'] if na_type in
                    self.mech_dict[sec_type]):
                self.modify_mech_param(sec_type, na_type, 'gbar', 0.)

    def set_terminal_branch_na_gradient(self, gmin=0.):
        """
        This is an admittedly ad-hoc procedure to implement a linearly decreasing gradient of sodium channels in
        terminal branches that is not easily accomplished by the general procedures implementing the mechanism
        dictionary.
        """
        na_type = (na_type for na_type in ['nas_kin', 'nat_kin', 'nas', 'nax']
                   if na_type in self.mech_dict['apical']).next()
        self.set_special_mech_param_linear_gradient(na_type, 'gbar', ['apical'],
                                                    self.is_terminal, gmin)

    def custom_inherit_by_branch_order(self, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
        """

        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param syn_type: str
        :param donor: :class:'SHocNode' or None
        """
        branch_order = int(rules['custom']['branch_order'])
        if self.is_terminal(node) or (branch_order is not None and self.get_branch_order(node) >= branch_order):
            if 'synapse' in mech_name:
                self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
            else:
                self._specify_mech_parameter(node, mech_name, param_name, baseline, rules, donor)

    def custom_gradient_by_branch_order(self, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
        """

        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param syn_type: str
        :param donor: :class:'SHocNode' or None
        """
        branch_order = int(rules['custom']['branch_order'])
        if self.get_branch_order(node) >= branch_order:
            if 'synapse' in mech_name:
                self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
            else:
                self._specify_mech_parameter(node, mech_name, param_name, baseline, rules, donor)

    def custom_gradient_by_terminal(self, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
        """

        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param syn_type: str
        :param donor: :class:'SHocNode' or None
        """
        if self.is_terminal(node):
            start_val = baseline
            if 'min' in rules:
                end_val = rules['min']
                direction = -1
            elif 'max' in rules:
                end_val = rules['max']
                direction = 1
            else:
                raise Exception('custom_gradient_by_terminal: no min or max target value specified for mechanism: %s '
                                'parameter: %s' % (mech_name, param_name))
            slope = (end_val - start_val)/node.sec.L
            if 'slope' in rules:
                if direction < 0.:
                    slope = min(rules['slope'], slope)
                else:
                    slope = max(rules['slope'], slope)
            for seg in node.sec:
                value = start_val + slope * seg.x * node.sec.L
                if direction < 0:
                    if value < end_val:
                        value = end_val
                else:
                    if value < end_val:
                        value = end_val
                setattr(getattr(seg, mech_name), param_name, value)
