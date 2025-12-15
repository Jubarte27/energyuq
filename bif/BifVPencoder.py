from easyvvuq.encoders import GenericEncoder
import math

# Custom encoder for dummy app
class BifVPEncoder(GenericEncoder):

    # Constructor
    def __init__(self, template_fname, delimiter='$', target_filename='app_input.txt', convert_vars=None, dependent_vars=None, physical_time=0.25):
        super().__init__(template_fname=template_fname, delimiter=delimiter, target_filename=target_filename)
        self.convert_vars = convert_vars
        self.dependent_vars = dependent_vars
        self.physical_time = physical_time

    # Parse simulation input (override)
    def encode(self, params=None, target_dir=''):
        if params is None:
            params = {}
        # Set config vars that need casting to integer
        if self.convert_vars is not None:
            for var in self.convert_vars:
                if var == 'steps':
                    params[var] = round(params[var])
        # Set config vars that depend on the uncertain inputs
        if self.dependent_vars is not None:
            for var in self.dependent_vars:
                if var == 'steps':
                    params[var] = math.ceil(self.physical_time/params['step_length'])
                if var == 'BifT':
                    params['radius'] = 20.1*params['voxel_size']
                    params['dim1'] = 23.0*params['voxel_size']
                    params['dim2'] = 103.0*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 203.0*params['voxel_size']
                if var == 'BifV':
                    params['radius'] = 20.1*params['voxel_size']
                    params['dim1'] = 23.0*params['voxel_size']
                    params['dim2'] = 88.0*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 203.0*params['voxel_size']
                    params['dim5'] = 103.0*params['voxel_size']
                    params['dim6'] = 17.16*params['voxel_size']
                    params['dim7'] = 173.95*params['voxel_size']
                if var == 'BifY':
                    params['radius'] = 20.1*params['voxel_size']
                    params['dim1'] = 23.0*params['voxel_size']
                    params['dim2'] = 88.0*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 103.0*params['voxel_size']
                    params['dim5'] = 17.17*params['voxel_size']
                    params['dim6'] = 174.0*params['voxel_size']
                    params['dim7'] = 158.83*params['voxel_size']
                if var == 'Bif2T':
                    params['radius'] = 40.1*params['voxel_size']
                    params['dim1'] = 43.0*params['voxel_size']
                    params['dim2'] = 203.0*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 403.0*params['voxel_size']
                if var == 'Bif2V':
                    params['radius'] = 40.1*params['voxel_size']
                    params['dim1'] = 43.0*params['voxel_size']
                    params['dim2'] = 173.0*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 403.5*params['voxel_size']
                    params['dim5'] = 203.3*params['voxel_size']
                    params['dim6'] = 31.3*params['voxel_size']
                    params['dim7'] = 344.9*params['voxel_size']
                if var == 'Bif2Y':
                    params['radius'] = 40.1*params['voxel_size']
                    params['dim1'] = 43.0*params['voxel_size']
                    params['dim2'] = 172.5*params['voxel_size']
                    params['dim3'] = 3.0*params['voxel_size']
                    params['dim4'] = 202.75*params['voxel_size']
                    params['dim5'] = 31.25*params['voxel_size']
                    params['dim6'] = 344.0*params['voxel_size']
                    params['dim7'] = 313.75*params['voxel_size']
        super().encode(params=params, target_dir=target_dir)
