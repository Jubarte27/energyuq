import os
import shutil
import sys
import chaospy as cp
import easyvvuq as uq
from easyvvuq.actions import Actions, CreateRunDirectory, Encode, Decode, ExecuteLocal

from ModCoders import DummyDecoder, BifVPEncoder

# Hardware resources
cpus_per_node = sys.argv[1]

#Get Job directory name
jobdir = sys.argv[2]

#Step size
#dx = float(sys.argv[3])

#Job type = T,V or Y
jobtype = sys.argv[3]

#BCs
BCs = 'VP'

# Define parameter space
params = {
    "step_length":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.000015},
    "steps":   {"type": "float",   "min": 100, "max": 1000000, "default": 20000},
    "voxel_size":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.000025},
    "inlet_max":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.001},
    "radius":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.002005},
    "outlet0_amplitude":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.0},
    "outlet0_mean":       {"type": "float",   "min": 0.0,  "max": 1.0,    "default": 0.0},
    "outlet0_phase":       {"type": "float",   "min": 0.0,  "max": 1.0,  "default": 0.0},
    "outlet0_period":    {"type": "float",   "min": 1.0,  "max": 2.0, "default": 1},
    "outlet1_amplitude":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.0},
    "outlet1_mean":       {"type": "float",   "min": 0.0,  "max": 1.0,    "default": 0.0},
    "outlet1_phase":       {"type": "float",   "min": 0.0,  "max": 1.0,  "default": 0.0},
    "outlet1_period":    {"type": "float",   "min": 1.0,  "max": 2.0, "default": 1},
    "dim1":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.00215},
    "dim2":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.00865},
    "dim3":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.00015},
    "dim4":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.020175},
    "dim5":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.010165},
    "dim6":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.001565},
    "dim7":   {"type": "float",   "min": 0.0, "max": 1.0, "default": 0.017245}
}

# Set distributions
# Main set of jobs
vary = {
    "steps":   cp.Uniform(10000, 20000),
    "step_length": cp.Uniform(0.000015,0.00003),
    "voxel_size": cp.Uniform(0.000025,0.000075)
}

## Long Runs
#vary = {
#    "steps":   cp.Uniform(100000, 200000),
#    "step_length": cp.Uniform(0.000015,0.00003),
#    "voxel_size": cp.Uniform(0.000025,0.000075)
#}

# Get cwd
cwd = os.getcwd()

# Reset run directory
if os.path.isdir(jobdir):
    shutil.rmtree(jobdir)
os.mkdir(jobdir)

# Create an encoder and decoder
generic_encoder = BifVPEncoder(
    template_fname="Bif2V_VP.template",
    delimiter="$",
    target_filename="dynamic_input.xml",
    convert_vars=['steps'],
    dependent_vars=['Bif2V']
)

copy_encoder = uq.encoders.CopyEncoder(os.path.join(cwd, 'UQJobSingle.py'), "HemePureJob.py")

encoder = uq.encoders.MultiEncoder(copy_encoder, generic_encoder)

if jobtype == 'Y':
    decoder = DummyDecoder(
        target_filename=["output_0.csv","output_1.csv","output_2.csv"],
        output_columns=["index", "pressure","velocityMag"]
    )

else:
    decoder = DummyDecoder(
        target_filename=["output_0.csv","output_1.csv"],
        output_columns=["index", "pressure","velocityMag"]
    )


# Set execution
command = 'python3 HemePureJob.py {} {} {} {}'.format(str(cpus_per_node), cwd, jobtype, BCs)

actions = Actions(CreateRunDirectory(root=os.path.join(cwd, jobdir), flatten=True),
                  Encode(encoder))

# Set campaign
campaign = uq.Campaign(name='HemePure', db_location='sqlite:///' + os.path.join(cwd, jobdir+'/campaign.db'), work_dir=os.path.join(cwd, jobdir))
campaign.add_app(name="hemelb", params=params, actions=actions)
campaign.set_sampler(uq.sampling.PCESampler(vary=vary, polynomial_order=9))
#campaign.set_sampler(uq.sampling.SCSampler(vary=vary, polynomial_order=8))
campaign.draw_samples()
print('Number of samples = {}'.format(campaign.get_active_sampler().count))

# Execute
campaign.execute().collate()

