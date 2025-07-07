#import sys
#sys.path.append('../')
#from SRFlowNIR.src.data import h5_utils
import h5_utils
import numpy as np
import h5py

full_h5_dir = "../../results/250209_PINN_sweep/Config_1t_1x_healthy_highSNR_PINN_SIREN_sweep_20250209-2055/Config_1t_1x_healthy_highSNR_PINN_SIREN_sweep_it35000/ref_data"

full_h5_filename = "healthy-05mm3_SR.h5"
#full_h5_filename = "Grei231113_invivo_06mm3_SR.h5"

full_h5_file = f"{full_h5_dir}/{full_h5_filename}"

output_dir = full_h5_dir
output_filename = "healthy-05mm3_SR_t25.h5"

with h5py.File(full_h5_file, mode = 'r') as hf:

    print([key for key in hf.keys()])

    u = hf.get('u')[25]
    v = hf.get('v')[25]
    w = hf.get('w')[25]
    p_normalized = hf.get('p_normalized')[25]

    #mask = np.asarray(hf['mask'])

    #_targetSNR = np.asarray(hf['_targetSNR'])
    #print(_targetSNR)


h5_utils.save_to_h5(f'{output_dir}/{output_filename}', 'u', u)#, compression='gzip')
h5_utils.save_to_h5(f'{output_dir}/{output_filename}', 'v', v)#, compression='gzip')
h5_utils.save_to_h5(f'{output_dir}/{output_filename}', 'w', w)#, compression='gzip')
h5_utils.save_to_h5(f'{output_dir}/{output_filename}', 'p_normalized', p_normalized)#, compression='gzip')

#h5_utils.save_to_h5(f'{output_dir}/{output_filename}', 'mask', mask, compression='gzip')
