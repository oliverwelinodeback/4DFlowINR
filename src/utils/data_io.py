import h5py
import numpy as np

def load_data(config):
    
    PA_PER_MM_TO_PA_PER_M = 1000.0

    # Load data
    with h5py.File(config.data_file, mode='r') as hf:
        
        # Crop the data
        if config.setup.include_time:
            u = np.asarray(hf['u'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['u'])

            v = np.asarray(hf['v'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['v'])
            w = np.asarray(hf['w'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['w'])
            
            if config.load_pressure_from_data:
                p = (np.asarray(hf['p'][config.domain.t_start:config.domain.t_end, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    ) if config.domain.crop else np.asarray(hf['p']))  if config.setup.include_pressure else None

                px = (np.asarray(hf['px'][config.domain.t_start:config.domain.t_end, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['px'])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None

                py = (np.asarray(hf['py'][config.domain.t_start:config.domain.t_end, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['py'])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None

                pz = (np.asarray(hf['pz'][config.domain.t_start:config.domain.t_end, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['pz'])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None
            else:
                p = None
                px = None
                py = None
                pz = None
            
        else:
            t_index = config.domain.t_start

            u = np.asarray(hf['u'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['u'][t_index])
            v = np.asarray(hf['v'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['v'][t_index])
            w = np.asarray(hf['w'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]) if config.domain.crop else np.asarray(hf['w'][t_index])
            
            if config.load_pressure_from_data:

                p = (np.asarray(hf['p'][t_index,
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    ) if config.domain.crop else np.asarray(hf['p'][t_index]))  if config.setup.include_pressure else None
                
                px = (np.asarray(hf['px'][t_index, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['px'][t_index])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None

                py = (np.asarray(hf['py'][t_index, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['py'][t_index])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None

                pz = (np.asarray(hf['pz'][t_index, 
                                    config.domain.x_start:config.domain.x_end, 
                                    config.domain.y_start:config.domain.y_end, 
                                    config.domain.z_start:config.domain.z_end]
                                    )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['pz'][t_index])*PA_PER_MM_TO_PA_PER_M) if (
                                        config.setup.include_pressure and config.training.predict_gradients) else None
            else:
                p = None
                px = None
                py = None
                pz = None
            
        # T×h×w×d = (126, 81, 57, 50)

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config.domain.x_start:config.domain.x_end, 
                    config.domain.y_start:config.domain.y_end, 
                    config.domain.z_start:config.domain.z_end] if config.domain.crop else mask
        # h×w×d = (81, 57, 50)

        if config.resolution.from_file:
            config.resolution.dx = hf.attrs['spacing'][0]
            config.resolution.dy = hf.attrs['spacing'][1]
            config.resolution.dz = hf.attrs['spacing'][2]
            config.resolution.dt = hf.attrs['dt']
            print(f"Loaded resolution from file: {config.resolution.dx}, "
                f"{config.resolution.dy}, {config.resolution.dz}, {config.resolution.dt}"
            )	

    return u, v, w, p, px, py, pz, mask, config

def load_ref_data(config):
    
    PA_PER_MM_TO_PA_PER_M = 1000.0

    # Load data
    with h5py.File(config.data_file_ref, mode='r') as hf:
        
        # Crop the data
        if config.setup.include_time:
            u = np.asarray(hf['u'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['u'])
            v = np.asarray(hf['v'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['v'])
            w = np.asarray(hf['w'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['w'])
            
            p = (np.asarray(hf['p'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                                ) if config.domain.crop else np.asarray(hf['p'])) if config.setup.include_pressure else None

            px = (np.asarray(hf['px'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor, 
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['px'])*PA_PER_MM_TO_PA_PER_M) if (
                    config.setup.include_pressure and config.training.reference_gradients) else None
            py = (np.asarray(hf['py'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['py'])*PA_PER_MM_TO_PA_PER_M) if (
                    config.setup.include_pressure and config.training.reference_gradients) else None
            pz = (np.asarray(hf['pz'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if config.domain.crop else np.asarray(hf['pz'])*PA_PER_MM_TO_PA_PER_M) if (
                    config.setup.include_pressure and config.training.reference_gradients) else None

            timesteps = len(u)
            print("Length REF DATA: ", timesteps)

            if timesteps % 2 == 0:
                print("Even number of timesteps detected — cropping the final frame.")
                u = u[:-1]
                v = v[:-1]
                w = w[:-1]
                if config.setup.include_pressure and p is not None:
                    p = p[:-1]
                if config.setup.include_pressure and config.training.reference_gradients:
                    px = px[:-1]
                    py = py[:-1]
                    pz = pz[:-1]
                timesteps -= 1  # Update counter
                print("Length cropped REF DATA: ", timesteps)

        else:
            t_index = config.domain.t_start

            u = np.asarray(hf['u'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['u'][t_index])
            v = np.asarray(hf['v'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['v'][t_index])
            w = np.asarray(hf['w'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]) if config.domain.crop else np.asarray(hf['w'][t_index])
            
            p = (np.asarray(hf['p'][t_index,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                                ) if config.domain.crop else np.asarray(hf['p'][t_index])) if config.setup.include_pressure else None
            
            # TODO - fix cropping for these alt. remove
            px = np.asarray(hf['px'][t_index, 
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if (config.setup.include_pressure and config.training.reference_gradients) else None
            py = np.asarray(hf['py'][t_index,
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if (config.setup.include_pressure and config.training.reference_gradients) else None
            pz = np.asarray(hf['pz'][t_index,
                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                )*PA_PER_MM_TO_PA_PER_M if (config.setup.include_pressure and config.training.reference_gradients) else None

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                    config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                    config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor] if config.domain.crop else mask

    return u, v, w, p, px, py, pz, mask