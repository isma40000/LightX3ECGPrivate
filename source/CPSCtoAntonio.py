from libs import * 
import configVars

def CPSCtoAntonio(ecg, sample_rate,new_len=4096, scale=1,
                   use_all_leads=True, remove_baseline=False):
    
    reduced_leads = ['DI', 'DII', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    all_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Remove baseline
    if remove_baseline:
        sos = remove_baseline_filter(sample_rate)
        ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype='constant', axis=-1)
    else:
        ecg_nobaseline = ecg

    # Rescale
    ecg_rescaled = scale * ecg_nobaseline

     # Resample
    if sample_rate != 400:
        ecg_resampled = sgn.resample_poly(ecg_rescaled, up=400, down=sample_rate, axis=-1)
    else:
        ecg_resampled = ecg_rescaled

    leads, length = ecg_resampled.shape

    # Add leads if needed
    target_leads = all_leads if use_all_leads else reduced_leads
    n_leads_target = len(target_leads)
    l2p = dict(zip(target_leads, range(n_leads_target)))
    ecg_targetleads = ecg_resampled

    if n_leads_target >= leads and use_all_leads:
        ecg_targetleads[l2p['DIII'], :] = ecg_targetleads[l2p['DII'], :] - ecg_targetleads[l2p['DI'], :]
        ecg_targetleads[l2p['AVR'], :] = -(ecg_targetleads[l2p['DI'], :] + ecg_targetleads[l2p['DII'], :]) / 2
        ecg_targetleads[l2p['AVL'], :] = (ecg_targetleads[l2p['DI'], :] - ecg_targetleads[l2p['DIII'], :]) / 2
        ecg_targetleads[l2p['AVF'], :] = (ecg_targetleads[l2p['DII'], :] + ecg_targetleads[l2p['DIII'], :]) / 2

    # Reshape
    if new_len is None or new_len == length:
        ecg_reshaped = ecg_targetleads
    elif new_len > length:
        ecg_reshaped = np.zeros([n_leads_target, new_len])
        pad = (new_len - length) // 2
        # print(f"Nleads: {n_leads_target}, pad: {pad}, new_len: {new_len}, len: {length}")
        # print(f"Izquierda:{ecg_reshaped[..., pad:length+pad].shape} Derecha:{ecg_targetleads.shape}")
        ecg_reshaped[..., pad:length+pad] = ecg_targetleads
    else:
        extra = (length - new_len) // 2
        ecg_reshaped = ecg_targetleads[:, extra:new_len + extra]

    return ecg_reshaped

def Antonizar(fsource,fdest):
    ecg = np.load(fsource)
    print(ecg.shape)
    ecgAntonizado = CPSCtoAntonio(ecg, 500, scale=2, use_all_leads=True, remove_baseline=False)
    print(ecgAntonizado.shape)
    np.save(fdest[:-4], ecgAntonizado)

def buclePorCarpeta(source,dest,func,source_file_Extension='npy',dest_file_Extension='npy'):
    for filename in os.listdir(source):
        fsource = os.path.join(source, filename)
        fdest= os.path.join(dest, filename)
        if os.path.isfile(fsource) and filename.endswith(f"{source_file_Extension}"):#and not os.path.isfile(f"{fdest[:-3]}{dest_file_Extension}")
            # try:
            func(fsource,fdest)
            # except Exception as e:
            #     print(f"Este archivo da error: {fsource}\n")
            #     print(e)

buclePorCarpeta(os.path.join("..","..","LightX3ECGPrivate","datasets","Casos","CPSC-2018","CasosNumpy"),os.path.join("..","..","Prueba"),Antonizar)
# ecg = np.load('../../Examenes_Antonio_NPY/100.npy')
# print(ecg.shape)
# ecg = ecg.T
# ecgAntonizado = CPSCtoAntonio(ecg, 500, scale=2, use_all_leads=True, remove_baseline=False)
# print(ecgAntonizado.shape)
# print(ecgAntonizado)

## Código que haría falta para mostrar un caso
# filename="A0001.npy"
# data = np.load(os.path.join("..","..","CPSCAntonizado",filename))
# data = data* 1
# #print(data)
# xml_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# ecg_plot.plot(data, sample_rate= 400,lead_index=xml_leads, title=f"{filename}")