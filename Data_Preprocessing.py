import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the .edf files
folder_path = 'dataset'

# Define the channels of interest
channels = ['Fp1.', 'Fp2.', 'Fz..', 'F3..', 'F4..', 'C3..', 'C4..', 'P3..', 'P4..']

# Define frequency bands
bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}

def preprocess_and_extract_features(file_path):
    # Load the .edf file
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # Resample the data to 200 Hz
    raw.resample(200)
    
    # Apply a bandpass filter
    raw.filter(0.5, 75., fir_design='firwin')
    
    # Select the channels of interest
    raw.pick_channels(channels)
    
    # Crop the first 60 seconds of data
    raw.crop(tmin=60)
    
    # Apply a final bandpass filter
    raw.filter(1., 50., fir_design='firwin')
    
    # Get the annotations
    annotations = raw.annotations
    events, event_id = mne.events_from_annotations(raw)
    
    # Define epochs for each event type
    epochs = {}
    for event_type in event_id:
        epochs[event_type] = mne.Epochs(raw, events, event_id=event_id[event_type], tmin=0, tmax=1, baseline=None, preload=True)
    
    # Initialize a dictionary to hold the differential entropy features for each event type
    all_de_features = {'filename': os.path.basename(file_path)}
    
    for event_type, epoch in epochs.items():
        # Compute PSD using the Welch method
        psds, freqs = mne.time_frequency.psd_array_welch(
            epoch.get_data(), sfreq=epoch.info['sfreq'], fmin=4, fmax=45, n_fft=200, n_per_seg=200, n_jobs=1
        )
        
        # Extract average power in each band
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psds[:, :, freq_mask], axis=-1)
            band_powers[band] = band_power
        
        # Compute Differential Entropy (DE)
        def compute_de(psd):
            return np.log(psd)
        
        de_features = {f"{band}_{event_type}": compute_de(band_powers[band]) for band in bands}
        all_de_features.update(de_features)
    
    return all_de_features

def main(folder_path):
    # List to hold all the extracted features
    all_features = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".edf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}")
            de_features = preprocess_and_extract_features(file_path)
            
            # Add filename and features to the list
            all_features.append(de_features)
    
    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save the features to a CSV file
    features_df.to_csv('extracted_features_with_events.csv', index=False)
    print("Feature extraction completed and saved to extracted_features_with_events.csv")

if __name__ == "__main__":
    main(folder_path)
