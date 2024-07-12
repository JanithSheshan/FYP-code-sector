import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.signal import welch

# Directory containing .edf files
edf_directory = 'dataset'  # Update this path
processed_directory = 'processed_dataset'
os.makedirs(processed_directory, exist_ok=True)

# Nodes to keep
nodes_to_keep = ['Fp1.', 'Fp2.', 'Fz..', 'F3..', 'F4..', 'C3..', 'C4..', 'P3..', 'P4..']

# Initialize lists to collect data and labels from all files
all_band_powers = []
all_labels = []

# Create and save original CSV files
for file_name in os.listdir(edf_directory):
    if file_name.endswith('.edf'):
        file_path = os.path.join(edf_directory, file_name)
        
        # Load .edf file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Convert to Pandas DataFrame and save original CSV
        eeg_df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
        original_csv_file_path = os.path.join(edf_directory, file_name.replace('.edf', '_original.csv'))
        eeg_df.to_csv(original_csv_file_path, index=False)

# Drop unwanted columns and save new CSV files
for file_name in os.listdir(edf_directory):
    if file_name.endswith('_original.csv'):
        original_csv_file_path = os.path.join(edf_directory, file_name)
        
        # Load original CSV file
        eeg_df = pd.read_csv(original_csv_file_path)
        
        # Drop unwanted columns
        eeg_df = eeg_df[nodes_to_keep]
        
        # Save new CSV file with only the desired nodes
        new_csv_file_path = os.path.join(processed_directory, file_name.replace('_original.csv', '_processed.csv'))
        eeg_df.to_csv(new_csv_file_path, index=False)

# Initialize MNE info based on processed data
info = mne.create_info(ch_names=nodes_to_keep, sfreq=250, ch_types='eeg')

# Load each processed CSV file
for file_name in os.listdir(processed_directory):
    if file_name.endswith('_processed.csv'):
        file_path = os.path.join(processed_directory, file_name)
        
        # Load CSV file
        eeg_df = pd.read_csv(file_path)
        
        # Convert DataFrame to MNE RawArray
        raw = mne.io.RawArray(eeg_df.T, info)
        
        # Apply band-pass filter
        raw.filter(l_freq=13, h_freq=30)
        
        # Apply ICA
        ica = ICA(n_components=3, random_state=42)
        ica.fit(raw)
        raw_corrected = ica.apply(raw)
        
        # Extract band powers
        band_powers = []
        for i in range(0, len(raw_corrected), 5):
            raw_segment = raw_corrected.copy().crop(tmin=0, tmax=60)
            psd, freqs = welch(raw_segment.get_data(), fs=raw.info['sfreq'], nperseg=256)
            band_powers.append(np.mean(psd, axis=0))
            
        # Save band powers and label
        all_band_powers.extend(band_powers)
        if 'left' in file_name.lower():
            all_labels.extend([1] * len(band_powers))
        elif 'right' in file_name.lower():
            all_labels.extend([2] * len(band_powers))
        else:
            all_labels.extend([0] * len(band_powers))

# Debugging statements to check the labels
print(f"Number of samples: {len(all_band_powers)}")
print(f"Number of labels: {len(all_labels)}")
print("Unique classes and counts in the dataset:")
print(np.unique(all_labels, return_counts=True))

# Convert lists to arrays for further processing
X = np.array(all_band_powers)
y = np.array(all_labels)

# Reshape X to be 2D if it's not already
if X.ndim == 1:
    X = X.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if there are multiple classes
if len(np.unique(y_train)) > 1:
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train score: {train_score:.2f}")
    print(f"Test score: {test_score:.2f}")

    # Save the model
    model_file_path = 'model.joblib'
    from joblib import dump
    dump(model, model_file_path)

    # Save the processed data
    processed_data_file_path = 'processed_data.npz'
    np.savez(processed_data_file_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
else:
    print("The dataset does not contain multiple classes. SVM training skipped.")
