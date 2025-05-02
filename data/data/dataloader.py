import sys,os,torch
import numpy as np
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    # batch est une liste de samples individuels
    batch_speed = [item['speed'] for item in batch]
    batch_fault = [item['fault'] for item in batch]
    batch_diameter = [item['diameter'] for item in batch]
    batch_end = [item['end'] for item in batch]
    batch_vibration_complete = [item['vibration_complete'] for item in batch]  # Liste de tensors de tailles variables
    batch_vibration_reduce = [item['vibration_reduce'] for item in batch]  # Liste de tensors de tailles variables
    batch_vibration_fft_complete = [item['vibration_fft_complete'] for item in batch]  # Liste de tensors de tailles variables  
    batch_vibration_fft_reduce = [item['vibration_fft_reduce'] for item in batch]  # Liste de tensors de tailles variables
    return {
        'speed': batch_speed,
        'fault': batch_fault,
        'diameter': batch_diameter,
        'end': batch_end,
        'vibration_complete': batch_vibration_complete,
        'vibration_reduce': batch_vibration_reduce,
        'vibration_fft_complete': batch_vibration_fft_complete,
        'vibration_fft_reduce': batch_vibration_fft_reduce
    }

class CWRUDataset:
    def __init__(self, end='FE', fault_filter=None, window_size=None, stride=None, downsample_factor=4):
        assert end in ['DE', 'FE', 'BA'], "source must be 'DE', 'FE' or 'BA'"
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter must be a list or None"
        
        self.end = end
        self.fault_filter = fault_filter 
        self.window_size = window_size
        self.stride = stride
        self.downsample_factor = downsample_factor

        fault_name = {'normal':'Normal',
         'inner':'IR', 
         'outer':'OR', 
         'ball':'B'}
        
        if self.fault_filter  is not None:
            self.fault_filter  = [fault_name[fault] for fault in self.fault_filter ]

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder_path = os.path.dirname(os.path.abspath(__file__))

        self.samples = []  # liste de dicts {filepath, speed, fault, diameter, end}
        for speed in os.listdir(folder_path):
            full_speed_path = os.path.join(folder_path, speed)
            if os.path.isdir(full_speed_path):
                for f in os.listdir(full_speed_path):
                    if f.endswith(".npz"):
                        filepath = os.path.join(full_speed_path, f)
                        filename = f.split('.')[0]
                        pattern = filename.split('_')
                        # On suppose que le nom de fichier est toujours au format speed_fault_diameter_end
                        # Par exemple : 1700_1_1_DE.npz ou 1700_1_1_1_DE.npz
                        if len(pattern) == 4:
                            speed_rpm, fault, diameter, end = pattern
                        else:
                            speed_rpm, fault = pattern
                            diameter = None
                            end = self.end  # Si pas de diamètre, on suppose que c'est la fin par défaut

                        if self.fault_filter is None or fault in self.fault_filter and self.end in end:
                            # On ne garde que les fichiers qui correspondent au filtre de défaut et à la fin
                            self.samples.append({
                                'filepath': filepath,
                                'speed': speed_rpm,
                                'fault': fault,
                                'diameter': diameter,
                                'end': end
                            })

        # Maintenant, préparer une liste de toutes les fenêtres
        self.windows = []  # liste de dicts {filepath, speed, fault, start_idx}
        for sample in self.samples:
            data = np.load(sample['filepath'])
            signal = torch.tensor(data[self.end]).float().squeeze()
            signal_length = signal.size(0)

            if self.window_size is None:
                self.windows.append({
                    'filepath': sample['filepath'],
                    'speed': sample['speed'],
                    'fault': sample['fault'],
                    'diameter': sample['diameter'],
                    'end': sample['end'],
                    'start_idx': 0
                })
            else:
                for start in range(0, signal_length - self.window_size + 1, self.stride):
                    self.windows.append({
                        'filepath': sample['filepath'],
                        'speed': sample['speed'],
                        'fault': sample['fault'],
                        'diameter': sample['diameter'],
                        'end': sample['end'],
                        'start_idx': start
                    })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window_info = self.windows[index]
        data = np.load(window_info['filepath'])
        signal = torch.tensor(data[self.end]).float().squeeze()

        if self.window_size is not None:
            start_idx = window_info['start_idx']
            vibration = signal[start_idx:start_idx + self.window_size]
        else:
            vibration = signal  # si pas de fenêtre glissante, tout le signal

        # Normalisation de la FFT
        N_complete = vibration.shape[-1]
        vibration_reduce = vibration[...,::self.downsample_factor]  # Réduction de la taille du signal
        N_reduce = vibration_reduce.shape[-1]

        sample = {
            'speed': window_info['speed'],
            'fault': window_info['fault'],
            'diameter': window_info['diameter'],
            'end': window_info['end'],
            'vibration_complete': vibration,
            'vibration_reduce':vibration_reduce,  # Réduction de la taille du signal
            'vibration_fft_complete': torch.abs(torch.fft.rfft(vibration, dim=-1))/N_complete,  # FFT du signal complet
            'vibration_fft_reduce': torch.abs(torch.fft.rfft(vibration_reduce, dim=-1))/N_reduce # FFT du signal réduit
        }
        return sample

if __name__ == "__main__":
    # Exemple d'utilisation
    train_dataset = CWRUDataset(
        fault_filter=['normal', 'inner', 'outer'],
        window_size=2048,
        stride=512)
    
    test_dataset = CWRUDataset(
        fault_filter=['ball'],
        window_size=2048,
        stride=512)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

