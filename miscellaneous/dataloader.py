import torch
import numpy as np
import os
import h5py
import pandas as pd
from torch.utils.data import Dataset
from functools import partial
from scipy.signal import resample

EEG_channels =['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

class EEGDataSet(Dataset):
    """
        Version 2
        Dataloader for FHA EEG corpus.

        Args:
            workspace_root: string root directory
            dataset_folder: list[str] Dataset folder
            seq_len: Target sequence length, smaller sequence will be zero-padded
            dtype: {'default' : all channels
                    'delayed_embedding': time delayed embedding
                    'single_channel': single channel EEG}
            args:
                randomChannel: boolean
                randomClip: boolean
                ERP: int event sequence length for ERP
                channel: eeg channel to choose, default 0
                lags: lag for time embedded delay 
                dim: dimension for time delayed embedding

    """
    def __init__(self, workspace_root:str, dataset_folder:list|str, seq_len:int = 512, dtype:str ='default', **args):
        self.size = 0
        self.root = workspace_root
        self.dataset_folder = dataset_folder if type(dataset_folder) == list else [dataset_folder]
        self.sLen = seq_len
        self.datasetType = dtype
        self.total_channels = 0
        self.eegFiles = None
        self.discription = None
        self.dataLoader = None
        self.preprocess = []
        self.cacheRemainTimes = 0
        self.dataCache = dict()

        default_args = {'randomChannel' : False, 'lags': 1, 'embeding_dim': 20, 'channel': 0, 'randomClip': False,
                        'onset':None, 'duratrion':None, 'cacheReuseTimes': 0, 'datasetType':'default', 'fileType':'hd5', 'keys':['EEG_Raw']}
        
        default_args.update(args)
        self._setup(**default_args)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        datasample = None
        try:
            data_dict, real_idx = self.dataLoader(idx = idx)
            data_raw = data_dict['EEG_Raw']
            for f in self.preprocess:
                data_raw = f(data_raw)
            else:
                datasample = {'EEG_Raw': torch.FloatTensor(data_raw), 'ID':  torch.LongTensor([real_idx])}
        except Exception as e:
            print(f'Error processing {self.eegFiles[idx]}: {e}')

        return datasample
    
    def getLabel(self, index):
        if self.discription is None:
            print("dataset discriptions are not provided")
            return None
        return self.discription.iloc[index]
        
    def resetSelectedChannel(self, channel):
        if type(channel) == str:
            self.ch = self.ch_names.index(channel)
        else:
            assert channel < len(self.ch_names), print(f"{channel} index out of rannge {len(self.ch_names)}")
            self.ch = channel
        return self.ch

    def loadLabel(self, fp):
        if self.scanID is None or self.discription is not None:
            return
        try:
            df = pd.read_csv(os.path.join(self.root, fp))
            self.discription = pd.DataFrame(columns=df.columns)
            for i in range(len(self.eegFiles)):
                ScanID = self.scanID[i]
                row = df[df['ScanID'] == ScanID]
                self.discription.loc[i] = None if row.empty else row.iloc[0]

        except:
            print(f"Load discription file {fp} failed")


    def sample(self, fileName):
        try:
            with h5py.File(os.path.join(self.root, fileName), 'r') as f:
                eeg_clip = np.array(f['EEG_Raw'])
                padded_feature = self._preProcess(eeg_clip)
                return torch.FloatTensor(padded_feature).unsqueeze(0)
        except Exception as e:
            print(f'Error processing {fileName}: {e}')
            return None
        
    '''
        Initialize dataLoader object
    '''
    def _setup(self, **args):
        steps = ['_loadDataDir', '_setDataloader', '_setPreprocessPipeline', '_setPostProcessPipeline']
        count = 0
        try:
            self.random_ch = args['randomChannel']
            self.lags = args['lags']
            self.embeding_dim = args['embeding_dim'] 
            self.cacheReuseTimes = args['cacheReuseTimes']

            #load data directory
            self._loadDataDir()
            count += 1

            #choose a channel by reference or index
            self.ch = self.resetSelectedChannel(args['channel']) if 'channel' in args else 0
            count += 1

            #setup data loader
            self._setDataloader(**args)
            count += 1

            #setup preprocess steps
            self._setPreprocessPipeline(**args)
            count += 1

            #setup postprocess steps
            self._setPostProcessPipeline(**args)
            count += 1

        except Exception as e:
            print('func <_setup> step <{}> error: {}'.format(steps[count], e))
            self.size = 0
    
    def _loadDataDir(self):
        self.eegFiles = []
        self.scanID = []
        for folder in self.dataset_folder:
            path = os.path.join(self.root, folder)
            temp = os.listdir(path)
            for file in temp:
                if not self._isValidFile(file):
                    continue
                scanID = file.split(".")[0]
                self.scanID.append(scanID)
                path2file = os.path.join(self.root, folder, file)
                self.eegFiles.append(path2file)
    
        #read meta data
        self.size = len(self.eegFiles)
        with h5py.File(os.path.join(self.root, 'Meta.hdf5'), 'r') as f: 
            channels = np.array(f['EEG_channels'])
            self.ch_names = [c.decode('UTF-8') for c in channels]
            self.total_channels = len(channels)

    def _isValidFile(self, fileName):
        return fileName.endswith('.hdf5')

    '''
        Data loaders
    '''  
    def _fifLoader(self, idx,  onset, duration):
        fileName = self.eegFiles[idx]
        raw = mne.io.read_raw_fif(fileName, verbose='ERROR', preload=True)
        raw.set_eeg_reference(['Fpz'],verbose='ERROR')
        raw.pick(['eeg'], exclude=['Fpz'], verbose='ERROR')
        if duration is not None:
            segment = raw.get_data(picks=raw.ch_names, tmin=onset, tmax=onset+duration)
        else:
            segment = raw.get_data(picks=raw.ch_names, tmin=onset)

        return {'EEG_Raw': segment}, idx
    
    def _hd5Loader(self, idx, entries):
        data = {}
        fileName = self.eegFiles[idx]
        with h5py.File(fileName) as f:
            for key in entries:
                data[key] = np.array(f[key])

        return data, idx
    
    def _genericLoader(self, loader_fn, **fargs):
        if self.cacheRemainTimes == 0:
            self.dataCache['Data'], idx = loader_fn(**fargs)
            self.dataCache['idx'] = idx
            self.cacheRemainTimes = self.cacheReuseTimes
        else:
            self.cacheRemainTimes -= 1

        return self.dataCache['Data'], self.dataCache['idx']
    
    def _setDataloader(self, **args):
        if args['fileType'] == 'hd5':
            import h5py
            self.dataLoader = partial(self._genericLoader, loader_fn=self._hd5Loader, entries = args['keys'])

        elif args['fileType'] == 'fif':
            import mne
            onset = args['onset'] if 'onset' in args else None
            duration = args['duration'] if 'duration' in args else None
            self.dataLoader = partial(self._genericLoader, loader_fn=self._fifLoader, onset = onset, duration = duration)


    '''
        Calculate time delayed embedding
    ''' 
    def _timeDelayEmbedding(self, data, lags):
        assert data.ndim == 1, \
            print(f"Time delay embedding can only be constructed for single channel, received shape {data.shape}")
        mean = np.mean(data)
        std = np.std(data) + 1e-9
        data_preNormalized = (data - mean) / std
        return self._delayEmbeding(data_preNormalized)
        

    '''
        Calculate the average of ERP
    '''
    def _ERP(self, seglen, data):
        assert data.shape[-1] % seglen == 0, print(f"For ERP data type, data length must be divisible by segment length")
        ERP = np.zeros((*data.shape[:-1], seglen))
        eventCount =  data.shape[-1] // seglen
        for i in range(eventCount):
            ERP += data[...,(eventCount-1) * seglen : eventCount*seglen]
        ERP /= eventCount
        return ERP
    
    '''
        Data Preprocess
    '''
    def _setPreprocessPipeline(self, **args):
        if 'randomClip' in args and args["randomClip"] == True:
            assert self.datasetType != 'ERP', print("Randomized clip is not supported for ERP data")
            self.preprocess.append(self._randomClip)
        else:
            self.preprocess.append(self._crop)


        if self.datasetType == 'default':
            self.preprocess.append(self._normalize)
        elif self.datasetType == 'delayed_embedding':
            self.preprocess.append(self._normalize) #pre-normalize prevent data leakage
            self.preprocess.append(partial(self._timeDelayEmbedding, args['lags']))
        elif self.datasetType == 'single_channel':
            self.preprocess.append(self._selectChannel)
            self.preprocess.append(self._normalize)
        elif self.datasetType == 'ERP':
            self.preprocess.append(self._selectChannel)
            self.preprocess.append(partial(self._ERP, self.sLen))
            self.preprocess.append(self._normalize)

    

    def _selectChannel(self, data):
        if self.random_ch:
            self.ch = np.random.randint(0,self.total_channels)
        return np.expand_dims(data[self.ch], axis=0)
    
    def _delayEmbeding(self, lags, data):
        requestLen = self.sLen + (self.embeding_dim-1) * lags
        if requestLen < data.shape[0]:
            max_itvl = data.shape[0] - requestLen
            rnd = np.random.randint(0,max_itvl)
            data = data[rnd : rnd+requestLen]

        buffer = np.zeros((self.embeding_dim, self.sLen))
        offset = 0
        for i in range(self.embeding_dim):
            if data.shape[0] < self.sLen + offset:
                len_pad = self.sLen + offset - data.shape[0]
                temp = data[offset:]
                buffer[i] = np.pad(temp, ((0,len_pad)), 'constant', constant_values = 0)
            else:
                buffer[i] = data[offset : self.sLen + offset]
            offset += lags 

        return buffer

    def _normalize(self, data_raw):
        mean = np.mean(data_raw, axis=-1, keepdims=True)
        std = np.std(data_raw, axis=-1, keepdims=True) + 1e-9
        return (data_raw - mean) / std

    def _crop(self, data_raw):
        return data_raw[..., :self.sLen]
    
    def _randomClip(self, data_raw):      
        if data_raw.shape[-1] < self.sLen:
            len_pad = self.sLen - data_raw.shape[-1]
            if data_raw.ndim == 2:
                data_raw = np.pad(data_raw, ((0, 0),(0,len_pad)), 'mean')
            else:
                data_raw = np.pad(data_raw, ((0,len_pad)), 'mean')
        else:
            max_itvl = data_raw.shape[-1] - self.sLen
            rnd = np.random.randint(0,max_itvl)
            data_raw = data_raw[...,rnd : rnd+self.sLen]

        return data_raw

    '''
        Data postprocess
    '''
    def _setPostProcessPipeline(self, **args):
        pass


class FHA_Supervised(Dataset):
    def __init__(self, workspace:str,  annotations_files: list[str] | str, filePostfix: str, fileType:str, dataset_path:str = '', labels:list[str]=[], **kwargs):
        self.root = workspace
        self.dataset_path = dataset_path
        self.annotations_files = annotations_files
        self.fileExtension = filePostfix
        self.labels = labels
        self.eeg_annotations = None
        self._setup(fileType = fileType, **kwargs)

    def __len__(self):
        return len(self.eeg_annotations)

    def __getitem__(self, idx):
        data = self._loadFile(idx)
        if data is not None:
            data = self._preprocess(data)
        return data

    def _loadFile(self, idx):
        fileName = self._getFilePath(idx)
        data = None
        if fileName is not None:
            data = self.loadData(fileName)
            for label in self.labels:
                data[label] = self.eeg_annotations.iloc[idx][label]
            data['ID'] = idx
        else:
            raise Warning("<_loadFile>: File name can not be none")
        return data        
    
    def __str__(self):
        return self.eeg_annotations['AdmittingStatus'].value_counts().to_string()
    
    def _filterInvalidFile(self):
        temp = []

        for i in range(len(self.eeg_annotations)):
            if self._loadFile(i) is not None:
                temp.append(i)
        self.eeg_annotations = self.eeg_annotations.iloc[temp]

    
    def _getFilePath(self, idx):
        hospital =  self.eeg_annotations.iloc[idx]['Hospital']
        ScanID = self.eeg_annotations.iloc[idx]['ScanID']
        fileName = ScanID + self.fileExtension
        filePath = os.path.join(self.root, self.dataset_path, hospital, fileName)
        return filePath
    
    def _loadhd5(self, fileName, entries):
        data = {}
        with h5py.File(fileName) as f:
            for key in entries:
                data[key] = np.array(f[key])
        return data 
    
    def _loadfif(self, fileName, onset, duration):
        raw = mne.io.read_raw_fif(str(fileName), verbose='ERROR', preload=True)
        raw.pick(['eeg'], verbose='ERROR')
        raw.reorder_channels(EEG_channels)
        raw.set_eeg_reference(ref_channels='average', projection=False,verbose='ERROR')
        segment = raw.get_data(tmin=onset, tmax=onset+duration)

        return {'EEG_Raw' : segment.astype(np.float32)}

    def _preprocess(self, data):
        if self.preprocessor is not None:
            return self.preprocessor(data)
        return data
    
    def _setup(self, **args):
        if args['fileType'] == 'hd5':
            import h5py
            self.loadData = partial(self._loadhd5, entries = args['keys'])

        if args['fileType'] == 'fif':
            onset = args['onset'] if 'onset' in args else 0
            duration = args['duration'] if 'duration' in args else 1
            self.loadData = partial(self._loadfif, onset = onset, duration = duration)

        if type(self.annotations_files) == list:
            for annotation in self.annotations_files:
                if self.eeg_annotations is None:
                    self.eeg_annotations = pd.read_csv(os.path.join(self.root, annotation))
                else:
                    temp = pd.read_csv(os.path.join(self.root, annotation))
                    self.eeg_annotations = pd.concat([self.eeg_annotations, temp], ignore_index=True, sort=False)
        else:
            self.eeg_annotations = pd.read_csv(os.path.join(self.root, self.annotations_files))

        if 'filter' in args:
            self.eeg_annotations = self.eeg_annotations.query(args['filter'])
        if 'ratio' in args:
            self.eeg_annotations = self.eeg_annotations.sample(frac=args['ratio'])
        if 'preprocessor' in args:
            self.preprocessor = args['preprocessor']
        if 'validation' in args and args['validation']:
            self._filterInvalidFile()


class FHA_Unsupervised(Dataset):
    def __init__(self, data_csv_path, dataset_folder, config):
        self.dataset_folder = dataset_folder
        self._set_config(config)
        self._load_csv(data_csv_path)
        self._sort_data() # sort the data by ChunksName for faster sequential loading
        self.start_timestamps = 0

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        file_path, ScanID = self._get_file_path(idx)
        data = self._load_EEG(file_path, ScanID)
        data = self._crop_data(data)
        data = self._pick_channels(data)
        data = self._resample_data(data)

        labels = self._extract_labels(idx)
        data = self._pack_data(data, labels)
        return self.preprocess(data)
    
    #convert the data to dictionary
    def _pack_data(self, data, labels):
        data_dict = {'EEG_Raw': data}
        for i, key in enumerate(self.label_keys):
            data_dict[key] = labels[i]
        return data_dict
    
    def _extract_labels(self, idx):
        return self.data_csv.iloc[idx][self.label_keys].values

    def _get_file_path(self, idx):
        ScanID = self.data_csv.iloc[idx]['ScanID']
        chunk_name = self.data_csv.iloc[idx]['ChunksName']
        return os.path.join(self.dataset_folder, chunk_name), ScanID
    
    def _pick_channels(self, data):
        if self.nchannels == 20:
            return data
        else:
            channels = np.random.choice(data.shape[0], self.nchannels, replace=False)
            return data[channels]
    
    def _crop_data(self, data):
        # (channels, timepoints)
        # crop the data to the window size
        if data.shape[-1] < self.window_size:
            return data
        
        if self.access_pattern == 'random':
            start = np.random.randint(0, data.shape[-1] - self.window_size)
            data = data[..., start:start+self.window_size]
        elif self.access_pattern == 'sequential':
            data = data[..., self.start_timestamps:self.start_timestamps+self.window_size]

        return data
    
    #for sequential access pattern, after each epoch, increment the start location by window size
    def on_epoch_end(self):
        self.total_iterations += 1
        if self.access_pattern == 'sequential':
            self.start_timestamps += self.window_size

    #for sequential access pattern, if the start location + window size is greater than the data length, reset the start location to 0
    def on_epoch_start(self):
        if self.access_pattern == 'sequential':
            if self.start_timestamps + self.window_size >= self.max_length:
                self.start_timestamps = 0

    def _load_EEG(self, file_path, scan_id):
        with h5py.File(file_path, 'r') as f:
            data = f[scan_id][:]
        return data
    
    def _load_csv(self, data_csv_path):
        self.data_csv = pd.read_csv(data_csv_path)
    
    def _sort_data(self):
        self.data_csv = self.data_csv.sort_values(by='ChunksName')
        return self.data_csv
    
    def _resample_data(self, data):
        if self.sampling_rate != 256:
            max_samples = min(data.shape[-1], self.window_size)
            data = resample(data, int(max_samples * self.sampling_rate / 256), axis=-1)
        return data.astype(np.float32)
    
    def _set_config(self, config):
        self.config = config
        self.nchannels = int(config['channels'])
        self.window_size = int(config['window_size'])
        self.sampling_rate = config['sampling_rate']
        self.access_pattern = config['access_pattern']
        self.label_keys = config['label']
        self.max_length = config['max_continuos_length'] if 'max_continuos_length' in config else 0
        self.preprocess = config['preprocess'] if 'preprocess' in config else lambda x: x

        assert self.nchannels < 21 and self.nchannels > 0, "channels must be between 1 and 20"
        assert self.sampling_rate <= 256, "sampling rate must be less than 256 Hz"

if __name__ == '__main__':
    dataset_config={'channels': 1, 'window_size':256*10, 'sampling_rate':256, 'access_pattern':'random', 'label':['CMG','Age', 'Sex', 'ComorbidityLevelDesc']}
    dataset = FHA_Unsupervised(r"E:\EEG\FHA\Resting\001_a01_01\Annotations\RCH_balanced.csv", r"E:\EEG\FHA\Resting\chunks", dataset_config)