import glob, os, sys
import h5py as h5
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modules import utils

# %% Default Settings
path_modules, path_csv, path_scalogram = utils.append_default_path(verbose=False)

path_identifier = utils.get_path_identifier()

selected_features = ['이름', '성별', 'age', 'BMI', 'TST-Total Sleep time (min)', 'Sleep latency (min)',\
                    'N2 sleep\nlatency(min)', 'REM latency (min)', 'WASO (min)', 'WASO (%)',\
                    'STAGE 1/TST(%)', 'STAGE 2/TST(%)', 'STAGE 3+4/TST(%)', 'REM (%)', 'AI - Arousal index',\
                    'REM Arousal \nindex(h)', 'NREM Arousal \nindex(h)', 'Sleep Efficiency (%)',\
                    'AHI - Total index A+H', 'ODI(%)', 'BD1', 'BD2', 'BD3', 'BD4', 'BD5', 'BD6',\
                    'BD7', 'BD8', 'BD9', 'BD10', 'BD11', 'BD12', 'BD13', 'BD14', 'BD15', 'BD16',\
                    'BD17', 'BD18', 'BD19', 'BD20', 'BD21', 'BDI Total-2', 'ISI  Total-2',\
                    'PSQI Total-2', 'SSS', 'ESS_total', '1-Subjective TST (hr)', '3-Subjective Sleep latency  (hr)']


# %% Function Definitions
def init_df_origin(excel_data):
    df = excel_data.copy() # 에러 방지를 위한 복제본 생성
    df = df.iloc[2:, 1:] # 비어 있는 cell + 필요 없는 초반 column --> 삭제
    df.rename(index={2:''}, inplace=True)
    df.columns = df.iloc[0,:] # columns 이름 설정
    df = df.iloc[1:, :] # columns 이름을 가져왔으니, 해당 row는 삭제

    df['age'] = df['age']*(-1) # 음수로 읽어들이는 age를 양수로 변환
    
    df['PSG study Number#'] = df['PSG study Number#'].str.strip()
    df['PSG study Number#'] = df['PSG study Number#'].str.upper() # 모든 sub_ID를 대문자로 변환 (혹시 소문자가 있으면 추후에 오류가 발생하기 때문)
    df.set_index('PSG study Number#', inplace=True)

    # 필요한 column만 include
    df = df.loc[:, selected_features]

    # abnormal value(NaN, .) 제거
    # 1. OSA
    con_abnormal_OSA = df['AHI - Total index A+H']=='.'
    con_NaN_OSA = df['AHI - Total index A+H'].isna()
    # 2. ISI
    con_abnormal_ISI = df['ISI  Total-2']=='.'
    con_NaN_ISI = df['ISI  Total-2'].isna()
    df = df[~(con_abnormal_OSA | con_NaN_OSA | con_abnormal_ISI | con_NaN_ISI)]

    # str value를 float로 변환
    df['AHI - Total index A+H'] = df['AHI - Total index A+H'].astype(float)
    df['ISI  Total-2'] = df['ISI  Total-2'].astype(float)

    return df   

def subject_exclusion(df_origin):
    # Exclude OSA and Include Insomnia
    con_OSA = df_origin['AHI - Total index A+H'] >= 15 # condition of being clinical OSA
    con_INS = df_origin['ISI  Total-2'] >= 15 # condition of being clinical Insomnia
    con = ~con_OSA & con_INS
    print("Number of included subjects: %d" % sum(con))

    df_origin_exclusion = df_origin[con]
    df_origin.to_csv(os.path.join(path_csv, 'df_origin.csv'), encoding='EUC-KR')
    df_origin_exclusion.to_csv(os.path.join(path_csv, 'df_origin_exclusion.csv'), encoding='EUC-KR')

    return df_origin_exclusion

def load_excel():
    if 'df_origin_exclusion.csv' in os.listdir(path_csv):
        df_origin_exclusion = pd.read_csv(os.path.join(path_csv, 'df_origin_exclusion.csv'), encoding='EUC-KR', index_col=0)
        
    else:
        print(path_csv)
        excel_data = pd.read_excel(os.path.join(path_csv, 'Brain age_PSG_raw_Total_201216(whole_data).xlsx'))
        df_origin = init_df_origin(excel_data)
        df_origin_exclusion = subject_exclusion(df_origin)
        
    return df_origin_exclusion

def match_subID_with_scalogram(df_origin_exclusion):
    
    '''
    h5 파일 중에서, label을 포함하고 있는 sub_id만 선택하는 함수
    '''
    f_names_scalogram = glob.glob(os.path.join(path_scalogram, '*.h5'))
    
    selected_sub_id = []; selected_f_names = []
    num_scalo_label = 0 # scalogram 중에서, df_origin_exclusion에 sub_id가 존재하는 파일의 개수
    for temp_sub_id in df_origin_exclusion.index.to_list():
        for temp_f_name in f_names_scalogram:
            if temp_sub_id in temp_f_name.split(path_identifier)[-1].split('.')[0]:
                selected_sub_id.append(temp_sub_id)
                selected_f_names.append(temp_f_name)
                num_scalo_label += 1
                
    df_label_scalogram = df_origin_exclusion.loc[selected_sub_id,:]
    return f_names_scalogram, selected_f_names, df_label_scalogram


# %% load_data class

class load_data():
    def __init__(self):            
        df_origin_exclusion = load_excel()
        f_names_scalogram, selected_f_names, df_origin_exclusion_scalogram = match_subID_with_scalogram(df_origin_exclusion)
        print('Total number of scalograms: {}'.format(len(os.listdir(path_scalogram))))
        print('Number of subjects that is enrolled in df_origin_exclusion: {}'.format(len(df_origin_exclusion)))
        print('Number of scalograms that is enrolled in df_origin_exclusion_scalogram: {}'.format(len(df_origin_exclusion_scalogram)))
        print('')
            
        # update self
        self.path_scalogram = path_scalogram
        self.df_origin_exclusion = df_origin_exclusion
        self.df_origin_exclusion_scalogram = df_origin_exclusion_scalogram
        
        self.path_np_data = os.path.join('./data')
        self.f_names_scalogram = f_names_scalogram
        self.selected_f_names = selected_f_names
        
    def stack_data_and_label(self):
        '''
        data(scalogram)를 하나의 numpy arrary에 쌓는다
        '''
        
        # 결과 data를 저장할 경로
        try:                     
            os.makedirs(self.path_np_data)
        except FileExistsError:
            pass
        
        if not('np_scalograms_1ch.npy' in os.listdir(self.path_np_data)):
            # - Data가 numpy 파일로 저장되어 있지 않은 경우, path_scalogram으로부터 scalogram을 read
            # - After remove sleep stage and reshape scalogram, save the numpy array as './data/np_scalograms.npy
            print('No existing Data --> create it !')
            data = []
            for temp_f_name in tqdm(self.selected_f_names, desc='concat scalograms'):
                f = h5.File(temp_f_name, 'r')
                scal = np.array(f['scalogram'])   
                data.append(scal)
        
            data = np.array(data) # (706, 16, 7, 1, 2000)
            data = data.astype(np.float32) # float 형으로 변환
            data = np.transpose(data, (0,4,1,2,3)) # reshape --> (706, 2000, 16, 7, 1)
            data = data[:,:,:,0,:] # F3 채널만 선택
            
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=43, shuffle=True)
            
            np.save(self.path_np_data+'/np_scalograms_1ch.npy', data)
            np.save(self.path_np_data+'/np_scalograms_1ch_train.npy', train_data)
            np.save(self.path_np_data+'/np_scalograms_1ch_test.npy', test_data)
        else:
            # data가 numpy 파일로 저장되어 있는 경우, 그것을 그대로 read
            print('FROM ./data --> LOAD np_scalograms_1ch.npy !')
            
            data = np.load(self.path_np_data+'/np_scalograms_1ch.npy')
            train_data = np.load(self.path_np_data+'/np_scalograms_1ch_train.npy')
            test_data = np.load(self.path_np_data+'/np_scalograms_1ch_test.npy')
        
        self.data = data
        self.train_data = train_data
        self.test_data = test_data
        print("Shape of data: {}".format(data.shape))


