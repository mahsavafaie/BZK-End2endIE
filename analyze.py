import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
import csv

df = pd.read_excel('transcription_BZK.xlsx')
df = df.loc[:,~df.columns.str.startswith('Unnamed:')]
df = df.fillna('')
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df = df.replace('\\n', ' ', regex=True)
df.loc[:, 'Filename'] = df.loc[:, 'Filename'].apply(lambda x: x + '.jpg')
df = df.rename(columns={'Filename': 'file_name'})

dict_replacements = {
    'Tabellen-Typ\xa0(abweichend)': 'Tabellen-Typ (abweichend)',
    'BY-Eigener-Typ' : 'BY-eigener-Typ',
    'BY-Spätphase (abweichend 1)': 'BY-Spätphase',
    'RLP-Hauptphase (abweichend 3)' : 'RLP-Hauptphase (abweichend 3 und 4)',
    'RLP-Hauptphase (abweichend 4)' : 'RLP-Hauptphase (abweichend 3 und 4)',

    # not perfect but good enough
    'BY-eigener-Typ (abweichend 1)' :  'BY-eigener-Typ (abweichend 1 und 2)',
    'BY-eigener-Typ (abweichend 2)' :  'BY-eigener-Typ (abweichend 1 und 2)',
    'Auskünfte_Statistisches_Landesamt_NRW (abweichend)': 'Auskünfte_Statistisches_Landesamt_NRW',
    'HH-NI-NRW-SH-Hauptphase (abweichend 2)' : 'HH-NI-NRW-SH-Hauptphase (abweichend 2 und 3)',
    'HH-NI-NRW-SH-Hauptphase (abweichend 3)' : 'HH-NI-NRW-SH-Hauptphase (abweichend 2 und 3)',

    # just put it into lower other category
    'Gerichtsurteile' : 'HH-NI-NRW-SH-Hauptphase (abweichend 1)',
    'NI-Frühe-Phase' : 'RLP-Hauptphase (abweichend 3 und 4)'
}

#print(df['Layout class'].replace(dict_replacements).value_counts())
df['Layout class'] = df['Layout class'].replace(dict_replacements)

selected_columns = ['file_name', 'CompensationOffice1', 'BZKNr', 
                    'Layout class', # just for stratification
                    'ApplicantFirstName', 'ApplicantLastName', 'ApplicantAltFirstName', 'ApplicantBirthName', 'ApplicantAltLastName', 'ApplicantBirthDate', 'ApplicantBirthPlace', 'ApplicantCurrentAddress',
                    'VictimFirstName',    'VictimLastName',    'VictimAltFirstName',    'VictimBirthName',    'VictimAltLastName',    'VictimBirthDate',    'VictimBirthPlace',    'VictimDeathDate', 'VictimDeathPlace']
df_selected = df[selected_columns]

train, test = train_test_split(df_selected, test_size=0.20, random_state=42, stratify=df_selected['Layout class'])
valid, test = train_test_split(test, test_size=0.50, random_state=42, stratify=test['Layout class'])

train = train.drop(columns=['Layout class'])
valid = valid.drop(columns=['Layout class'])
test = test.drop(columns=['Layout class'])

def create_image_folder_dataset(df, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for index, row in df.iterrows():
        file_name = './IE-GT/' + str(row['file_name'])
        if os.path.exists(file_name):
            shutil.copy(file_name, folder_name)
        else:
            print(f"File {file_name} does not exist")
    df.to_csv(folder_name + '/metadata.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

create_image_folder_dataset(train, './inferable/data/BZK/train')
create_image_folder_dataset(valid, './inferable/data/BZK/valid')
create_image_folder_dataset(test, './inferable/data/BZK/test')