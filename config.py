# dataset root path
root = '/vol/vssp/datasets/audio/dcase2016/task3'

# development data
meta_txt = root + '/TUT-sound-events-2016-development/meta.txt'

meta_home_fd = root + "/TUT-sound-events-2016-development/meta/home"
meta_resi_fd = root + "/TUT-sound-events-2016-development/meta/residential_area"

wav_home_fd = root + '/TUT-sound-events-2016-development/audio/home'
wav_resi_fd = root + '/TUT-sound-events-2016-development/audio/residential_area'

fe_mel_home_fd = 'Fe/Mel/home'
fe_mel_resi_fd = 'Fe/Mel/residential_area'

development_fd = root + '/TUT-sound-events-2016-development/evaluation_setup'
results_fd = 'Results'

# privative evaluation data
eva_ann_fd = ''
eva_wav_home_fd = root + '/TUT-sound-events-2016-evaluation/audio/home'
eva_wav_resi_fd = root + '/TUT-sound-events-2016-evaluation/audio/residential_area'
eva_fe_mel_home_fd = 'Fe_eva/Mel/home'
eva_fe_mel_resi_fd = 'Fe_eva/Mel/resi'


labels_home = [ '(object) rustling', '(object) snapping', 'cupboard', 'cutlery', 'dishes', 'drawer', 'glass jingling', 'object impact', 'people walking', 'washing dishes', 'water tap running', 'bg' ]
labels_resi = [ '(object) banging', 'bird singing', 'car passing by', 'children shouting', 'people speaking', 'people walking', 'wind blowing', 'bg' ]
lb_to_id_home = { ch:i for i, ch in enumerate(labels_home) }
id_to_lb_home = { i:ch for i, ch in enumerate(labels_home) }
lb_to_id_resi = { ch:i for i, ch in enumerate(labels_resi) }
id_to_lb_resi = { i:ch for i, ch in enumerate(labels_resi) }



fs = 44100.
win = 1024.