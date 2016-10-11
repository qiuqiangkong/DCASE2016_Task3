# development data
dev_root = '/vol/vssp/AP_datasets/audio/dcase2016/task3/TUT-sound-events-2016-development'
dev_meta_txt = dev_root + '/meta.txt'
dev_meta_home_fd = dev_root + "/meta/home"
dev_meta_resi_fd = dev_root + "/meta/residential_area"
dev_wav_home_fd = dev_root + '/audio/home'
dev_wav_resi_fd = dev_root + '/audio/residential_area'
dev_evaluation_fd = dev_root + '/evaluation_setup'

# privative evaluation data
eva_root = '/vol/vssp/AP_datasets/audio/dcase2016/task3/TUT-sound-events-2016-evaluation'
eva_wav_home_fd = eva_root + '/audio/home'
eva_wav_resi_fd = eva_root + '/audio/residential_area'

# your workspace
scrap_fd = '/vol/vssp/msos/qk/DCASE2016_task3_scrap'

dev_fe_fd = scrap_fd + '/Fe_dev'
dev_fe_mel_fd = dev_fe_fd + '/Mel'
dev_fe_mel_home_fd = dev_fe_mel_fd + '/home'
dev_fe_mel_resi_fd = dev_fe_mel_fd + '/residential_area'
dev_md_fd = scrap_fd + '/Md_dev'
dev_results_fd = scrap_fd + '/Results_dev'

eva_fe_fd = scrap_fd + '/Fe_eva'
eva_fe_mel_fd = eva_fe_fd + '/Mel'
eva_fe_mel_home_fd = eva_fe_mel_fd + '/home'
eva_fe_mel_resi_fd = eva_fe_mel_fd + '/residential_area'
eva_md_fd = scrap_fd + '/Md_eva'
eva_results_fd = scrap_fd + '/Results_eva'


# global configurations
labels_home = [ '(object) rustling', '(object) snapping', 'cupboard', 'cutlery', 'dishes', 'drawer', 'glass jingling', 'object impact', 'people walking', 'washing dishes', 'water tap running', 'bg' ]
labels_resi = [ '(object) banging', 'bird singing', 'car passing by', 'children shouting', 'people speaking', 'people walking', 'wind blowing', 'bg' ]
lb_to_id_home = { ch:i for i, ch in enumerate(labels_home) }
id_to_lb_home = { i:ch for i, ch in enumerate(labels_home) }
lb_to_id_resi = { ch:i for i, ch in enumerate(labels_resi) }
id_to_lb_resi = { i:ch for i, ch in enumerate(labels_resi) }

fs = 44100.
win = 1024.