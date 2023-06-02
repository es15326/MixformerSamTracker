from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/usr/mvl2/esdft/MixFormer/data/got10k_lmdb'
    settings.got10k_path = '/usr/mvl2/esdft/MixFormer/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/usr/mvl2/esdft/MixFormer/data/lasot_lmdb'
    settings.lasot_path = '/usr/mvl2/esdft/MixFormer/data/lasot'
    settings.network_path = '/usr/mvl2/esdft/MixFormer/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/usr/mvl2/esdft/MixFormer/data/nfs'
    settings.otb_path = '/usr/mvl2/esdft/MixFormer/data/OTB2015'
    settings.prj_dir = '/usr/mvl2/esdft/MixFormer'
    settings.result_plot_path = '/usr/mvl2/esdft/MixFormer/test/result_plots'
    settings.results_path = '/usr/mvl2/esdft/MixFormer/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/usr/mvl2/esdft/MixFormer'
    settings.segmentation_path = '/usr/mvl2/esdft/MixFormer/test/segmentation_results'
    settings.tc128_path = '/usr/mvl2/esdft/MixFormer/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/usr/mvl2/esdft/MixFormer/data/trackingNet'
    settings.uav_path = '/usr/mvl2/esdft/MixFormer/data/UAV123'
    settings.vot_path = '/usr/mvl2/esdft/MixFormer/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

