#!/usr/bin/env python

"""Tests for `sandpyper` package."""


import unittest
#from pathlib import Path
import os
import pickle
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

import sandpyper
from sandpyper.sandpyper import ProfileSet, ProfileDynamics
from sandpyper.common import get_sil_location, get_opt_k, create_transects, sensitivity_tr_rbcd

pd.options.mode.chained_assignment = None  # default='warn'


# create global variables used across the test analysis
loc_codes=["mar","leo"]

loc_full={'mar': 'Marengo',
         'leo': 'St. Leonards'}

loc_specs={'mar':{'thresh':6,
       'min_points':6}}

loc_search_dict = {'leo': ['St', 'Leonards', 'leonards', 'leo'], 'mar': ['Marengo', 'marengo', 'mar'] }
crs_dict_epsg = {'mar': 32754}, 'leo':32755 }

labels=["Undefined", "Small", "Medium", "High", "Extreme"]

appendix=["_deposition", "_erosion"]

relabel_dict={"Undefined_erosion":"ue",
            "Small_erosion":"se",
            "Medium_erosion":"me",
            "High_erosion":"he",
            "Extreme_erosion":"ee",
             "Undefined_deposition":"ud",
             "Small_deposition":"sd",
             "Medium_deposition":"md",
             "High_deposition":"hd",
             "Extreme_deposition":"ed"
            }

transects_spacing=20

feature_set=["band1","band2","band3","distance"]

water_dict={'leo_20180606':[0,9,10],
'leo_20180713':[0,3,4,7],
'leo_20180920':[0,2,6,7],
'leo_20190211':[0,2,5],
'leo_20190328':[2,4,5],
'leo_20190731':[0,2,8,6],
'mar_20180601':[1,6],
'mar_20180621':[4,6],
'mar_20180727':[0,5,9,10],
'mar_20180925':[6],
'mar_20181113':[1],
'mar_20181211':[4],
'mar_20190205':[],
'mar_20190313':[],
'mar_20190516':[4,7]}

no_sand_dict={'leo_20180606':[5],
'leo_20180713':[],
'leo_20180920':[],
'leo_20190211':[1],
'leo_20190328':[],
'leo_20190731':[1],
'mar_20180601':[4,5],
'mar_20180621':[3,5],
'mar_20180727':[4,7],
'mar_20180925':[5],
'mar_20181113':[0],
'mar_20181211':[0],
'mar_20190205':[0,5],
'mar_20190313':[4],
'mar_20190516':[2,5]}

veg_dict={'leo_20180606':[1,3,7,8],
'leo_20180713':[1,5,9],
'leo_20180920':[1,4,5],
'leo_20190211':[4],
'leo_20190328':[0,1,6],
'leo_20190731':[3,7],
'mar_20180601':[0,7],
'mar_20180621':[1,7],
'mar_20180727':[1,3],
'mar_20180925':[1,3],
'mar_20181113':[3],
'mar_20181211':[2],
'mar_20190205':[3],
'mar_20190313':[1,5],
'mar_20190516':[0]}

sand_dict={'leo_20180606':[2,4,6],
'leo_20180713':[2,6,8],
'leo_20180920':[3],
'leo_20190211':[3],
'leo_20190328':[3],
'leo_20190731':[4,5],
'mar_20180601':[2,3],
'mar_20180621':[0,2],
'mar_20180727':[2,6,8],
'mar_20180925':[0,4,2],
'mar_20181113':[2,4],
'mar_20181211':[3,1],
'mar_20190205':[1,2,4],
'mar_20190313':[0,2,3],
'mar_20190516':[1,3,6]}


l_dicts={'no_sand': no_sand_dict,
         'sand': sand_dict,
        'water': water_dict,
        'veg':veg_dict}



class TestCreateProfiles(unittest.TestCase):
    """Tests the creation profile function."""

    def setUp(self):
        """Load shorelines examples and create necessary objects"""
        shoreline_leo_path = os.path.abspath("examples/test_data/shorelines/leo_shoreline_short.gpkg")
        shoreline_mar_path = os.path.abspath('examples/test_data/shorelines/mar_shoreline_short.gpkg')

        self.leo_shoreline = gpd.read_file(shoreline_leo_path)
        self.mar_shoreline = gpd.read_file(shoreline_mar_path)


    def tearDown(self):
        """Tear down test datasets, if any."""
        self.leo_shoreline
        self.mar_shoreline

    def test_000_CreateTransects(self):
        """Test create transects function."""
        transects_leo=create_transects(self.leo_shoreline,
                           sampling_step=20,
                           tick_length=50,
                           location='leo',
                           crs=crs_dict_epsg['leo'],
                           side='both'
                          )
        self.assertEqual(transects_leo.shape, (59, 3))
        self.assertEqual(transects_leo.length.mean(), 49.99999999999309)
        self.assertEqual(transects_leo.crs.name, 'WGS 84 / UTM zone 55S')
        self.assertIsInstance(transects_leo,  gpd.GeoDataFrame)


    def test_001_CreateTransects(self):
        transects_mar=create_transects(self.mar_shoreline,
                       sampling_step=20,
                       tick_length=50,
                       location='mar',
                       crs=crs_dict_epsg['mar'],
                       side='both'
                      )

        self.assertEqual(transects_mar.shape, (27, 3))
        self.assertEqual(transects_mar.length.mean(), 49.999999999999716)
        self.assertEqual(transects_mar.crs.name, 'WGS 84 / UTM zone 54S')
        self.assertIsInstance(transects_mar,  gpd.GeoDataFrame)

    def test_002_CreateTransects(self):
        with self.assertRaises(ValueError) as cm:
            create_transects(self.leo_shoreline,
                           sampling_step=0,
                           tick_length=100,
                           location='leo' ,crs=crs_dict_epsg['leo'],
                           side='both'
                          )

    def test_003_CreateTransects(self):
        right=create_transects(self.mar_shoreline,
                           sampling_step=150,
                           tick_length=50,
                           location='mar' ,crs=crs_dict_epsg['mar'],
                           side='right'
                          )
        left=create_transects(self.mar_shoreline,
                                   sampling_step=150,
                                   tick_length=50,
                                   location='mar' ,crs=crs_dict_epsg['mar'],
                                   side='left'
                                  )
        self.assertTrue(right.touches(left).all())

class TestProfileSet(unittest.TestCase):
    """Tests the ProfileSet pipeline."""

    @classmethod
    def setUpClass(cls):
        ############################# Profile extraction ######################

        dsms_dir_path = os.path.abspath("examples/test_data/dsm_1m/")
        orthos_dir_path = os.path.abspath("examples/test_data/orthos_1m")

        transects_path = os.path.abspath("examples/test_data/transects")
        lod_mode=os.path.abspath("examples/test_data/lod_transects/")

        label_corrections_path=os.path.abspath("examples/test_data/clean/label_corrections.gpkg")
        watermasks_path=os.path.abspath("examples/test_data/clean/watermasks.gpkg")
        shoremasks_path=os.path.abspath("examples/test_data/clean/shoremasks.gpkg")

        cls.P = ProfileSet(dirNameDSM=dsms_dir_path,
                        dirNameOrtho=orthos_dir_path,
                        dirNameTrans=transects_path,
                        transects_spacing=transects_spacing,
                        loc_codes=loc_codes,
                        loc_search_dict=loc_search_dict,
                        crs_dict_epsg=crs_dict_epsg,
                        check="all")



        np.random.seed(42)

        cls.P.extract_profiles(mode='all',tr_ids='tr_id',sampling_step=1,add_xy=True,lod_mode=lod_mode)

        print(f"PROFILES SHAPE: {cls.P.profiles.shape}")
        ############################# Iterative Silhouette Analysis with inflexion point search ######################
        np.random.seed(10)
        cls.sil_df = get_sil_location(cls.P.profiles,
                                ks=(2,15),
                                feature_set=feature_set,
                               random_state=10)

        cls.opt_k = get_opt_k(cls.sil_df, sigma=0 )
        print(f"opt_k SHAPE: {len(cls.opt_k)}")

        cls.P.kmeans_sa(cls.opt_k, feature_set=feature_set)


        cls.check_no_sand_mar_preclean=cls.P.profiles.query("location == 'mar' and raw_date==20190205 and label_k in [0,5]")
        cls.check_no_sand_leo_preclean=cls.P.profiles.query("location == 'leo' and raw_date==20190211 and label_k ==1")

        cls.check_veg_mar_preclean=cls.P.profiles.query("location == 'mar' and raw_date==20180925 and label_k in [1,3]")
        cls.check_veg_leo_preclean=cls.P.profiles.query("location == 'leo' and raw_date==20180713 and label_k in [1,5,9]")

        cls.check_water_mar_preclean=cls.P.profiles.query("location == 'mar' and raw_date==20190516 and label_k in [4,7]")
        cls.check_water_leo_preclean=cls.P.profiles.query("location == 'leo' and raw_date==20180920 and label_k in [0,2,6,7]")

        cls.check_sand_mar_preclean=cls.P.profiles.query("location == 'mar' and raw_date==20180621 and label_k in [0,2]")
        cls.check_sand_leo_preclean=cls.P.profiles.query("location == 'leo' and raw_date==20190328 and label_k == 3")

        cls.check_watermask_mar_pre=cls.P.profiles.query("location == 'mar' and raw_date==20181113 and distance < 10").shape
        cls.check_watermask_leo_pre=cls.P.profiles.query("location == 'leo' and raw_date==20180920 and distance < 10").shape
        ############################# Cleaning ######################

        cls.P.cleanit(l_dicts,
                watermasks_path=watermasks_path,
                shoremasks_path=shoremasks_path,
                label_corrections_path=label_corrections_path)

        cls.check_watermask_mar_post=cls.P.profiles.query("location == 'mar' and raw_date==20181113 and distance < 10").shape
        cls.check_watermask_leo_post=cls.P.profiles.query("location == 'leo' and raw_date==20180920 and distance < 10").shape

        cls.check_no_sand_mar_postclean=cls.P.profiles.query("location == 'mar' and raw_date==20190205 and label_k in [0,5]")
        cls.check_no_sand_leo_postclean=cls.P.profiles.query("location == 'leo' and raw_date==20190211 and label_k ==1")

        cls.check_veg_mar_postclean=cls.P.profiles.query("location == 'mar' and raw_date==20180925 and label_k in [1,3]")
        cls.check_veg_leo_postclean=cls.P.profiles.query("location == 'leo' and raw_date==20180713 and label_k in [1,5,9]")

        cls.check_water_mar_postclean=cls.P.profiles.query("location == 'mar' and raw_date==20190516 and label_k in [4,7]")
        cls.check_water_leo_postclean=cls.P.profiles.query("location == 'leo' and raw_date==20180920 and label_k in [0,2,6,7]")

        cls.check_sand_mar_postclean=cls.P.profiles.query("location == 'mar' and raw_date==20180621 and label_k in [0,2]")
        cls.check_sand_leo_postclean=cls.P.profiles.query("location == 'leo' and raw_date==20190328 and label_k == 3")

    @classmethod
    def tearDownClass(cls):
        cls.P
        cls.sil_df
        cls.opt_k
        cls.check_no_sand_mar_preclean
        cls.check_no_sand_leo_preclean
        cls.check_water_mar_preclean
        cls.check_water_leo_preclean
        cls.check_sand_mar_preclean
        cls.check_sand_leo_preclean
        cls.check_veg_mar_preclean
        cls.check_veg_leo_preclean
        cls.check_watermask_mar_pre
        cls.check_watermask_leo_pre
        cls.check_watermask_leo_post
        cls.check_watermask_mar_post

    def test_004_check_dataframe(self):
        """Test the check dataframe creation."""

        self.assertEqual(self.P.check.shape, (15, 8))
        self.assertTrue((self.P.check.iloc[:,-2]==self.P.check.iloc[:,-1]).all())
        self.assertIsInstance(self.P.check, pd.DataFrame)

    def test_005_extractions(self):

        # check sampling step of points on first, last and random transects
        test_slice=self.P.profiles.query("location=='mar' & survey_date=='2019-05-16'").copy()
        test_slice.loc[:,'next_point']=test_slice.loc[:,'coordinates'].shift(1)
        test_slice_tr0=test_slice.query(f"tr_id=={test_slice.tr_id.iloc[-1]}").iloc[1:,:].copy()
        test_slice_last=test_slice.query(f"tr_id=={max(test_slice.tr_id.unique())}").copy()
        test_slice_rand=test_slice.query(f"tr_id=={np.random.randint(min(test_slice.tr_id.unique()),max(test_slice.tr_id.unique()))}").iloc[1:,:].copy()

        pts_spacing_0=np.nanmean(test_slice_tr0.coordinates.set_crs(epsg=32754).distance(test_slice_tr0.next_point.set_crs(epsg=32754)))
        pts_spacing_last=np.nanmean(test_slice_last.coordinates.set_crs(epsg=32754).distance(test_slice_last.next_point.set_crs(epsg=32754)))
        pts_spacing_rand=np.nanmean(test_slice_rand.coordinates.set_crs(epsg=32754).distance(test_slice_rand.next_point.set_crs(epsg=32754)))


        self.assertEqual(self.P.profiles.shape[0], 17508)
        self.assertEqual(self.P.profiles.isna()['z'].sum(), 0)
        self.assertIsInstance(self.P.profiles,  gpd.GeoDataFrame)
        self.assertAlmostEqual(pts_spacing_0, 1) # check if the sampling point spacing is actually 1m, as specified in the fn parameter
        self.assertAlmostEqual(pts_spacing_last, 1) # check if the sampling point spacing is actually 1m, as specified in the fn parameter
        self.assertAlmostEqual(pts_spacing_rand, 1) # check if the sampling point spacing is actually 1m, as specified in the fn parameter

    def test_006_iter_sil_location(self):
        self.assertFalse(self.sil_df.empty)
        self.assertEqual(self.sil_df.shape, (195, 4))
        self.assertIsInstance(self.sil_df,  pd.DataFrame)
        self.assertTrue(self.P.profiles.raw_date.unique().sort()==self.sil_df.raw_date.unique().sort())
        self.assertEqual(self.sil_df.k.max(), 14)
        self.assertEqual(self.sil_df.k.min(), 2)

    def test_007_optimal_k(self):
        self.assertEqual(len(self.opt_k), 15)
        self.assertIsInstance(self.opt_k,  dict)
        self.assertTrue(len(self.opt_k)==self.P.profiles.groupby(["location","raw_date"]).raw_date.count().shape[0])
        self.assertEqual(self.opt_k['leo_20190328'], 7)

    def test_008_correction(self):

        self.assertIsInstance(self.P.profiles.label_k[0],  np.int32)
        self.assertEqual(self.check_watermask_mar_pre,(219, 14))
        self.assertEqual(self.check_watermask_leo_pre,(300, 14))
        self.assertEqual(self.check_watermask_leo_post,(24, 15))
        self.assertEqual(self.check_watermask_mar_post,(219, 15))

        self.assertTrue(self.check_no_sand_mar_postclean.pt_class.unique()=='no_sand')
        self.assertTrue(self.check_no_sand_leo_postclean.pt_class.unique()=='no_sand')



        self.assertTrue(np.all(self.check_veg_mar_postclean.pt_class.unique()==np.array(['veg', 'water']))==True) # due to polygon corrections
        self.assertTrue(self.check_veg_leo_postclean.pt_class.unique()=='veg')

        self.assertTrue(self.check_water_mar_postclean.pt_class.unique()=='water')
        self.assertTrue(np.all(self.check_water_leo_postclean.pt_class.unique()==np.array(['veg', 'water', 'sand']))==True)

        self.assertTrue(np.all(self.check_sand_mar_postclean.pt_class.unique()==np.array(['water', 'sand']))==True)
        self.assertTrue(self.check_sand_leo_postclean.pt_class.unique()=='sand')



class TestProfileDynamics(unittest.TestCase):
    """Tests the ProfileDynamics pipeline."""

    @classmethod
    def setUpClass(cls):

        transects_path = os.path.abspath("examples/test_data/transects/")
        lod_mode=os.path.abspath("examples/test_data/lod_transects/")
        test_pickled=os.path.abspath("examples/test_data/test.p")
        P_test=pickle.load(open(test_pickled, "rb"))

        cls.D2 = ProfileDynamics(P_test, bins=5, method="JenksCaspall", labels=labels)
        cls.D2.compute_multitemporal(loc_full=loc_full, filter_class='sand')
        cls.D2.compute_volumetrics(lod=cls.D2.lod_df)
        cls.D2.LISA_site_level(mode="distance", distance_value=35)
        cls.D2.discretise(absolute=True, print_summary=True, lod=cls.D2.lod_df, appendix=appendix)
        cls.D2.infer_weights()
        cls.D2.BCD_compute_location("geometry","all",True, filterit='lod')
        cls.D2.BCD_compute_transects(loc_specs=loc_specs,reliable_action='keep',dirNameTrans=transects_path)

    @classmethod
    def tearDownClass(cls):
        cls.D2

    def test_009_ProfileDynamics(self):
        self.assertEqual(self.D2.bins,  5)
        self.assertEqual(self.D2.labels,  ['Undefined', 'Small', 'Medium', 'High', 'Extreme'])
        self.assertIsInstance(self.D2,  sandpyper.sandpyper.ProfileDynamics)
        self.assertIsInstance(self.D2.ProfileSet,  sandpyper.sandpyper.ProfileSet)

    def test_010_ProfileDynamics_sand(self):
        self.assertEqual(self.D2.dh_df.shape, (5112, 13))
        self.assertEqual(self.D2.dh_df.dh.sum(),  265.5305953633506)
        self.assertEqual(list(self.D2.dh_df.query("location=='mar'").dt.unique()),['dt_7', 'dt_6', 'dt_5', 'dt_4', 'dt_3', 'dt_2', 'dt_1', 'dt_0'])
        self.assertEqual(list(self.D2.dh_df.query("location=='leo'").dt.unique()),['dt_4', 'dt_3', 'dt_2', 'dt_1', 'dt_0'])

        self.assertIsInstance(self.D2.dh_df.geometry.iloc[0],  shapely.geometry.point.Point)

    def test_011_dh_class_filters(self):
        self.D2.compute_multitemporal(loc_full=loc_full, filter_class=['veg','no_sand'])
        self.assertEqual(self.D2.dh_df.shape, (1245, 11))
        self.assertEqual(self.D2.dh_df.dh.sum(),  -48.3220699429512)
        self.assertEqual(list(self.D2.dh_df.class_filter.unique()),  ['veg_no_sand'])

        self.D2.compute_multitemporal(loc_full=loc_full, filter_class=None)
        self.assertEqual(self.D2.dh_df.shape, (14800, 11))
        self.assertEqual(self.D2.dh_df.dh.sum(),  144.2691128794686)
        self.assertEqual(list(self.D2.dh_df.class_filter.unique()),  ['no_filters_applied'])

    def test_012_dh_details(self):
        prepostcheck=[str(pre)+str(post) for pre,post in zip(self.D2.dh_details.iloc[:5]['date_pre'],self.D2.dh_details.iloc[:5]['date_post'])]

        self.assertEqual(self.D2.dh_details.shape, (13, 6))
        self.assertEqual(self.D2.dh_details.n_days.sum(), 769)
        self.assertEqual(prepostcheck, ['2018060620180713','2018071320180920','2018092020190211','2019021120190328','2019032820190731'])

    def test_013_lod_tables(self):
        self.assertEqual(self.D2.lod_dh.shape, (1155, 12))
        self.assertEqual(self.D2.lod_df.shape, (13, 18))

        self.assertEqual(self.D2.lod_dh.dh_abs.sum(), 193.05977356433868)
        self.assertEqual(self.D2.lod_df.rrmse.sum(),1.2861174673896938)

        self.assertEqual(self.D2.lod_df.isna().sum().sum(),0)
        self.assertEqual(self.D2.lod_dh.isna().sum().sum(),0)

    def test_014_LISA_notebook(self):
        self.assertEqual(self.D2.hotspots.shape, (5112, 23))
        self.assertEqual(self.D2.hotspots.isna().sum().sum(),0)
        self.assertEqual(self.D2.hotspots.lisa_I.sum(),1187.7767310601516)
        self.assertEqual(self.D2.hotspots.lisa_opt_dist.unique()[0],35)
        self.assertEqual(self.D2.hotspots.lisa_dist_mode.unique()[0],'distance_band')
        self.assertEqual(self.D2.hotspots.decay.unique()[0],0)

    def test_015_LISA_idw(self):
        self.D2.compute_multitemporal(loc_full=loc_full, filter_class='sand')
        self.D2.LISA_site_level(mode="idw", distance_value=100, decay=-2)

        self.assertEqual(self.D2.hotspots.shape, (5112, 21))
        self.assertEqual(self.D2.hotspots.isna().sum().sum(),0)
        self.assertEqual(self.D2.hotspots.lisa_I.sum(),3693.3532825625425)
        self.assertEqual(self.D2.hotspots.lisa_opt_dist.unique()[0],100)
        self.assertEqual(self.D2.hotspots.lisa_dist_mode.unique()[0],'idw')
        self.assertEqual(self.D2.hotspots.decay.unique()[0],-2)

    def test_016_LISA_knn(self):
        self.D2.LISA_site_level(mode="knn", k_value=50)

        self.assertEqual(self.D2.hotspots.shape, (5112, 21))
        self.assertEqual(self.D2.hotspots.isna().sum().sum(),0)
        self.assertAlmostEqual(self.D2.hotspots.lisa_I.sum(),1594.5027919977902, places=2)
        self.assertEqual(self.D2.hotspots.lisa_opt_dist.unique()[0],50)
        self.assertEqual(self.D2.hotspots.lisa_dist_mode.unique()[0],'k')
        self.assertEqual(self.D2.hotspots.decay.unique()[0],0)

    def test_017_discretiser(self):
        self.assertTrue(list(set(np.isin(['markov_tag', 'magnitude_class', 'spatial_id'], self.D2.df_labelled.columns)))[0])
        self.assertEqual(self.D2.df_labelled.shape, (5112, 29))
        self.assertEqual(self.D2.df_labelled.markov_tag.value_counts()['Medium_deposition'], 629)
        self.assertTrue(len(self.D2.df_labelled.markov_tag.unique())==len(labels)*len(appendix))

    def test_018_BCDs(self):

        # infer_weights
        self.assertTrue(list(self.D2.weights_dict.values())==[0.05, 0.05, 0.17, 0.17, 0.33, 0.33, 0.58, 0.58, 1.62, 1.62])
        self.assertTrue(len(list(self.D2.weights_dict.keys()))==len(labels)*len(appendix))

        # location_ebcd
        self.assertEqual(self.D2.location_ebcds.shape, (8, 6))
        self.assertEqual(self.D2.location_ebcds.coastal_markov_idx.sum(),12.714)
        self.assertEqual(self.D2.location_ebcds.isna().sum().sum(),0)
        # check whether sign and trend makes sense
        self.assertTrue(list(set((np.select([[self.D2.location_ebcds.trend > 0 , self.D2.location_ebcds.sign == '+'],
          [self.D2.location_ebcds.trend < 0 , self.D2.location_ebcds.sign == '-']], [ True, True], default=False)).flatten()))[0])
        # location_ss
        self.assertEqual(self.D2.location_ss.shape, (13, 2))
        self.assertAlmostEqual(self.D2.location_ss.leo.sum(), 0.7210473593811956, places=2)
        self.assertEqual(self.D2.location_ss.isna().sum().sum(),0)
        self.assertEqual(self.D2.location_ss.loc['Extreme_deposition','leo'], 0.0)
        # transects_r-bcds
        self.assertEqual(self.D2.transects_rbcd.shape, (51, 6))
        self.assertAlmostEqual(self.D2.transects_rbcd.residual.sum(), 1.2806354281813492, places=2)
        self.assertEqual(self.D2.transects_rbcd.isna().sum().sum(),0)
        self.assertIsInstance(self.D2.transects_rbcd,  gpd.GeoDataFrame)
        self.assertEqual(self.D2.transects_rbcd.crs.to_epsg(),32754)
        self.assertEqual(list(self.D2.transects_rbcd.location.unique()),['mar', 'leo'])

    def test_019_volumetrics_with_LODs(self):
        # location_volumetrics
        self.assertEqual(self.D2.location_volumetrics.shape, (13, 18))
        self.assertEqual(self.D2.location_volumetrics.isna().sum().sum(),0)
        self.assertEqual(self.D2.location_volumetrics[['abs_in','abs_net_change','tot_vol_ero','n_days','n_obs_valid']].sum().sum(), -3443.7290466835893)
        self.assertIsInstance(self.D2.location_volumetrics,  pd.DataFrame)
        self.assertIsInstance(self.D2.location_volumetrics.dt.iloc[0],  str)
        self.assertEqual(list(self.D2.location_volumetrics.location_full.unique()),['St. Leonards', 'Marengo'])
        # transects_volumetrics
        self.assertEqual(self.D2.transects_volumetrics.shape, (309, 17))
        self.assertEqual(self.D2.transects_volumetrics.isna().sum().sum(),0)
        self.assertEqual(self.D2.transects_volumetrics[['abs_in','abs_net_change','n_days','n_obs_valid']].sum().sum(), 25226.484578590374)
        self.assertIsInstance(self.D2.transects_volumetrics,  pd.DataFrame)
        self.assertIsInstance(self.D2.transects_volumetrics.dt.iloc[0],  str)
        self.assertEqual(list(self.D2.transects_volumetrics.location_full.unique()),['St. Leonards', 'Marengo'])

if __name__ == '__main__':
    unittest.main()
