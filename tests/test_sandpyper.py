#!/usr/bin/env python

"""Tests for `sandpyper` package."""


import unittest
import os
import pickle
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely


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
crs_dict_string = {'mar': {'init': 'epsg:32754'}, 'leo':{'init': 'epsg:32755'} }

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

if os.getcwdb() != b'C:\\my_packages\\sandpyper\\tests': # if the script is running in github action as a workflow and not locally

    shoreline_leo_path = os.path.abspath("tests/test_data/shorelines/leo_shoreline_short.gpkg")
    shoreline_mar_path = os.path.abspath('tests/test_data/shorelines/mar_shoreline_short.gpkg')
    dsms_dir_path = os.path.abspath('tests/test_data/dsm_1m/')
    orthos_dir_path = os.path.abspath('tests/test_data/orthos_1m')
    transects_path = os.path.abspath('tests/test_data/transects')
    lod_mode=os.path.abspath('tests/test_data/lod_transects')
    label_corrections_path=os.path.abspath("tests/test_data/clean/label_corrections.gpkg")
    watermasks_path=os.path.abspath("tests/test_data/clean/watermasks.gpkg")
    shoremasks_path=os.path.abspath("tests/test_data/clean/shoremasks.gpkg")
    test_pickled=os.path.abspath("tests/test_data/test.p")

else:

    shoreline_leo_path = os.path.abspath("test_data/shorelines/leo_shoreline_short.gpkg")
    shoreline_mar_path = os.path.abspath('test_data/shorelines/mar_shoreline_short.gpkg')
    dsms_dir_path = os.path.abspath('test_data/dsm_1m/')
    orthos_dir_path = os.path.abspath('test_data/orthos_1m/')
    transects_path = os.path.abspath('test_data/transects/')
    lod_mode=os.path.abspath('test_data/lod_transects')
    label_corrections_path=os.path.abspath("test_data/clean/label_corrections.gpkg")
    watermasks_path=os.path.abspath("test_data/clean/watermasks.gpkg")
    shoremasks_path=os.path.abspath("test_data/clean/shoremasks.gpkg")
    test_pickled=os.path.abspath("test_data/tests/test_data/test.p")


class TestCreateProfiles(unittest.TestCase):
    """Tests the creation profile function."""

    def setUp(self):
        """Load shorelines examples and create necessary objects"""

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
                           crs=crs_dict_string['leo'],
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
                       crs=crs_dict_string['mar'],
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
                           location='leo' ,crs=crs_dict_string['leo'],
                           side='both'
                          )
        the_exception = cm.exception
        print(the_exception)
        self.assertEqual(the_exception.args, ("Maximum allowed size exceeded",))

    def test_003_CreateTransects(self):
        right=create_transects(self.mar_shoreline,
                           sampling_step=150,
                           tick_length=50,
                           location='mar' ,crs=crs_dict_string['mar'],
                           side='right'
                          )
        left=create_transects(self.mar_shoreline,
                                   sampling_step=150,
                                   tick_length=50,
                                   location='mar' ,crs=crs_dict_string['mar'],
                                   side='left'
                                  )
        self.assertTrue(right.touches(left).all())

class TestProfileSet(unittest.TestCase):
    """Tests the ProfileSet pipeline."""

    @classmethod
    def setUpClass(cls):
        ############################# Profile extraction ######################

        cls.P = ProfileSet(dirNameDSM=dsms_dir_path,
                        dirNameOrtho=orthos_dir_path,
                        dirNameTrans=transects_path,
                        transects_spacing=transects_spacing,
                        loc_codes=loc_codes,
                        loc_search_dict=loc_search_dict,
                        crs_dict_string=crs_dict_string,
                        check="all")

        cls.P.extract_profiles(mode='all',tr_ids='tr_id',sampling_step=1,add_xy=True,lod_mode=lod_mode)


        ############################# Iterative Silhouette Analysis with inflexion point search ######################


        cls.sil_df = get_sil_location(cls.P.profiles,
                                ks=(2,15),
                                feature_set=feature_set,
                               random_state=10)

        cls.opt_k = get_opt_k(cls.sil_df, sigma=0 )

        cls.P.kmeans_sa(cls.opt_k, feature_set=feature_set)
        cls.check_pre_cleanit=(cls.P.profiles.label_k.iloc[171],cls.P.profiles.label_k.iloc[666],cls.P.profiles.label_k.iloc[0],cls.P.profiles.label_k.iloc[-1])

        ############################# Cleaning ######################

        cls.P.cleanit(l_dicts=l_dicts,
          watermasks_path=watermasks_path,
          shoremasks_path=shoremasks_path,
          label_corrections_path=label_corrections_path)

        cls.check_post_cleanit=(cls.P.profiles.label_k.iloc[171],cls.P.profiles.label_k.iloc[666],cls.P.profiles.label_k.iloc[0],cls.P.profiles.label_k.iloc[-1])


    @classmethod
    def tearDownClass(cls):
        cls.P
        cls.sil_df
        cls.opt_k

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

        pts_spacing_0=np.nanmean(test_slice_tr0.coordinates.set_crs(32754).distance(test_slice_tr0.next_point.set_crs(32754)))
        pts_spacing_last=np.nanmean(test_slice_last.coordinates.set_crs(32754).distance(test_slice_last.next_point.set_crs(32754)))
        pts_spacing_rand=np.nanmean(test_slice_rand.coordinates.set_crs(32754).distance(test_slice_rand.next_point.set_crs(32754)))


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
        self.assertEqual(check_pre_cleanit, (7, 1, 8, 0))
        self.assertEqual(check_post_cleanit, (9, 0, 8, 2))

class TestProfileDynamics(unittest.TestCase):
    """Tests the ProfileDynamics pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.P = pickle.load(open(test_pickled, "rb"))
        cls.D = ProfileDynamics(cls.P, bins=5, method="JenksCaspall", labels=labels)
        cls.D.compute_multitemporal(loc_full=loc_full, filter_class='sand')
        cls.D.compute_volumetrics(lod=cls.D.lod_df)
        cls.D.LISA_site_level(mode="distance", distance_value=35)
        cls.D.discretise(absolute=True, print_summary=True, lod=cls.D.lod_df, appendix=appendix)
        cls.D.infer_weights()
        cls.D.BCD_compute_location("geometry","all",True, filterit='lod')
        cls.D.BCD_compute_transects(loc_specs=loc_specs,reliable_action='keep', dirNameTrans=cls.D.ProfileSet.dirNameTrans)

    @classmethod
    def tearDownClass(cls):
        cls.P
        cls.D

    def test_009_ProfileDynamics(self):
        self.assertEqual(self.D.bins,  5)
        self.assertEqual(self.D.labels,  ['Undefined', 'Small', 'Medium', 'High', 'Extreme'])
        self.assertIsInstance(self.D.ProfileDynamics,  sandpyper.sandpyper.ProfileDynamics)
        self.assertIsInstance(self.D.ProfileSet,  sandpyper.sandpyper.ProfileSet)

    def test_010_ProfileDynamics_sand(self):
        self.assertEqual(self.D.dh_df.shape, (5112, 11))
        self.assertEqual(self.D.dh_df.dh.sum(),  265.5305953633506)
        self.assertEqual(list(D.dh_df.query("location=='mar'").dt.unique()),['dt_7', 'dt_6', 'dt_5', 'dt_4', 'dt_3', 'dt_2', 'dt_1', 'dt_0'])
        self.assertEqual(list(D.dh_df.query("location=='leo'").dt.unique()),['dt_4', 'dt_3', 'dt_2', 'dt_1', 'dt_0'])

        self.assertIsInstance(self.D.dh_df.geometry.iloc[0],  shapely.geometry.point.Point)

    def test_011_dh_class_filters(self):
        self.D.compute_multitemporal(loc_full=loc_full, filter_class=['veg','no_sand'])
        self.assertEqual(self.D.dh_df.shape, (1245, 11))
        self.assertEqual(self.D.dh_df.dh.sum(),  -48.3220699429512)
        self.assertEqual(list(self.D.dh_df.class_filter.unique()),  ['veg_no_sand'])

        self.D.compute_multitemporal(loc_full=loc_full, filter_class=None)
        self.assertEqual(self.D.dh_df.shape, (14800, 11))
        self.assertEqual(self.D.dh_df.dh.sum(),  144.2691128794686)
        self.assertEqual(list(self.D.dh_df.class_filter.unique()),  ['no_filters_applied'])

    def test_012_dh_details(self):
        prepostcheck=[str(pre)+str(post) for pre,post in zipself.(D.dh_details.iloc[:5]['date_pre'],self.D.dh_details.iloc[:5]['date_post'])]
        
        self.assertEqual(self.D.dh_details.shape, (13, 6))
        self.assertEqual(self.D.dh_details.n_days.sum(), 769)
        self.assertEqual(prepostcheck, ['2018060620180713','2018071320180920','2018092020190211','2019021120190328','2019032820190731'])

    def test_013_lod_tables(self):
        self.assertEqual(self.D.lod_dh.shape, (1155, 12))
        self.assertEqual(self.D.lod_dh.shape, (1155, 12))

        self.assertEqual(self.D.lod_dh.dh_abs.sum(), 193.05977356433868)
        self.assertEqual(self.D.lod_df.rrmse.sum(),1.2861174673896938)


        self.assertEqual(prepostcheck, ['2018060620180713','2018071320180920','2018092020190211','2019021120190328','2019032820190731'])





if __name__ == '__main__':
    unittest.main()
