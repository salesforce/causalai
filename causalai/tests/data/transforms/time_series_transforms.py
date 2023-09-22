import unittest
import numpy as np
import math
from causalai.data.transforms.time_series import StandardizeTransform, DifferenceTransform


class TestTimeSeriesTransforms(unittest.TestCase):
    def test_standardize_transform_class(self):
        data = np.array([[-0.42889981,  0.33311697, -2.29588071],
                         [-0.39757703, -0.6893588, -1.46107175],
                         [ 0.06473446,  1.53470192, 0.71391537],
                         [-0.29572794,  0.52484768, 1.45721869],
                         [-1.68834491, -0.20526237, -0.34289961]])
        
        # test when a single data array is provided
        transform = StandardizeTransform(with_mean = True, with_std = True)
        transform.fit(data)
        self.assertTrue(np.allclose(transform.global_mean,[-0.54916305,  0.29960908, -0.3857436 ], atol=1e-7))
        self.assertTrue(np.allclose(transform.global_var,[0.35525528, 0.56205244, 1.88250997], atol=1e-7))
        
        # test when 2 data arrays are provided
        data2 = data[:3]
        transform = StandardizeTransform(with_mean = True, with_std = True)
        transform.fit(data, data2)
        self.assertTrue(np.allclose(transform.global_mean, [-0.4384447,   0.33456319, -0.62146939], atol=1e-7))
        self.assertTrue(np.allclose(transform.global_var, [0.26156496, 0.6631403,  1.87277762], atol=1e-7))
        
        # test if transformed array is correct
        data_transformed = transform.transform(data)
        
        data_transformed_gt = np.array([[ 1.86629855e-02, -1.77594901e-03, -1.22354243e+00],
                                         [ 7.99079567e-02, -1.25737298e+00, -6.13522557e-01],
                                         [ 9.83858815e-01,  1.47376659e+00,  9.75805581e-01],
                                         [ 2.79051985e-01,  2.33668760e-01,  1.51895955e+00],
                                         [-2.44391150e+00, -6.62904086e-01,  2.03559268e-01]])
        self.assertTrue(np.allclose(data_transformed, data_transformed_gt, atol=1e-7))
        
        
        # repeat above tests when data has NaNs
        
        data = np.array([[-0.42889981,  0.33311697, -2.29588071],
                         [-0.39757703, -0.6893588, -1.46107175],
                         [ 0.06473446,  1.53470192, 0.71391537],
                         [-0.29572794,  0.52484768, 1.45721869],
                         [-1.68834491, -0.20526237, -0.34289961]])
        
        data[0,2] = math.nan
        data[1,0] = math.nan
        # test when a single data array is provided
        transform = StandardizeTransform(with_mean = True, with_std = True)
        transform.fit(data)
        self.assertTrue(np.allclose(transform.global_mean,[-0.58705955,  0.29960908,  0.09179068], atol=1e-7))
        self.assertTrue(np.allclose(transform.global_var,[0.43688837, 0.56205244, 1.21294254], atol=1e-7))
        
        # test when 2 data arrays are provided
        data2 = data[:3]
        transform = StandardizeTransform(with_mean = True, with_std = True)
        transform.fit(data, data2)
        self.assertTrue(np.allclose(transform.global_mean, [-0.45206726,  0.33456319, -0.06333228], atol=1e-7))
        self.assertTrue(np.allclose(transform.global_var, [0.34801098, 0.6631403,  1.2509687 ], atol=1e-7))
        
        # test if transformed array is correct
        
        data_transformed = transform.transform(data)
        
        data_transformed_gt = np.array([[ 3.92718775e-02, -1.77594901e-03,        math.nan],
                                         [       math.nan, -1.25737298e+00, -1.24969200e+00],
                                         [ 8.76047008e-01,  1.47376659e+00,  6.94922188e-01],
                                         [ 2.65015744e-01,  2.33668760e-01,  1.35949540e+00],
                                         [-2.09565351e+00, -6.62904086e-01, -2.49955777e-01]])
        
        data_transformed = transform.transform(data, data)
        
        # test if transformed array is correct when 2 arrays are to be transformed
        
        data_transformed_gt = np.array([
                                [[ 3.92718775e-02, -1.77594901e-03,        math.nan],
                                 [       math.nan, -1.25737298e+00, -1.24969200e+00],
                                 [ 8.76047008e-01,  1.47376659e+00,  6.94922188e-01],
                                 [ 2.65015744e-01,  2.33668760e-01,  1.35949540e+00],
                                 [-2.09565351e+00, -6.62904086e-01, -2.49955777e-01]],
                                [[ 3.92718775e-02, -1.77594901e-03,        math.nan],
                                 [       math.nan, -1.25737298e+00, -1.24969200e+00],
                                 [ 8.76047008e-01,  1.47376659e+00,  6.94922188e-01],
                                 [ 2.65015744e-01,  2.33668760e-01,  1.35949540e+00],
                                 [-2.09565351e+00, -6.62904086e-01, -2.49955777e-01]]
                                ])
        
        
        for d,d_gt in zip(data_transformed, data_transformed_gt):
            self.assertTrue(np.allclose(d, d_gt, atol=1e-7, equal_nan=True))
        
    def test_difference_transform_class(self):
        data = np.array([[-0.42889981,  0.33311697, -2.29588071],
                         [-0.39757703, -0.6893588, -1.46107175],
                         [ 0.06473446,  1.53470192, 0.71391537],
                         [-0.29572794,  0.52484768, 1.45721869],
                         [-1.68834491, -0.20526237, -0.34289961]])
        
        # test when a single data array is provided
        transform = DifferenceTransform(order=1)
        transform.fit(data)
        data_transformed = transform.transform(data)
        data_transformed_gt = np.array([[ 0.03132278, -1.02247577,  0.83480896],
                                 [ 0.46231149,  2.22406072,  2.17498712],
                                 [-0.3604624 , -1.00985424,  0.74330332],
                                 [-1.39261697, -0.73011005, -1.8001183 ]])
        
        self.assertTrue(np.allclose(data_transformed, data_transformed_gt, atol=1e-7, equal_nan=True))
        
        # test when a 2 data arrays are provided
        transform = DifferenceTransform(order=1)
        transform.fit(data)
        data_transformed = transform.transform(data, data[:3])
        data_transformed_gt = [
                              [[ 0.03132278, -1.02247577,  0.83480896],
                               [ 0.46231149,  2.22406072,  2.17498712],
                               [-0.3604624 , -1.00985424,  0.74330332],
                               [-1.39261697, -0.73011005, -1.8001183 ]],
                              [[ 0.03132278, -1.02247577,  0.83480896],
                               [ 0.46231149,  2.22406072,  2.17498712]]
                                ]
                              
            
        for d,d_gt in zip(data_transformed, data_transformed_gt):
            self.assertTrue(np.allclose(d, d_gt, atol=1e-7, equal_nan=True))

# if __name__ == "__main__":
#     unittest.main()