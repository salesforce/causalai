import unittest
import numpy as np
import math
from causalai.data.time_series import TimeSeriesData


class TestTimeSeriesData(unittest.TestCase):
    def test_data_class(self):
        data = np.array([[-0.42889981,  0.33311697, -2.29588071],
                         [-0.39757703, -0.6893588, -1.46107175],
                         [ 0.06473446,  1.53470192, 0.71391537],
                         [-0.29572794,  0.52484768, 1.45721869],
                         [-1.68834491, -0.20526237, -0.34289961]])

        data_obj = TimeSeriesData(data, var_names=['A', 'B', 'C'])

         # test for the case when X,Y are specified and Z = None
        x,y,z = data_obj.extract_array(X='A', Y=('B', -1), Z=[], max_lag=2)

        self.assertTrue(all(x==[ 0.06473446, -0.29572794, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   1.53470192,  0.52484768]))
        self.assertTrue(z == None)

        x,y,z = data_obj.extract_array(X=0, Y=(1, -1), Z=[], max_lag=2)

        self.assertTrue(all(x==[ 0.06473446, -0.29572794, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   1.53470192,  0.52484768]))
        self.assertTrue(z == None)

        # test for the case when all X,Y,Z are specified
        x,y,z = data_obj.extract_array(X='A', Y=('B', -1), Z=[('C',-2)], max_lag=2)
        self.assertTrue(all(z==[[-2.29588071], [-1.46107175], [ 0.71391537]]))
        
        x,y,z = data_obj.extract_array(X=0, Y=(1, -1), Z=[('C',-2)], max_lag=2)
        self.assertTrue(all(z==[[-2.29588071], [-1.46107175], [ 0.71391537]]))

        data[1,-1] = math.nan
        data_obj = TimeSeriesData(data, var_names=['A', 'B', 'C'], contains_nans=True)

        # test for the case when X,Y are specified and Z = None and data has NaN
        x,y,z = data_obj.extract_array(X='A', Y=('B', -1), Z=[], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -0.29572794, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   1.53470192,  0.52484768]))
        self.assertTrue(z==None)
        
        x,y,z = data_obj.extract_array(X=0, Y=(1, -1), Z=[], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -0.29572794, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   1.53470192,  0.52484768]))
        self.assertTrue(z==None)

        # test for the case when all X,Y,Z are specified
        x,y,z = data_obj.extract_array(X='A', Y=('B', -1), Z=[('C',-2)], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   0.52484768]))
        self.assertTrue(all(z==[[-2.29588071], [ 0.71391537]]))
        
        x,y,z = data_obj.extract_array(X=0, Y=(1, -1), Z=[(2,-2)], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -1.68834491]))
        self.assertTrue(all(y==[-0.6893588,   0.52484768]))
        self.assertTrue(all(z==[[-2.29588071], [ 0.71391537]]))
        
        # test for when Y=None
        x,y,z = data_obj.extract_array(X='A', Y=None, Z=[('C',-2)], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -1.68834491]))
        self.assertTrue((y==None))
        self.assertTrue(all(z==[[-2.29588071], [ 0.71391537]]))
        
        x,y,z = data_obj.extract_array(X=0, Y=None, Z=[(2,-2)], max_lag=2)
        self.assertTrue(all(x==[ 0.06473446, -1.68834491]))
        self.assertTrue((y==None))
        self.assertTrue(all(z==[[-2.29588071], [ 0.71391537]]))
        
        
        # test to_var_index()
        self.assertTrue(data_obj.to_var_index([('A',-1)], [('B',-1)])==[[(0,-1)], [(1,-1)]])
        self.assertTrue(data_obj.to_var_index([('A',-1)])==[(0,-1)])
        
        self.assertTrue(data_obj.var_name2index('A')==0)
        self.assertTrue(data_obj.var_name2index(0)==0)
        
        # test data object length and dim
        data = np.array([[-0.42889981,  0.33311697, -2.29588071],
                         [-0.39757703, -0.6893588, -1.46107175],
                         [ 0.06473446,  1.53470192, 0.71391537],
                         [-0.29572794,  0.52484768, 1.45721869],
                         [-1.68834491, -0.20526237, -0.34289961]])
        data1=data
        data2 = data[:3]
        data_obj = TimeSeriesData(data1, data2, var_names=['A', 'B', 'C'])
        
        self.assertTrue(data_obj.length==[5,3])
        self.assertTrue(data_obj.dim==3)
        

# if __name__ == "__main__":
#     unittest.main()