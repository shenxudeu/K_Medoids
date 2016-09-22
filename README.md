# K Medoids Algorithm in Python 

K-Medoid is similar to K-means, it can be applied to any customized distance function. All it requires is that there is a distance function that return a real value when defining some distance between two data points. How it works if fairly simple. It randomly pick K centers, and clusters each other points to the nearest center. Iteratively then we would swap non-center points with center point and try to minimize the total distance cost function.

Reference: 
 - Wiki: https://en.wikipedia.org/wiki/K-medoids
 - http://blog.otoro.net/2015/08/23/k-medoids-clustering-algorithm/

Author: Shen Xu
Date: Sep, 2016
