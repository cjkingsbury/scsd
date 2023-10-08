import numpy as np

def ellipsoid_from_df(v1, v2, df): return (np.array([np.mean(df[v1].values),np.mean(df[v2].values)]), np.cov(df[v1].values, df[v2].values)) 
def ellipsoid_eq(testvals, means, invcov): return  np.sqrt(np.dot(np.dot(np.subtract(testvals,means).T,invcov),np.subtract(testvals,means)))

def ellipsoid_eq_for_web(testvals, parameters): return  np.sqrt(np.dot(np.dot(np.subtract(testvals,parameters[0:2]).T,np.array(parameters)[[2,4,4,3]].reshape(2,2)),np.subtract(testvals,parameters[0:2])))
def return_ellipsoid_pars(means, cov): return  np.round(means.tolist() + np.linalg.inv(cov).flatten()[[0,3,2]].tolist(),4)
def print_ellipsoid_pars(label,v1,v2, ellipsoid_pars): return '	'.join([label,v1,v2] + ['%.3f' % x for x in ellipsoid_pars[0:2]] + ['%d' % x if (np.abs(x)>20) else '%.3f' % x  for x in ellipsoid_pars[2:]])

def check_ellipsoid(testvals, params, threshold): return  ellipsoid_eq_for_web(testvals, params) < threshold

def ellipsoid_from_df_reject_outliers(v1, v2, df, max_sig=3):
    means, cov = (np.array([np.mean(df[v1].values),np.mean(df[v2].values)]), np.cov(df[v1].values, df[v2].values)) 
    el_pars = return_ellipsoid_pars(means,cov)
    sig_passes = [check_ellipsoid((x1,x2),el_pars,max_sig) for x1,x2 in zip(df[v1].values,df[v2].values)]
    return ellipsoid_from_df(v1,v2,df[sig_passes])

#ellipsoid equation for plotting it 
def ellipsoid_eq_2d(parameters, threshold, num_x = 100): 
    x0,y0,a,b,c = parameters
    d = threshold**2
    el_1 = lambda x : (-np.sqrt(np.abs(b*(d - a*(x0 - x)**2) + c**2*(x0 - x)**2)) + b*y0 + c*(x0 - x))/b
    el_2 = lambda x : (np.sqrt(np.abs(b*(d - a*(x0 - x)**2) + c**2*(x0 - x)**2)) + b*y0 + c*(x0 - x))/b
    xmin,xmax  =  x0+np.sqrt( -b*d / (c**2 - b*a)), x0-np.sqrt( -b*d / (c**2 - b*a))
    xs = np.linspace(xmin,xmax,num_x)
    return np.hstack([xs,xs[::-1],xs[0]]),np.hstack([el_1(xs),el_2(xs[::-1]),el_1(xs[0])])
