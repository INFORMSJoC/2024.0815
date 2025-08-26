from imports import *

# See the following website for lots of ideas:
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm

# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page288.htm
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/TestGO_files/TestCodes/beale.m
# Note: Formulas don't seem to match on web page and Matlab code. Using web page
def beale(x, y):
    # Beale has global minumum of 0 at (3, 0.5)
    return (
        (1.5   - x + x*y)**2 +
        (2.25  - x + x*(y**2))**2 + 
        (2.625 - x + x*(y**3))**2
    )

# https://www.mathworks.com/help/matlab/ref/peaks.html#mw_46aeee28-390e-4373-aa47-e4a52447fc85
def peaks(xx, yy):
    # peaks has global minimum of -6.55113 at (0.2283, -1.6256)
    #xx = xx + 0.2283
    #yy = yy - 1.6256
    return (
        3 * ((1 - xx) ** 2.0) * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**3 - yy**5) * np.exp(-(xx**2) - yy**2)
        - (1 / 3) * np.exp(-((xx + 1) ** 2) - yy**2) + 6.55113*0
    )

# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page1905.htm
def griewank(x1, x2, x3, x4):
    result = (x1**2 + x2**2 + x3**2 + x4**2)/4000
    result -= np.cos(x1/np.sqrt(2)) * np.cos(x2/np.sqrt(3)) * np.cos(x3/np.sqrt(4)) * np.cos(x4/np.sqrt(5))
    result += 1
    return result

def griewank2d(x1, x2):
    result = (x1**2 + x2**2)/4000
    result -= np.cos(x1/np.sqrt(1)) * np.cos(x2/np.sqrt(2)) # Note: Sqrt operand is different than above. Does it matter?
    result += 1
    return result

# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2720.htm
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/TestGO_files/TestCodes/powell.m
# https/www.sfu.ca/~ssurjano/powell.html:/
# The function is usually evaluated on the hypercube xi ∈ [-4, 5], for all i = 1, …, d. 
def powell(x1, x2, x3, x4):
    return (
        (x1 + 10*x2)**2 + 5*(x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4
    )

# https://en.wikipedia.org/wiki/Rastrigin_function
def rastrigin10d(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    result = 0
    for item in [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]:
        result += 10 + item**2 - 10*np.cos(2*np.pi*item)
    return result

def rastrigin2d(x1, x2):
    result = 0
    for item in [x1, x2]:
        result += 10 + item**2 - 10*np.cos(2*np.pi*item)
    return result

def qing(x1, x2, x3, x4, x5, x6, x7, x8):
    result = 0
    l = [x1, x2, x3, x4, x5, x6, x7, x8]
    for i in range(8):
        result += (l[i]**2 - i - 1)**2
    return result

def quintic(x1, x2, x3, x4, x5):
    result = 0
    for item in [x1, x2, x3, x4, x5]:
        result += np.abs(item**5 - 3*item**4 + 4*item**3 +2*item**2 - 10*item -4)
    return result

def rotated_hyper_ellipsoid(*args):
    result = 0
    n = len(args)
    for i in range(n):
        temp = sum(args[:i+1])
        result += temp**2
    return result 

def generate_function(func_id):
    
    if func_id == 1:
        obj_func = beale
        mean = [3, 0.5]
        domain_l = [-4.5, -4.5]
        domain_r = [4.5, 4.5]
        #  diam = 1 # Necessary?
        global_min = [3, 0.5]
    elif func_id == 2:
        obj_func = peaks
        mean = [0.2283, -1.6256]
        global_min = [0.2283, -1.6256]
        domain_l = [-3, -3]
        domain_r = [3, 3]
    elif func_id == 3:
        obj_func = griewank
        mean = [0, 0, 0, 0]
        global_min = [0, 0, 0, 0]
        domain_l = -5 * np.ones(4)
        domain_r =  5 * np.ones(4)
        #  obj_func = griewank2d
        #  mean = [0, 0]
        #  global_min = [0, 0]
        #  domain_l = [-5, -5]
        #  domain_r = [5, 5]
    elif func_id == 4:
        obj_func = powell
        mean = [0, 0, 0, 0]
        global_min = [0, 0, 0, 0]
        domain_l = [-4, -4, -4, -4]
        domain_r = [5, 5, 5, 5]
    elif func_id == 5:
        obj_func = quintic
        mean = [-1, -1, -1, -1, -1]
        global_min = [-1, -1, -1, -1, -1]
        domain_l = [-2, -2, -2, -2, -2]
        domain_r = [0, 0, 0, 0, 0]
    elif func_id == 6:
        obj_func = qing
        mean = [np.sqrt(i) for i in range(1,9)]
        global_min = [np.sqrt(i) for i in range(1,9)]
        domain_l = [0, 0, 0, 0, 0, 0, 0, 0]
        domain_r = [5, 5, 5, 5, 5, 5, 5, 5]
    elif func_id == 7:
        obj_func = rastrigin2d
        mean = [0, 0]
        global_min = [0, 0]
        domain_l = [-5.12, -5.12]
        domain_r = [5.12, 5.12]
    elif func_id == 8:
        obj_func = rastrigin10d
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        global_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        domain_l = -5.12 * np.ones(10)
        domain_r =  5.12 * np.ones(10)
    elif func_id == 9:
        obj_func = rotated_hyper_ellipsoid
        mean = [0] * 20
        global_min = [0] * 20
        domain_l = [-10] * 20
        domain_r = [10] * 20
        
    mean = np.array(mean)
    global_min = np.array(global_min)
    domain_l = np.array(domain_l)
    domain_r = np.array(domain_r)
    diam_x = np.linalg.norm(domain_r - domain_l)
    
    global_min_val = obj_func(*global_min)
    
    return obj_func, mean, global_min, domain_l, domain_r, diam_x, global_min_val
