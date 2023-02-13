import numpy as np 
import itertools

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    
    return vector[:]/ np.linalg.norm(vector[:])

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    

def midpoints(x,y):
    """ Returns the midpoints of the edges of the polygon with  vertices (x,y) 
    """
    x_v=np.append(np.array(x),x[0])
    y_v=np.append(np.array(y),y[0])
    mid_x=(x_v[0:-2]+x_v[1:-1])/2
    mid_y=(y_v[0:-2]+y_v[1:-1])/2
    return mid_x,mid_y



def get_positive_heading(x1,y1,x2,y2):
    """ Returns the heading of the vector from the point (x1,y1) to the point  (x2,y2)
    """
    x_v=np.array(x2)-np.array(x1)
    y_v=np.array(y2)-np.array(y2)
    
    heading=np.degrees(np.arctan2(x_v,y_v))
    heading[heading<0]+=360

    
    return heading
    

def footprint_simplification(x,y):
    """ Given a list of Vertices of a Polygon, this function discards all edges 
        with length zero and returns the vertices of the simplified Polygon
        
        TO DO: Merge edges with angles really close to 180 degrees
               and edges with comparatively small lengths.
    """
    n=len(x)
    x_pool=itertools.cycle(x)
    y_pool=itertools.cycle(y)
    remove_index=[True]*n
    x_0=next(x_pool)
    y_0=next(y_pool)
    x_1=next(x_pool)
    y_1=next(y_pool)
    for i in range(n):
        x_2=next(x_pool)
        y_2=next(y_pool)
        v1=[x_1-x_0,y_1-y_0]
        v2=[x_2-x_1,y_2-y_1]
        v1_norm=np.linalg.norm(v1)
        v2_norm=np.linalg.norm(v2)
        if v1_norm==0 :
            remove_index[(i+1)%n]=False
        x_0=x_1
        y_0=y_1
        x_1=x_2
        y_1=y_2
    #print(remove_index)
    x=[xv for xv, ri in zip(x, remove_index) if ri]
    y=[yv for yv, ri in zip(y, remove_index) if ri]
    
    return x,y




def get_fov(x,y,cam_x,cam_y):
    """ Returns the field of view  (FOV) from camera position towards  
        corresponding line segments.
    """
    x_v=np.append(np.array(x),x[0])
    y_v=np.append(np.array(y),y[0])
    ca=np.vstack([cam_x-x_v[0:-2],cam_y-y_v[0:-2]]).T
    cb=np.vstack([cam_x-x_v[1:-1],cam_y-y_v[1:-1]]).T
    print(ca-cb)
    fov=[]
    for v1,v2 in zip(ca,cb):
        fov.append(np.degrees(angle_between(v1,v2)))

    return fov


def edge_partition(x,y,n=5):
    """ Returns the points on an edge to partition it in n equal segments 
       
    """
    x_v=np.append(np.array(x),x[0])
    y_v=np.append(np.array(y),y[0])

    ref_x=np.zeros((len(x),n))
    ref_y=np.zeros((len(x),n))
    
    for i in range(len(x)):
        ref_x[i]=np.linspace(x_v[i],x_v[i+1],num=n)
        ref_y[i]=np.linspace(y_v[i],y_v[i+1],num=n)


    return ref_x[:,1:-1],ref_y[:,1:-1]




def intersects(p1, p2, p3, p4):
    """"
    Returns if the line segments p1p2 and p3p4 intersect and their intersection point
    """
    IntersectionPoint=[None,None]
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return IntersectionPoint,False
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return IntersectionPoint,False
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return IntersectionPoint,False
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    IntersectionPoint=[x,y]
    #print((x,y,ua,ub))
    return IntersectionPoint,True



def LineOfsight(pointA,pointB, vertices):
    """"
    Function to check if the line of sight between to Points A and B is obstructed by a polygon
    Since all points of interest are either outside of the polygon or worst case scenario on its edges 
    We can check if LoS is obstructed by checking how many times the segment AB intersects with the polygon vertices
    """
    intersectsNum: int = 0
    #vertices=np.array(vertices)
    xA,yA=pointA
    xB,yB=pointB

    AB=[xB-xA,yB-yA]
    ab_norm=np.linalg.norm(AB)
    unitAB=unit_vector(AB)
    xB+=0.01*ab_norm*unitAB[0]
    yB+=0.01*ab_norm*unitAB[1]
    pointB=[xB,yB]
    vertices.append(vertices[0])
    for v1,v2 in zip(vertices[0:-2],vertices[1:-1]):
        intersectionPoint,intersectionBool =intersects(pointA,pointB,v1,v2)
        if intersectionBool: #and not(intersectionPoint==pointA or intersectionPoint==pointB ) :
            intersectsNum=intersectsNum+1
            #print(intersectsNum,end='\t')
    
    if intersectsNum>1:
        return False
    else:
        return True
   
#To be removed as verticality of north is a safe assumption in our usecase
def northUnitVector(x,y):
    northX=500000
    northY=9997964.9429

    north_v=np.vstack([northX-x,northY-y]).T
    north_uv=unit_vector(north_v[:])

    return north_uv


if __name__=="__main__" :
    print(f'{ __file__} contains Building Footprint utility functions')