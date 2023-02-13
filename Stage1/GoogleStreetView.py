#Version 4 of script to scrape Google Street View for 360 panoramas


import requests
import itertools
import shutil
import time
import cv2 
import numpy as np
#from PIL import Image
from io import BytesIO
from os import getcwd,mkdir


def removeBlacktiles(img):
    """ Returns an image after removing its black borders.

        Black borders are removed by finding the biggest contour in the
        grey-scale converted panorama.
    """
    # Convert RGB to BGR 
    #img = img[:, :, ::-1].copy() 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    #find contours in it. There will be only one object, so find bounding rectangle for it.
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)

    #crop the image
    crop = img[y:y+h,x:x+w]
    #cv2.imshow('crop',crop)

    return crop
    

class Panorama:
    def __init__(self,latitude,longitude,api_key):
        
        self.lat=latitude
        self.lng=longitude
        self._api_key=api_key
        self.status ,self.panoid , self.cam_lat , self.cam_lng=self.get_metadata()
        self.img=np.zeros(())
        
        
        
    
    def get_metadataUrl(self):
        """Returns the URL to request Metadata for the panorama """
    
        url="https://maps.googleapis.com/maps/api/streetview/metadata?location={},{}&key={}"
        return url.format(self.lat,self.lng,self._api_key)
    
    def _pano_metadata_request(self):
    
        url=self.get_metadataUrl()
        return requests.get(url,proxies=None)
    
    def get_metadata(self):
        """Returns metadata for the  requested Panorama """

        resp=self._pano_metadata_request()
        if resp.status_code==200:
            status=resp.json().get("status")
            if resp.json().get("status")=='OK':
                
                panoid=resp.json().get("pano_id")
                cam_lat=resp.json().get("location").get("lat")
                cam_lng=resp.json().get("location").get("lng")
            else:
                panoid="None"
                cam_lat="None"
                cam_lng="None"
        else:
            print("gsv3.py\nResponse Return Error Code: "+ str(resp.status_code))
        del resp
        return status, panoid, cam_lat ,cam_lng

    def _tiles_info(self,zoom=4,nbt=0,fover=2):
        """
        Generate a list of a panorama's tiles and their position.
        The format is (x, y, filename, fileurl)
        """
        #Old url : image_url = 'http://maps.google.com/cbk?output=tile&panoid={}&zoom={}&x={}&y={}'
        image_url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={}&x={}&y={}&zoom={}&nbt={}&fover={}"

        # The tiles positions
        coord = list(itertools.product(range(2**zoom), range(2**(zoom-1))))
    
        tiles = [(x, y, "%s_%dx%d.jpg" % (self.panoid, x, y), image_url.format(self.panoid,x, y, zoom, nbt, fover)) for x, y in coord]
        return tiles

    def download_panorama(self,zoom=4):
        """Returns the requested Panoramic image  """

        #Size of each tile that makes the panorama (subject to change)
        if zoom==5:
            tile_width = 256
            tile_height = 256
        else:
            tile_width = 512
            tile_height = 512

        # https://developers.google.com/maps/documentation/javascript/streetview#CreatingPanoramas
        img_w, img_h = 512*(2**zoom), 512*( 2**(zoom-1) )
        panorama= np.zeros(shape=[img_h, img_w, 3], dtype=np.uint8)

        tiles=self._tiles_info(zoom=zoom)
        valid_tiles=[]
        for x,y,fname,url in tiles:
            if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
                # Try to download the image file
                while True:
                    try:
                        #print(url)
                        response = requests.get(url, stream=True)
                        print(f'ID: {self.panoid} | Tile x: {x+1}/{2**zoom} y:{y+1}/{2**(zoom-1)} |', end='\r')
                        break
                    except requests.ConnectionError:
                        print("Connection error. Trying again in 2 seconds.")
                        time.sleep(2)
                #print(response.json())
                #img=Image.open(BytesIO(response.content))
                #panorama.paste(im=img,box=(x*tile_width, y*tile_height))
                img=cv2.imdecode(np.frombuffer(BytesIO(response.content).read(), np.uint8), 1)
                try:
                    panorama[y*img.shape[1]:(y+1)*img.shape[1],x*img.shape[0]:(x+1)*img.shape[0],:]=img
                except:
                    print("Stitching error. Trying again.")
                del response
        panorama=removeBlacktiles(panorama)
        self.img=panorama
        return panorama
        

    def _static_url(self,img_h=600, img_w=600,heading=0, pitch=0 ,fov=120,radius=50):
        """Returns the URL to request a non-panoramic image from the official Google Street View API"""

        parameters=dict.fromkeys(["pano","size","heading","fov","pitch","radius","return_error_code"])
    
        parameters["pano"]=str(self.panoid)
        parameters["size"]=str(img_w)+"x"+str(img_h)
        parameters["heading"]=str(heading)
        parameters["fov"]=str(fov)
        parameters["pitch"]=str(pitch)
        parameters["radius"]=str(radius)
        parameters["return_error_code"]="true"

        url="https://maps.googleapis.com/maps/api/streetview?"
        for kw in parameters.keys():
            url= url + kw +"="+parameters[kw]+"&"
        url= url +"key="+self._api_key
        
        return url

    def streetview_flat(self,img_height=400, img_width=600,heading=0, pitch=0 ,fov=90,radius=50):
        """Returns a non-panoramic image from the official Google Street View API"""
        
        img=np.zeros((img_width,img_height))
        url=self._static_url(img_h=img_height,img_w=img_width,heading=heading,pitch=pitch,fov=fov,radius=radius)
        while True:
            try:
                #print(url)
                response = requests.get(url, stream=True)
                #print(response.content)
                break
            except requests.ConnectionError:
                print("Connection error. Trying again in 2 seconds.")
                time.sleep(2)
        img=cv2.imdecode(np.frombuffer(BytesIO(response.content).read(), np.uint8), 1)
        del response
        #self.img=img
        return img

    
    def streetview_flat_ws(self,img_height=768, img_width=1024,heading=0, pitch=0 ,fov=120):
        """Returns a non-panoramic image by web-scraping on the Google Street View site"""
        
        img=np.zeros((img_width,img_height))
        img_url="https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={}&cb_client=search.revgeo_and_fetch.gps&w={}&h={}&yaw={}&pitch={}&thumbfov={}"
        img_url=img_url.format(self.panoid,img_width,img_height,heading,pitch,fov)
        print(img_url)
        while True:
            try:
                print(img_url)
                response = requests.get(img_url, stream=True)
                #print(response.content)
                break
            except requests.ConnectionError:
                print("Connection error. Trying again in 2 seconds.")
                time.sleep(2)
        img=cv2.imdecode(np.frombuffer(BytesIO(response.content).read(), np.uint8), 1)
        del response
        self.img=img
        return img

    def save(self,directory,fname=None,extension='jpg', rmvbt=True):
        """Writes Panorama to file
        
            TO DO:
            Embed and save metadata such as north orientation as EXIF within the image.
        """
       
        #self.img=removeBlacktiles(self.img)
        
        if not fname:
            fname = "pano_%s" % (self.panoid)
        else:
            fname , ext =fname.split(".",1) 
        image_format = extension if extension != 'jpg' else 'jpeg'    
        try:
                filename="%s/%s.%s" % (directory,fname, extension)
                cv2.imwrite(filename,self.img)      
        except:
                print("Image not saved")

    def getNorth(self):
        """ Returns the x position of north within the image.
            [0,2pi) <--> [0,image width)
        """

        if self.status!="OK":
            print('getNorth(): Error: Panorama not found')
            return None
    
        panoImg=self.img
        northHeadingTile=self.streetview_flat(img_height=400, img_width=600,heading=0, pitch=0 ,fov=30)

        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(panoImg,None)
        kp2, des2 = orb.detectAndCompute(northHeadingTile,None)

        brf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = brf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(panoImg,kp1,northHeadingTile,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        list_kp1 = []
        list_kp2 = []

        # For each match...
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

        true_north=np.mean(np.array(list_kp1), axis=0)
        x_north=true_north[0]
        #print(x_north)
        return x_north




if __name__=="__main__" :
    print(f'{ __file__} :\n Module to retrieve flat and panoramic images from Google Street View')