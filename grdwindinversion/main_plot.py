import xarray as xr

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt 

import os,sys,re
sys.path.append("/home1/datawork/vlheureu/tools/")
from palette import *
palette = '/home1/datawork/vlheureu/tools/high_wind_speed.pal'
cwnd = getColorMap(rgbFile = palette)

root_path = "/home/datawork-cersat-public/cache/public/ftp/project/sarwing/processings/c39e79a/default"


def work(input_file):
    print(input_file)
    if os.path.exists(input_file) == False:
        return;
    nc_new = xr.open_dataset(input_file)
    basename = os.path.basename(os.path.dirname(input_file))

    compare_plot = True
    if ('RS2' in input_file):
        list_ = glob.glob(os.path.join(root_path, "RS2","L1","*","*","*","RS2*","rs2*.nc"))
    elif ('S1A' in input_file):
        list_ = glob.glob(os.path.join(root_path, "sentinel-1a","L1","*","*","*","*","*","s1*.nc"))
    elif ('S1B' in input_file):
        list_ = glob.glob(os.path.join(root_path, "sentinel-1b","L1","*","*","*","*","*","s1*.nc"))
    else : 
        compare_plot = False
    list_ = [f for f in list_ if not (f.endswith('filter.nc') or f.endswith('winddir.nc'))]


    # plot 1 
    fig, axs = plt.subplots(nrows=1,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(22,7))

    vars__ = ["owiWindSpeed_co","owiWindSpeed_cross","owiWindSpeed"]
    titles__ = ["wind speed VV","wind speed VH","wind speed VV+VH"]

    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    for idx,ax in enumerate(axs) :

        im = ax.pcolormesh(nc_new.owiLon,nc_new.owiLat,nc_new[vars__[idx]],shading='auto' ,vmin = 0, vmax = 80,cmap=cwnd)
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1+0.01, pos.y0, 0.02, pos.height])

        # Ajoutez la colorbar à l'axe "fantôme"
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("wind speed [m/s]")

        ax.coastlines('10m')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(titles__[idx])
        #ax.grid()
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    fig.savefig(input_file.replace('.nc','.jpeg'))
    print("OK 1")

    if compare_plot:
        file_sarwing=""
        searched_file = basename
        for files in list_:
            if searched_file in files:
                file_sarwing=files
        if (file_sarwing == ""):
            print("ERROR cant file sarwing file for " + input_file)
            return
        else:
            print(file_sarwing)
            nc_sarwing = xr.open_dataset(file_sarwing)
            #nc_sarwing = nc_sarwing.rename({"owiAzSize":"line","owiRaSize":"sample"})

            import numpy as np
            new_max = np.nanmax(nc_new.owiWindSpeed)
            old_max = np.nanmax(nc_sarwing.owiWindSpeed)
            diff = new_max - old_max

            z__ = [nc_new.owiWindSpeed,nc_sarwing.owiWindSpeed,nc_new.owiWindSpeed-nc_sarwing.owiWindSpeed]
            titles__ = ["new wind speed VV+VH","sarwing wind speed VV+VH","diff"]
            cmaps__ = [cwnd,cwnd,"jet"]
            vmaxs__ = [80,80,+np.abs(diff)]
            vmins__ = [0,0,-np.abs(diff)]
            cbars__ = ["wind speed [m/s]","wind speed [m/s]","wind speed difference [m/s]"]

            if np.abs(diff) < 1e-5:
                diff = 5

            fig, axs = plt.subplots(nrows=1,ncols=3,
                                    subplot_kw={'projection': ccrs.PlateCarree()},
                                    figsize=(22,7))
            plt.subplots_adjust(wspace=0.4, hspace=0.3)

            for idx,ax in enumerate(axs) :
                im = ax.pcolormesh(nc_new.owiLon,nc_new.owiLat,z__[idx],shading='auto' ,vmin = vmins__[idx], vmax = vmaxs__[idx]
                                   ,cmap=cmaps__[idx])

                pos = ax.get_position()
                cbar_ax = fig.add_axes([pos.x1+0.01, pos.y0, 0.02, pos.height])

                # Ajoutez la colorbar à l'axe "fantôme"
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(cbars__[idx])


                ax.coastlines('10m')
                #ax.grid()
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(titles__[idx])

                gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                gl.top_labels = False
                gl.right_labels = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            fig.savefig(input_file.replace('.nc','compare_.jpeg'))

            print("OK 2")
            
            
if __name__ == '__main__':      
    import argparse, os
    from pathlib import Path
    import glob
    
    parser = argparse.ArgumentParser(description = 'plotting new L2 sw')
    parser.add_argument('--input_file',help='input file path')

    args = parser.parse_args()
    input_file = args.input_file
    
    work(input_file)
        
 