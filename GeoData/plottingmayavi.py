#!/usr/bin/env python
"""

"""

from __future__ import division, absolute_import
import logging
import numpy as np
import scipy as sp
import scipy.interpolate as spinterp
import time
import datetime as dt

import matplotlib.pyplot as plt
try:
    from mayavi import mlab
except ImportError:
    mlab = None
except (ValueError,RuntimeError) as e:
    mlab = None
    print('Mayavi not imported due to {}'.format(e))



try:
    plt.get_cmap('viridis')
    defmap3d = 'viridis'
except ValueError:
    defmap3d = 'jet'
    #%%
def plot3Dslice(geodata, surfs, vbounds, titlestr='', time=0, gkey=None, cmap=defmap3d,
                ax=None, fig=None, method='linear', fill_value=np.nan, view=None, units='',
                colorbar=False, outimage=False):
    """
    This function create 3-D slice image given either a surface or list of
    coordinates to slice through.
    Inputs:
    geodata - A geodata object that will be plotted in 3D
    surfs - This is a three element list. Each element can either be
        altlist - A list of the altitudes that RISR parameter slices will be taken at
        xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the
                data will be interpolated over.
                ie, xyvecs=[np.linspace(-100.0,500.0), np.linspace(0.0,600.0)]
    vbounds = a list of bounds for the geodata objec's parameters. ie, vbounds=[500,2000]
    title - A string that holds for the overall image
    ax - A handle for an axis that this will be plotted on.

    Returns a mayavi image with a surface
    """

    if mlab is None:
        print('mayavi was not successfully imported')
        return

    assert geodata.coordnames.lower() == 'cartesian'

    datalocs = geodata.dataloc

    xvec = sp.unique(datalocs[:, 0])
    yvec = sp.unique(datalocs[:, 1])
    zvec = sp.unique(datalocs[:, 2])

    assert len(xvec)*len(yvec)*len(zvec) == datalocs.shape[0]

    #determine if the ordering is fortran or c style ordering
    diffcoord = sp.diff(datalocs, axis=0)

    if diffcoord[0, 1] != 0.0:
        ar_ord = 'f'
    elif diffcoord[0, 2] != 0.0:
        ar_ord = 'c'
    elif diffcoord[0, 0] != 0.0:
        if len(np.where(diffcoord[:, 1])[0]) == 0:
            ar_ord = 'f'
        elif len(np.where(diffcoord[:, 2])[0]) == 0:
            ar_ord = 'c'

    matshape = (len(yvec), len(xvec), len(zvec))
    # reshape the arrays into a matricies for plotting
    x, y, z = [sp.reshape(datalocs[:, idim], matshape, order=ar_ord) for idim in range(3)]

    if gkey is None:
        gkey = geodata.datanames()[0]
    porig = geodata.data[gkey][:, time]

    mlab.figure(fig)
    #determine if list of slices or surfaces are given

    islists = isinstance(surfs[0], list)
    if isinstance(surfs[0], np.ndarray):
        onedim = surfs[0].ndim == 1
    #get slices for each dimension out
    surflist = []
    if islists or onedim:
        p = np.reshape(porig, matshape, order=ar_ord)
        xslices = surfs[0]
        for isur in xslices:
            indx = sp.argmin(sp.absolute(isur-xvec))
            xtmp = x[:, indx]
            ytmp = y[:, indx]
            ztmp = z[:, indx]
            ptmp = p[:, indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)] = True
            surflist.append(mlab.mesh(xtmp, ytmp, ztmp, scalars=ptmp, vmin=vbounds[0],
                                      vmax=vbounds[1], colormap=cmap, mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0

        yslices = surfs[1]
        for isur in yslices:
            indx = sp.argmin(sp.absolute(isur-yvec))
            xtmp = x[indx]
            ytmp = y[indx]
            ztmp = z[indx]
            ptmp = p[indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)] = True
            surflist.append(mlab.mesh(xtmp, ytmp, ztmp, scalars=ptmp, vmin=vbounds[0],
                                      vmax=vbounds[1], colormap=cmap, mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
        zslices = surfs[2]
        for isur in zslices:
            indx = sp.argmin(sp.absolute(isur-zvec))
            xtmp = x[:, :, indx]
            ytmp = y[:, :, indx]
            ztmp = z[:, :, indx]
            ptmp = p[:, :, indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)] = True
            surflist.append(mlab.mesh(xtmp, ytmp, ztmp, scalars=ptmp, vmin=vbounds[0],
                                      vmax=vbounds[1], colormap=cmap, mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    else:
        # For a general surface.
        xtmp, ytmp, ztmp = surfs[:]
        gooddata = ~np.isnan(porig)
        curparam = porig[gooddata]
        curlocs = datalocs[gooddata]
        new_coords = np.column_stack((xtmp.flatten(), ytmp.flatten(), ztmp.flatten()))
        ptmp = spinterp.griddata(curlocs, curparam, new_coords, method, fill_value)
        pmask = sp.zeros_like(ptmp).astype(bool)
        pmask[sp.isnan(ptmp)] = True
        surflist.append(mlab.mesh(xtmp, ytmp, ztmp, scalars=ptmp, vmin=vbounds[0],
                                  vmax=vbounds[1], colormap=cmap, mask=pmask))
        surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    mlab.title(titlestr, color=(0, 0, 0))
    #mlab.outline(color=(0,0,0))
    mlab.axes(color=(0, 0, 0), x_axis_visibility=True, xlabel='x in km', y_axis_visibility=True,
              ylabel='y in km', z_axis_visibility=True, zlabel='z in km')

    mlab.orientation_axes(xlabel='x in km', ylabel='y in km', zlabel='z in km')

    if view is not None:
        # order of elevation is changed between matplotlib and mayavi
        mlab.view(view[0], view[1])
    if colorbar:
        if units == '':
            titlestr = gkey
        else:
            titlstr = gkey +' in ' +units
        mlab.colorbar(surflist[-1], title=titlstr, orientation='vertical')

    if outimage:
        arr = mlab.screenshot(fig, antialiased=True)
        mlab.close(fig)
        return arr
    else:
        return surflist
