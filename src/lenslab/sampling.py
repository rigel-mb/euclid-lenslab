"""Shared selection checks and exact WCS-based VIS stamp extraction."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

STAMP_SIZE = 96
PIXEL_ARCSEC = 0.1
FLUX_MIN = 3.6307805477
RADIUS_DEG = 0.03
MAG_CALIPER = 0.3
AREA_CALIPER = 0.2
MIN_KNOWN_SEPARATION_ARCSEC = 20.0
COLUMNS = ['object_id', 'right_ascension', 'declination', 'segmentation_area',
           'flux_detection_total', 'fluxerr_detection_total']


def separation_deg(ra, dec, ra0, dec0):
    """Great-circle separation; safe for tiny distances and RA wraparound."""
    a, d, a0, d0 = np.deg2rad([ra, dec, ra0, dec0]) if np.isscalar(ra) else (
        np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra0), np.deg2rad(dec0))
    h = np.sin((d-d0)/2)**2 + np.cos(d)*np.cos(d0)*np.sin((a-a0)/2)**2
    return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def eligible(frame):
    return (frame.vis_det.eq(1) & frame.segmentation_area.ge(300)
            & frame.flux_detection_total.gt(FLUX_MIN)
            & np.isfinite(frame.right_ascension) & np.isfinite(frame.declination))


def add_matching_columns(frame):
    frame = frame.copy()
    frame['mag_total'] = 23.9 - 2.5*np.log10(frame.flux_detection_total)
    frame['log_area'] = np.log10(frame.segmentation_area)
    return frame


def reproject_stamp(path:Path, ra:float, dec:float):
    """Bilinear image sampling onto identical north-up TAN grids, in native units."""
    with fits.open(path, memmap=False) as hdus:
        candidates = [h for h in hdus if h.data is not None and np.asarray(h.data).ndim==2]
        if not candidates:
            raise ValueError('FITS has no 2D image')
        hdu = candidates[0]
        original = np.asarray(hdu.data,dtype=np.float64)
        source_wcs = WCS(hdu.header).celestial
        target = WCS(naxis=2)
        target.wcs.ctype = ['RA---TAN','DEC--TAN']
        target.wcs.crval = [ra,dec]
        target.wcs.crpix = [(STAMP_SIZE+1)/2,(STAMP_SIZE+1)/2]
        target.wcs.cdelt = [-PIXEL_ARCSEC/3600,PIXEL_ARCSEC/3600]
        yy,xx = np.indices((STAMP_SIZE,STAMP_SIZE),dtype=float)
        world = target.pixel_to_world_values(xx,yy)
        xsrc,ysrc = source_wcs.world_to_pixel_values(*world)
        image = map_coordinates(original,[ysrc,xsrc],order=1,mode='constant',cval=np.nan,prefilter=False)
        finite = np.isfinite(image)
        if not finite.all():
            raise ValueError(f'incomplete 9.6arcsec stamp: finite fraction {finite.mean():.6f}')
        if float(np.std(image)) == 0:
            raise ValueError('constant image')
        return image.astype(np.float32)
