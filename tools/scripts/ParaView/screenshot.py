#### import the simple module from the paraview
from paraview.simple import *

from optparse import OptionParser
import os.path

parser = OptionParser()
parser.add_option("--state-file", type="string", default="",dest="stateFile", help="paraview state file (.pvsm)")
parser.add_option("--data-dir", type="string", default="",dest="dataDir", help="data directory")
parser.add_option("--output-file", type="string", default="",dest="outputFile", help="outupt file")
(options, args) = parser.parse_args()

stateFile=options.stateFile;
dataDir=options.dataDir;
outputFile=options.outputFile;
if stateFile == "" :
    print "ERROR : option --state-file is required"
    sys.exit(1);
if not os.path.isfile(stateFile):
    print "ERROR : file "+stateFile+" does not exist"
    sys.exit(1);
if dataDir == "" :
    print "ERROR : option --data-dir is required"
    sys.exit(1);
if not os.path.isdir(dataDir):
    print "ERROR : directory "+dataDir+" does not exist"
    sys.exit(1);
if outputFile == "" :
    print "ERROR : option --output-file is required"
    sys.exit(1);

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2150, 1176]

# destroy renderView1
Delete(renderView1)
del renderView1

# load state
print "LoadState start ...";
LoadState(stateFile, #LoadStateDataFileOptions='Use File Names From State',
          LoadStateDataFileOptions='Search files under specified directory',
          #DataDirectory='/feel/',
          DataDirectory=dataDir,
          #OnlyUseFilesInDataDirectory=1,
          #ExportcaseCaseFileName='/feel/applications/models/solid/cantilever/P1G1/np_1/solid.exports/Export.case'
)
print "LoadState done";


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2150, 1176]
# print "viewsize="+str(renderView1.ViewSize);

# renderView1.UseOffscreenRendering = 1

# save screenshot
print "SaveScreenshot start ...";
SaveScreenshot(outputFile, renderView1,
               #ImageResolution=[2150, 1176],
               #ImageResolution=[renderView1.ViewSize[0], renderView1.ViewSize[1]],
               magnification=1,
               FontScaling='Scale fonts proportionally',
               OverrideColorPalette='',
               StereoMode='No change',
               TransparentBackground=0,
               ImageQuality=100)
print "SaveScreenshot done";
