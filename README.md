# FloodMAGNet Workflow

## Data Collection
### Study Area
- Select area; delineate watershed boundary using USGS StreamStats (or equivalent, e.g. HEC-HMS)
https://streamstats.usgs.gov/ss/
- Obtain hydrologic geospatial datasets:
    - Land Cover Map (USGS NLCD):
    https://www.usgs.gov/centers/eros/science/national-land-cover-database
    - Soils Hydraulic Group (WebSoilSurvey):
    https://websoilsurvey.nrcs.usda.gov/app/
    - Parameters for Manning's Roughness Coefficient (HEC-RAS Guide):
    https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.6/developing-a-terrain-model-and-geospatial-layers/creating-land-cover-mannings-n-values-and-impervious-layers
    - Parameters for % Impervious - may need calibration to match expected flood behaviors:
    https://rashms.com/blog/import-nlcd-impervious-raster-file-as-hec-ras-land-cover-layer-for-percent-impervious-values/
    - Parameters for infiltration, dependent on method selected. SCS method was used for the study. For New Jersey, the data was retrieved from Resilient NJ Floodplain Mapping Methodology.
    - Digital Elevation Model (DEM, USGS National Map):
    https://apps.nationalmap.gov/downloader/
    - Alternatively, download terrain data directly from HEC-RAS 2D
- Extract the following from the HEC-RAS 2D model for FloodMAGNet training:
    - Infiltration raster.
    - Imperviousness % raster.
    - Manning's n raster
    - Computation Points
    - Cells
- Additional data requirements:
    - Streams data (National Hydrology Dataset):
    https://www.usgs.gov/national-hydrography/national-hydrography-dataset
    - Parcels MOD IV - this is dependant on area of interest, for New Jersey:
    https://njogis-newjersey.opendata.arcgis.com/documents/newjersey::parcels-and-mod-iv-composite-of-nj-download/about
- Optionally, match naming convention within Graph Construction and Model Training and Evaluation scripts.

__All spatial datasets were projected to a common projected coordinate system (e.g., EPSG:3424 for New Jersey).__

### Rainfall
- Rain on grid was used for the study: uniform rainfall across the watershed.
- For New Jersey, observed/historical data was acquired from (nearest gauge):
https://njdep.rutgers.edu/rainfall/
- Design storms were generated from:
https://hdsc.nws.noaa.gov/pfds/
    - Select state and city, the corresponding table for rainfall cumulative values can be used to construct time-series data.
    - Default durations are 24-hours, this study utilized 6-hour storms. 
    - Physical simulations do not require consistent temporal resolution; however, the data-driven model does. To account for this, the higher temporal resolution data (design storms) was downsampled to achieve uniform resolution.
    - Synthetic storms were constructed by combining a smaller (2Y) storm sequence with a larger one (100Y) in both orders (2 synthetic storms).

### Simulation
- Import watershed as a geometry perimiter to create a computational mesh (dx dy were set as 80ft x 80ft, use lower values for higher model resolution).
- Import DEM data as terrain.
- Import Soils and Land Cover as map layers, optionally add a visual layer as well (Google Maps Hybrid preferred).
- Generate an Infiltration layer from Soils and Land Cover
- Fill Infiltration and Land Cover parameters per corresponding documents/reports.
- Refine cells along streams to avoid incorrect terrain assumptions - Streams shapefile can be used to streamline process.
- Optionally, model bridges where streams are incorrecly represented below roads in the terrain data.
- Define simulation parameters
    - Unsteady flow data: precipitation across the watershed
    - Unsteady flow analysis: computational step of 1s was used (optional but preferred); SWE-ELM numerical method.
- Run all necessary simulations for data collection and extract maximum water depth rasters (used as target for FloodMAGNet).

## Model Training and Testing

__Download libraries noted in *requirements.txt*__

### Graph Construction
- Compile and organize data for each watershed to be evaluated.
- Update directories within the Graph Construction script.
- Update high-occupancy parameters to match needs and area of interest. For New Jersey, MOD-IV parameters can be found here: https://www.nj.gov/treasury/taxation/pdf/lpt/modIVmanual.pdf
- Run the Graph Construction script for each watershed, ensure saved artifacts are named appropriately.
- Modify the KNN distance threshold if using a metric-based projection system. The evaluated CRS is measured in feet.

### Model Training and Testing
- Similarly update directories within the script. Rainfall data must have uniform temporal resolution.
- Update training and testing rainfall scenarios. It is preferred to keep synthetic storms (2Y-100Y, 100Y-2Y) within training subset as per the study findings.
- Update the *unit* variable: the code predefined the conversion for meters in loss calculation - additional logic statements are necessary for different depth units.
- The model should now be ready for training and evaluation. IWMSE hyperparameters may need watershed-specific tuning.
