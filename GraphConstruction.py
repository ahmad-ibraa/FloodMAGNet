import os, math, json, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.mask
import networkx as nx

from shapely.geometry import Point, MultiPoint
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rasterio.transform import rowcol
from scipy import ndimage
import torch
from torch_geometric.data import Data
from shapely.validation import make_valid

from rasterstats import zonal_stats

## Imports & Paths
## Adjust file names as needed.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")

terrain_path      = os.path.join(data_path, "Terrain.tif")
curvature_path    = os.path.join(data_path, "Curvature.tif")
slope_path        = os.path.join(data_path, "Slope.tif")
mannings_path     = os.path.join(data_path, "Mannings.tif")
impervious_path   = os.path.join(data_path, "Impervious.tif")
infiltration_path = os.path.join(data_path, "Infiltration.tif")
cmp_points_path   = os.path.join(data_path, "ComputationPoints.shp")
cells_path = os.path.join(data_path, "cells.shp")

watershed_path = os.path.join(data_path, "watershed.shp")
streams_path   = os.path.join(data_path, "NHDStreams.shp")
parcels_path   = os.path.join(data_path, "Parcels.shp")
rainfall_path  = os.path.join(data_path, "Rainfall.csv")
projection = "EPSG:3424" # This is for New Jersey, adjust if evaluating for a different area. Ensure all layers are in the same CRS.


watershed = gpd.read_file(watershed_path).to_crs(projection)
city = "Englewood"
# Depth raster naming convention:
# Each storm in DEPTH_STORMS must have a corresponding raster named:
#   f"{storm}_Depth.tif"
DEPTH_STORMS = [
    "2Y6h",
    "5Y6h",
    "10Y6h",
    "25Y6h",
    "50Y6h",
    "100Y6h",
    "200Y6h",
    "500Y6h",
    "1000Y6h",
    "2Y_100Y6h",
    "100Y_2Y6h",
]
TARGET_STORM = DEPTH_STORMS[0] # Placeholder, used to fill data.y in tensor

## -------------------------------------------------------------------- Load Mesh -------------------------------------------------------------------- 
watershed["geometry"] = watershed["geometry"].buffer(0)

cmp = gpd.read_file(cmp_points_path).to_crs(projection)
cmp = gpd.clip(cmp, watershed)

streams = gpd.read_file(streams_path).to_crs(projection)
streams = gpd.clip(streams, watershed)

def flatten_points(gdf):
    out = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if isinstance(geom, Point):
            out.append(geom)
        elif isinstance(geom, MultiPoint):
            out.extend(list(geom.geoms))
        else:
            out.append(geom.representative_point())
    return out

flat_points = flatten_points(cmp)
coords = np.array([[p.x, p.y] for p in flat_points], dtype=float)
print(f"FV nodes: {len(coords)}")

## -------------------------------------------------------------------- KNN + Direction Filter -------------------------------------------------------------------- 
eps = 1e-6
K   = min(10, max(1, len(coords) - 1))
DIST_MAX = 113.2 # The Projection System is in feet, so the distance threshold is also in feet. For reference, 100ft ≈ 30.48m. Adjust if using a different CRS.
TOL_DEG  = 15
TOL_RAD  = np.deg2rad(TOL_DEG)

def circ_diff(a, b):
    d = abs((a - b) % (2*np.pi))
    return min(d, 2*np.pi - d)

edge_w = {}

nn = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree')
nn.fit(coords)
dist, idxs = nn.kneighbors(coords)

## -------------------------------------------------------------------- Parcels -> Risk Weights -------------------------------------------------------------------- 
RISK_CLASSES = {"2", "4A", "15A", "15C"} # These are the PROP_CLASS values that we consider high-importance for the areas studied, and can be adjusted to different exposure priorities.
RISK_MULT    = 5

pts_unique = gpd.GeoDataFrame(
    {"node_id": np.arange(len(coords))},
    geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
    crs=projection,
)

parcels = gpd.read_file(parcels_path).to_crs(pts_unique.crs)

parcels["geometry"]   = parcels["geometry"].apply(make_valid)
watershed["geometry"] = watershed["geometry"].apply(make_valid)

parcels   = parcels[parcels.is_valid].copy()
watershed = watershed[watershed.is_valid].copy()

parcels = gpd.clip(parcels, watershed)
parcels["geometry"] = parcels.buffer(0)

if "PROP_CLASS" not in parcels.columns:
    raise KeyError("Parcels layer missing 'PROP_CLASS' field.")

def _norm_propclass(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s

parcels["_PC"] = parcels["PROP_CLASS"].apply(_norm_propclass)
parcels_sel    = parcels[parcels["_PC"].isin(RISK_CLASSES)].copy()

node_weight = np.ones(len(pts_unique), dtype=np.float32)

if len(parcels_sel):
    sj = gpd.sjoin(
        pts_unique,
        parcels_sel[["geometry", "_PC"]],
        how="left",
        predicate="within",
    )
    in_risk_by_node = (~sj["index_right"].isna()).groupby(sj.index).any()
    in_risk_flag = in_risk_by_node.reindex(pts_unique.index, fill_value=False).to_numpy()
    node_weight[in_risk_flag] = RISK_MULT

print(f"Risk-weighted nodes: {int((node_weight > 1).sum())} / {len(node_weight)} (×{RISK_MULT})")

## -------------------------------------------------------------------- Static Feature Extraction -------------------------------------------------------------------- 
def scale_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        arr = arr[~np.isnan(arr)]
        s = StandardScaler()
        if arr.size > 0:
            s.fit(arr.reshape(-1, 1))
        else:
            s.fit(np.array([[0.0], [1.0]]))
        return s

terrain_scaler      = scale_tiff(terrain_path)
curvature_scaler    = scale_tiff(curvature_path)
slope_scaler        = scale_tiff(slope_path)
mannings_scaler     = scale_tiff(mannings_path)
infiltration_scaler = scale_tiff(infiltration_path)
impervious_scaler   = scale_tiff(impervious_path)

streams = gpd.read_file(streams_path).to_crs(pts_unique.crs)
streams = gpd.clip(streams, watershed)
stream_union = streams.union_all()

dist_to_stream_raw = np.array([g.distance(stream_union) for g in pts_unique.geometry], dtype=float)
dist_scaler = MinMaxScaler()
dist_to_stream_feat = dist_scaler.fit_transform(dist_to_stream_raw.reshape(-1, 1)).ravel()


cells_raw  = gpd.read_file(cells_path).to_crs(pts_unique.crs)
cells = cells_raw[
    cells_raw.geometry.notna()
    & (~cells_raw.geometry.is_empty)
    & (cells_raw.geom_type.isin(["Polygon", "MultiPolygon"]))
].copy()

cells["geometry"] = cells.buffer(0)
cells = gpd.clip(cells, watershed)
cells = cells.reset_index(drop=True)

if len(cells) == 0:
    raise ValueError("No valid polygon cells found after cleaning. Check cells.shp.")

cell_area_ft2 = cells.geometry.area.values

sj_cells      = gpd.sjoin(pts_unique, cells[["geometry"]], how="left", predicate="within")
node2cell     = sj_cells["index_right"].to_numpy()
has_cell      = ~pd.isna(node2cell)

node2cell_int = np.full(len(node2cell), -1, dtype=int)
node2cell_int[has_cell] = node2cell[has_cell].astype(int)

cell_area_node = np.full(len(pts_unique), np.nan, dtype=float)
m = node2cell_int >= 0
cell_area_node[m] = cell_area_ft2[node2cell_int[m]]

area_scaler = StandardScaler()
cell_area_node_scaled = area_scaler.fit_transform(
    np.nan_to_num(cell_area_node, nan=np.nanmedian(cell_area_node)).reshape(-1, 1)
).ravel()



def zonal_means_per_cell(tiff_path, cells_gdf, all_touched=True):
    # This reads the raster in chunks and is heavily optimized
    stats = zonal_stats(
        cells_gdf, 
        tiff_path, 
        stats="mean", 
        all_touched=all_touched
    )
    
    # Extract the 'mean' from the dictionary returned by rasterstats
    means = np.array([s['mean'] if s['mean'] is not None else np.nan for s in stats])
    return means

def point_sample_feature(tiff_path, pts_xy):
    with rasterio.open(tiff_path) as src:
        pts_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pts_xy[:, 0], pts_xy[:, 1]), crs=projection
        ).to_crs(src.crs)

        xy_r = np.column_stack([pts_gdf.geometry.x, pts_gdf.geometry.y])

        band = src.read(1).astype(np.float64)
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)

        nan_mask = np.isnan(band)
        if np.any(nan_mask):
            idx  = ndimage.distance_transform_edt(
                nan_mask, return_distances=False, return_indices=True
            )
            band = band[tuple(idx)]

        rows, cols = rowcol(src.transform, xy_r[:, 0], xy_r[:, 1], op=np.rint)
        rows = np.clip(rows, 0, src.height - 1).astype(int)
        cols = np.clip(cols, 0, src.width - 1).astype(int)

        vals = band[rows, cols]
    return vals

def node_feature_from_cell_mean(tiff_path, node2cell_idx, cells_gdf, coords_xy):
    cell_means = zonal_means_per_cell(tiff_path, cells_gdf, all_touched=True)
    feat = np.full(len(coords_xy), np.nan, dtype=float)
    m = node2cell_idx >= 0
    feat[m] = cell_means[node2cell_idx[m]]
    if (~m).any():
        feat[~m] = point_sample_feature(tiff_path, coords_xy[~m])
    return feat

terrain_raw_node = node_feature_from_cell_mean(terrain_path, node2cell_int, cells, coords)

## -------------------------------------------------------------------- Build Node Attributes -------------------------------------------------------------------- 
raw_feats = {
    "terrain":      node_feature_from_cell_mean(terrain_path,      node2cell_int, cells, coords),
    "curvature":    node_feature_from_cell_mean(curvature_path,    node2cell_int, cells, coords),
    "slope":        node_feature_from_cell_mean(slope_path,        node2cell_int, cells, coords),
    "mannings":     node_feature_from_cell_mean(mannings_path,     node2cell_int, cells, coords),
    "infiltration": node_feature_from_cell_mean(infiltration_path, node2cell_int, cells, coords),
    "impervious":   node_feature_from_cell_mean(impervious_path,   node2cell_int, cells, coords),
}

node_attrs = {i: {} for i in range(len(coords))}

for k, v in raw_feats.items():
    arr    = np.asarray(v, dtype=float)
    finite = np.isfinite(arr)
    scaler = StandardScaler()
    if finite.any():
        arr_scaled = np.full_like(arr, np.nan, dtype=float)
        arr_scaled[finite] = scaler.fit_transform(arr[finite].reshape(-1,1)).ravel()
    else:
        arr_scaled = arr
    raw_feats[k] = arr_scaled

coord_scaler  = StandardScaler()
coords_scaled = coord_scaler.fit_transform(coords)

for i in range(len(coords)):
    node_attrs[i]["x_coord"]        = coords_scaled[i, 0]
    node_attrs[i]["y_coord"]        = coords_scaled[i, 1]
    node_attrs[i]["dist_to_stream"] = float(dist_to_stream_feat[i])
    node_attrs[i]["cell_area"]      = float(cell_area_node_scaled[i])

for i in range(len(coords)):
    for k, arr in raw_feats.items():
        node_attrs[i][k] = float(arr[i])

for i in range(len(coords)):
    node_attrs[i]["terrain_raw"] = float(terrain_raw_node[i])

## -------------------------------------------------------------------- Build Directed Edges -------------------------------------------------------------------- 
def add_edge_attr(i, j, d):
    if i == j:
        return
    w_inv = 1.0 / (d + eps)
    slope_raw = float((terrain_raw_node[j] - terrain_raw_node[i]) / (terrain_raw_node[i] + eps))
    dx, dy = coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]
    theta = math.atan2(dy, dx)
    if theta < 0:
        theta += 2*np.pi
    edge_w[(i, j)] = np.array([w_inv, slope_raw, np.sin(theta), np.cos(theta)], dtype=float)

for i in range(len(coords)):
    angles_i = []
    for k in range(1, idxs.shape[1]):
        j = int(idxs[i, k])
        d_ij = float(dist[i, k])
        if d_ij > DIST_MAX:
            continue
        theta = math.atan2(coords[j, 1] - coords[i, 1],
                           coords[j, 0] - coords[i, 0])
        if theta < 0:
            theta += 2*np.pi
        if all(circ_diff(theta, a) > TOL_RAD for a in angles_i):
            angles_i.append(theta)
            add_edge_attr(i, j, d_ij)

edges     = np.array(list(edge_w.keys()), dtype=int)
edge_attr = np.vstack([edge_w[e] for e in edge_w])
print(f"Built {len(edges)} directed edges with K={K} and tol={TOL_DEG}°")

## -------------------------------------------------------------------- Build NetworkX Graph -------------------------------------------------------------------- 
G = nx.DiGraph()
G.add_nodes_from(node_attrs.keys())
nx.set_node_attributes(G, node_attrs)
G.add_edges_from(edges)

attr_dict = {
    (u, v): {
        "weight":     edge_w[(u, v)].tolist(),
        "w_inv_dist": float(edge_w[(u, v)][0]),
        "slope":      float(edge_w[(u, v)][1]),
        "theta_rad":  float(edge_w[(u, v)][2]),
        "theta_deg":  float(np.degrees(edge_w[(u, v)][2])),
    }
    for (u, v) in edge_w.keys()
}
nx.set_edge_attributes(G, attr_dict)

for i, wv in enumerate(node_weight):
    G.nodes[i]["risk_weight"] = float(wv)

print(f"Created DIRECTED graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

## -------------------------------------------------------------------- Depth Targets -------------------------------------------------------------------- 
rainfall_df = pd.read_csv(rainfall_path)
depth_paths = {s: os.path.join(data_path, f"{s}_Depth.tif") for s in DEPTH_STORMS}

def zonal_means_per_cell_depth(tiff_path, cells_gdf, all_touched=True):
    means = np.full(len(cells_gdf), np.nan, dtype=float)
    with rasterio.open(tiff_path) as src:
        cells_r = cells_gdf.to_crs(src.crs)
        nodata  = src.nodata
        for i, geom in enumerate(cells_r.geometry):
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                out_img, _ = rasterio.mask.mask(
                    src, [geom], crop=True, filled=False, all_touched=all_touched
                )
            except Exception:
                continue
            arr = out_img[0].astype(float)
            total_pixels = arr.size
            if total_pixels == 0:
                continue
            if hasattr(arr, "mask"):
                valid    = ~arr.mask
                arr_data = np.where(valid, arr.data, 0.0)
            else:
                arr_data = np.where(np.isfinite(arr), arr, 0.0)
            if nodata is not None:
                arr_data = np.where(arr_data == nodata, 0.0, arr_data)
            total_sum = np.nansum(arr_data)
            means[i]  = total_sum / total_pixels
    return means

def node_avg_depth_for_storm(depth_tif_path, node2cell_idx, cells_gdf, coords_xy):
    cell_means = zonal_means_per_cell_depth(depth_tif_path, cells_gdf, all_touched=True)
    feat = np.full(len(coords_xy), np.nan, dtype=float)
    m = node2cell_idx >= 0
    feat[m] = cell_means[node2cell_idx[m]]
    if (~m).any():
        feat[~m] = point_sample_feature(depth_tif_path, coords_xy[~m])
    return feat

depth_by_storm = {}
for storm, path in depth_paths.items():
    depth_cell_mean = node_avg_depth_for_storm(path, node2cell_int, cells, coords)
    depth_by_storm[storm] = np.asarray(depth_cell_mean, dtype=float)

target_vec_scaled = depth_by_storm[TARGET_STORM]

for i, v in enumerate(target_vec_scaled):
    node_attrs[i]["target_depth"] = float(v)

nx.set_node_attributes(G, {i: node_attrs[i]["target_depth"] for i in node_attrs}, "target_depth")

FEAT_KEYS = [
    "terrain","curvature","slope","mannings",
    "infiltration","impervious",
    "x_coord","y_coord","dist_to_stream","cell_area"
]
TARGET_KEY = "target_depth"

## -------------------------------------------------------------------- PyG Tensors -------------------------------------------------------------------- 
N = G.number_of_nodes()
X = np.array([[G.nodes[i][k] for k in FEAT_KEYS] for i in range(N)], dtype=float)
y = np.array([G.nodes[i][TARGET_KEY] for i in range(N)], dtype=float)

x          = torch.tensor(X, dtype=torch.float32)
y          = torch.tensor(y, dtype=torch.float32).view(-1, 1)
edge_index = torch.tensor(edges.T, dtype=torch.long)
edge_attr  = torch.tensor(edge_attr, dtype=torch.float32)
node_w_t   = torch.tensor(node_weight, dtype=torch.float32)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
    node_weight=node_w_t
)

print(data)

storm_cols = rainfall_df.columns.tolist()
rain_series = {
    c: torch.tensor(rainfall_df[c].values, dtype=torch.float32).view(-1, 1)
    for c in storm_cols
}

## -------------------------------------------------------------------- Save Artifacts -------------------------------------------------------------------- 
save_dir = os.path.join(data_path, f"artifacts_{city}")
os.makedirs(save_dir, exist_ok=True)

pyg_tensors = {
    "x":           data.x,
    "edge_index":  data.edge_index,
    "edge_attr":   getattr(data, "edge_attr", None),
    "y":           getattr(data, "y", None),
    "node_weight": getattr(data, "node_weight", None),
    "pos":         getattr(data, "pos", None),
}

torch.save(pyg_tensors, os.path.join(save_dir, "pyg_tensors.pt"))

nx_path = os.path.join(save_dir, "nx_graph.pkl")
with open(nx_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save(os.path.join(save_dir, "coords.npy"), coords)
np.save(os.path.join(save_dir, "node_weight.npy"), node_weight)
np.savez_compressed(os.path.join(save_dir, "depth.npz"), **depth_by_storm)

meta = {
    "feat_keys":     FEAT_KEYS,
    "target_key":    TARGET_KEY,
    "storm_cols":    storm_cols,
    "crs":           projection,
    "depth_storms":  DEPTH_STORMS,
    "target_storm":  TARGET_STORM,
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

try:
    import joblib
    joblib.dump(dist_scaler,  os.path.join(save_dir, "dist_scaler.joblib"))
    joblib.dump(area_scaler,  os.path.join(save_dir, "area_scaler.joblib"))
    joblib.dump(coord_scaler, os.path.join(save_dir, "coord_scaler.joblib"))
except Exception:
    pass

try:
    rainfall_df.to_parquet(os.path.join(save_dir, "rainfall.parquet"))
except Exception:
    pass

print("Saved: pyg_tensors.pt, nx_graph.pkl, coords.npy, node_weight.npy, depth.npz, meta.json")
