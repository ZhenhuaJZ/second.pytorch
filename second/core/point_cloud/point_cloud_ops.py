import time

import numba
import numpy as np


# voxel_size (0.16,0.16,4) 4 is the total Z range (-3,1) !!!
# change voxel size Z axis to evenly grids, such as 0.4 which means 4/0.4 = 10 devided Z into 10 grids
@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel_avg(
                                    points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=4,
                                    max_voxels=200000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    print("grid_size : ", grid_size[2])
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c # inverse!!!!!! coor[0] = z, coor[1] = y, coor[2] = x
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]] # coor_to_voxelidx value is -1
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            #place the point to corespoding index
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx # assign voxelid(voxel_num) to coor_to_voxelid[z,y,x]
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]

        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num

    ## ############################ Voxel -> Pillar ############################

    # coors = coors[:voxel_num] # coors --> loacation
    # voxels = voxels[:voxel_num] # voxels -> points
    # num_points_per_voxel = num_points_per_voxel[:voxel_num]

    ############################# V1 ###########################################

    # pillar_xy_plane = coors[:,1:] # all the xy plane coordinates
    # unique(pillar_xy_plane)
    # pillar_coors = np.unique(pillar_xy_plane, axis=0) # get unique xy plane
    # print("runed -- leo")
    # print(pillar_coors)
    #
    # max_pillars = len(pillar_coors)
    # max_points_per_pillar = max_points * grid_size[2] # grids_size[2] equal z grid size
    #
    # pillars = np.zeros(shape=(max_pillars, max_points_per_pillar, points.shape[-1]), dtype=points.dtype)
    # pillars_coors = np.zeros(shape=(max_pillars, 3), dtype=np.int32)
    # num_points_per_pillar = np.zeros(shape=(max_pillars, ), dtype=np.int32)
    #
    # for p_index, pillar_c in enumerate(pillar_coors):
    #     pillar_voxel_index = pillar_xy_plane == pillar_c
    #     pillar_voxel_index = np.logical_and(pillar_voxel_index[:,0], pillar_voxel_index[:,1]) # logical and [y,x]
    #
    #     pillars_coors[p_index] = np.array([0, pillar_c[0], pillar_c[1]]) # z,y,x
    #     pillars[p_index] = voxels[pillar_voxel_index].reshape(-1, points.shape[-1])
    #     num_points_per_pillar[p_index] = np.sum(num_points_per_voxel[pillar_voxel_index])
    #
    # print("[debug] pillars : ",pillars)
    # print("[debug] pillars_coors : ", pillars_coors)
    # print("[debug] num_points_per_pillar : ", num_points_per_pillar)

    ############################# V2 ###########################################
    # print(coors.shape)
    # print(voxels.shape)
    #
    # x_shape = coor_to_voxelidx.shape[2] #x
    # y_shape = coor_to_voxelidx.shape[1] #y
    # print("[debug] total shape: ", x_shape*y_shape)
    #
    # max_pillars = x_shape*y_shape
    # max_points_per_pillar = max_points * grid_size[2]
    # print("[debug] max_pillars: ", max_pillars)
    #
    # pillars = np.zeros(shape=(max_pillars, max_points_per_pillar, points.shape[-1]), dtype=points.dtype)
    # pillars_coors = np.zeros(shape=(max_pillars, 3), dtype=np.int32)
    # num_points_per_pillar = np.zeros(shape=(max_pillars, ), dtype=np.int32)
    #
    # pillar_index = 0
    # for i in range (x_shape):
    #     for j in range(y_shape):
    #         voxel_to_pillar_index = coor_to_voxelidx[:,j,i]
    #         # print("[debug] voxel_to_pillar_index length: ", voxel_to_pillar_index.shape)
    #         # print("[debug] voxel_to_pillar_index: ", voxel_to_pillar_index.)
    #         # print("[debug] voxel_to_pillar_index_shape: ", voxel_to_pillar_index.shape)
    #         # print("[debug] voxels[voxel_to_pillar_index]: ", voxels[voxel_to_pillar_index].shape)
    #         pillars[pillar_index] = voxels[voxel_to_pillar_index].reshape(max_points_per_pillar, points.shape[-1])
    #         # # print("[debug] coors[voxel_to_pillar_index] shape : ", coors[voxel_to_pillar_index].shape)
    #         pillars_coors[pillar_index] = np.array([0,j,i])
    #         # # print("num_points_per_pillar[pillar_index] : ", num_points_per_voxel[voxel_to_pillar_index])
    #         num_points_per_pillar[pillar_index] = np.sum(num_points_per_voxel[voxel_to_pillar_index])
    #         #
    #         pillar_index +=1
    #         # print(pillar_index)
    # print("[debug] out of loop")
    # return pillars, pillars_coors, num_points_per_pillar

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points= 4, # for averge sampling change max_points to 4
                            max_voxels= 200000): # for averge sampling change max_voxels = 200000
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            #the index of each points respectively
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c #map index [index_x,index_y,index_z]
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx # assgin voxel index to coor_to_voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]

        if num < max_points:
            """
            max_voxels = 6000+
            max_points = 35 or 100
            points = number of features (3 -> xyz)
            voxels = np.zeros(max_voxels, max_points, points.shape[-1])
            voxels init with (M,N,4) -> fill with points[i] (x,y,z)
            put point to the coresponding index of voxel of the number of points

            """
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points=4, # for averge sampling change max_points to 4
                     reverse_index=True,
                     max_voxels=200000): # for averge sampling change max_voxels = 200000
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel_avg(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    pillar_xy_plane = coors[:,1:] # x,y plane ignore z axis
    pillar_coors = np.unique(pillar_xy_plane, axis=0)

    max_pillars = len(pillar_coors)
    print("[debug] voxelmap_shape[2]:", voxelmap_shape[2])
    max_points_per_pillar = max_points * voxelmap_shape[2] # grid_size[2] is Z (not reversed)
    print("[debug] max_points_per_pillar:", max_points_per_pillar)

    # pillars = np.zeros(shape=(max_pillars, max_points_per_pillar, points.shape[-1]), dtype=points.dtype)
    pillars = np.zeros(shape=(max_pillars, max_points_per_pillar, points.shape[-1]), dtype=points.dtype)
    pillars_coors = np.zeros(shape=(max_pillars, 3), dtype=np.int32)
    num_points_per_pillar = np.zeros(shape=(max_pillars, ), dtype=np.int32)

    for p_index, pillar_c in enumerate(pillar_coors):
        print("[debug] p_index : ", p_index)
        voxel_to_pillar_index = pillar_xy_plane == pillar_c
        voxel_to_pillar_index = np.logical_and(voxel_to_pillar_index[:,0], voxel_to_pillar_index[:,1]) # logical and [y,x]

        pillars_coors[p_index] = np.array([0, pillar_c[0], pillar_c[1]]) # z,y,x
        num_points_per_pillar[p_index] = np.sum(num_points_per_voxel[voxel_to_pillar_index]) # total points in one pillar (sum all voxels points)
        # print("voxels[voxel_to_pillar_index] shape ", voxels[voxel_to_pillar_index].shape)
        pillars = voxels[voxel_to_pillar_index].reshape(-1, points.shape[-1])
        pillars[p_index][:pillars.shape[0]] = pillars # put voxels to pillars container
        # pillars[p_index] = voxels[voxel_to_pillar_index].reshape(-1, points.shape[-1])
        # print("voxels[voxel_to_pillar_index] reshape ", voxels[voxel_to_pillar_index].reshape(-1, points.shape[-1]).shape)

    return voxels, coors, num_points_per_voxel
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    # return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
