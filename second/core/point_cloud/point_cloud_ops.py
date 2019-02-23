import time

import numba
import numpy as np
import pcl
import pcl.pcl_visualization

def pcl_viewer(points):
    print(points.shape)
    if points.shape[1] > 3:
        p_cloud = pcl.PointCloud_PointXYZI()
        p_cloud.from_array(points[:,:4])
    else:
        p_cloud = pcl.PointCloud_pointXYZ()
        p_cloud.from_array(points[:,:3])
    viewer = pcl.pcl_visualization.CloudViewing()
    viewer.ShowGrayCloud(p_cloud)
    flag = True
    while flag:
        flag = not viewer.WasStopped()

@numba.jit(nopython = True)
def dense_sampling(voxels, dense_smp_voxels, coors, num_points_per_voxel, voxel_size, max_points, voxel_ratio = 0.8):
    voxel_indexes = voxels.shape[0]
    num_points = voxels.shape[1]
    ndim = voxels.shape[2]
    points = np.zeros(shape = (num_points,ndim),dtype = np.float32)
    most_points = np.zeros(shape = (max_points,ndim),dtype = np.float32)
    tmp_points = np.zeros(shape = (max_points,ndim),dtype = np.float32)
    zero_point = np.zeros(shape = (ndim,), dtype = np.float32)
    cluster_radius = voxel_size[0]/2 * voxel_ratio

    for index in range(voxel_indexes):
        points = voxels[index]
        most_points[:] = 0
        # Center point index
        num_max_points_in_radius = -1
        for c_idx in range(num_points):
            if (points[c_idx] == zero_point).all():
                continue
            #surrounding points index
            tmp_points[:] = 0
            num_points_in_radius = 0
            for s_idx in range(num_points):
                if (points[s_idx] == zero_point).all():
                    continue
                distance = np.sqrt(np.sum(np.square(points[c_idx][:2] - points[s_idx][:2])))
                if distance < cluster_radius:
                    tmp_points[num_points_in_radius] = points[s_idx]
                    num_points_in_radius += 1
                # if stored points are already exceed maximum points, then break
                if num_points_in_radius >= max_points :
                    break
            # if this c_idx has the most point then store its points
            if num_points_in_radius > num_max_points_in_radius:
                num_max_points_in_radius = num_points_in_radius
                most_points = np.copy(tmp_points)
        # After gone through all the c_idx, store only the maximum number of points
        # to its voxel index
        dense_smp_voxels[index] = most_points
        if num_max_points_in_radius > max_points:
            num_points_per_voxel[index] = max_points
        else:
            num_points_per_voxel[index] = num_max_points_in_radius
    return dense_smp_voxels

@numba.jit(nopython = True)
def dense_sampling_v2(voxels, num_points_per_voxel, voxel_size, max_points, voxel_ratio = 0.8):
    voxel_indexes = voxels.shape[0]
    num_points = voxels.shape[1]
    ndim = voxels.shape[2]
    points = np.zeros(shape = (num_points,ndim),dtype = np.float32)
    tmp_points = np.zeros(shape = (max_points,ndim),dtype = np.float32)
    zero_point = np.zeros(shape = (ndim,), dtype = np.float32)
    cluster_radius = voxel_size[0]/2 * voxel_ratio

    for index in range(voxel_indexes):
        points = voxels[index]
        # Center point index
        num_max_points_in_radius = -1
        for c_idx in range(num_points):
            if (points[c_idx] == zero_point).all():
                continue
            #surrounding points index
            tmp_points[:] = 0
            num_points_in_radius = 0
            for s_idx in range(num_points):
                if (points[s_idx] == zero_point).all():
                    continue
                distance = np.sqrt(np.sum(np.square(points[c_idx][:2] - points[s_idx][:2])))
                if distance < cluster_radius:
                    tmp_points[num_points_in_radius] = points[s_idx]
                    num_points_in_radius += 1
                # if stored points are already exceed maximum points, then break
                if num_points_in_radius >= max_points :
                    break
            # if this c_idx has the most point then store its points
            if num_points_in_radius > num_max_points_in_radius:
                num_max_points_in_radius = num_points_in_radius
                voxels[index] = tmp_points
        # After gone through all the c_idx, store only the maximum number of points
        # to its voxel index
        if num_max_points_in_radius > max_points:
            num_points_per_voxel[index] = max_points
        else:
            num_points_per_voxel[index] = num_max_points_in_radius

@numba.jit(nopython = True)
def _points_to_voxel_dense_sample(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_range = np.zeros(shape=(6,), dtype=np.float32)
    mask = np.zeros(shape = (N,), dtype = np.bool_)
    mask_xyz = np.zeros(shape = (N, 3), dtype = np.bool_)
    # distant_point = np.zeros(shape = (points.shape[-1]), dtype = np.float32)
    cluster_radius = voxel_size[0]/2 * 0.8
    # voxel_points =
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            # get xyz into voxel coordinates
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            # Obtain range of the voxel
            voxel_range[j] = c * voxel_size[j] + coors_range[j]
            voxel_range[j + 3] = c * voxel_size[j] + voxel_size[j] + coors_range[j]
            # print(voxel_range)
            # reverse voxel coordinate
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            # Assign voxel index in order
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            # Assign index to voxel coordinate
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
            ######################### Dense Sampling ############################
            # Obtain points within the voxel from voxel range
            mask_xyz = ((points[:,:3] >= voxel_range[:3]) & (points[:,:3] <= voxel_range[3:]))
            mask = (mask_xyz[:,0]*mask_xyz[:,1]*mask_xyz[:,2])#.astype(np.bool)
            voxel_points = points[mask,:]
            max_points_in_radius = -1
            index = voxel_points.shape[0]
            # Create a temprarely container for sampling
            if index < 100:
                temp_points = np.zeros(shape = (100 ,points.shape[-1]), dtype = points.dtype)
            else:
                temp_points = np.zeros(shape = (index ,points.shape[-1]), dtype = points.dtype)

            for i in range(index):
                temp_points[:] = 0
                num_point_in_radius = 0
                for j in range(index):
                    distance = np.sqrt(np.sum(np.square(voxel_points[i][:2] - voxel_points[j][:2])))
                    if distance < cluster_radius:
                        temp_points[num_point_in_radius] = voxel_points[j]
                        num_point_in_radius += 1
                if num_point_in_radius > max_points_in_radius:
                    voxels[voxelidx] = temp_points[:max_points]
                    max_points_in_radius = num_point_in_radius

            if max_points_in_radius > max_points:
                num_points_per_voxel[voxelidx] = max_points
            else:
                num_points_per_voxel[voxelidx] = max_points_in_radius
            #
            # if max_points_in_radius > 100:
            #     print("*"*20)
            #     print("[debug] max_points_in_radius: ", max_points_in_radius)
            #     print("[debug] all voxel_points: ", voxel_points.shape[0])
            #     print("[debug] num_points_per_voxel[voxelidx]: ", num_points_per_voxel[voxelidx])
            #     print("[debug] voxels[voxelidx]: ", voxels[voxelidx])

    return voxel_num

            # print(voxel_points)

@numba.jit(nopython = True)
def _points_to_voxel_dense_sample_v2(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    N = points.shape[0]
    # print("[debug] number of points ", N)
    # print("[debug] max_points ", max_points)
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_range = np.zeros(shape=(6,), dtype=np.float32)
    mask = np.zeros(shape = (N,), dtype = np.bool_)
    mask_xyz = np.zeros(shape = (N, 3), dtype = np.bool_)
    # distant_point = np.zeros(shape = (points.shape[-1]), dtype = np.float32)
    xy_plane_orth = np.sqrt(np.square(voxel_size[0]/2) + np.square(voxel_size[1]/2))
    cluster_radius = np.sqrt(np.square(xy_plane_orth) + np.square(voxel_size[2]/2)) * 0.8
    # voxel_points =
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            # get xyz into voxel coordinates
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            # Obtain range of the voxel
            voxel_range[j] = c * voxel_size[j] + coors_range[j]
            voxel_range[j + 3] = c * voxel_size[j] + voxel_size[j] + coors_range[j]
            # print(voxel_range)
            # reverse voxel coordinate
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            # Assign voxel index in order
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            # Assign index to voxel coordinate
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
            ######################### Dense Sampling ############################
            # Obtain points within the voxel from voxel range
            mask_xyz = ((points[:,:3] >= voxel_range[:3]) & (points[:,:3] <= voxel_range[3:]))
            mask = (mask_xyz[:,0]*mask_xyz[:,1]*mask_xyz[:,2])#.astype(np.bool)
            voxel_points = points[mask,:]
            # max_points_in_radius = -1
            index = voxel_points.shape[0]
            # print("[debug] number points in voxel > 100 : ", voxel_points.shape)
            # Create a temprarely container for sampling max_point default = 100
            # if index < max_points:
            #     temp_points = np.zeros(shape = (max_points ,points.shape[-1]), dtype = points.dtype)
            # else:
            #     temp_points = np.zeros(shape = (index ,points.shape[-1]), dtype = points.dtype)


            ##########################pillar center version#####################

            pillar_center = np.sum(voxel_points[:,:3], axis=0)/index # center of xyz in pillar

            #####v1###
            # num_point_in_radius = 0
            #
            # for i in range(index):
            #     distance = np.sqrt(np.sum(np.square(voxel_points[i][:3]-pillar_center)))
            #     if distance < cluster_radius:
            #         temp_points[num_point_in_radius] = voxel_points[i]
            #         num_point_in_radius += 1
            #
            #     if num_point_in_radius >= max_points:
            #         break
            #
            # voxels[voxelidx] = temp_points[:max_points] # put points in temp container back to voxels
            # num_points_per_voxel[voxelidx] = num_point_in_radius
            #
            # if num_point_in_radius >= 100:
            #     print("[debug] voxelidx ", voxelidx)
            #     print("[debug] num_points_per_voxel[voxelidx] ", num_points_per_voxel[voxelidx])

            ###v2###
            """
            step1 calculate the points in voxels nearest to the center point
            step2 ascending sorting all the points, select the number of points
                  which less than max points


            pro: do not need to calculate radius center
            cro: allocate more memory
            """

            """!!!!! need to be fixed : only need to create temp array length = max_points"""
            temp_points = np.zeros(shape = (max_points ,points.shape[-1]), dtype = points.dtype)
            #
            # if index < max_points:
            #     temp_points[:index] = voxel_points
            #     voxels[voxelidx] = temp_points # put points in temp container back to voxels
            #     num_points_per_voxel[voxelidx] = index

            distance_matrix = np.sqrt(np.sum(np.square(voxel_points[:,:3]-pillar_center), axis=1))
            dis_flag = np.argsort(distance_matrix)[:max_points]
            num_point_in_radius = len(dis_flag)
            temp_points[:num_point_in_radius] = voxel_points[dis_flag]

            # to be fixed temp_points is enough
            voxels[voxelidx] = temp_points # put points in temp container back to voxels
            num_points_per_voxel[voxelidx] = num_point_in_radius

            # if num_point_in_radius > 80:
            #     print("[debug] voxels", voxels[voxelidx])
            #     print("[debug] num_points_per_voxel ", num_points_per_voxel[voxelidx])
            #     print("[debug] Found points > 80 -- break")


            ###################### loop all the points #########################

            # for i in range(index):
            #     distance = np.sqrt(np.sum(np.square(voxel_points[:,:2]-voxel_points[i][:2]), axis=1))
            #     seleted = np.sqrt(np.sum(np.square(voxel_points[:,:2]-voxel_points[i][:2]), axis=1)) < cluster_radius
            #     num_point_in_radius = len(distance[seleted])
            #
            #     if num_point_in_radius > max_points_in_radius:
            #         temp_points[:num_point_in_radius] = voxel_points[seleted]
            #         voxels[voxelidx] = temp_points[:max_points]
            #         max_points_in_radius = num_point_in_radius
            #
            # if max_points_in_radius > max_points:
            #     num_points_per_voxel[voxelidx] = max_points
            # else:
            #     num_points_per_voxel[voxelidx] = max_points_in_radius

            ####################jim version#################################
            # for i in range(index):
            #     temp_points[:] = 0
            #     num_point_in_radius = 0
            #     for j in range(index):
            #         distance = np.sqrt(np.sum(np.square(voxel_points[i][:2] - voxel_points[j][:2])))
            #         if distance < cluster_radius:
            #             temp_points[num_point_in_radius] = voxel_points[j]
            #             num_point_in_radius += 1
            #     if num_point_in_radius > max_points_in_radius:
            #         voxels[voxelidx] = temp_points[:max_points]
            #         max_points_in_radius = num_point_in_radius
            #
            # if max_points_in_radius > max_points:
            #     num_points_per_voxel[voxelidx] = max_points
            # else:
            #     num_points_per_voxel[voxelidx] = max_points_in_radius


            # if max_points_in_radius > 100:
            #     print("*"*20)
            #     print("[debug] max_points_in_radius: ", max_points_in_radius)
            #     print("[debug] all voxel_points: ", voxel_points.shape[0])
            #     print("[debug] num_points_per_voxel[voxelidx]: ", num_points_per_voxel[voxelidx])
                # print("[debug] voxelidx: ", voxelidx)
                # print("[debug] voxels[voxelidx]: ", voxels[voxelidx])

    return voxel_num


@numba.jit(nopython = True)
def _points_to_voxel_dense_sample_v3(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_range = np.zeros(shape=(6,), dtype=np.float32)
    mask = np.zeros(shape = (N,), dtype = np.bool_)
    mask_xyz = np.zeros(shape = (N, 3), dtype = np.bool_)
    # distant_point = np.zeros(shape = (points.shape[-1]), dtype = np.float32)
    # xy_plane_orth = np.sqrt(np.square(voxel_size[0]/2) + np.square(voxel_size[1]/2))
    # cluster_radius = np.sqrt(np.square(xy_plane_orth) + np.square(voxel_size[2]/2)) * 0.5 #1.6
    cluster_radius = voxel_size[0]/2 * 0.8
    # voxel_points =
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            # get xyz into voxel coordinates
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            # Obtain range of the voxel
            voxel_range[j] = c * voxel_size[j] + coors_range[j]
            voxel_range[j + 3] = c * voxel_size[j] + voxel_size[j] + coors_range[j]
            # print(voxel_range)
            # reverse voxel coordinate
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            # Assign voxel index in order
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            # Assign index to voxel coordinate
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
            ######################### Dense Sampling ############################
            # Obtain points within the voxel from voxel range
            mask_xyz = ((points[:,:3] >= voxel_range[:3]) & (points[:,:3] <= voxel_range[3:]))
            mask = (mask_xyz[:,0]*mask_xyz[:,1]*mask_xyz[:,2])#.astype(np.bool)
            voxel_points = points[mask,:]
            max_points_in_radius = -1
            index = voxel_points.shape[0]

            pillar_center = np.sum(voxel_points[:,:2], axis=0)/index # center of xyz in pillar
            # Create a temprarely container for sampling
            if index < 100:
                temp_points = np.zeros(shape = (100 ,points.shape[-1]), dtype = points.dtype)
            else:
                temp_points = np.zeros(shape = (index ,points.shape[-1]), dtype = points.dtype)

            num_point_in_radius = 0
            for i in range(index):
                distance = np.sqrt(np.sum(np.square(voxel_points[i][:2] - pillar_center)))
                if distance < cluster_radius:
                    temp_points[num_point_in_radius] = voxel_points[i]
                    num_point_in_radius += 1

            voxels[voxelidx] = temp_points[:max_points]

            if num_point_in_radius >= max_points:
                num_points_per_voxel[voxelidx] = max_points
            else:
                num_points_per_voxel[voxelidx] = num_point_in_radius

    return voxel_num

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
            # get xyz into voxel coordinates
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            # reverse voxel coordinate
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        # Obtain voxel coordinate voxel index
        # Voxel index is only to monitor if the voxel is assigned
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        # if index not exist
        if voxelidx == -1:
            # Assign voxel index in order
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            # Assign index to voxel coordinate
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
                            max_points=35,
                            max_voxels=20000):
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
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
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


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points=35,
                     reverse_index=True,
                     max_voxels=20000,
                     dense_sample=True):
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
    # pcl_viewer(points)
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
    pre_sample_max_points = max_points
    if dense_sample:
        pre_sample_max_points = max_points + 100
    voxels = np.zeros(
        shape=(max_voxels, pre_sample_max_points, points.shape[-1]), dtype=points.dtype)
    # print("[debug] pre_sample_max_points : ", pre_sample_max_points)
    # print("[debug] voxels shape", voxels.shape)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        # Ran here
        # voxel_num = _points_to_voxel_dense_sample(
        voxel_num =_points_to_voxel_dense_sample_v3(
        # voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, pre_sample_max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # print("[debug] voxels : ", voxels)
    #########Dense Sample###########
    # if dense_sample:
        # dense_smp_voxels = np.zeros(shape=(voxel_num,max_points,points.shape[-1]), dtype = points.dtype)
        # voxels = dense_sampling(voxels, dense_smp_voxels, coors, num_points_per_voxel, voxel_size, max_points)
        # dense_sampling_v2(voxels, num_points_per_voxel, voxel_size, max_points)
    # pcl_viewer(voxels.reshape(-1,points.shape[-1]))
    return voxels, coors, num_points_per_voxel


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
