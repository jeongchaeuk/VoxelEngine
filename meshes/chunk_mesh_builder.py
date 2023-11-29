from settings import *
import numpy as np


def get_voxel_id(voxel_pos, chunk_voxels):
    x, y, z = voxel_pos
    return chunk_voxels[x + z * CHUNK_SIZE + y * CHUNK_AREA]


def is_void(voxel_pos, chunk_voxels):
    x, y, z = voxel_pos
    if 0 <= x < CHUNK_SIZE and 0 <= y < CHUNK_SIZE and 0 <= z < CHUNK_SIZE:
        if get_voxel_id(voxel_pos, chunk_voxels):
            return False
    return True


def add_data(vertex_data, index, *vertices):
    for vertex in vertices:
        for attr in vertex:
            vertex_data[index] = attr
            index += 1
    return index


def build_chunk_mesh(chunk_voxels, format_size):
    """
    :param chunk_voxels:
    :param format_size: vertex's attributes's size in bytes.
    :return: vertex_data which is ndarray.
    """
    # Volxel's visible face is 3 (top, front, right).
    # Each face has 2 triangles, each triangle has 3 vertices.
    NUM_VOXEL_VERTICES = 18  # why not 36?
    # Vertex attributes's format: x, y, z, voxel_id, face_id
    ARRAY_SIZE = CHUNK_VOL * NUM_VOXEL_VERTICES * format_size
    vertex_data = np.empty(ARRAY_SIZE, dtype='uint8')
    index = 0

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                voxel_id = get_voxel_id((x, y, z), chunk_voxels)
                if not voxel_id:
                    continue

                # Top face.
                if is_void((x, y + 1, z), chunk_voxels):
                    v0 = (x,     y + 1, z,     voxel_id, TOP)
                    v1 = (x + 1, y + 1, z,     voxel_id, TOP)
                    v2 = (x + 1, y + 1, z + 1, voxel_id, TOP)
                    v3 = (x,     y + 1, z + 1, voxel_id, TOP)
                    # counter_clock_wise
                    vertices = (v0, v3, v2, v0, v2, v1)
                    index = add_data(vertex_data, index, *vertices)

                # Bottom face
                if is_void((x, y - 1, z), chunk_voxels):
                    v0 = (x,     y, z,     voxel_id, BOTTOM)
                    v1 = (x + 1, y, z,     voxel_id, BOTTOM)
                    v2 = (x + 1, y, z + 1, voxel_id, BOTTOM)
                    v3 = (x,     y, z + 1, voxel_id, BOTTOM)
                    vertices = (v0, v2, v3, v0, v1, v2)
                    index = add_data(vertex_data, index, *vertices)

                # Right face.
                if is_void((x + 1, y, z), chunk_voxels):
                    v0 = (x + 1, y,     z,     voxel_id, RIGHT)
                    v1 = (x + 1, y + 1, z,     voxel_id, RIGHT)
                    v2 = (x + 1, y + 1, z + 1, voxel_id, RIGHT)
                    v3 = (x + 1, y,     z + 1, voxel_id, RIGHT)
                    vertices = (v0, v1, v2, v0, v2, v3)
                    index = add_data(vertex_data, index, *vertices)

                # Left face.
                if is_void((x - 1, y, z), chunk_voxels):
                    v0 = (x, y,     z,     voxel_id, LEFT)
                    v1 = (x, y + 1, z,     voxel_id, LEFT)
                    v2 = (x, y + 1, z + 1, voxel_id, LEFT)
                    v3 = (x, y,     z + 1, voxel_id, LEFT)
                    vertices = (v0, v2, v1, v0, v3, v2)
                    index = add_data(vertex_data, index, *vertices)

                # Back face.
                if is_void((x, y, z - 1), chunk_voxels):
                    v0 = (x,     y,     z, voxel_id, BACK)
                    v1 = (x,     y + 1, z, voxel_id, BACK)
                    v2 = (x + 1, y + 1, z, voxel_id, BACK)
                    v3 = (x + 1, y,     z, voxel_id, BACK)
                    vertices = (v0, v1, v2, v0, v2, v3)
                    index = add_data(vertex_data, index, *vertices)

                # Front face.
                if is_void((x, y, z + 1), chunk_voxels):
                    v0 = (x,     y,     z + 1, voxel_id, FRONT)
                    v1 = (x,     y + 1, z + 1, voxel_id, FRONT)
                    v2 = (x + 1, y + 1, z + 1, voxel_id, FRONT)
                    v3 = (x + 1, y,     z + 1, voxel_id, FRONT)
                    vertices = (v0, v2, v1, v0, v3, v2)
                    index = add_data(vertex_data, index, *vertices)

    return vertex_data[:index]  # [:index + 1]
