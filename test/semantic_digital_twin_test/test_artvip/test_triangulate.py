import numpy as np

from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader


def test_triangulate_keeps_a_triangle_as_is():
    faces = ArtVipDatasetLoader._triangulate([3], [0, 1, 2])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_triangulate_fans_a_quad_into_two_triangles():
    faces = ArtVipDatasetLoader._triangulate([4], [0, 1, 2, 3])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_triangulate_fans_a_pentagon_into_three_triangles():
    faces = ArtVipDatasetLoader._triangulate([5], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3], [0, 3, 4]])


def test_triangulate_handles_multiple_faces_of_different_sizes():
    # A triangle followed by a quad, sharing no vertex indices.
    faces = ArtVipDatasetLoader._triangulate([3, 4], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5], [3, 5, 6]])
