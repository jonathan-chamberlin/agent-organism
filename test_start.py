from start import coordinates_to_q_table_index
from start import q_table_width


def test_coordinates_to_q_table_index() -> None:
    assert coordinates_to_q_table_index([0,0]) == 0
    assert coordinates_to_q_table_index([1,0]) == 1 + 0 * q_table_width
    assert coordinates_to_q_table_index([0,1]) == 0 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,1]) == 5 + 1 * q_table_width
    assert coordinates_to_q_table_index([5,9]) == 5 + 9 * q_table_width
    assert coordinates_to_q_table_index([-5,0]) == -1
    assert coordinates_to_q_table_index([0,-5]) == -1
    assert coordinates_to_q_table_index([11,0]) == -1
    assert coordinates_to_q_table_index([0,11]) == -1
    assert coordinates_to_q_table_index([-5,-50]) == -1

    