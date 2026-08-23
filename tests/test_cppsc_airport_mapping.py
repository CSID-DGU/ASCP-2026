import unittest

from baselines.tahir.airport_mapping import remap_cppsc_airports


class CppscAirportMappingTests(unittest.TestCase):
    def test_maps_by_airport_code_not_numeric_position(self):
        flights = [{"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0}]
        mapped, bases = remap_cppsc_airports(
            flights,
            {"ATL": 0, "JFK": 1},
            [0],
            {"JFK": 3, "ATL": 7},
        )
        self.assertEqual(mapped[0]["origin"], 7)
        self.assertEqual(mapped[0]["dest"], 3)
        self.assertEqual(bases, [7])

    def test_unknown_airport_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "학습 당시 없던 공항"):
            remap_cppsc_airports(
                [{"id": 0, "origin": 0, "dest": 1}],
                {"ATL": 0, "UNKNOWN": 1},
                [0],
                {"ATL": 4},
            )


if __name__ == "__main__":
    unittest.main()
