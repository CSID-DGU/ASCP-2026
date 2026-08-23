import unittest

from RL.loader import airport_map_hash, validate_airport_map, bases_to_ids


class CheckpointAirportMapTest(unittest.TestCase):
    def test_hash_depends_on_airport_id_meaning(self):
        self.assertNotEqual(
            airport_map_hash({"ATL": 0, "LAX": 1}),
            airport_map_hash({"ATL": 1, "LAX": 0}),
        )

    def test_validate_rejects_non_contiguous_ids(self):
        with self.assertRaisesRegex(ValueError, "연속 범위"):
            validate_airport_map({"ATL": 0, "LAX": 2})

    def test_validate_rejects_embedding_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "embedding 크기"):
            validate_airport_map({"ATL": 0}, n_airports=2)

    def test_validate_normalizes_serialized_values(self):
        self.assertEqual(
            validate_airport_map({"ATL": "0", "LAX": "1"}, n_airports=2),
            {"ATL": 0, "LAX": 1},
        )

    def test_missing_configured_base_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "configured base"):
            bases_to_ids(["ATL", "LAX"], {"ATL": 0})


if __name__ == "__main__":
    unittest.main()
