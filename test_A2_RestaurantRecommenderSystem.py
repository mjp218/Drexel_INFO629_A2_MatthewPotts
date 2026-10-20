import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from A2_RestaurantRecommenderSystem import (
    distance_miles,
    geocode_address,
    load_restaurants,
    main,
    recommend_restaurants,
)


class RestaurantRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.origin = (39.9526, -75.1652)

    def business(
        self,
        name,
        latitude,
        longitude,
        rating,
        review_count,
        categories="Restaurants, Italian",
    ):
        return {
            "business_id": name,
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
            "stars": rating,
            "review_count": review_count,
            "categories": categories,
            "address": "1 Main St",
            "city": "Philadelphia",
            "state": "PA",
            "postal_code": "19103",
        }

    def test_distance_is_zero_at_same_coordinates(self):
        self.assertAlmostEqual(
            distance_miles(*self.origin, *self.origin),
            0,
        )

    def test_geocodes_exact_local_address(self):
        other_address = self.business(
            "same ZIP other address", 39.96, -75.16, 4.5, 10
        )
        other_address["address"] = "2 Elm St"
        businesses = [
            self.business("matching address", *self.origin, 4.5, 10),
            other_address,
        ]

        latitude, longitude, exact_match = geocode_address(
            "1 Main St, Philadelphia, pa, 19103", businesses
        )

        self.assertEqual((latitude, longitude), self.origin)
        self.assertTrue(exact_match)

    def test_uses_zip_area_median_when_street_is_not_in_dataset(self):
        businesses = [
            self.business("first", 39.94, -75.17, 4.5, 10),
            self.business("second", 39.96, -75.15, 4.5, 10),
        ]

        latitude, longitude, exact_match = geocode_address(
            "99 Unknown Rd, Philadelphia, PA, 19103", businesses
        )

        self.assertAlmostEqual(latitude, 39.95)
        self.assertAlmostEqual(longitude, -75.16)
        self.assertFalse(exact_match)

    def test_rejects_invalid_address_format_or_unresolved_zip(self):
        with self.assertRaisesRegex(ValueError, "address line 1"):
            geocode_address("Philadelphia, PA", [])
        with self.assertRaisesRegex(ValueError, "could not be resolved"):
            geocode_address("1 Main St, Unknown, ZZ, 00000", [])

    def test_filters_by_cuisine_and_radius(self):
        businesses = [
            self.business("near italian", *self.origin, 4.5, 100),
            self.business("far italian", 40.5, -75.1652, 5.0, 500),
            self.business(
                "near mexican",
                *self.origin,
                5.0,
                500,
                categories="Restaurants, Mexican",
            ),
        ]

        results = recommend_restaurants(
            businesses, *self.origin, "italian", 3, 5
        )

        self.assertEqual([item["name"] for item in results], ["near italian"])

    def test_rating_must_be_strictly_above_minimum(self):
        businesses = [
            self.business(
                "exactly four",
                *self.origin,
                4.0,
                100,
                categories="Restaurants, Indian",
            ),
            self.business(
                "above four",
                *self.origin,
                4.5,
                10,
                categories="Restaurants, Indian",
            ),
        ]

        results = recommend_restaurants(
            businesses, *self.origin, "Indian", 100, 5, minimum_rating=4.0
        )

        self.assertEqual([item["name"] for item in results], ["above four"])

    def test_matches_any_requested_cuisine(self):
        businesses = [
            self.business(
                "Indian restaurant",
                *self.origin,
                4.5,
                20,
                categories="Restaurants, Indian",
            ),
            self.business(
                "Mexican restaurant",
                *self.origin,
                4.6,
                25,
                categories="Restaurants, Mexican",
            ),
            self.business(
                "Other restaurant",
                *self.origin,
                5.0,
                100,
                categories="Restaurants, Italian",
            ),
        ]

        results = recommend_restaurants(
            businesses, *self.origin, "Indian,Mexican", 100, 100
        )

        self.assertEqual(
            {item["name"] for item in results},
            {"Indian restaurant", "Mexican restaurant"},
        )

    def test_ranks_by_rating_then_distance_and_obeys_limit(self):
        businesses = [
            self.business("lower rating", 39.96, -75.1652, 4.0, 10000),
            self.business("farther", 39.97, -75.1652, 4.5, 10),
            self.business("nearer", 39.953, -75.1652, 4.5, 1),
            self.business("highest rating", *self.origin, 5.0, 1),
        ]

        results = recommend_restaurants(
            businesses, *self.origin, "Italian", 2, 3
        )

        self.assertEqual(
            [item["name"] for item in results],
            ["highest rating", "nearer", "farther"],
        )

    def test_returns_empty_when_no_business_matches(self):
        results = recommend_restaurants(
            [], *self.origin, "Italian", 3, 5
        )
        self.assertEqual(results, [])

    def test_loads_json_lines_and_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "businesses.json"
            data_file.write_text('{"name": "Valid"}\n', encoding="utf-8")
            self.assertEqual(load_restaurants(data_file), [{"name": "Valid"}])

            data_file.write_text("{invalid json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_restaurants(data_file)

    def test_cli_runs_with_a_kaggle_shaped_business_record(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "businesses.json"
            data_file.write_text(
                json.dumps(
                    self.business(
                        "Demo Indian",
                        *self.origin,
                        4.5,
                        25,
                        categories="Restaurants, Indian",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--data-file",
                        str(data_file),
                        "--address",
                        "1 Main St, Philadelphia, PA, 19103",
                        "--cuisine",
                        "Indian",
                        "--radius",
                        "100",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Demo Indian", output.getvalue())
        self.assertIn("4.5/5", output.getvalue())
        self.assertIn("25", output.getvalue())
        self.assertIn("matched address", output.getvalue())
        self.assertIn("does not record spice levels", output.getvalue())
        output_lines = output.getvalue().splitlines()
        header = next(line for line in output_lines if line.startswith("Rank"))
        self.assertEqual(
            [column.strip() for column in header.split("|")],
            [
                "Rank",
                "Name",
                "Rating",
                "Review Count",
                "Distance",
                "Address",
                "Cuisine",
            ],
        )
        record = next(line for line in output_lines if "Demo Indian" in line)
        self.assertEqual(len(record.split("|")), len(header.split("|")))
        self.assertEqual(record.split("|")[0].strip(), "1")
        self.assertEqual(record.split("|")[5].strip(), "Philadelphia, PA, 19103")
        self.assertEqual(record.split("|")[6].strip(), "Indian")
        self.assertNotIn("Main St", record)


if __name__ == "__main__":
    unittest.main()
