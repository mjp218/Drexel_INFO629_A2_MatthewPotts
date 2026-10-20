import argparse
import json
import math
import re
import sys
from statistics import median
from pathlib import Path

from ortools.sat.python import cp_model


DEFAULT_DATA_FILE = (
    Path(__file__).resolve().parent / "yelp_academic_dataset_business.json"
)
KAGGLE_DOWNLOAD_COMMAND = (
    "kaggle datasets download -d yelp-dataset/yelp-dataset "
    "-f yelp_academic_dataset_business.json -p . --unzip"
)


def distance_miles(lat1, lon1, lat2, lon2):
    """Return the great-circle distance between two coordinates in miles."""
    earth_radius = 3958.8
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    return earth_radius * 2 * math.atan2(
        math.sqrt(a), math.sqrt(1 - a)
    )


def load_restaurants(data_file):
    """Read Yelp's one-business-per-line JSON file."""
    restaurants = []
    with Path(data_file).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                business = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {data_file}: {error}"
                ) from error
            if not isinstance(business, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {data_file}"
                )
            restaurants.append(business)
    return restaurants


def _normalized_address(value):
    return " ".join(value.casefold().split())


def geocode_address(address, restaurants):
    """Resolve an address from local Yelp records without an external service."""
    parts = [part.strip() for part in address.split(",")]
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            "Enter an address as: address line 1, city, state, ZIP code."
        )

    street, city, state, postal_code = parts
    if not re.fullmatch(r"\d{5}(?:-\d{4})?", postal_code):
        raise ValueError("ZIP code must be 5 digits, optionally followed by +4.")

    postal_code = postal_code[:5]
    matching_location = [
        business
        for business in restaurants
        if str(business.get("postal_code", "")).strip() == postal_code
        and str(business.get("city", "")).strip().casefold() == city.casefold()
        and str(business.get("state", "")).strip().casefold() == state.casefold()
    ]
    if not matching_location:
        raise ValueError(
            "That city, state, and ZIP code could not be resolved from the local "
            "Yelp business dataset."
        )

    exact_matches = [
        business
        for business in matching_location
        if _normalized_address(str(business.get("address", "")))
        == _normalized_address(street)
    ]
    matches = exact_matches or matching_location
    coordinates = []
    for business in matches:
        try:
            latitude = float(business["latitude"])
            longitude = float(business["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            coordinates.append((latitude, longitude))

    if not coordinates:
        raise ValueError("No usable coordinates found for that address.")

    return (
        median(latitude for latitude, _ in coordinates),
        median(longitude for _, longitude in coordinates),
        bool(exact_matches),
    )


def _category_tokens(categories):
    if not isinstance(categories, str):
        return []
    return [category.strip().casefold() for category in categories.split(",")]


def _cuisine_tokens(cuisines):
    if isinstance(cuisines, str):
        return [
            cuisine.strip().casefold()
            for cuisine in cuisines.split(",")
            if cuisine.strip()
        ]
    return [cuisine.strip().casefold() for cuisine in cuisines if cuisine.strip()]


def _has_required_categories(business, cuisines):
    categories = _category_tokens(business.get("categories"))
    return "restaurants" in categories and any(
        cuisine in categories for cuisine in cuisines
    )


def _candidate(business, latitude, longitude):
    try:
        restaurant_latitude = float(business["latitude"])
        restaurant_longitude = float(business["longitude"])
        rating = float(business["stars"])
        review_count = int(business["review_count"])
    except (KeyError, TypeError, ValueError):
        return None

    categories = _category_tokens(business.get("categories"))
    address = ", ".join(
        value
        for value in (
            business.get("city"),
            business.get("state"),
            business.get("postal_code"),
        )
        if isinstance(value, str) and value.strip()
    )
    return {
        "business_id": business.get("business_id", ""),
        "name": business.get("name") or "Unnamed restaurant",
        "rating": rating,
        "review_count": review_count,
        "distance": distance_miles(
            latitude,
            longitude,
            restaurant_latitude,
            restaurant_longitude,
        ),
        "categories": categories,
        "address": address,
    }


def recommend_restaurants(
    restaurants,
    latitude,
    longitude,
    cuisines,
    radius_miles,
    number_to_recommend=5,
    minimum_rating=4.0,
):
    """Apply cuisine, rating, and radius constraints, then rank with CP-SAT."""
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not math.isfinite(radius_miles) or radius_miles <= 0:
        raise ValueError("Radius must be greater than zero miles.")
    if not math.isfinite(minimum_rating) or not 0 <= minimum_rating <= 5:
        raise ValueError("Minimum rating must be between 0 and 5.")
    if number_to_recommend <= 0:
        raise ValueError("The number of recommendations must be greater than zero.")
    cuisine_options = _cuisine_tokens(cuisines)
    if not cuisine_options:
        raise ValueError("At least one cuisine must be provided.")

    eligible = []
    for business in restaurants:
        if not _has_required_categories(business, cuisine_options):
            continue
        candidate = _candidate(business, latitude, longitude)
        if (
            candidate is not None
            and candidate["rating"] > minimum_rating
            and candidate["distance"] <= radius_miles
        ):
            candidate["cuisine"] = ", ".join(
                cuisine.title()
                for cuisine in cuisine_options
                if cuisine in candidate["categories"]
            )
            eligible.append(candidate)

    if not eligible:
        return []

    eligible.sort(
        key=lambda item: (
            -item["rating"],
            item["distance"],
            item["name"].casefold(),
        )
    )
    model = cp_model.CpModel()
    selected = [model.new_bool_var(f"restaurant_{i}") for i in range(len(eligible))]
    model.add(sum(selected) == min(number_to_recommend, len(eligible)))

    # Scores preserve the requested rating-descending, distance-ascending order.
    scores = range(len(eligible), 0, -1)
    model.maximize(sum(score * variable for score, variable in zip(scores, selected)))

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT could not find recommendations (status {status}).")

    recommendations = [
        restaurant
        for restaurant, variable in zip(eligible, selected)
        if solver.value(variable)
    ]
    recommendations.sort(
        key=lambda item: (
            -item["rating"],
            item["distance"],
            item["name"].casefold(),
        )
    )
    return recommendations


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recommend restaurants from Yelp that match cuisine, rating, "
            "and distance constraints."
        )
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--address",
        help="Origin formatted as: address line 1, city, state, ZIP code.",
    )
    parser.add_argument(
        "--cuisine",
        default="Indian,Mexican",
        help="One or more comma-separated cuisine categories (default: Indian,Mexican).",
    )
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        restaurants = load_restaurants(args.data_file)
        address = args.address or input(
            "Enter your address (address line 1, city, state, ZIP code): "
        )
        latitude, longitude, exact_match = geocode_address(address, restaurants)
        recommendations = recommend_restaurants(
            restaurants,
            latitude,
            longitude,
            args.cuisine,
            args.radius,
            args.limit,
            args.min_rating,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        if not args.data_file.exists():
            print(
                "Download the Kaggle business data after accepting its terms and "
                "configuring Kaggle API credentials:\n"
                f"  {KAGGLE_DOWNLOAD_COMMAND}",
                file=sys.stderr,
            )
        return 1

    resolution = "matched address" if exact_match else "ZIP-area estimate"
    print(
        f"Origin resolved as a {resolution} "
        f"({latitude:.5f}, {longitude:.5f})."
    )
    if not exact_match:
        print(
            "The local Yelp file does not geocode this street address; "
            "distances use the median business coordinates for its ZIP area."
        )
    print(
        f"\nTop {len(recommendations)} restaurants matching "
        f"{' or '.join(_cuisine_tokens(args.cuisine))}, rated above "
        f"{args.min_rating:g} within {args.radius:g} miles"
    )
    print("=" * 72)
    print(
        "Note: Yelp's business file does not record spice levels. "
        "Confirm spicy options with the restaurant."
    )
    if not recommendations:
        print(
            "No restaurants matched the cuisine, rating, and radius constraints."
        )
        return 0

    headers = (
        "Rank",
        "Name",
        "Rating",
        "Review Count",
        "Distance",
        "Address",
        "Cuisine",
    )
    rows = [
        (
            str(rank),
            restaurant["name"],
            f"{restaurant['rating']:.1f}/5",
            f"{restaurant['review_count']:,}",
            f"{restaurant['distance']:.2f} mi",
            restaurant["address"] or "Not available",
            restaurant["cuisine"],
        )
        for rank, restaurant in enumerate(recommendations, start=1)
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(
        " | ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
