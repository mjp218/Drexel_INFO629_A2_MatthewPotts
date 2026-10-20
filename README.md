# Restaurant Recommender Demo

This demo uses the **Yelp Dataset** on Kaggle:
[kaggle.com/datasets/yelp-dataset/yelp-dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset).
It reads only `yelp_academic_dataset_business.json`; the business records
provide restaurant categories, coordinates, star ratings, review counts, and
addresses. The Yelp dataset's terms of use apply.

## Get the data

Accept the dataset terms on Kaggle and configure the Kaggle API credentials
for your account. Then install the Kaggle CLI and download just the business
file:

```sh
python -m pip install kaggle
kaggle datasets download -d yelp-dataset/yelp-dataset \
  -f yelp_academic_dataset_business.json -p . --unzip
```

The downloaded file should be at
`yelp_academic_dataset_business.json` in the project directory.
The full user and review files are not needed for this demo.

## Run

```sh
python -m pip install -r requirements.txt
python RestaurantReviewData.py \
  --address "address line 1, city, state, ZIP code" \
  --cuisine Indian,Mexican --radius 100 --min-rating 4 --limit 100
```

Enter your origin as an address line 1, city, state, and ZIP code. The address
can be supplied with `--address` or entered when prompted. To avoid sending the
address to an external geocoding service, the program resolves it against the
local Yelp business file. If the exact street address is present, its recorded
coordinates are used; otherwise it uses the median business coordinates for
the matching city/state/ZIP area, which is an approximation. The defaults
include either Indian or Mexican cuisine, a 100-mile radius, a strict rating
above 4, and up to 100 recommendations. Pass `--help` for all options. Results
must be categorized as both `Restaurants` and at least one requested cuisine,
have a rating strictly above the minimum, and fall inside the Haversine
distance radius. OR-Tools CP-SAT selects up to the requested number,
prioritizing star rating in descending order and distance in ascending order.
Review count is shown as context but does not affect rank.

## Method and basic tests

This is a **knowledge-based, constraint-based recommender** rather than a
collaborative-filtering model. A user's location, Indian or Mexican cuisine
preference, 100-mile radius, and minimum rating are explicit constraints. Business
categories and coordinates determine cuisine and distance eligibility; Yelp
stars and review counts provide the ranking evidence. No user-item rating
matrix is available or needed for this task. The business file does not contain
a spice-level field, so the recommender cannot verify whether a restaurant
serves spicy food; check its menu or contact the restaurant.

The tests use small synthetic business records to check distance handling,
multi-cuisine/rating constraints, ranking, and the result limit. Run them with:

```sh
python -m unittest
```
