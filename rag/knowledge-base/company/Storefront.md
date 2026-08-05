# Storefront

Fretwork's own web storefront, built and run by Priya Raghunathan's engineering
team. It replaced an off-the-shelf shop in 2021.

## What it does

- Sells the six instrument lines and the accessories range, with options —
  string gauge, finish, left-handed, hard case — priced and configured in the
  basket.
- Quotes a build-to-ship date per instrument, taken live from the stock system
  rather than a generic "ships in 2-4 weeks".
- Runs the guided recommender: five questions that end in a line and a gauge,
  built after support noticed most tickets were "which guitar should I buy".
- Carries the setup card archive, so a customer can pull up the card for their
  own serial number.

## How it is built

- Python and Postgres, deployed on two servers in London and one in Frankfurt.
- The stock system is the same service: it holds workshop capacity, the bench
  queue, and warehouse stock, and every promised date comes from it.
- Payments are the one third-party dependency, and are isolated so the rest of
  the site stays up if the provider is down.

## Notes

- Hard rule from Priya Raghunathan: no customer-facing page may depend on a
  third-party service that Fretwork cannot run without.
- Peak traffic is Black Friday week and the first week of January.
- The recommender's conversion is reported monthly next to the return rate,
  because the point of it was fewer returns rather than more orders.
