# Mushroom Forecast

A daily, colour-coded map of where mushrooms are most likely to be fruiting,
built from habitat (trees or grassland, soil acidity, altitude, damp ground) and
recent weather (rain in the right window, temperature, frost).

**Live map:** https://rkmvzm9nmd-cpu.github.io/mushroom-forecast/

Covered species are listed in `config/species.yaml`; regions in `config/regions.yaml`.

## How it works

A GitHub Actions job (`.github/workflows/daily.yml`) runs every morning:

1. **Habitat layers** (built once, then cached): Copernicus DEM elevation, ESA
   WorldCover land cover, ISRIC SoilGrids topsoil pH, and in France the IGN
   BD Forêt tree-species map.
2. **Weather**: Open-Meteo past 31 days + 9-day forecast on a ~10 km lattice,
   interpolated with an altitude correction.
3. **Scores**: habitat × weather per species and day, written as PNG layers.
4. **Checks**: public GBIF/iNaturalist records are used to test how well the
   habitat map picks out real finds (shown in each species' info panel).
5. **Alerts**: optional push notifications for saved spots (see below).
6. The site in `site/` plus the generated data is published to GitHub Pages.

The build log of the latest run is at `data/status.json` on the live site.

## Alerts (optional)

1. Install the free **ntfy** app and subscribe to a long, hard-to-guess topic name.
2. In the repo: Settings → Secrets and variables → Actions → New repository secret:
   - `NTFY_TOPIC` = your topic name
   - `SPOTS_JSON` = paste from the map's *Saved spots → Copy spots for alerts*
3. You'll get one message on mornings when a saved spot scores well in the next 9 days.

## Adding a region

Copy the block in `config/regions.yaml`, change `id`, `name`, `bbox` and
`timezone`, and set `bdforet: false` outside France. Push, and the next run builds it.

## Caveats

These are rule-of-thumb models, not guarantees. Never eat a mushroom based on
this map; get finds checked by an expert. Psilocybe species are controlled
substances in France: observe only.
