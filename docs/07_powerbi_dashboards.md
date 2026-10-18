# Power BI dashboards

Three report pages. Each one matches a slide in the CTO deck, so a screenshot can replace the Python chart on that slide (or the page can be shown live).

| Page | Shows | Replaces in the CTO deck |
|---|---|---|
| A · Fleet today | How far the fleet is from the targets, by scale type and setting | slide 4 (and the chart on slide 5) |
| B · Factories | The same setting gives very different results per factory | slide 10 |
| C · KPI 2.0 watch list | Which scales need attention this week | slide 14 (optional) |

## Data

The pipeline writes the Power BI pack to `reports/bossard/powerbi/` (not in the repository: it holds one row per real scale).

| File | Grain | Use |
|---|---|---|
| `fact_device_sensitivity.csv` | one scale on one setting | every page |
| `cleaning_funnel.csv` | one cleaning rule | optional cleaning page |
| `reports/bossard/tables/m2_sensitivity__loop_flags.csv` | one scale on one setting | loop flag for page C |

Import with *File › Options › Current file › Regional settings › Locale for import = English (United States)* so that `0.92` is read as a number.

Power Query steps:
1. `fact_device_sensitivity`: filter `analysis_group = "main"` (this is what the deck uses).
2. `loop_flags`: keep `deviceid`, `sensitivity`, `loop`, `auto_per_day`, `share_retrigger`. Merge into the fact table on `device_id = deviceid` and `sensitivity = sensitivity` (left outer). Replace empty `loop` with `FALSE`.
3. Add a column `setting_label = "Setting " & Text.From([sensitivity])` and sort it by `sensitivity` descending (8 first).

## Measures (DAX)

Put them in a measure table. `F` = `fact_device_sensitivity`.

```DAX
Scales = DISTINCTCOUNT ( F[device_id] )

Typical empty wake-ups per week = MEDIAN ( F[too_many_per_week] )

Typical real picks per week = MEDIANX ( F, DIVIDE ( F[n_auto_ok], F[weeks_observed] ) )

Typical Bossard KPI = MEDIAN ( F[too_many_fraction] )

% meet Bossard empty target = DIVIDE ( COUNTROWS ( FILTER ( F, F[meets_too_many_target] ) ), COUNTROWS ( F ) )

% meet Bossard miss target = DIVIDE ( COUNTROWS ( FILTER ( F, F[meets_too_few_target] ) ), COUNTROWS ( F ) )

-- KPI 2.0 (targets: <= 7 empty wake-ups a week, <= 10% of picks missed, <= 2 missed picks a week)
% KPI2 empty ok = DIVIDE ( COUNTROWS ( FILTER ( F, F[too_many_per_week] <= 7 ) ), COUNTROWS ( F ) )

% KPI2 miss ok = DIVIDE ( COUNTROWS ( FILTER ( F, COALESCE ( F[miss_rate], 0 ) <= 0.10 ) ), COUNTROWS ( F ) )

% KPI2 in target =
DIVIDE (
    COUNTROWS ( FILTER ( F, F[too_many_per_week] <= 7 && COALESCE ( F[miss_rate], 0 ) <= 0.10 && F[too_few_per_week] <= 2 ) ),
    COUNTROWS ( F )
)

Missed picks = SUM ( F[n_too_few] )

Looping scales = CALCULATE ( [Scales], F[loop] = TRUE () )
```

Calculated column for the watch list (the 10% of scales with the most missed picks, per scale type and setting):

```DAX
Watch list =
VAR grp = FILTER ( F, F[scale_type] = EARLIER ( F[scale_type] ) && F[sensitivity] = EARLIER ( F[sensitivity] ) )
VAR r   = RANKX ( grp, F[n_too_few], , DESC, Dense )
RETURN  F[n_too_few] > 0 && r <= 0.10 * COUNTROWS ( grp )
```

## Page A · Fleet today

Slicers: scale type (6 kg / 20 kg), setting (default: 8).

| Visual | Fields | Note |
|---|---|---|
| 4 cards | Typical empty wake-ups per week · Typical real picks per week · % meet Bossard empty target · % meet Bossard miss target | the headline of slide 4 |
| Clustered bar | axis: scale_type · values: the two "% meet" measures | "one-sided problem" |
| Clustered column | axis: setting_label · values: Typical empty wake-ups per week · legend: scale_type | slide 5, left |
| Clustered column + constant line 0.5 | axis: setting_label · values: Typical Bossard KPI · legend: scale_type | slide 5, right: the KPI hardly moves |
| 100% stacked column (optional) | axis: setting_label · legend: too_many_fraction_band · values: Scales | spread inside every setting |

Do not use a dual axis. Turn the setting slicer off for the two column charts (Edit interactions).

## Page B · Factories

Filter: setting = 8, scales per factory >= 20 (visual-level filter on `Scales`).

| Visual | Fields | Note |
|---|---|---|
| Column chart, sorted ascending | axis: location · values: Typical Bossard KPI · constant line 0.5 | one bar per factory, slide 10 |
| Matrix with colour scale | rows: location · columns: setting_label · values: Typical empty wake-ups per week | same setting, different factory |
| Scatter | details: location · x: Typical real picks per week · y: Typical Bossard KPI · size: Scales | busy factories look better on the old KPI |

## Page C · KPI 2.0 watch list

| Visual | Fields | Note |
|---|---|---|
| 3 cards | % KPI2 empty ok · % KPI2 miss ok · % KPI2 in target | KPI 2.0 for the fleet |
| Table, sorted by n_too_few | device_id · location · scale_type · setting_label · too_many_per_week · miss_rate · n_too_few · Watch list · loop | conditional formatting: miss_rate > 0.10 red, too_many_per_week > 7 amber, loop = TRUE red |
| Card | Looping scales | the FLAG step |
| Slicer | Watch list = TRUE | "show only the scales to check" |

## Style

Use the deck theme so screenshots fit: background `#0A1020`, text `#B8C4D9`, 20 kg scales `#5EEAD4`, 6 kg scales `#8EA2FF`, warnings `#FF8FA3`, fonts DM Sans (text) and Space Grotesk (titles). Export each page with *File › Export › PowerPoint › Embed an image* or a screenshot at 1920 × 1080.

## Privacy

The `.pbix` file embeds the data. It stays outside the repository (`*.pbix` is in `.gitignore`). Only screenshots go into the deck.
