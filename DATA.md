# Data

This document explains where the REFIT data comes from, what it really contains, and why each cleaning decision was made. The numbers come from pipeline outputs in [outputs/reports/](outputs/reports/). The full figures are in [cleaning_report.md](outputs/reports/cleaning_report.md).

Findings from the data are in [RESULTS.md](RESULTS.md), and what they mean in practice is in [Recommendation.md](Recommendation.md).

## Key points

- Raw House 1 is cleaned to a 1-minute table of 920,031 minutes, with nothing filled or interpolated.
- Raw timestamps are UK local time, despite the documented "UCT". The official cleaned files use a clock that changes during the study, so their hour-of-day values need correcting.
- `Issues` depends on the appliance readings. It flags up to 69% of high-power washing machine rows, so discarding those rows would bias the data, and it must never be a model feature.
- Sensors sometimes freeze on one value, which missing-value checks can't detect. House 10's official aggregate is frozen for 45.8 days. Frozen minutes stay flagged in the data, and are excluded from primary analysis and modelling.

## Sources

| Dataset | Files used | DOI |
|---|---|---|
| REFIT raw | `REFIT_RAW_081116.7z`, `RAW_READ_ME_081116.txt` | [10.15129/31da3ece-f902-4e95-a093-e0a9536983c4](https://doi.org/10.15129/31da3ece-f902-4e95-a093-e0a9536983c4) |
| REFIT cleaned | `CLEAN_REFIT_081116.7z`, `CLEAN_READ_ME_081116.txt`, `MetaData_Tables.xlsx` | [10.15129/9ab14b0e-19ac-4279-938f-27f643078cec](https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec) |

Both are published by the University of Strathclyde under CC BY 4.0. The sources ask for different citations, so both are given:

- Murray, D., Liao, J., Stankovic, L., Stankovic, V., Hauxwell-Baldwin, R., Wilson, C., Coleman, M., Kane, T. and Firth, S. (2015). "A data management platform for personalised real-time energy feedback." *Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting.* Requested by the readmes.
- Murray, D., Stankovic, L. and Stankovic, V. (2017). "An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study." *Scientific Data.* [10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122). Requested by the dataset pages. This paper could not be read during this work, because automated access was blocked. Everything here rests on the readmes, the metadata and the data itself.

## Getting the data

The Strathclyde file links require a normal browser, so download these five files by hand from the [raw](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) and [cleaned](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) dataset pages:

| File | Size | Location | SHA-256 |
|---|---|---|---|
| `REFIT_RAW_081116.7z` | 666 MB | `data/raw/archives/` | `f294fe673532220dc527b5e7d7a25e99568bd24acf9fb5be0a0def14af64b2cf` |
| `CLEAN_REFIT_081116.7z` | 490 MB | `data/raw/archives/` | `4dc4dde763da1c843e9d565fc68f1a63752d858980615635994d7f701fdfcc4b` |
| `RAW_READ_ME_081116.txt` | 10.8 KB | `data/raw/docs/` | `6b20b821a36af681e460bde5d24b933aa58e191c4c3f8de05218941e6d34937d` |
| `CLEAN_READ_ME_081116.txt` | 11.7 KB | `data/raw/docs/` | `c5f5236fd5a7b29771913480fcef6e9be8a85d07511939c5a35160b7afc97bfb` |
| `MetaData_Tables.xlsx` | 28.5 KB | `data/raw/docs/` | `3d071013b178409e543d558e837ef806f1f2b108f665ab78e015a4411519fccb` |

The older files on those pages (`REFITPowerData111215.7z`, `Processed_Data_CSV.7z` and their readmes) are not used.

The pipeline stops if any checksum differs. It never writes to the archives, and it extracts only the two raw House 1 files into `data/raw/refit_raw/`. Other files are read through memory, because the archives unpack to 8.31 GB and 6.94 GB.

## Reproducing the results

Tested on Python 3.14.4 with the exact versions in [requirements.txt](requirements.txt).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline                 # all stages
python -m src.pipeline --stage clean   # or any one stage
python -m pytest                       # tests
```

[README.md](README.md) lists every stage and its run time. Each stage reads only files on disk and names the stage to run first if an input is missing. Rerunning `clean` produces byte-identical Parquet files.

The `inspect` stage regenerates the [dataset inventory](outputs/reports/dataset_inventory.csv) and [household screening](outputs/reports/household_screening.csv).

## Selected households

Washing machine channels were checked against both readmes, the Excel metadata and the data. The sources agree, and each channel shows the same pattern: a heating plateau of about 2 kW, long low-power tumbling, then a spin rise. Machine details and the full screening are in [household_screening.csv](outputs/reports/household_screening.csv).

| House | Washing machine channel | Occupants | Machine | Main limitation |
|---|---|---|---|---|
| 1 | Appliance 5 | 2 | Beko WMC6140 | Weakest of the six: see below |
| 2 | Appliance 2 | 4 | LG F1289TD | Most missing data, with 465 data days in about 617 |
| 8 | Appliance 4 | 2 | Not documented | Washer-dryer in the aggregate, 1.36 uses a week |
| 10 | Appliance 5 | 4 | Beko WI1382 | Aggregate frozen for 45.8 days |
| 15 | Appliance 3 | 1 | Beko WMB91242LB | Tumble dryer in the aggregate, about once a week |
| 16 | Appliance 5 | 6 | Bosch WAB24262GB/01 | Heaters and dehumidifier in the aggregate, 212 rows above 23 kW |

**House 1 limitations.** House 1 is included because it links directly to the raw cleaning work, despite being the weakest of the six:
- At heating edges the aggregate rises only 0.85 W per 1 W of washing machine, so the two sensors disagree on scale.
- The sensors' total exceeds the aggregate in 19.4% of heating minutes, close to the 20% screening limit, which was itself set after viewing the data.
- `Issues` flags 26.55% of rows with the machine at 500 W or more.
- A washer-dryer is also present, though rarely used.

**Not selected.**
- **House 12:** no documented washing machine.
- **House 4:** two active machines.
- **House 11:** the aggregate barely registers its machine.
- **House 13:** an official change note on its channel, and a "Freezer" channel that behaves like a dryer.
- **Houses 3, 5, 6, 19, 21:** the aggregate tracks the machine too poorly to model.
- **Houses 7, 9, 17, 18, 20:** pass every check and are held in reserve.

The six were chosen by screening on the data quality of all 20 houses, so they are not a random sample. These checks can disprove a label but can't prove one. The readmes warn that occupants may have moved or swapped appliances.

## Where the documents and the data disagree

| Documents say | The data shows |
|---|---|
| Timestamps are "UNIX TIMESTAMP (UCT)" | Raw `Time` and `Unix` are both UK local clock time |
| Cleaned files correct for daylight saving | Only part of the study (see "The official cleaned clock") |
| Dataset page: gaps over 2 minutes were zeroed | Whole-house outages are missing rows. In House 1, a silent washing machine sensor is held at its last value at any length |
| Data was recorded only on a change in load | Part 2 `NaN`s look like missed reports: 7.27% after an active washing machine reading, 8.23% after an off reading |
| Sampled every 8 seconds | Part 2 polls about 9 times a minute, but Part 1 only about 4, repeated across rows |
| IAM spikes above 4000 W were removed | True, but bit errors below 4000 W survive, as do aggregate spikes up to 68 kW across houses |
| Study period 2013 to 2014 | Data runs to July 2015 |

## The `Issues` column

The cleaned readme says `Issues` "is set to 1 if the sum of the sub-metering (IAMs) is greater than that of the household aggregate", and that such data "should be discarded or noted". It only exists in the cleaned files. It isn't a missing-data flag.

This rule reproduces the column on 100% of rows in all 20 houses. It flags 0.24% to 5.84% of all rows, but far more when the machine is running:

| House | All rows flagged | Rows flagged with the machine at 500 W or more |
|---|---|---|
| 1 | 0.84% | 26.55% |
| 2 | 0.50% | 1.97% |
| 8 | 0.41% | 4.89% |
| 10 | 0.45% | 3.07% |
| 15 | 0.38% | 9.09% |
| 16 | 0.26% | 2.10% |

**Decision:** keep these rows and note them. Evaluation will report metrics on all rows and again with `Issues = 1` excluded, and explain the differences. `Issues` is calculated from the appliance readings, so it is never a model feature.

## Raw House 1

The raw data is two CSVs with CRLF line endings. Columns are `Time, Unix, Aggregate, Appliance1..9` in whole watts, with no negative values.

| | `RAW_House1_Part1.csv` | `RAW_House1_Part2.csv` |
|---|---|---|
| Raw clock range | 2013-10-09 to 2014-10-01 | 2014-06-27 to 2015-07-10 |
| Rows | 4,377,877 | 4,140,886 |
| Unavailable sensor recorded as | 0 | `NaN` |
| Rows repeating a timestamp | 1,004,404 | 14,755 |
| Independent polls per minute | 4 | 9 |
| Rows with an appliance above 4000 W | 3,038 | 911 |
| Washing machine `NaN` | 0% | 12.01% |

### Row counts through cleaning

| Step | Rows |
|---|---|
| Read from both parts | 8,518,763 |
| Part 1 rows not used, at or after the switch | -1,257,865 |
| Exact duplicates removed (508,381 from Part 1, 393 from Part 2) | -508,774 |
| Rows sharing a timestamp, averaged into one | -87,954 |
| **Unique timestamps** | **6,664,170** |
| **1-minute rows, including empty minutes** | **920,031** |

### Cleaning decisions

| Decision | Evidence | Effect |
|---|---|---|
| **Timestamp key is `Unix`.** `Time` stays in the source files. | `Time` and `Unix` differ by 1 second in about half of rows. | Rows are grouped by one consistent label. |
| **Convert UK local time to UTC, keeping the raw label.** | The raw clock jumps back from 01:59:58 to 01:00:01 on 2013-10-27, and the 01:00 hour is missing on 2014-03-30 and 2015-03-29. | The UTC timeline is continuous, so spring changes create no false outage. |
| **Part 1 before the first Part 2 timestamp, Part 2 from there.** The switch is 2014-06-27 21:57:49 raw clock. A midnight cut would discard about 22 hours of good data. | The step at the switch is 7.1 W against a 95th-percentile step of 151.7 W, and lag is 0 on the first full day. Nine of the ten sensors match themselves across the boundary, and Appliance 9 can't be tested because Part 2 barely reports it. Part 2 also separates silence from zero. | Gains 7,085 aggregate minutes only Part 2 has, and gives up 4,549 only Part 1 has. |
| **Remove exact duplicates,** meaning rows identical in timestamp and all ten readings. | They carry no extra information. | 508,774 rows removed. |
| **Average conflicting readings at one timestamp, sensor by sensor.** A `NaN` never erases a reading. | See "Why conflicts are averaged". | 80,478 conflicts, across 72,871 timestamps and 61,952 minutes, are all logged with their resolved value. |
| **Appliance readings above 4000 W become missing and are counted.** The official files set them to 0, but zero would claim the appliance was off. | The readme removes spikes above 4000 W, and the dataset page calls this the sensor's rated limit. 59% to 69% of over-limit readings in the raw files are exact bit patterns such as 4096 and 8192. | 2,622 readings removed. In 175 sensor-minutes, no reading remained. |
| **Aggregate above 23 kW is kept, flagged and investigated.** | See "Aggregate above 23 kW". | 63 readings across 24 minutes. |
| **1-minute values are the mean of available readings, with no filling or interpolation.** `_n` holds the sample count and `_missing` the indicator. | A typical minute has several readings. Holding values creates load that wasn't measured (see "Comparison with the official file"). | Minutes with no reading stay `NaN`. |
| **Part 1 zeros are preserved.** `source_part` records the origin. | The readme says Part 1 sets unavailable sensors to 0, and the data can't separate that from off. | 310,522 minutes come from Part 1. |
| **Outages stay missing and are flagged,** with run length per sensor. | The readme describes outages from internet and hardware failures. | 82.99 days across 1,012 runs, of which 448 last 3 minutes or more and are `long_outage`. The longest is 41.6 days from 2014-01-24. |
| **Clock changes are handled explicitly.** Autumn 2013's hour is split by file order. Autumn 2014's is flagged. | In Part 1 a backward step separates the passes. In Part 2 they are interleaved and can't be split reliably. | 120 minutes are flagged `dst_ambiguous`. In autumn 2014, all readings sit in the first pass, and the second pass stays empty and isn't counted as an outage. |
| **The Part 2 drift period is flagged, not corrected.** | See "Clock uncertainty between the parts". | 79,200 minutes are `clock_uncertain`. |
| **Frozen sensors are flagged, not removed.** Added during this work and awaiting a decision. | See "Frozen sensors". | Values are unchanged. |

### Why conflicts are averaged

Rows sharing a `Unix` label often hold different readings: the washing machine shows 3 W and 2,372 W under one label on 2014-08-04, with `Time` one second apart. The evidence says these are real, consecutive readings, not errors:

- For the washing machine in Part 1, the first reading matches the previous timestamp 28.9% of the time, against 5.4% for the last. The last matches the next timestamp 30.7% of the time, against 4.8% for the first.
- `Time` differs under one `Unix` label at 64.8% of Part 1 conflict timestamps and 100% of Part 2's.
- Choosing first or last could move a washing machine minute by up to 552 W. The typical effect of a merge is about 3 W, and total energy changes by about 0.02%.

Averaging keeps every reading and doesn't depend on row order. The log is `data/cleaned/house_1_duplicate_conflicts.parquet`.

### Aggregate above 23 kW

23 kW is about what a 100 A supply can deliver at 230 V. That threshold is a general assumption, not from the REFIT documents. Nothing is deleted. The readings fall into two groups:

- **Sustained high load:** 59 readings, 50 of them at exactly 24,000 W, in episodes where neighbouring readings are also about 9 to 23 kW, while metered appliances total only about 30 W to 2.2 kW. This looks like a sensor ceiling during some large unmetered load.
- **Isolated single readings:** 4 readings of 24.7 to 29.2 kW that return to 155 to 480 W on the next reading. These look like sensor errors.

The flagged readings raise the affected minutes' means by a median of 2,257 W, and at most 4,729 W. This is unresolved.

### Clock uncertainty between the parts

From about 11 July to 27 August 2014, Part 2 runs later than Part 1 by a growing amount, up to 7 minutes, then snaps back. Part 1's row rate stays flat at the snap on 7 August while Part 2's wobbles, which suggests Part 2 is the late one, but this isn't proven. Houses 1, 2, 8 and 10 all show a 6-minute lag on 5 August, so the cause looks like shared logging. Sensors stay aligned within each part.

This only affects timing by minutes, so it has little effect on hourly results. It remains in the cleaned table, flagged `clock_uncertain` from 2014-07-11 to 2014-09-03. The pipeline stops if measured drift of 2 minutes or more falls outside that window.

### Frozen sensors

A sensor repeating one exact 1-minute mean for 60 minutes or more is flagged `<sensor>_frozen`. For appliances this applies only at 20 W or above, because a flat reading is normal when idle. Values stay in the table.

The clearest case is 8 February 2015, when several sensors froze together for 2.6 hours, with the aggregate at 12,505 W and the washing machine at 2,256 W. Those 158 minutes hold 5.94 kWh, or 4.17% of the washing machine's measured energy. They are almost certainly false, and they would count as real usage in any analysis that ignores the flag.

The flag means "needs review", not "wrong". The electric heater's flat 1,020 W for 5 hours could be a steady load.

The official cleaned files have the same problem, in their aggregate:

| House | Frozen periods | Total days | Share of minutes | Longest |
|---|---|---|---|---|
| 1 | 3 | 0.4 | 0.1% | 0.21 days |
| 2 | 6 | 3.5 | 0.8% | 1.77 days |
| 8 | 21 | 3.9 | 0.8% | 1.78 days |
| 10 | 30 | 45.8 | 8.8% | 10.03 days |
| 15 | 2 | 3.7 | 0.7% | 1.89 days |
| 16 | 10 | 8.2 | 1.8% | 2.81 days |

House 10's aggregate sits at exactly 2,169 W for about 30 days around March 2014 and at 602 W for about 12 days in June 2015, with its washing machine reading zero in both stretches. They would look like ordinary "no wash" time to a model. Houses 2, 8, 10, 15 and 16 also share a freeze of about 1.8 days from 30 August 2014, when Part 1 had an outage. Every period is in [official_frozen_aggregate_periods.csv](outputs/reports/official_frozen_aggregate_periods.csv).

### Comparison with the official file

Compared on the same clock, using days with a usable clock offset:

| Period | Aggregate r | Washing machine r | Washing machine mean abs diff |
|---|---|---|---|
| Both from Part 1, to 2014-06-27 | 0.9994 | 0.9996 | 0.14 W |
| Ours from Part 2, official mostly from Part 1 | 0.7935 | 0.8037 | 3.99 W |
| Both from Part 2, from 2014-10-01 | 0.9992 | 0.9983 | 0.27 W |

Agreement is close where both use the same part. The middle period is lower because of the clock drift between parts.

Two differences are deliberate:
- **Spikes:** in the 62 washing machine minutes with a spike removed, the official file reads exactly 0 in 52 of them. This table holds the remaining readings, or leaves the minute empty if none remain.
- **Silent sensors:** while both use Part 2, the official file has a washing machine value in 7,505 minutes where the sensor reported nothing here. 567 of those are non-zero, totalling 2.36 kWh that was held and not measured. Across 1,440 silences while other sensors kept reporting, the official file repeats one value in 1,439, including all 18 lasting an hour or more. It repeats a non-zero value in 50 of them.

## Output table

`data/cleaned/house_1_1min.parquet` has one row per UTC minute from 2013-10-09 13:06 to 2015-07-10 10:56.

| Column | Meaning |
|---|---|
| `timestamp_utc` | The minute in UTC, continuous and unique |
| `raw_clock_minute` | The same minute on the raw file's UK local clock. Repeats in autumn |
| `utc_offset_min` | 0 for GMT, 60 for BST |
| `source_part` | 1 or 2. Empty if there are no rows. The one minute containing the switch reads 2 |
| `dst_ambiguous` | In a clock hour whose two passes couldn't be separated |
| `clock_uncertain` | In the Part 2 drift window |
| `Aggregate`, `Appliance1`..`Appliance9` | Mean watts of available readings. `NaN` if none |
| `<sensor>_n` | Number of timestamps with a reading in that minute |
| `<sensor>_missing` | True if the sensor had no reading |
| `<sensor>_gap_minutes` | Length of the missing run containing this minute, otherwise 0 |
| `<sensor>_frozen` | In a frozen run (values unchanged) |
| `Appliance*_spikes` | Readings above 4000 W removed from that minute |
| `Aggregate_over_23kw` | Aggregate readings above 23 kW in that minute (values kept) |
| `n_conflicts` | Conflicts resolved in that minute |
| `outage_minutes` | Length of the whole-house outage containing this minute, otherwise 0 |
| `long_outage` | True if that outage is 3 minutes or longer |

House 1 appliances: 1 Fridge, 2 Freezer (1), 3 Freezer (2), 4 Washer Dryer, 5 Washing Machine, 6 Dishwasher, 7 Computer, 8 Television Site, 9 Electric Heater.

## Prepared official data

Usage analysis and modelling use the official cleaned files, read from the archive in memory and never changed. The `prepare` stage writes one table per household to `data/cleaned/official_house_<n>_1min.parquet`.

| Column | Meaning |
|---|---|
| `official_minute` | The minute on the official file's own clock, exactly as supplied |
| `uk_local_minute` | Derived UK local time, for hourly analysis only. Empty on transition and uncertain days |
| `clock_status` | `verified`, `inferred`, `transition`, `uncertain` or `no_offset` |
| `aggregate_w`, `washing_machine_w` | Mean watts of the rows in that minute. Empty if there are none |
| `other_appliance<n>_w` | Other channels, used only to explain false alarms and never as model inputs |
| `n_readings` | Rows in that minute |
| `issues_rows`, `issues_any` | Rows with `Issues` = 1, and whether there were any |
| `aggregate_over_23kw` | Aggregate readings above 23 kW in that minute |
| `missing` | No rows, so no values |
| `aggregate_frozen`, `washing_machine_frozen`, `frozen` | One exact value for 60 minutes or more. Machine frozen only counts at 20 W or above |
| `valid` | Not missing and not frozen. This defines the primary analysis |

**Decisions, with the effect on the data:**

- **1-minute resolution:** the mean of the official rows in each minute, on a complete grid with nothing filled. The aggregate and appliances come from the same rows, so they stay aligned.
- **Frozen minutes:** excluded from primary analysis and modelling, and kept in the tables. Their values also become missing before model features are calculated, so they can't enter the rolling windows of nearby minutes. [RESULTS.md](RESULTS.md) compares results with and without this exclusion.
- **`Issues`:** kept in primary analysis. A minute counts as flagged if any of its rows has `Issues` = 1. It is used only for the secondary metrics, and never as a model feature.
- **Clock:** the correction is used only for hour and weekday results. Model training and cycle detection use the official row order, which keeps aggregate and appliance readings aligned.

| House | Minutes in span | Missing | Frozen, excluded | Valid minutes |
|---|---|---|---|---|
| 1 | 920,091 | 116,318 | 644 | 803,129 |
| 2 | 889,078 | 218,914 | 5,075 | 665,089 |
| 8 | 799,284 | 82,063 | 5,622 | 711,599 |
| 10 | 845,257 | 94,671 | 65,908 | 684,678 |
| 15 | 817,007 | 111,626 | 5,493 | 699,888 |
| 16 | 782,833 | 135,663 | 11,808 | 635,362 |

## The official cleaned clock

Measured by correlating each household's official and raw aggregate day by day. The official files shift at raw midnight on clock-change days, and whenever their data source changes part. House 1 shows the pattern, and the other houses differ by when their data starts and stops:

| Official clock period, House 1 | Official minus raw | To get UK local |
|---|---|---|
| Start to 2013-10-26 | -60 min | Add 60 min |
| 2013-10-27 to 2014-03-29 | 0 | No change |
| 2014-03-30 to 2014-08-30 | -60 min | Add 60 min |
| 2014-09-01 to 2014-09-03 | 0 | No change |
| 2014-09-05 to 2014-09-30 | -60 min | Add 60 min |
| 2014-10-01 to end | 0 | No change |

**How it is used.** For hourly analysis only, a derived UK local time is added by the `prepare` stage. Official files and their timestamps are never modified, so aggregate and appliance values stay on the same row, and modelling isn't affected. In the household screening, the aggregate and all 20 washing machine channels stay aligned at lag 0 in every period tested.

**Uncertain periods.** Days are classed in [clock_daily_offsets.csv](outputs/reports/clock_daily_offsets.csv), and periods are in [clock_boundaries.csv](outputs/reports/clock_boundaries.csv):

| Status | Days | Use in hourly analysis |
|---|---|---|
| `verified` | 2,946 | Used |
| `inferred` | 115 | Used, with the offset taken from agreeing neighbouring days |
| `transition` | 48 | Left out, because the day can hold two offsets |
| `uncertain` | 64 | Left out, usually because a frozen aggregate gives nothing to correlate |

Specific oddities in House 1's official file include what looks like an interleaved hour at 23:00 on 2014-03-29, a stretch of Part 2 data at a different offset from about 1 to 4 September 2014, and a change of source and offset around midnight on 2014-10-01. The 2-hour evidence is in the cleaning report.

**Validation.**
- After correction, clock-change hours match raw in 18 of 19 household and date combinations with data. The exception is House 10 in spring 2014, when its aggregate was frozen.
- Against cleaned raw House 1, the corrected official clock has lag 0 in every month with data. Correlation is at least 0.97, except July and August 2014, where it is about 0.71 because of the Part 2 drift.
- The same-season routine shift for the corrected official clock equals raw in all six houses. This check is coarse, because routines also change between years.

## Open issues

- **Frozen sensors:** runs shorter than 60 minutes aren't flagged, and a held value of zero can't be told apart from an idle machine.
- **Bit errors below 4000 W:** values such as 1024 and 2048 aren't removed. They are rare in the washing machine channel, but implausible values remain elsewhere, such as a fridge minute at 3,072 W.
- **Aggregate above 23 kW:** the isolated single readings look like errors and raise some minute means.
- **Part 2 drift:** up to 7 minutes in summer 2014, with the cause unproven.
- **Part 1 zeros:** off and unavailable can't be separated.
- **Part 1 sampling:** about 4 independent readings a minute before the switch, against about 9 after.
- **Electric heater in House 1:** Part 2 has almost no readings until late 2014. Part 1 shows it at about 1 W during the overlap, so little is lost.
