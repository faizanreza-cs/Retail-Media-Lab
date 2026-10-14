# Power BI report kit

Everything needed to build the Retail Media Lab report in Power BI Desktop.

## 1. Export the data

```bash
rml build            # if you haven't already
rml export-powerbi   # writes powerbi/data/*.csv
```

| Table | Grain | Use |
|---|---|---|
| `dim_date` | day | date table (mark it as the date table on `date`) |
| `dim_geo` | geo | region, size, holdout or control group |
| `dim_channel` | channel | names, funnel stage, sort order |
| `fact_media_daily` | day x geo x channel | spend, impressions |
| `fact_sales_daily` | day x geo | sales, orders |
| `fact_attribution` | method x channel | credited purchases and share, including the truth |
| `fact_channel_roas` | channel | MMM ROAS and marginal ROAS with intervals, true values |
| `fact_response_curve` | channel x spend level | MMM response curves |
| `fact_budget_plan` | channel | current vs recommended daily spend |
| `fact_mmm_contribution` | week x component | weekly sales by baseline and channel |
| `fact_geo_test_weekly` | week | holdout geos vs synthetic control |

## 2. Load and model

1. **Get data > Text/CSV** for each file. In Power Query, set `date` and `week_start` to Date.
2. **Mark `dim_date` as a date table** (Table tools > Mark as date table, column `date`).
3. **Relationships** (single direction, many-to-one):
   - `fact_media_daily[date]` and `fact_sales_daily[date]` to `dim_date[date]`
   - `fact_media_daily[geo_id]` and `fact_sales_daily[geo_id]` to `dim_geo[geo_id]`
   - `channel` in `fact_media_daily`, `fact_attribution`, `fact_channel_roas`,
     `fact_response_curve` and `fact_budget_plan` to `dim_channel[channel]`
4. Sort `dim_channel[channel_name]` by `dim_channel[sort_order]`.
5. **Measures:** create each one from [`measures.dax`](measures.dax) (Modeling > New measure).
6. **Theme:** View > Themes > Browse for themes > [`theme.json`](theme.json). The colours are a
   colour-blind-checked palette; keep labels on, since two of the colours are light.

## 3. Suggested pages

| Page | Visuals |
|---|---|
| Overview | Cards: Total Spend, Total Sales, Blended ROAS. Line: Total Spend by `week_start` and `channel_name`. Bar: Spend Share by channel. |
| Attribution | Clustered bar: Attribution Share by `channel_name`, legend `method`. Highlight `truth`. |
| Geo test | Line: Holdout Sales and Synthetic Control Sales by `week_start`, shaded where `in_test`. Card: Holdout Gap %. |
| Mix model | Error bars: MMM ROAS with MMM ROAS Low/High and True ROAS by channel. Stacked area: Modelled Contribution by `week_start`, legend `component`. Line: `fact_response_curve` incremental sales by daily spend, legend channel. |
| Budget | Clustered bar: Current vs Recommended Daily Spend by channel. Card: Expected Incremental Sales Change. |

Add your screenshots to `docs/img/` once the report is built.
